"""Stage-by-stage sample export and diagnosis for alpha debugging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB-like frame for debug export, got shape {rgb.shape}")
    if rgb.shape[2] > 3:
        rgb = rgb[..., :3]
    out = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        out /= float(np.iinfo(rgb.dtype).max)
    elif float(out.max(initial=0.0)) > 1.0:
        out /= max(float(out.max(initial=1.0)), 1.0)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def _compute_alpha_metrics(alpha: np.ndarray, rgb_u8: np.ndarray | None = None) -> dict[str, float]:
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    h, w = a.shape[:2]
    hw = max(1, h * w)
    fg = (a >= 0.5).astype(np.uint8)
    fg_pixels = int(fg.sum())
    coverage = float((a > 0.01).mean())
    edge_cov = float(((a > 0.05) & (a < 0.95)).mean())

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if num_labels > 1 else np.asarray([], dtype=np.int64)
    num_components = int(len(areas))
    largest_area = int(areas.max()) if areas.size else 0
    largest_component_ratio = float(largest_area / max(1, fg_pixels))

    small_threshold = max(16, int(0.00005 * hw))
    small_area = int(areas[areas < small_threshold].sum()) if areas.size else 0
    speckle_ratio = float(small_area / max(1, fg_pixels))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    hole_pixels = max(0, int(closed.sum()) - fg_pixels)
    hole_ratio = float(hole_pixels / max(1, fg_pixels))

    metrics = {
        "coverage": coverage,
        "edge_band_coverage": edge_cov,
        "fg_pixels": float(fg_pixels),
        "num_components": float(num_components),
        "largest_component_ratio": largest_component_ratio,
        "speckle_ratio": speckle_ratio,
        "hole_ratio": hole_ratio,
    }
    if rgb_u8 is not None:
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)

        agx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        agy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        agrad = np.hypot(agx, agy)
        alpha_edge = agrad > 0.03
        rgb_edge = grad > np.percentile(grad, 85)
        rgb_edge_dil = cv2.dilate(rgb_edge.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        edge_alignment = float((alpha_edge & rgb_edge_dil).sum() / max(1, alpha_edge.sum()))

        boundary_grad = float(grad[alpha_edge].mean()) if alpha_edge.any() else 0.0
        global_grad = float(grad.mean())
        boundary_grad_ratio = float(boundary_grad / max(global_grad, 1e-6))

        fg_soft = a > 0.5
        fg_eroded = cv2.erode(fg_soft.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1) > 0
        fg_solidity = float(a[fg_eroded].mean()) if fg_eroded.any() else 0.0

        fg_dilated = cv2.dilate(fg_soft.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1) > 0
        outer_bg = ~fg_dilated
        outer_bg_leak = float(a[outer_bg].mean()) if outer_bg.any() else 0.0

        metrics.update(
            {
                "edge_alignment": edge_alignment,
                "boundary_grad_ratio": boundary_grad_ratio,
                "fg_solidity": fg_solidity,
                "outer_bg_leak_mean_alpha": outer_bg_leak,
            }
        )
    return metrics


def resolve_sample_local_frames(
    num_frames: int,
    frame_start: int,
    sample_frames: list[int] | None,
    sample_count: int,
) -> list[int]:
    if num_frames <= 0:
        return []

    resolved: set[int] = set()
    for raw in sample_frames or []:
        frame = int(raw)
        for cand in (frame, frame - int(frame_start)):
            if 0 <= cand < num_frames:
                resolved.add(int(cand))

    if not resolved:
        count = max(1, int(sample_count))
        if count >= num_frames:
            resolved = set(range(num_frames))
        else:
            positions = np.linspace(0, num_frames - 1, num=count, dtype=int).tolist()
            resolved = set(int(p) for p in positions)

    return sorted(resolved)


def _save_overlay(
    rgb_u8: np.ndarray,
    alpha: np.ndarray,
    out_path: Path,
) -> None:
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    overlay = rgb_u8.astype(np.float32).copy()
    overlay[..., 1] = overlay[..., 1] * (1.0 - 0.6 * a) + 255.0 * (0.6 * a)
    edge = (a > 0.05) & (a < 0.95)
    overlay[edge] = np.array([255.0, 48.0, 48.0], dtype=np.float32)
    ov_u8 = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
    ok = cv2.imwrite(str(out_path), cv2.cvtColor(ov_u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to write debug overlay: {out_path}")


def export_stage_samples(
    *,
    output_dir: Path,
    stage_dir_name: str,
    stage_name: str,
    source: Any,
    alphas: list[np.ndarray],
    sample_local_frames: list[int],
    frame_start: int,
    confidences: list[np.ndarray] | None = None,
    save_rgb: bool = True,
    save_overlay: bool = True,
) -> dict[int, dict[str, float]]:
    stage_root = output_dir / stage_dir_name
    alpha_dir = stage_root / stage_name
    rgb_dir = stage_root / "rgb"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    if save_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    stage_metrics: dict[int, dict[str, float]] = {}
    for local_idx in sample_local_frames:
        if local_idx < 0 or local_idx >= len(alphas):
            continue
        abs_idx = int(frame_start + local_idx)
        alpha = np.clip(np.asarray(alphas[local_idx], dtype=np.float32), 0.0, 1.0)
        alpha_u16 = (alpha * 65535.0).round().astype(np.uint16)

        alpha_path = alpha_dir / f"{abs_idx:06d}_alpha.png"
        ok = cv2.imwrite(str(alpha_path), alpha_u16)
        if not ok:
            raise IOError(f"Failed to write stage alpha sample: {alpha_path}")

        rgb_u8 = _to_rgb_u8(source[local_idx])
        metrics = _compute_alpha_metrics(alpha, rgb_u8=rgb_u8)
        if confidences is not None and 0 <= local_idx < len(confidences):
            conf = np.clip(np.asarray(confidences[local_idx], dtype=np.float32), 0.0, 1.0)
            metrics["mean_confidence"] = float(conf.mean())
            conf_u8 = (conf * 255.0).round().astype(np.uint8)
            conf_path = alpha_dir / f"{abs_idx:06d}_conf.png"
            ok = cv2.imwrite(str(conf_path), conf_u8)
            if not ok:
                raise IOError(f"Failed to write stage confidence sample: {conf_path}")

        if save_rgb:
            rgb_path = rgb_dir / f"{abs_idx:06d}.png"
            if not rgb_path.exists():
                ok = cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise IOError(f"Failed to write debug RGB sample: {rgb_path}")
        if save_overlay:
            overlay_path = alpha_dir / f"{abs_idx:06d}_overlay.png"
            _save_overlay(rgb_u8, alpha, overlay_path)

        stage_metrics[abs_idx] = metrics
    return stage_metrics


def _quality_score(m: dict[str, float]) -> float:
    edge_align = float(m.get("edge_alignment", 0.0))
    bgrad_norm = float(np.clip(m.get("boundary_grad_ratio", 0.0) / 2.0, 0.0, 1.0))
    fg_solidity = float(m.get("fg_solidity", 0.0))
    bg_leak = float(m.get("outer_bg_leak_mean_alpha", 0.0))
    speckle = float(m.get("speckle_ratio", 0.0))
    holes = float(m.get("hole_ratio", 0.0))
    return edge_align + 0.5 * bgrad_norm + 0.6 * fg_solidity - 2.2 * bg_leak - 3.0 * speckle - 2.0 * holes


def write_stage_diagnosis_report(
    *,
    output_dir: Path,
    stage_dir_name: str,
    stage_order: list[str],
    per_stage_metrics: dict[str, dict[int, dict[str, float]]],
) -> tuple[Path, Path]:
    stage_root = output_dir / stage_dir_name
    stage_root.mkdir(parents=True, exist_ok=True)

    all_frames: set[int] = set()
    for stage in stage_order:
        all_frames.update(per_stage_metrics.get(stage, {}).keys())
    frames = sorted(all_frames)

    transition_rows: list[dict[str, float | str | int | bool]] = []
    per_frame_rows: list[dict[str, Any]] = []
    stage_quality_rows: dict[str, list[float]] = {k: [] for k in stage_order}

    for frame in frames:
        frame_detail: dict[str, Any] = {"frame": frame, "stages": {}}
        quality_by_stage: dict[str, float] = {}
        for stage in stage_order:
            m = per_stage_metrics.get(stage, {}).get(frame)
            if m is not None:
                frame_detail["stages"][stage] = m
                q = _quality_score(m)
                quality_by_stage[stage] = q
                stage_quality_rows.setdefault(stage, []).append(q)
        per_frame_rows.append(frame_detail)
        if quality_by_stage:
            worst_stage = min(quality_by_stage.items(), key=lambda x: x[1])
            frame_detail["worst_stage"] = {"stage": worst_stage[0], "quality_score": float(worst_stage[1])}
            frame_detail["quality_by_stage"] = {k: float(v) for k, v in quality_by_stage.items()}

        for prev, curr in zip(stage_order[:-1], stage_order[1:]):
            pm = per_stage_metrics.get(prev, {}).get(frame)
            cm = per_stage_metrics.get(curr, {}).get(frame)
            if pm is None or cm is None:
                continue
            d_speckle = float(cm.get("speckle_ratio", 0.0) - pm.get("speckle_ratio", 0.0))
            d_largest = float(cm.get("largest_component_ratio", 0.0) - pm.get("largest_component_ratio", 0.0))
            d_cov = float(cm.get("coverage", 0.0) - pm.get("coverage", 0.0))
            d_holes = float(cm.get("hole_ratio", 0.0) - pm.get("hole_ratio", 0.0))
            d_quality = _quality_score(cm) - _quality_score(pm)
            score = (
                max(0.0, d_speckle * 200.0)
                + max(0.0, -d_largest * 80.0)
                + max(0.0, d_holes * 120.0)
                + max(0.0, -d_quality * 80.0)
            )
            suspect = bool(d_speckle > 0.003 or d_largest < -0.04 or d_holes > 0.01 or d_quality < -0.08)
            transition_rows.append(
                {
                    "frame": int(frame),
                    "transition": f"{prev}->{curr}",
                    "delta_speckle": d_speckle,
                    "delta_largest_component_ratio": d_largest,
                    "delta_coverage": d_cov,
                    "delta_hole_ratio": d_holes,
                    "delta_quality_score": float(d_quality),
                    "noise_score": score,
                    "suspect": suspect,
                }
            )

    summary: dict[str, Any] = {
        "stage_order": stage_order,
        "frames": frames,
        "per_frame": per_frame_rows,
        "transitions": transition_rows,
    }

    if transition_rows:
        worst = max(transition_rows, key=lambda r: float(r.get("noise_score", 0.0)))
        summary["worst_transition"] = worst

        grouped: dict[str, list[float]] = {}
        for row in transition_rows:
            key = str(row["transition"])
            grouped.setdefault(key, []).append(float(row["noise_score"]))
        summary["transition_score_mean"] = {
            key: float(np.mean(vals)) for key, vals in grouped.items()
        }
    else:
        summary["worst_transition"] = None
        summary["transition_score_mean"] = {}

    summary["stage_quality_mean"] = {
        stage: float(np.mean(vals)) if vals else 0.0
        for stage, vals in stage_quality_rows.items()
    }

    json_path = stage_root / "diagnosis.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage Diagnosis")
    lines.append("")
    lines.append(f"- Stage order: `{', '.join(stage_order)}`")
    if summary.get("worst_transition") is not None:
        wt = summary["worst_transition"]
        lines.append(
            f"- Worst transition: `{wt['transition']}` at frame `{wt['frame']}` "
            f"(noise_score={float(wt['noise_score']):.3f})"
        )
    if summary.get("stage_quality_mean"):
        ranked = sorted(summary["stage_quality_mean"].items(), key=lambda x: x[1])
        lines.append(f"- Worst mean stage quality: `{ranked[0][0]}` ({ranked[0][1]:.3f})")
    lines.append("")
    lines.append("## Mean Transition Noise Score")
    for key, val in summary["transition_score_mean"].items():
        lines.append(f"- {key}: {float(val):.3f}")
    lines.append("")
    lines.append("## Suspect Transitions")
    suspects = [r for r in transition_rows if bool(r.get("suspect"))]
    if not suspects:
        lines.append("- none")
    else:
        for row in suspects:
            lines.append(
                f"- frame {int(row['frame'])} {row['transition']}: "
                f"d_speckle={float(row['delta_speckle']):.4f}, "
                f"d_largest={float(row['delta_largest_component_ratio']):.4f}, "
                f"d_holes={float(row['delta_hole_ratio']):.4f}, "
                f"d_quality={float(row['delta_quality_score']):.4f}, "
                f"score={float(row['noise_score']):.3f}"
            )

    md_path = stage_root / "diagnosis.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
