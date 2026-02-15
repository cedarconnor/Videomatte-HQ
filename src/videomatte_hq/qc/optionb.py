"""Option B QC metrics, regression gates, and report output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videomatte_hq.io.reader import read_frame
from videomatte_hq.qc.metrics import compute_edge_confidence


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _edge_band(alpha: np.ndarray, lo: float, hi: float, radius: int) -> np.ndarray:
    band = (alpha > lo) & (alpha < hi)
    return _dilate(band, radius)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), 95.0))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _safe_alpha(alpha: np.ndarray) -> np.ndarray:
    return np.nan_to_num(alpha.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def _resolve_output_path(base_dir: Path, pattern: str, frame_idx: int) -> Path:
    try:
        name = pattern % frame_idx
    except TypeError:
        name = pattern.format(frame_idx)
    return base_dir / name


def evaluate_optionb_qc(
    alphas: list[np.ndarray],
    confidences: list[np.ndarray],
    cfg: Any,
) -> dict[str, Any]:
    """Compute per-frame QC metrics and evaluate regression gates."""

    if len(alphas) != len(confidences):
        raise ValueError("alphas/confidences length mismatch for QC evaluation")

    num_frames = len(alphas)
    edge_lo = float(np.clip(cfg.temporal_cleanup.edge_bg_threshold, 0.0, 0.49))
    edge_hi = float(np.clip(cfg.temporal_cleanup.edge_fg_threshold, 0.51, 1.0))
    edge_radius = max(int(cfg.temporal_cleanup.edge_band_radius_px), 0)
    alpha_eps = float(max(cfg.qc.alpha_range_eps, 0.0))

    frames: list[dict[str, Any]] = []
    coverages: list[float] = []
    edge_confidences: list[float] = []
    flickers: list[float] = []
    edge_flickers: list[float] = []

    invalid_alpha_frames = 0
    range_violation_frames = 0

    prev_alpha: np.ndarray | None = None
    prev_band: np.ndarray | None = None

    for idx in range(num_frames):
        alpha_raw = np.asarray(alphas[idx], dtype=np.float32)
        conf = np.asarray(confidences[idx], dtype=np.float32)

        finite_ok = bool(np.isfinite(alpha_raw).all())
        if not finite_ok:
            invalid_alpha_frames += 1

        alpha = _safe_alpha(alpha_raw)
        alpha_min = float(alpha.min())
        alpha_max = float(alpha.max())
        range_ok = alpha_min >= -alpha_eps and alpha_max <= (1.0 + alpha_eps)
        if not range_ok:
            range_violation_frames += 1

        band = _edge_band(alpha, lo=edge_lo, hi=edge_hi, radius=edge_radius)
        band_coverage = float(band.mean())
        edge_conf = float(compute_edge_confidence(np.clip(alpha, 0.0, 1.0)))
        mean_conf = float(np.clip(conf, 0.0, 1.0).mean())

        flicker: float | None = None
        edge_flicker: float | None = None
        if prev_alpha is not None and prev_band is not None:
            diff = np.abs(alpha - prev_alpha)
            flicker = float(diff.mean())
            union_band = band | prev_band
            edge_flicker = float(diff[union_band].mean()) if union_band.any() else 0.0
            flickers.append(flicker)
            edge_flickers.append(edge_flicker)

        frames.append(
            {
                "frame": idx,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "finite_ok": finite_ok,
                "range_ok": range_ok,
                "band_coverage": band_coverage,
                "edge_confidence": edge_conf,
                "mean_confidence": mean_conf,
                "flicker": flicker,
                "edge_flicker": edge_flicker,
            }
        )
        coverages.append(band_coverage)
        edge_confidences.append(edge_conf)
        prev_alpha = alpha
        prev_band = band

    # Spike detection: coverage relative to running mean.
    coverages_arr = np.asarray(coverages, dtype=np.float32)
    running_avg = np.cumsum(coverages_arr) / np.arange(1, max(num_frames, 1) + 1)
    spike_frames: list[dict[str, float | int]] = []
    for idx in range(1, num_frames):
        baseline = max(float(running_avg[idx - 1]), 1e-6)
        ratio = float(coverages_arr[idx] / baseline)
        frames[idx]["band_spike_ratio"] = ratio
        if idx >= 3 and ratio > float(cfg.qc.band_spike_ratio):
            spike_frames.append({"frame": idx, "ratio": ratio})
    if frames:
        frames[0]["band_spike_ratio"] = None

    summary = {
        "num_frames": num_frames,
        "invalid_alpha_frames": invalid_alpha_frames,
        "range_violation_frames": range_violation_frames,
        "mean_flicker": _mean(flickers),
        "p95_flicker": _p95(flickers),
        "mean_edge_flicker": _mean(edge_flickers),
        "p95_edge_flicker": _p95(edge_flickers),
        "mean_edge_confidence": _mean(edge_confidences),
        "p95_band_coverage": _p95(coverages),
        "band_spike_frames": len(spike_frames),
    }

    gates: list[dict[str, Any]] = [
        {
            "name": "alpha_finite_range",
            "passed": invalid_alpha_frames == 0 and range_violation_frames == 0,
            "details": {
                "invalid_alpha_frames": invalid_alpha_frames,
                "range_violation_frames": range_violation_frames,
            },
        },
        {
            "name": "temporal_flicker",
            "passed": summary["p95_flicker"] <= float(cfg.qc.max_p95_flicker),
            "details": {
                "value": summary["p95_flicker"],
                "max": float(cfg.qc.max_p95_flicker),
            },
        },
        {
            "name": "edge_flicker",
            "passed": summary["p95_edge_flicker"] <= float(cfg.qc.max_p95_edge_flicker),
            "details": {
                "value": summary["p95_edge_flicker"],
                "max": float(cfg.qc.max_p95_edge_flicker),
            },
        },
        {
            "name": "edge_confidence",
            "passed": summary["mean_edge_confidence"] >= float(cfg.qc.min_mean_edge_confidence),
            "details": {
                "value": summary["mean_edge_confidence"],
                "min": float(cfg.qc.min_mean_edge_confidence),
            },
        },
        {
            "name": "band_spikes",
            "passed": len(spike_frames) <= int(cfg.qc.max_band_spike_frames),
            "details": {
                "value": len(spike_frames),
                "max": int(cfg.qc.max_band_spike_frames),
                "ratio_threshold": float(cfg.qc.band_spike_ratio),
            },
        },
    ]

    problem_frames: list[dict[str, Any]] = []
    for item in frames:
        issues: list[str] = []
        if not item["finite_ok"]:
            issues.append("non_finite_alpha")
        if not item["range_ok"]:
            issues.append("alpha_out_of_range")
        if item["flicker"] is not None and float(item["flicker"]) > float(cfg.qc.max_p95_flicker):
            issues.append("high_flicker")
        if item["edge_flicker"] is not None and float(item["edge_flicker"]) > float(cfg.qc.max_p95_edge_flicker):
            issues.append("high_edge_flicker")
        spike_ratio = item.get("band_spike_ratio")
        if spike_ratio is not None and float(spike_ratio) > float(cfg.qc.band_spike_ratio):
            issues.append("band_coverage_spike")
        if issues:
            severity = "critical" if ("non_finite_alpha" in issues or "alpha_out_of_range" in issues) else "warning"
            problem_frames.append(
                {
                    "frame": int(item["frame"]),
                    "severity": severity,
                    "issues": issues,
                }
            )

    return {
        "version": "optionb_qc_v1",
        "summary": summary,
        "gates": gates,
        "problem_frames": problem_frames,
        "spike_frames": spike_frames,
        "frames": frames,
    }


def add_output_roundtrip_gate(
    metrics: dict[str, Any],
    *,
    output_dir: Path,
    output_pattern: str,
    alphas: list[np.ndarray],
    frame_start: int,
    sample_count: int,
    max_mae: float,
) -> dict[str, Any]:
    """Validate written outputs against in-memory alphas using sampled frames."""

    num_frames = len(alphas)
    sample_count = max(int(sample_count), 1)
    if num_frames == 0:
        roundtrip = {
            "samples": [],
            "max_mae": 0.0,
            "max_abs_error": 0.0,
            "failed_samples": 0,
            "passed": True,
        }
    else:
        sample_local_indices = np.linspace(
            0,
            num_frames - 1,
            num=min(sample_count, num_frames),
            dtype=np.int32,
        ).tolist()
        sample_local_indices = sorted(set(int(i) for i in sample_local_indices))

        samples: list[dict[str, Any]] = []
        failed_samples = 0
        max_seen_mae = 0.0
        max_seen_abs = 0.0

        for local_idx in sample_local_indices:
            output_idx = frame_start + local_idx
            path = _resolve_output_path(output_dir, output_pattern, output_idx)

            item: dict[str, Any] = {
                "frame": output_idx,
                "path": str(path),
                "passed": False,
            }

            if not path.exists():
                item["error"] = "missing_output_file"
                failed_samples += 1
                samples.append(item)
                continue

            try:
                written = read_frame(path, as_float=True)
                if written.ndim == 3:
                    if written.shape[2] >= 4:
                        alpha_written = written[..., 3]
                    else:
                        alpha_written = written[..., 0]
                else:
                    alpha_written = written
                alpha_written = np.squeeze(alpha_written).astype(np.float32)

                alpha_ref = np.clip(np.asarray(alphas[local_idx], dtype=np.float32), 0.0, 1.0)
                if alpha_written.shape != alpha_ref.shape:
                    item["error"] = f"shape_mismatch:{alpha_written.shape}!={alpha_ref.shape}"
                    failed_samples += 1
                    samples.append(item)
                    continue

                diff = np.abs(alpha_written - alpha_ref)
                mae = float(diff.mean())
                max_abs = float(diff.max())
                max_seen_mae = max(max_seen_mae, mae)
                max_seen_abs = max(max_seen_abs, max_abs)

                item["mae"] = mae
                item["max_abs_error"] = max_abs
                item["passed"] = mae <= float(max_mae)
                if not item["passed"]:
                    failed_samples += 1
            except Exception as exc:
                item["error"] = f"read_failed:{exc}"
                failed_samples += 1

            samples.append(item)

        roundtrip = {
            "samples": samples,
            "max_mae": max_seen_mae,
            "max_abs_error": max_seen_abs,
            "failed_samples": failed_samples,
            "passed": failed_samples == 0,
        }

    metrics["output_roundtrip"] = roundtrip
    metrics.setdefault("gates", []).append(
        {
            "name": "output_roundtrip",
            "passed": bool(roundtrip["passed"]),
            "details": {
                "max_mae": float(roundtrip["max_mae"]),
                "max_allowed_mae": float(max_mae),
                "failed_samples": int(roundtrip["failed_samples"]),
            },
        }
    )
    return metrics


def failed_gate_names(metrics: dict[str, Any]) -> list[str]:
    return [str(g["name"]) for g in metrics.get("gates", []) if not bool(g.get("passed", False))]


def write_optionb_qc_artifacts(
    metrics: dict[str, Any],
    *,
    output_dir: Path,
    output_subdir: str,
    metrics_filename: str,
    report_filename: str,
) -> tuple[Path, Path]:
    qc_dir = output_dir / output_subdir
    qc_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = qc_dir / metrics_filename
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_path = qc_dir / report_filename
    summary = metrics.get("summary", {})
    failed = failed_gate_names(metrics)
    lines = [
        "# Option B QC Report",
        "",
        "## Summary",
        f"- Frames: {summary.get('num_frames', 0)}",
        f"- p95 flicker: {summary.get('p95_flicker', 0.0):.6f}",
        f"- p95 edge flicker: {summary.get('p95_edge_flicker', 0.0):.6f}",
        f"- mean edge confidence: {summary.get('mean_edge_confidence', 0.0):.6f}",
        f"- band spike frames: {summary.get('band_spike_frames', 0)}",
        f"- invalid alpha frames: {summary.get('invalid_alpha_frames', 0)}",
        "",
        "## Gates",
    ]
    for gate in metrics.get("gates", []):
        status = "PASS" if gate.get("passed", False) else "FAIL"
        lines.append(f"- [{status}] {gate.get('name')}")
    if failed:
        lines.extend(["", "## Failed Gates", *(f"- {name}" for name in failed)])
    else:
        lines.extend(["", "## Failed Gates", "- none"])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_path, report_path
