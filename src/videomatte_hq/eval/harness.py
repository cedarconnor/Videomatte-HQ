"""v1-v2 comparison harness for temporal stability metrics."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from videomatte_hq.io.reader import FrameSource
from videomatte_hq.pipeline.stage_qc import compute_temporal_metrics


def _as_alpha(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    elif np.issubdtype(arr.dtype, np.integer):
        out = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        out = arr.astype(np.float32)
        max_val = float(out.max()) if out.size else 1.0
        if max_val > 1.0:
            out = out / max(max_val, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_alpha_sequence(
    pattern: str,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> list[np.ndarray]:
    source = FrameSource(
        pattern=pattern,
        frame_start=frame_start,
        frame_end=frame_end,
        prefetch_workers=0,
    )
    try:
        return [_as_alpha(source[i]) for i in range(len(source))]
    finally:
        source.close()


def summarize_alpha_sequence(alphas: list[np.ndarray], threshold: float = 0.5) -> dict[str, float]:
    masks = [(np.asarray(alpha, dtype=np.float32) >= float(threshold)).astype(np.float32) for alpha in alphas]
    metrics = compute_temporal_metrics(masks)
    coverages = np.asarray([float(mask.mean()) for mask in masks], dtype=np.float32) if masks else np.zeros((0,))
    return {
        **asdict(metrics),
        "num_frames": float(len(alphas)),
        "mean_coverage": float(coverages.mean()) if coverages.size else 0.0,
        "min_coverage": float(coverages.min()) if coverages.size else 0.0,
        "max_coverage": float(coverages.max()) if coverages.size else 0.0,
    }


def compare_alpha_sequences(
    reference_alphas: list[np.ndarray],
    candidate_alphas: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, object]:
    if not reference_alphas or not candidate_alphas:
        raise ValueError("Both reference and candidate alpha sequences are required.")

    count = min(len(reference_alphas), len(candidate_alphas))
    ref = [np.asarray(a, dtype=np.float32) for a in reference_alphas[:count]]
    cand = [np.asarray(a, dtype=np.float32) for a in candidate_alphas[:count]]

    ref_summary = summarize_alpha_sequence(ref, threshold=threshold)
    cand_summary = summarize_alpha_sequence(cand, threshold=threshold)

    maes = np.asarray([np.abs(r - c).mean() for r, c in zip(ref, cand)], dtype=np.float32)
    maxes = np.asarray([np.abs(r - c).max() for r, c in zip(ref, cand)], dtype=np.float32)
    diffs = {
        key: float(cand_summary[key] - ref_summary[key])
        for key in (
            "temporal_iou_mean",
            "temporal_iou_std",
            "temporal_iou_p95",
            "area_jitter_mean",
            "area_jitter_p95",
            "centroid_jitter_mean",
            "centroid_jitter_p95",
            "mean_coverage",
        )
    }

    return {
        "frames_compared": count,
        "reference": ref_summary,
        "candidate": cand_summary,
        "diff": diffs,
        "frame_mae_mean": float(maes.mean()),
        "frame_mae_p95": float(np.percentile(maes, 95.0)),
        "frame_abs_max": float(maxes.max()),
    }


def run_v1_v2_comparison(
    reference_pattern: str,
    candidate_pattern: str,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    threshold: float = 0.5,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    ref = load_alpha_sequence(reference_pattern, frame_start=frame_start, frame_end=frame_end)
    cand = load_alpha_sequence(candidate_pattern, frame_start=frame_start, frame_end=frame_end)
    result = compare_alpha_sequences(ref, cand, threshold=threshold)
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
