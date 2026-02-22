"""Temporal QC helpers for drift/jitter detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
    a = np.asarray(mask_a, dtype=np.float32) >= float(threshold)
    b = np.asarray(mask_b, dtype=np.float32) >= float(threshold)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 1.0
    return float(intersection / union)


@dataclass(slots=True)
class DriftCheck:
    iou: float
    area_ratio: float
    drift: bool


def check_drift(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    iou_threshold: float = 0.70,
    area_change_threshold: float = 0.40,
) -> DriftCheck:
    iou = compute_iou(current_mask, previous_mask)
    cur_area = float((np.asarray(current_mask, dtype=np.float32) >= 0.5).sum())
    prev_area = float((np.asarray(previous_mask, dtype=np.float32) >= 0.5).sum())
    area_ratio = cur_area / max(prev_area, 1.0)
    drift = bool(iou < float(iou_threshold) or abs(1.0 - area_ratio) > float(area_change_threshold))
    return DriftCheck(iou=iou, area_ratio=area_ratio, drift=drift)


def area_jitter(masks: list[np.ndarray]) -> np.ndarray:
    if len(masks) < 2:
        return np.zeros((0,), dtype=np.float32)
    areas = np.asarray([(np.asarray(m, dtype=np.float32) >= 0.5).sum() for m in masks], dtype=np.float32)
    return np.abs(np.diff(areas)) / np.maximum(areas[:-1], 1.0)


def centroid_jitter(masks: list[np.ndarray]) -> np.ndarray:
    if len(masks) < 2:
        return np.zeros((0,), dtype=np.float32)
    centers: list[tuple[float, float]] = []
    for mask in masks:
        fg = np.argwhere(np.asarray(mask, dtype=np.float32) >= 0.5)
        if fg.size == 0:
            centers.append((0.0, 0.0))
            continue
        yx = fg.mean(axis=0)
        centers.append((float(yx[1]), float(yx[0])))
    points = np.asarray(centers, dtype=np.float32)
    diffs = np.diff(points, axis=0)
    return np.sqrt((diffs * diffs).sum(axis=1))


@dataclass(slots=True)
class TemporalMetrics:
    temporal_iou_mean: float
    temporal_iou_std: float
    temporal_iou_p95: float
    area_jitter_mean: float
    area_jitter_p95: float
    centroid_jitter_mean: float
    centroid_jitter_p95: float


def compute_temporal_metrics(masks: list[np.ndarray]) -> TemporalMetrics:
    if len(masks) < 2:
        return TemporalMetrics(
            temporal_iou_mean=1.0,
            temporal_iou_std=0.0,
            temporal_iou_p95=1.0,
            area_jitter_mean=0.0,
            area_jitter_p95=0.0,
            centroid_jitter_mean=0.0,
            centroid_jitter_p95=0.0,
        )

    ious = np.asarray(
        [compute_iou(masks[i], masks[i - 1]) for i in range(1, len(masks))],
        dtype=np.float32,
    )
    areas = area_jitter(masks)
    centers = centroid_jitter(masks)
    return TemporalMetrics(
        temporal_iou_mean=float(ious.mean()),
        temporal_iou_std=float(ious.std()),
        temporal_iou_p95=float(np.percentile(ious, 95.0)),
        area_jitter_mean=float(areas.mean()) if areas.size else 0.0,
        area_jitter_p95=float(np.percentile(areas, 95.0)) if areas.size else 0.0,
        centroid_jitter_mean=float(centers.mean()) if centers.size else 0.0,
        centroid_jitter_p95=float(np.percentile(centers, 95.0)) if centers.size else 0.0,
    )
