"""Mask-to-prompt adapters used by stage-1 segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from videomatte_hq.protocols import PromptAdapter, SegmentPrompt


def _normalize_mask(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    out = np.asarray(mask, dtype=np.float32)
    if out.ndim == 3:
        out = out[..., 0]
    if out.ndim != 2:
        raise ValueError(f"Mask must be 2D or HxWx1, got shape={out.shape}")

    h, w = int(frame_shape[0]), int(frame_shape[1])
    if out.shape != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _bbox_from_binary(binary: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(binary)
    if ys.size == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return (x0, y0, x1, y1)


def _expand_bbox(
    bbox: tuple[float, float, float, float] | None,
    frame_shape: tuple[int, int],
    *,
    expand_ratio: float,
    min_expand_px: int,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    expand = max(float(min_expand_px), max(bw, bh) * float(expand_ratio))

    ex0 = max(0.0, x0 - expand)
    ey0 = max(0.0, y0 - expand)
    ex1 = min(float(w), x1 + expand)
    ey1 = min(float(h), y1 + expand)
    return (ex0, ey0, ex1, ey1)


def _nearest_background_point(binary: np.ndarray, x: int, y: int, max_radius: int = 96) -> tuple[int, int] | None:
    h, w = binary.shape
    if 0 <= x < w and 0 <= y < h and not bool(binary[y, x]):
        return (x, y)

    for r in range(1, max_radius + 1):
        x0 = max(0, x - r)
        x1 = min(w - 1, x + r)
        y0 = max(0, y - r)
        y1 = min(h - 1, y + r)

        for px in (x0, x1):
            for py in range(y0, y1 + 1):
                if not bool(binary[py, px]):
                    return (px, py)
        for py in (y0, y1):
            for px in range(x0, x1 + 1):
                if not bool(binary[py, px]):
                    return (px, py)
    return None


def _sample_interior_points(
    binary: np.ndarray,
    *,
    k: int = 5,
    suppression_ratio: float = 0.3,
    min_suppression_radius: int = 10,
) -> list[tuple[float, float]]:
    if k <= 0 or not bool(binary.any()):
        return []

    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    points: list[tuple[float, float]] = []
    max_samples = min(int(k), int(binary.sum()))

    for _ in range(max_samples):
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        peak = float(dist[y, x])
        if peak <= 0.0:
            break
        points.append((float(x), float(y)))
        radius = max(min_suppression_radius, int(max(1.0, peak * suppression_ratio)))
        cv2.circle(dist, (int(x), int(y)), radius, 0.0, thickness=-1)

    if points:
        return points

    coords = np.argwhere(binary)
    if coords.size == 0:
        return []
    y, x = coords[len(coords) // 2]
    return [(float(x), float(y))]


def _sample_negative_points(
    binary: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    *,
    margin_ratio: float = 0.05,
    min_margin_px: int = 8,
) -> list[tuple[float, float]]:
    h, w = binary.shape
    margin = max(int(max(h, w) * margin_ratio), int(min_margin_px))

    if bbox is None:
        candidates = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    else:
        x0, y0, x1, y1 = bbox
        candidates = [
            (int(round(x0)) - margin, int(round(y0)) - margin),
            (int(round(x1)) + margin, int(round(y0)) - margin),
            (int(round(x0)) - margin, int(round(y1)) + margin),
            (int(round(x1)) + margin, int(round(y1)) + margin),
        ]

    points: list[tuple[float, float]] = []
    for cx, cy in candidates:
        px = int(np.clip(cx, 0, w - 1))
        py = int(np.clip(cy, 0, h - 1))
        bg = _nearest_background_point(binary, px, py)
        if bg is None:
            continue
        bx, by = bg
        points.append((float(bx), float(by)))
    return points


@dataclass(slots=True)
class MaskPromptAdapter(PromptAdapter):
    """Convert user masks to bbox + multi-point prompts."""

    interior_points: int = 5
    negative_margin_ratio: float = 0.05
    min_negative_margin_px: int = 8
    suppression_ratio: float = 0.3
    min_suppression_radius: int = 10
    bbox_expand_ratio: float = 0.08
    min_bbox_expand_px: int = 12

    def adapt(self, mask: np.ndarray, frame_shape: tuple[int, int]) -> SegmentPrompt:
        mask_f = _normalize_mask(mask, frame_shape)
        binary = mask_f >= 0.5
        bbox = _expand_bbox(
            _bbox_from_binary(binary),
            frame_shape,
            expand_ratio=float(self.bbox_expand_ratio),
            min_expand_px=int(self.min_bbox_expand_px),
        )

        positive_points = _sample_interior_points(
            binary,
            k=int(self.interior_points),
            suppression_ratio=float(self.suppression_ratio),
            min_suppression_radius=int(self.min_suppression_radius),
        )
        negative_points = _sample_negative_points(
            binary,
            bbox,
            margin_ratio=float(self.negative_margin_ratio),
            min_margin_px=int(self.min_negative_margin_px),
        )
        return SegmentPrompt(
            bbox=bbox,
            positive_points=positive_points,
            negative_points=negative_points,
            mask=mask_f,
        )
