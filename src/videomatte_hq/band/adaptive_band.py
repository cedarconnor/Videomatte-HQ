"""Adaptive band definition — multi-signal union per design doc §10.1.

Three edge signals: alpha gradient, RGB edges (directional-aligned),
and BG-sub edges (temporal-coherent), with area cap safeguards.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from videomatte_hq.config import BandConfig

logger = logging.getLogger(__name__)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude via Sobel."""
    if img.ndim == 3:
        from videomatte_hq.io.colorspace import rgb_to_luma
        img = rgb_to_luma(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def _gradient_direction(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized gradient direction (nx, ny)."""
    if img.ndim == 3:
        from videomatte_hq.io.colorspace import rgb_to_luma
        img = rgb_to_luma(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) + 1e-8
    return gx / mag, gy / mag


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Morphological dilation with circular kernel."""
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def compute_adaptive_band(
    alpha: np.ndarray,
    rgb: np.ndarray,
    cfg: "BandConfig",
    bg_confidence: Optional[np.ndarray] = None,
    fg_diff: Optional[np.ndarray] = None,
    fg_diff_history: Optional[list[np.ndarray]] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the adaptive uncertainty band.

    Args:
        alpha: (H, W) float32 alpha from A0prime_8k.
        rgb: (H, W, C) float32 RGB frame.
        cfg: Band configuration.
        bg_confidence: (H, W) float32 BG confidence (for signal 3 gating).
        fg_diff: (H, W) float32 current frame fg_diff.
        fg_diff_history: List of recent fg_diff maps (for temporal coherence).
        roi_mask: (H, W) bool ROI mask (for signal 3 edge margin filter).

    Returns:
        (H, W) bool band mask.
    """
    h, w = alpha.shape

    # Signal 1: Alpha gradient band
    alpha_grad = _gradient_magnitude(alpha)
    alpha_edges = alpha_grad > cfg.alpha_grad_threshold
    signal1 = _dilate(alpha_edges, cfg.dilate_alpha_px)

    # Signal 2: RGB edges with directional alignment + alpha proximity
    rgb_gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32) * 255
    rgb_edges_raw = cv2.Canny(rgb_gray.astype(np.uint8), 50, 150)
    rgb_edges = rgb_edges_raw > 0

    # Filter: proximity to alpha gradient pixels
    alpha_edge_dilated = _dilate(alpha_edges, cfg.rgb_proximity_px)
    rgb_edges = rgb_edges & alpha_edge_dilated

    # Filter: directional alignment
    if cfg.edge_alignment_threshold > 0 and alpha_edges.any():
        rgb_nx, rgb_ny = _gradient_direction(rgb[..., :3])
        alpha_nx, alpha_ny = _gradient_direction(alpha)
        alignment = np.abs(rgb_nx * alpha_nx + rgb_ny * alpha_ny)
        rgb_edges = rgb_edges & (alignment > cfg.edge_alignment_threshold)

    # Filter: alpha proximity
    lo, hi = cfg.rgb_alpha_range
    alpha_near_boundary = (alpha > lo) & (alpha < hi)
    alpha_near_dilated = _dilate(alpha_near_boundary, cfg.dilate_rgb_px)
    rgb_edges = rgb_edges & alpha_near_dilated

    signal2 = _dilate(rgb_edges, cfg.dilate_rgb_px)

    # Signal 3: BG subtraction edges (locked-off only)
    signal3 = np.zeros((h, w), dtype=bool)
    if cfg.bg_enabled and fg_diff is not None and bg_confidence is not None:
        fg_diff_edges = _gradient_magnitude(fg_diff) > 0.02
        # Gate by bg_confidence
        fg_diff_edges = fg_diff_edges & (bg_confidence > cfg.bg_confidence_gate)

        # Temporal coherence: require edge to persist across N frames
        if fg_diff_history and len(fg_diff_history) >= cfg.bg_edge_persist_frames:
            persist_count = np.zeros((h, w), dtype=np.int32)
            for hist_diff in fg_diff_history[-cfg.bg_edge_persist_frames:]:
                hist_edges = _gradient_magnitude(hist_diff) > 0.02
                persist_count += hist_edges.astype(np.int32)
            fg_diff_edges = fg_diff_edges & (persist_count >= cfg.bg_edge_persist_frames)

        signal3 = _dilate(fg_diff_edges, cfg.dilate_bg_px)

    # Union band
    band = signal1 | signal2 | signal3

    # Area cap: auto-tighten if band exceeds coverage limit
    if cfg.auto_tighten:
        roi_area = h * w if roi_mask is None else roi_mask.sum()
        if roi_area > 0:
            coverage = band.sum() / roi_area
            if coverage > cfg.band_max_coverage:
                logger.warning(
                    f"Band coverage {coverage:.2%} exceeds cap {cfg.band_max_coverage:.0%}, auto-tightening"
                )
                # Progressive tightening: remove weaker signals
                band = signal1 | signal2  # drop BG-sub
                coverage = band.sum() / roi_area
                if coverage > cfg.band_max_coverage:
                    band = signal1  # alpha gradient only
                    coverage = band.sum() / roi_area
                    if coverage > cfg.band_max_coverage:
                        # Tighten alpha gradient threshold
                        for mult in [2, 4, 8]:
                            tight_edges = alpha_grad > (cfg.alpha_grad_threshold * mult)
                            band = _dilate(tight_edges, cfg.dilate_alpha_px)
                            coverage = band.sum() / roi_area
                            if coverage <= cfg.band_max_coverage:
                                break

    return band
