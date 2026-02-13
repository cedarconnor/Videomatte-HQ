"""QC metrics — per-frame flicker, seam, flow, and edge confidence metrics."""

from __future__ import annotations

import numpy as np


def compute_flicker_score(alpha_t: np.ndarray, alpha_prev: np.ndarray) -> float:
    """Per-frame temporal flicker score (L1 diff)."""
    return float(np.abs(alpha_t - alpha_prev).mean())


def compute_edge_confidence(alpha: np.ndarray) -> float:
    """Edge confidence: fraction of edge pixels with alpha near 0 or 1.

    Higher = cleaner edges (bimodal). Lower = soft/uncertain edges.
    """
    edge_pixels = (alpha > 0.01) & (alpha < 0.99)
    if not edge_pixels.any():
        return 1.0
    near_binary = ((alpha > 0.9) | (alpha < 0.1)) & edge_pixels
    return float(near_binary.sum() / edge_pixels.sum())


def compute_band_coverage_ratio(band: np.ndarray, roi_area: int) -> float:
    """Band area as fraction of ROI."""
    if roi_area == 0:
        return 0.0
    return float(band.sum() / roi_area)
