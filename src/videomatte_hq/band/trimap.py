"""Distance-transform-based trimap generation per design doc §10.5.

Preserves thin structures (hair wisps, fingers) better than erosion-based approaches.
Supports adaptive unknown width and histogram-based thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from videomatte_hq.config import TrimapConfig


def generate_trimap(
    alpha_prior: np.ndarray,
    cfg: "TrimapConfig",
    hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate trimap from alpha prior using distance transforms.

    Args:
        alpha_prior: (H, W) float32 alpha from Pass A or A′.
        cfg: Trimap configuration.
        hair_mask: (H, W) optional bool mask for hair-aware width.

    Returns:
        (H, W) float32 trimap: 0.0=BG, 0.5=unknown, 1.0=FG.
    """
    t_fg = cfg.t_fg
    t_bg = cfg.t_bg

    # Adaptive thresholds from histogram
    if cfg.adaptive_thresholds:
        t_fg, t_bg = _adaptive_thresholds(alpha_prior, t_fg, t_bg)

    # Binary masks
    fg_mask = (alpha_prior > t_fg).astype(np.uint8)
    bg_mask = (alpha_prior < t_bg).astype(np.uint8)

    # Distance transforms
    dist_from_fg_boundary = ndimage.distance_transform_edt(fg_mask)
    dist_from_bg_boundary = ndimage.distance_transform_edt(bg_mask)

    # Determine unknown width per pixel
    unknown_width = np.full(alpha_prior.shape, cfg.unknown_width, dtype=np.float32)

    if cfg.adaptive_width and hair_mask is not None:
        unknown_width[hair_mask] = cfg.unknown_width_hair
        unknown_width[~hair_mask] = cfg.unknown_width_body

    # Definite regions: pixels far enough from boundary
    definite_fg = dist_from_fg_boundary > unknown_width
    definite_bg = dist_from_bg_boundary > unknown_width

    # Build trimap
    trimap = np.full(alpha_prior.shape, 0.5, dtype=np.float32)
    trimap[definite_fg] = 1.0
    trimap[definite_bg] = 0.0

    return trimap


def _adaptive_thresholds(
    alpha: np.ndarray,
    default_fg: float = 0.95,
    default_bg: float = 0.05,
) -> tuple[float, float]:
    """Compute adaptive FG/BG thresholds from alpha histogram.

    If distribution is bimodal (clean edges) → tight thresholds.
    If significant mid-range mass (motion blur, sheer fabric) → wider thresholds.
    """
    hist, bins = np.histogram(alpha.ravel(), bins=50, range=(0, 1))
    total = hist.sum()

    # Mid-range mass: fraction of pixels between 0.1 and 0.9
    mid_bins = hist[5:45]  # bins for [0.1, 0.9]
    mid_fraction = mid_bins.sum() / max(total, 1)

    if mid_fraction > 0.15:
        # Significant mid-range mass — widen thresholds
        return 0.85, 0.15
    else:
        return default_fg, default_bg
