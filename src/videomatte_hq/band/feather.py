"""Feather mask generation from band using distance transform."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_feather_mask(band: np.ndarray, feather_px: int = 64) -> np.ndarray:
    """Build feather mask M for band compositing.

    M = 0 outside band (use A0prime),
    M = 1 inside band core,
    smooth ramp at boundary.

    Args:
        band: (H, W) bool band mask.
        feather_px: Width of the feather ramp in pixels.

    Returns:
        (H, W) float32 mask in [0, 1].
    """
    if not band.any():
        return np.zeros(band.shape, dtype=np.float32)

    # Distance from outside into the band
    dist_inside = ndimage.distance_transform_edt(band)

    # Normalize: 0 at edge, 1 at feather_px depth
    if feather_px > 0:
        feather = np.clip(dist_inside / feather_px, 0.0, 1.0)
    else:
        feather = band.astype(np.float32)

    return feather.astype(np.float32)
