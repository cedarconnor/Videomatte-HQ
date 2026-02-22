"""Hann (raised cosine) 2D window for tile blending."""

from __future__ import annotations

import numpy as np


def hann_2d(height: int, width: int) -> np.ndarray:
    """Generate a 2D Hann window for overlap blending.

    Args:
        height: Tile height in pixels.
        width: Tile width in pixels.

    Returns:
        (H, W) float32 window in [0, 1], maximum at center.
    """
    wy = np.hanning(height).astype(np.float32)
    wx = np.hanning(width).astype(np.float32)
    return np.outer(wy, wx)
