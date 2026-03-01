"""Hann (raised cosine) 2D window for tile blending."""

from __future__ import annotations

import numpy as np


def hann_2d(height: int, width: int, *, floor: float = 0.01) -> np.ndarray:
    """Generate a 2D Hann window for overlap blending.

    The window is clamped to a minimum *floor* so that tile edges always
    contribute some weight — pure-zero edges cause coverage gaps in the
    tile stitcher and amplify numerical noise at seams.

    Args:
        height: Tile height in pixels.
        width: Tile width in pixels.
        floor: Minimum window value (default 0.01).

    Returns:
        (H, W) float32 window in [floor, 1], maximum at center.
    """
    wy = np.hanning(height).astype(np.float32)
    wx = np.hanning(width).astype(np.float32)
    window = np.outer(wy, wx)
    if floor > 0.0:
        np.maximum(window, float(floor), out=window)
    return window
