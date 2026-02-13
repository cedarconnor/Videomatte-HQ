"""Foreground difference computation.

fg_diff = |luma(frame) - luma(bg_plate_normalized)|

Used for ROI validation, adaptive band definition, and refiner input.
"""

from __future__ import annotations

import numpy as np

from videomatte_hq.io.colorspace import rgb_to_luma
from videomatte_hq.background.photometric import (
    estimate_photometric_params,
    normalize_bg_plate,
)


def compute_fg_diff(
    frame: np.ndarray,
    bg_plate: np.ndarray,
    bg_confidence: np.ndarray | None = None,
    photometric_normalize: bool = True,
    space: str = "luma",
) -> np.ndarray:
    """Compute foreground difference between frame and background plate.

    Args:
        frame: (H, W, C) float32 RGB frame.
        bg_plate: (H, W, C) float32 BG plate.
        bg_confidence: (H, W) float32 confidence map (for photometric fitting).
        photometric_normalize: Whether to compensate for exposure drift.
        space: 'luma' for luminance diff, 'rgb' for per-channel L1.

    Returns:
        (H, W) float32 foreground difference map.
    """
    if photometric_normalize and bg_confidence is not None:
        gain, offset = estimate_photometric_params(frame, bg_plate, bg_confidence)
        bg_norm = normalize_bg_plate(bg_plate, gain, offset)
    else:
        bg_norm = bg_plate

    if space == "luma":
        fg_diff = np.abs(rgb_to_luma(frame) - rgb_to_luma(bg_norm))
    elif space == "rgb":
        fg_diff = np.abs(frame - bg_norm).mean(axis=-1)
    elif space == "normalized_rgb":
        # Normalize RGB to unit length before diffing (robust to brightness changes)
        frame_norm = frame / (np.linalg.norm(frame, axis=-1, keepdims=True) + 1e-8)
        bg_norm_rgb = bg_norm / (np.linalg.norm(bg_norm, axis=-1, keepdims=True) + 1e-8)
        fg_diff = np.abs(frame_norm - bg_norm_rgb).mean(axis=-1)
    else:
        fg_diff = np.abs(rgb_to_luma(frame) - rgb_to_luma(bg_norm))

    return fg_diff.astype(np.float32)
