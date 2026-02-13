"""Photometric normalization — compensate for lighting/exposure drift.

Before computing fg_diff, estimate per-frame gain and offset from
high-confidence BG pixels, then normalize the BG plate to match
each frame's exposure.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_photometric_params(
    frame: np.ndarray,
    bg_plate: np.ndarray,
    bg_confidence: np.ndarray,
    confidence_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-channel linear gain and offset for a frame.

    Models: frame_bg ≈ gain * plate_bg + offset
    Fits only on pixels where bg_confidence > confidence_threshold.

    Args:
        frame: (H, W, C) float32 current frame.
        bg_plate: (H, W, C) float32 BG plate.
        bg_confidence: (H, W) float32 confidence map.
        confidence_threshold: Only use pixels above this confidence.

    Returns:
        (gain, offset): each (C,) float32 arrays for per-channel normalization.
    """
    mask = bg_confidence > confidence_threshold
    if mask.sum() < 100:
        # Not enough reliable BG pixels — return identity transform
        logger.warning("Photometric: too few high-confidence BG pixels, using identity")
        c = frame.shape[2] if frame.ndim == 3 else 1
        return np.ones(c, dtype=np.float32), np.zeros(c, dtype=np.float32)

    frame_bg = frame[mask]  # (N, C)
    plate_bg = bg_plate[mask]  # (N, C)

    num_channels = frame_bg.shape[1] if frame_bg.ndim == 2 else 1
    gain = np.ones(num_channels, dtype=np.float32)
    offset = np.zeros(num_channels, dtype=np.float32)

    for c in range(num_channels):
        if frame_bg.ndim == 2:
            x = plate_bg[:, c]
            y = frame_bg[:, c]
        else:
            x = plate_bg
            y = frame_bg

        # Least squares: y = gain * x + offset
        # Using normal equations: [sum(x^2) sum(x); sum(x) N] [gain; offset] = [sum(xy); sum(y)]
        n = len(x)
        sx = x.sum()
        sy = y.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()

        det = n * sxx - sx * sx
        if abs(det) < 1e-10:
            gain[c] = 1.0
            offset[c] = 0.0
        else:
            gain[c] = (n * sxy - sx * sy) / det
            offset[c] = (sxx * sy - sx * sxy) / det

    return gain, offset


def normalize_bg_plate(
    bg_plate: np.ndarray,
    gain: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Apply photometric normalization to the BG plate.

    normalized = gain * bg_plate + offset

    Args:
        bg_plate: (H, W, C) float32 BG plate.
        gain: (C,) per-channel gain.
        offset: (C,) per-channel offset.

    Returns:
        (H, W, C) float32 normalized BG plate.
    """
    return (bg_plate * gain[np.newaxis, np.newaxis, :] + offset[np.newaxis, np.newaxis, :]).astype(np.float32)
