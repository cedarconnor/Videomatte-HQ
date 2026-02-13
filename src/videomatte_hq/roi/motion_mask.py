"""Motion mask from background subtraction, gated by bg_confidence."""

from __future__ import annotations

import numpy as np

from videomatte_hq.background.fg_diff import compute_fg_diff


def compute_motion_mask(
    frame: np.ndarray,
    bg_plate: np.ndarray,
    bg_confidence: np.ndarray,
    fg_diff_threshold: float = 0.08,
    confidence_gate: float = 0.5,
    photometric_normalize: bool = True,
) -> np.ndarray:
    """Compute binary motion mask from BG subtraction.

    Only trusts motion in regions where bg_confidence exceeds the gate.

    Args:
        frame: (H, W, C) float32 RGB frame.
        bg_plate: (H, W, C) float32 BG plate.
        bg_confidence: (H, W) float32 confidence map.
        fg_diff_threshold: Difference threshold for motion detection.
        confidence_gate: Minimum bg_confidence to trust motion.
        photometric_normalize: Compensate for exposure drift.

    Returns:
        (H, W) bool motion mask.
    """
    fg_diff = compute_fg_diff(
        frame, bg_plate, bg_confidence,
        photometric_normalize=photometric_normalize,
    )

    motion = (fg_diff > fg_diff_threshold) & (bg_confidence > confidence_gate)
    return motion
