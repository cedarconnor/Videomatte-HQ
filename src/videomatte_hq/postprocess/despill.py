"""Edge color despill / decontamination gated by bg_confidence."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def despill_frame(
    frame: np.ndarray,
    alpha: np.ndarray,
    bg_plate: np.ndarray,
    bg_confidence: np.ndarray,
    strength: float = 1.0,
    min_confidence: float = 0.4,
) -> np.ndarray:
    """Remove background color contamination from edge pixels.

    Args:
        frame: (H, W, 3) float32 RGB.
        alpha: (H, W) float32 alpha.
        bg_plate: (H, W, 3) float32 background plate.
        bg_confidence: (H, W) float32 confidence map.
        strength: Despill strength (0=off, 1=full).
        min_confidence: Don't despill where bg_confidence < this.

    Returns:
        (H, W, 3) float32 despilled foreground.
    """
    # Edge pixels
    edge_mask = (alpha > 0.05) & (alpha < 0.95)

    # Gate by bg_confidence
    conf_mask = bg_confidence > min_confidence
    active = edge_mask & conf_mask

    fg = frame.copy()

    if active.any():
        # Premultiplied foreground
        fg_premul = frame * alpha[..., np.newaxis]

        # Estimate spill
        spill = (1.0 - alpha[..., np.newaxis]) * bg_plate

        # Effective strength: scale by bg_confidence
        effective_strength = strength * bg_confidence[..., np.newaxis]

        # Corrected foreground
        fg_clean = fg_premul - spill * effective_strength
        fg_clean = fg_clean / np.maximum(alpha[..., np.newaxis], 1e-8)
        fg_clean = np.clip(fg_clean, 0.0, 1.0)

        # Only apply in edge region
        fg[active] = fg_clean[active]

    return fg


def run_despill(
    source,
    alphas: list[np.ndarray],
    bg_plate: np.ndarray,
    bg_confidence: np.ndarray,
    strength: float = 1.0,
    min_confidence: float = 0.4,
) -> list[np.ndarray]:
    """Run despill on all frames.

    Returns list of despilled foreground frames.
    """
    results = []
    for t in range(len(alphas)):
        frame = source[t]
        fg = despill_frame(frame, alphas[t], bg_plate, bg_confidence, strength, min_confidence)
        results.append(fg)
    return results
