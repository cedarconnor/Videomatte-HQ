"""Motion-adaptive temporal alpha smoothing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TemporalSmoothConfig:
    enabled: bool = True
    strength: float = 0.55
    motion_threshold: float = 0.03


def apply_temporal_smooth(
    alphas: list[np.ndarray],
    cfg: TemporalSmoothConfig,
) -> list[np.ndarray]:
    """Apply temporal smoothing to an alpha sequence.

    Uses a two-pass approach:
    1. **Temporal median** over a sliding window to kill isolated frame spikes
    2. **Forward-backward EMA** blend to smooth remaining jitter on static pixels

    Motion-adaptive: pixels with large frame-to-frame delta pass through
    unsmoothed to preserve genuine movement.
    """
    if not cfg.enabled or cfg.strength == 0.0 or len(alphas) <= 1:
        return alphas

    n = len(alphas)
    strength = float(np.clip(cfg.strength, 0.0, 1.0))
    motion_thr = max(float(cfg.motion_threshold), 1e-8)

    # --- Pass 1: temporal median over a small window ---
    # Window radius scales with strength: 1 at low, up to 3 at high
    radius = max(1, min(3, int(round(strength * 4))))
    logger.info("Temporal smooth pass 1/2: median filter, radius=%d, %d frames.", radius, n)

    for t in range(n):
        lo = max(0, t - radius)
        hi = min(n, t + radius + 1)
        if hi - lo < 3:
            continue  # not enough neighbours
        current = np.asarray(alphas[t], dtype=np.float32)
        window = np.stack([np.asarray(alphas[i], dtype=np.float32) for i in range(lo, hi)], axis=0)
        median = np.median(window, axis=0)
        # Only apply median where pixel is relatively static across the window
        spread = np.max(window, axis=0) - np.min(window, axis=0)
        static_mask = (spread < motion_thr).astype(np.float32)
        alphas[t] = current * (1.0 - static_mask) + median * static_mask
        if (t + 1) % 100 == 0:
            logger.info("  median frame %d/%d", t + 1, n)

    # --- Pass 2: forward-backward EMA ---
    logger.info("Temporal smooth pass 2/2: forward-backward EMA, strength=%.2f, threshold=%.3f.", strength, motion_thr)

    # Forward pass
    fwd = [np.asarray(alphas[0], dtype=np.float32)]
    for t in range(1, n):
        current = np.asarray(alphas[t], dtype=np.float32)
        delta = np.abs(current - fwd[t - 1])
        motion = np.clip(delta / motion_thr, 0.0, 1.0)
        weight = 1.0 - strength * (1.0 - motion)
        fwd.append(fwd[t - 1] * (1.0 - weight) + current * weight)
        if (t + 1) % 100 == 0:
            logger.info("  forward EMA frame %d/%d", t + 1, n)

    # Backward pass
    bwd = [None] * n
    bwd[n - 1] = np.asarray(alphas[n - 1], dtype=np.float32)
    for t in range(n - 2, -1, -1):
        current = np.asarray(alphas[t], dtype=np.float32)
        delta = np.abs(current - bwd[t + 1])
        motion = np.clip(delta / motion_thr, 0.0, 1.0)
        weight = 1.0 - strength * (1.0 - motion)
        bwd[t] = bwd[t + 1] * (1.0 - weight) + current * weight
        if (t + 1) % 100 == 0:
            logger.info("  backward EMA frame %d/%d", t + 1, n)

    # Average forward and backward, write back in-place
    for t in range(n):
        alphas[t] = np.clip((fwd[t] + bwd[t]) * 0.5, 0.0, 1.0).astype(np.float32)
    # Free intermediate lists
    del fwd, bwd

    logger.info("Temporal smooth complete: %d frames, strength=%.2f, threshold=%.3f.",
                n, strength, motion_thr)
    return alphas
