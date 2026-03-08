"""Temporal smoothing for SAM binary masks and logits before trimap generation.

Smoothing the segmentation masks *before* trimap construction prevents
single-frame mask jitter from propagating into the unknown band and
causing MEMatte to produce inconsistent alpha edges frame-to-frame.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def smooth_masks_temporal(
    masks: list[np.ndarray],
    radius: int = 1,
) -> list[np.ndarray]:
    """Apply temporal median filter to binary segmentation masks.

    For each pixel, takes the median over a sliding window of
    ``2 * radius + 1`` frames.  On binary (0/1) masks this is equivalent
    to a majority vote — a pixel must be foreground in more than half the
    window frames to stay foreground.  This eliminates isolated single-frame
    mask pops without blurring genuine motion boundaries.

    Operates in-place on the input list.

    Parameters
    ----------
    masks : list[np.ndarray]
        Binary masks (0.0 / 1.0 float32 or uint8).  Modified in-place.
    radius : int
        Temporal window half-size.  1 → 3-frame window, 2 → 5-frame window.
    """
    n = len(masks)
    if n < 3 or radius < 1:
        return masks

    radius = min(radius, 2, n // 2)
    logger.info("Temporal mask smoothing: %d frames, radius=%d (window=%d).", n, radius, 2 * radius + 1)

    # Work on float32 copies so median produces clean 0/1 on binary data
    originals = [np.asarray(m, dtype=np.float32) for m in masks]

    for t in range(n):
        lo = max(0, t - radius)
        hi = min(n, t + radius + 1)
        if hi - lo < 3:
            continue
        window = np.stack(originals[lo:hi], axis=0)
        median = np.median(window, axis=0)
        # Re-threshold to stay strictly binary
        masks[t] = (median >= 0.5).astype(np.float32)

    del originals
    logger.info("Temporal mask smoothing complete.")
    return masks


def smooth_logits_temporal(
    logits: list[np.ndarray],
    radius: int = 1,
) -> list[np.ndarray]:
    """Apply temporal median filter to soft segmentation logits.

    Unlike binary masks, logits carry confidence information.  Temporal
    median in logit space preserves soft edges while removing single-frame
    outlier values.

    Operates in-place on the input list.

    Parameters
    ----------
    logits : list[np.ndarray]
        Raw logit arrays (unbounded float).  Modified in-place.
    radius : int
        Temporal window half-size.
    """
    n = len(logits)
    if n < 3 or radius < 1:
        return logits

    radius = min(radius, 2, n // 2)
    logger.info("Temporal logit smoothing: %d frames, radius=%d (window=%d).", n, radius, 2 * radius + 1)

    originals = [np.asarray(lg, dtype=np.float32) for lg in logits]

    for t in range(n):
        lo = max(0, t - radius)
        hi = min(n, t + radius + 1)
        if hi - lo < 3:
            continue
        window = np.stack(originals[lo:hi], axis=0)
        logits[t] = np.median(window, axis=0).astype(np.float32)

    del originals
    logger.info("Temporal logit smoothing complete.")
    return logits
