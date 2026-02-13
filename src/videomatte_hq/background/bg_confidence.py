"""Background confidence map and occlusion detection.

bg_confidence indicates per-pixel reliability of the estimated background plate.
High variance across samples → low confidence (person was there most of the time).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from videomatte_hq.config import BackgroundConfig

logger = logging.getLogger(__name__)


def compute_bg_confidence(
    sampled_frames: np.ndarray,
    bg_plate: np.ndarray,
    cfg: "BackgroundConfig",
) -> np.ndarray:
    """Compute per-pixel confidence that the BG plate is valid.

    Args:
        sampled_frames: (N, H, W, C) float32 sampled frames.
        bg_plate: (H, W, C) float32 estimated background.
        cfg: Background config with variance_threshold and occlusion_threshold.

    Returns:
        (H, W) float32 confidence in [0, 1]. High = BG plate is reliable.
    """
    from videomatte_hq.io.colorspace import rgb_to_luma

    # Compute per-pixel temporal variance in luma space
    luma_frames = np.stack([rgb_to_luma(f) for f in sampled_frames], axis=0)  # (N, H, W)
    variance = np.var(luma_frames, axis=0)  # (H, W)

    # High variance = person was there most of the time = low confidence
    bg_confidence = 1.0 - np.clip(variance / cfg.variance_threshold, 0.0, 1.0)
    bg_confidence = bg_confidence.astype(np.float32)

    logger.info(
        f"BG confidence: mean={bg_confidence.mean():.3f}, "
        f"min={bg_confidence.min():.3f}, "
        f"high_conf_pct={100 * (bg_confidence > 0.8).mean():.1f}%"
    )

    return bg_confidence


def detect_occlusion_mask(
    bg_confidence: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Detect persistently occluded regions.

    Args:
        bg_confidence: (H, W) float32 confidence map.
        threshold: Pixels below this confidence are flagged as occluded.

    Returns:
        (H, W) bool mask — True = persistently occluded.
    """
    mask = bg_confidence < threshold
    occluded_pct = mask.sum() / mask.size * 100
    logger.info(f"Occlusion mask: {occluded_pct:.1f}% of pixels flagged (threshold={threshold})")
    return mask
