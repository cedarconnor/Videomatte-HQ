"""Reference frame auto-selection based on sharpness and edge confidence."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_frame_quality(frame: np.ndarray, alpha: np.ndarray) -> float:
    """Score a frame's quality for reference selection.

    Combines sharpness (Laplacian variance) with edge confidence.

    Args:
        frame: (H, W, C) float32 RGB.
        alpha: (H, W) float32 alpha.

    Returns:
        Quality score (higher = better reference candidate).
    """
    gray = (0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2])
    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

    # Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = laplacian.var()

    # Edge quality: concentrate on edge region
    edge_mask = (alpha > 0.05) & (alpha < 0.95)
    if edge_mask.any():
        edge_sharpness = laplacian[edge_mask].var()
    else:
        edge_sharpness = sharpness

    return float(sharpness * 0.3 + edge_sharpness * 0.7)


def select_reference_frames(
    source,
    alphas: list[np.ndarray],
    count: int = 5,
) -> list[int]:
    """Auto-select reference frames by quality score.

    Picks well-distributed high-quality frames across the sequence.

    Args:
        source: FrameSource.
        alphas: Alpha per frame (from any pass).
        count: Number of reference frames to select.

    Returns:
        Sorted list of frame indices.
    """
    num_frames = len(alphas)
    if num_frames <= count:
        return list(range(num_frames))

    # Score every Nth frame
    sample_step = max(1, num_frames // (count * 10))
    scored = []
    for t in range(0, num_frames, sample_step):
        frame = source[t]
        score = compute_frame_quality(frame, alphas[t])
        scored.append((t, score))

    # Distribute: divide into `count` segments, pick best in each
    segment_size = len(scored) // count
    selected = []
    for i in range(count):
        segment = scored[i * segment_size: (i + 1) * segment_size]
        if segment:
            best = max(segment, key=lambda x: x[1])
            selected.append(best[0])

    selected.sort()
    logger.info(f"Selected {len(selected)} reference frames: {selected}")
    return selected
