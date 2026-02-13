"""Reference frame propagation with dynamic range gating."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def propagate_reference(
    ref_alpha: np.ndarray,
    ref_frame_idx: int,
    target_frame_idx: int,
    cumulative_flow_error: float,
    cumulative_motion: float,
    error_limit: float = 15.0,
    motion_limit: float = 50.0,
    range_max: int = 30,
) -> Optional[np.ndarray]:
    """Determine if reference frame result should propagate to target.

    Returns the reference alpha if propagation is valid, None otherwise.

    Args:
        ref_alpha: (H, W) float32 reference frame alpha.
        ref_frame_idx: Index of reference frame.
        target_frame_idx: Index of target frame.
        cumulative_flow_error: Sum of flow consistency errors from ref to target.
        cumulative_motion: Sum of motion magnitudes from ref to target.
        error_limit: Max cumulative flow error for propagation.
        motion_limit: Max cumulative motion for propagation.
        range_max: Hard cap on propagation distance.

    Returns:
        Reference alpha if propagation is valid, None if target should
        use standard pipeline instead.
    """
    distance = abs(target_frame_idx - ref_frame_idx)

    if distance > range_max:
        logger.debug(f"Propagation stopped: distance {distance} > max {range_max}")
        return None

    if cumulative_flow_error > error_limit:
        logger.debug(f"Propagation stopped: flow error {cumulative_flow_error:.1f} > limit {error_limit}")
        return None

    if cumulative_motion > motion_limit:
        logger.debug(f"Propagation stopped: motion {cumulative_motion:.1f} > limit {motion_limit}")
        return None

    return ref_alpha
