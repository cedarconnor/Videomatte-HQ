"""Safe math utilities for logit-space operations throughout the pipeline.

All logit-space operations use LOGIT_EPS = 1e-6 to preserve true 0s and 1s
as closely as possible. Logit operations are only applied inside the band
region — outside the band, raw alpha values pass through unclamped.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Minimal clamp for logit transforms — preserves true 0/1 as closely as possible
LOGIT_EPS: float = 1e-6

# Division stability epsilon
DIV_EPS: float = 1e-8


def safe_logit(x: Tensor) -> Tensor:
    """Logit transform with epsilon clamping.

    x is clamped to [LOGIT_EPS, 1 - LOGIT_EPS] before computing log(x / (1-x)).
    This should ONLY be applied inside the band region.
    """
    x_clamped = torch.clamp(x, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return torch.log(x_clamped / (1.0 - x_clamped))


def safe_sigmoid(x: Tensor) -> Tensor:
    """Sigmoid — thin wrapper for consistency / readability."""
    return torch.sigmoid(x)


def logit_blend(alphas: list[Tensor], weights: list[Tensor]) -> Tensor:
    """Weighted blend of alpha values in logit space.

    Used for chunk crossfade and tile stitching inside the band only.

    Args:
        alphas: List of (1, H, W) or (H, W) alpha tensors.
        weights: List of matching spatial weight tensors (same shape as alphas).

    Returns:
        Blended alpha via sigmoid(sum(w * logit(a)) / sum(w)).
    """
    logit_num = torch.zeros_like(alphas[0])
    logit_den = torch.zeros_like(alphas[0])

    for alpha, weight in zip(alphas, weights):
        L = safe_logit(alpha)
        logit_num = logit_num + weight * L
        logit_den = logit_den + weight

    return safe_sigmoid(logit_num / logit_den.clamp(min=DIV_EPS))
