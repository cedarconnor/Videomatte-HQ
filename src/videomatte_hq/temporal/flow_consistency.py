"""Forward-backward flow consistency check and band-scoped metrics."""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from videomatte_hq.temporal.warp import warp

logger = logging.getLogger(__name__)


def compute_flow_confidence(
    flow_forward: Tensor,
    flow_backward: Tensor,
    sigma: float = 2.0,
) -> Tensor:
    """Compute per-pixel flow confidence via forward-backward consistency.

    If flow is consistent, flow_forward + warp(flow_backward, flow_forward) ≈ 0.

    Args:
        flow_forward: (1, 2, H, W) flow from t-1 to t.
        flow_backward: (1, 2, H, W) flow from t to t-1.
        sigma: Consistency sigma (larger = more tolerant).

    Returns:
        (1, 1, H, W) confidence in [0, 1].
    """
    # Warp backward flow into frame t-1's coordinate system
    flow_backward_warped = warp(flow_backward, flow_forward)

    # If consistent: forward + backward_warped ≈ 0
    consistency_error = torch.norm(flow_forward + flow_backward_warped, dim=1, keepdim=True)

    # Soft confidence
    confidence = torch.exp(-consistency_error / sigma)
    return confidence


def compute_band_flow_metrics(
    flow_confidence: Tensor,
    band: Tensor,
) -> dict:
    """Compute flow sanity metrics scoped to the band region.

    Args:
        flow_confidence: (1, 1, H, W) confidence map.
        band: (H, W) bool band mask.

    Returns:
        Dict with metrics for QC logging.
    """
    if not band.any():
        return {"avg_fb_error_band": 0.0, "pct_confident_band": 1.0}

    band_mask = band.unsqueeze(0).unsqueeze(0) if band.dim() == 2 else band
    conf_in_band = flow_confidence[band_mask.expand_as(flow_confidence)]

    avg_conf = float(conf_in_band.mean())
    pct_confident = float((conf_in_band > 0.5).float().mean())

    return {
        "avg_fb_error_band": 1.0 - avg_conf,
        "pct_confident_band": pct_confident,
    }
