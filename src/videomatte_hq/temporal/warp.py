"""Grid-sample warp implementation with explicit coordinate conventions.

Conventions (from design doc §12.3):
- Pixel coordinates: 0-indexed, pixel-center (pixel (0,0) has center (0.0, 0.0))
- Flow direction: flow_{src→dst} maps src coords to dst coords
- Positive flow_x = rightward, positive flow_y = downward
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def warp(image: Tensor, flow: Tensor) -> Tensor:
    """Warp an image using optical flow via grid_sample.

    Samples `image` (defined at source coordinates) at positions displaced
    by `flow` to produce a result aligned with destination coordinates.

    Args:
        image: (B, C, H, W) tensor to warp.
        flow: (B, 2, H, W) flow field mapping src→dst (dx, dy in pixels).

    Returns:
        (B, C, H, W) warped image.
    """
    B, C, H, W = image.shape

    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=flow.dtype),
        torch.arange(W, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Apply flow displacement
    sample_x = grid_x + flow[:, 0]  # (B, H, W)
    sample_y = grid_y + flow[:, 1]

    # Normalize to [-1, 1] for grid_sample
    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0

    grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(
        image, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
