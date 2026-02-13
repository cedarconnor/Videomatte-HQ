"""VRAM-aware tile size probing and backoff."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

TILE_SIZES = [2048, 1536, 1024]


def available_vram(device: str = "cuda") -> float:
    """Get available VRAM in bytes."""
    if not torch.cuda.is_available():
        return 0.0
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else 0
    free, total = torch.cuda.mem_get_info(idx)
    return float(free)


def estimate_vram(model_name: str, tile_size: int, channels: int = 4) -> float:
    """Rough VRAM estimate for a refiner model at given tile size.

    Conservative estimate: ~4 bytes/param * activation_factor.
    """
    # Rough estimates based on model architecture
    # ViTMatte-small: ~40M params → ~160MB weights
    # Activation memory scales with tile area
    activation_bytes = tile_size * tile_size * channels * 4 * 8  # 8x activations heuristic
    weight_bytes = 200 * 1024 * 1024  # ~200MB conservative

    return activation_bytes + weight_bytes


def select_tile_size(
    model_name: str = "vitmatte",
    device: str = "cuda",
    headroom: float = 0.85,
    tile_sizes: list[int] | None = None,
) -> int:
    """Select largest tile size that fits in VRAM.

    Args:
        model_name: Refiner model name.
        device: CUDA device.
        headroom: VRAM usage limit as fraction (0.85 = use at most 85%).
        tile_sizes: Ordered list of sizes to try (default: [2048, 1536, 1024]).

    Returns:
        Selected tile size.

    Raises:
        RuntimeError: If even the smallest tile size doesn't fit.
    """
    if tile_sizes is None:
        tile_sizes = TILE_SIZES

    vram = available_vram(device)
    budget = vram * headroom

    if vram == 0:
        logger.warning("No CUDA device; defaulting to smallest tile size")
        return tile_sizes[-1]

    for size in tile_sizes:
        est = estimate_vram(model_name, size)
        if est <= budget:
            logger.info(
                f"Tile size selected: {size} "
                f"(est. {est / 1e9:.1f}GB / {budget / 1e9:.1f}GB budget)"
            )
            return size

    raise RuntimeError(
        f"Insufficient VRAM for minimum tile size {tile_sizes[-1]}. "
        f"Available: {vram / 1e9:.1f}GB, needed: {estimate_vram(model_name, tile_sizes[-1]) / 1e9:.1f}GB"
    )
