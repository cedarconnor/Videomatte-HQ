"""Tile planning — boundary-only tiles with VRAM-aware sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from videomatte_hq.config import TileConfig
    from videomatte_hq.roi.detect import BBox

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """A single tile within the frame."""
    x0: int
    y0: int
    x1: int
    y1: int
    band_coverage: float = 0.0
    has_hair: bool = False

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def plan_tiles(
    band: np.ndarray,
    roi: "BBox",
    cfg: "TileConfig",
    hair_mask: Optional[np.ndarray] = None,
) -> list[Tile]:
    """Generate boundary-only tiles covering the band region.

    Args:
        band: (H, W) bool band mask.
        roi: ROI bounding box (tiles are generated within ROI).
        cfg: Tile configuration.
        hair_mask: (H, W) optional hair mask for priority sorting.

    Returns:
        List of Tile objects, sorted by priority.
    """
    tile_size = cfg.tile_size
    overlap = cfg.overlap
    min_coverage = cfg.min_band_coverage

    h, w = band.shape

    # Clip ROI to frame bounds
    rx0 = max(0, roi.x0)
    ry0 = max(0, roi.y0)
    rx1 = min(w, roi.x1)
    ry1 = min(h, roi.y1)

    # Generate grid tiles within ROI
    tiles: list[Tile] = []
    if overlap >= tile_size:
        logger.error(
            "Invalid tiling: overlap=%d >= tile_size=%d. Clamping overlap to tile_size - 1.",
            overlap,
            tile_size,
        )
        overlap = max(0, tile_size - 1)
    step = max(1, tile_size - overlap)

    y = ry0
    while y < ry1:
        x = rx0
        while x < rx1:
            tx0 = x
            ty0 = y
            tx1 = min(x + tile_size, rx1)
            ty1 = min(y + tile_size, ry1)

            # Calculate band coverage
            tile_band = band[ty0:ty1, tx0:tx1]
            tile_area = (ty1 - ty0) * (tx1 - tx0)
            coverage = tile_band.sum() / max(tile_area, 1)

            if coverage >= min_coverage:
                has_hair = False
                if hair_mask is not None:
                    tile_hair = hair_mask[ty0:ty1, tx0:tx1]
                    has_hair = tile_hair.any()

                tiles.append(Tile(
                    x0=tx0, y0=ty0, x1=tx1, y1=ty1,
                    band_coverage=coverage,
                    has_hair=has_hair,
                ))

            x += step
        y += step

    # Priority sort: hair-region tiles first, then by coverage
    if cfg.priority == "hair_first":
        tiles.sort(key=lambda t: (-int(t.has_hair), -t.band_coverage))
    else:
        tiles.sort(key=lambda t: -t.band_coverage)

    logger.debug(f"Tile plan: {len(tiles)} tiles from grid, tile_size={tile_size}, overlap={overlap}")
    return tiles
