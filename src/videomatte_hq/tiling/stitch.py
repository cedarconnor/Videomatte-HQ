"""Band-scoped tile stitching in logit space.

Tiles are blended using Hann windows in logit space INSIDE the band only.
Outside the band, raw alpha from A0prime passes through unclamped.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from videomatte_hq.safe_math import LOGIT_EPS, DIV_EPS
from videomatte_hq.tiling.planner import Tile
from videomatte_hq.tiling.windows import hann_2d

logger = logging.getLogger(__name__)


def _np_safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(x / (1.0 - x))


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stitch_tiles(
    tile_alphas: list[tuple[Tile, np.ndarray]],
    a0prime: np.ndarray,
    band: np.ndarray,
    feather: np.ndarray,
) -> np.ndarray:
    """Stitch refined tile alphas into full-frame result.

    Logit-space blending inside the band; raw alpha outside.

    Args:
        tile_alphas: List of (Tile, alpha) pairs. Alpha is (H, W) float32.
        a0prime: (H, W) float32 backbone alpha.
        band: (H, W) bool band mask.
        feather: (H, W) float32 feather mask [0, 1].

    Returns:
        (H, W) float32 stitched alpha (A1_8k).
    """
    full_h, full_w = a0prime.shape
    logit_num = np.zeros((full_h, full_w), dtype=np.float64)
    logit_den = np.zeros((full_h, full_w), dtype=np.float64)

    for tile, alpha_tile in tile_alphas:
        th = tile.y1 - tile.y0
        tw = tile.x1 - tile.x0

        # Ensure alpha_tile matches tile dimensions
        if alpha_tile.shape != (th, tw):
            import cv2
            alpha_tile = cv2.resize(alpha_tile, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Hann window
        window = hann_2d(th, tw)

        # Logit transform (inside band only)
        L_tile = _np_safe_logit(alpha_tile)

        # Accumulate
        logit_num[tile.y0:tile.y1, tile.x0:tile.x1] += window * L_tile
        logit_den[tile.y0:tile.y1, tile.x0:tile.x1] += window

    # Stitched result inside band
    has_coverage = logit_den > DIV_EPS
    stitched = np.zeros((full_h, full_w), dtype=np.float32)
    stitched[has_coverage] = _np_sigmoid(logit_num[has_coverage] / logit_den[has_coverage])

    # Compose with backbone using feather mask
    # Outside band: raw alpha from A0prime (no logit, no epsilon, no clamping)
    # Inside band: stitched tiles via feather mask M
    M = feather
    A1 = (1.0 - M) * a0prime + M * stitched

    # Where tiles had no coverage inside band, fall back to a0prime
    no_tile_in_band = band & ~has_coverage
    A1[no_tile_in_band] = a0prime[no_tile_in_band]

    return A1.astype(np.float32)
