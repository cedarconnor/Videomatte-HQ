from __future__ import annotations

import numpy as np

from videomatte_hq.tiling.planner import Tile
from videomatte_hq.tiling.stitch import stitch_tiles


def test_stitch_tiles_accumulates_only_inside_band() -> None:
    a0prime = np.zeros((5, 5), dtype=np.float32)
    band = np.zeros((5, 5), dtype=bool)
    band[2, 2] = True

    # Intentionally set feather=1 everywhere to make any outside-band accumulation visible.
    # stitch_tiles should still keep outside-band output at the backbone alpha because
    # tile accumulation is restricted to the band.
    feather = np.ones((5, 5), dtype=np.float32)

    tile = Tile(x0=0, y0=0, x1=5, y1=5)
    alpha_tile = np.full((5, 5), 0.9, dtype=np.float32)

    out = stitch_tiles([(tile, alpha_tile)], a0prime=a0prime, band=band, feather=feather)

    outside = out[~band]
    assert float(outside.max()) == 0.0
    assert float(out[2, 2]) > 0.5
