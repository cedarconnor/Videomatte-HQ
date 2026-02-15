
import numpy as np
import pytest
from videomatte_hq.tiling.planner import Tile
from videomatte_hq.tiling.stitch import stitch_tiles

def test_stitch_tiles_repro():
    """Reproduce potential crash in stitching logic."""
    full_h, full_w = 2160, 3840
    a0prime = np.zeros((full_h, full_w), dtype=np.float32)
    band = np.zeros((full_h, full_w), dtype=bool)
    feather = np.zeros((full_h, full_w), dtype=np.float32)
    
    # Mock band and feather
    band[1000:1100, 1000:1100] = True
    feather[1000:1100, 1000:1100] = 1.0
    
    # Mock tiles
    tiles = [
        Tile(x0=1000, y0=1000, x1=1100, y1=1100),
    ]
    
    tile_alphas = []
    for tile in tiles:
        # Create tile alpha matching tile size
        th, tw = tile.y1 - tile.y0, tile.x1 - tile.x0
        tile_alpha = np.ones((th, tw), dtype=np.float32) * 0.5
        tile_alphas.append((tile, tile_alpha))
        
    # Run stitching
    result = stitch_tiles(tile_alphas, a0prime, band, feather)
    assert result.shape == (full_h, full_w)
    print("Stitching OK")

if __name__ == "__main__":
    test_stitch_tiles_repro()
