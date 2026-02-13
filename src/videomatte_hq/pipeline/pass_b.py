"""Pass B — Edge refinement with trimap-guided tiling.

Runs the edge refiner on boundary-only tiles at native resolution,
then stitches results via band-scoped logit blending.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.tiling.planner import Tile
from videomatte_hq.tiling.stitch import stitch_tiles
from videomatte_hq.tiling.vram_probe import select_tile_size

logger = logging.getLogger(__name__)


def run_pass_b(
    source,
    a0prime_results: list[np.ndarray],
    per_frame_data: list[dict],
    cfg: VideoMatteConfig,
    bg_plate: Optional[np.ndarray] = None,
    bg_confidence: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Run Pass B: edge refinement on boundary tiles.

    Args:
        source: FrameSource.
        a0prime_results: Pass A′ alpha results (H, W) per frame.
        per_frame_data: Dicts with 'band', 'trimap', 'feather', 'tiles' per frame.
        cfg: Pipeline config.
        bg_plate: Optional background plate.
        bg_confidence: Optional BG confidence map.

    Returns:
        List of (H, W) float32 alpha (A1_8k) per frame.
    """
    num_frames = source.num_frames

    # Select tile size based on VRAM
    try:
        tile_size = select_tile_size(
            cfg.refine.model, cfg.runtime.device,
            cfg.tiles.vram_headroom, cfg.tiles.tile_size_backoff,
        )
    except RuntimeError:
        tile_size = cfg.tiles.tile_size_backoff[-1]
        logger.warning(f"VRAM probe failed, using minimum tile size {tile_size}")

    # Load refiner model
    model_name = cfg.refine.model
    if model_name == "vitmatte":
        from videomatte_hq.models.edge_vitmatte import ViTMatteModel
        refiner = ViTMatteModel(device=cfg.runtime.device, precision=cfg.runtime.precision)
    else:
        raise ValueError(f"Unknown refiner model: {model_name}")

    refiner.load_weights(cfg.runtime.device)

    a1_results = []

    for t in range(num_frames):
        frame = source[t]
        a0prime = a0prime_results[t]
        data = per_frame_data[t]
        band = data["band"]
        trimap = data["trimap"]
        feather = data["feather"]
        tiles = data["tiles"]

        if not tiles:
            # No tiles needed — use backbone alpha
            a1_results.append(a0prime.copy())
            continue

        # Process each tile
        tile_alphas: list[tuple[Tile, np.ndarray]] = []

        for tile in tiles:
            # Extract tile crops
            rgb_crop = frame[tile.y0:tile.y1, tile.x0:tile.x1]
            trimap_crop = trimap[tile.y0:tile.y1, tile.x0:tile.x1]
            alpha_crop = a0prime[tile.y0:tile.y1, tile.x0:tile.x1]

            # Convert to tensors
            rgb_tensor = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float()
            trimap_tensor = torch.from_numpy(trimap_crop).unsqueeze(0).float()
            alpha_tensor = torch.from_numpy(alpha_crop).unsqueeze(0).float()

            # Optional BG plate tile (only where confident)
            bg_tensor = None
            if (
                bg_plate is not None
                and bg_confidence is not None
                and cfg.refine.use_bg_plate
            ):
                bg_crop = bg_plate[tile.y0:tile.y1, tile.x0:tile.x1]
                conf_crop = bg_confidence[tile.y0:tile.y1, tile.x0:tile.x1]
                # Mask out low-confidence regions
                bg_crop = bg_crop.copy()
                bg_crop[conf_crop < cfg.refine.bg_confidence_gate] = 0.0
                bg_tensor = torch.from_numpy(bg_crop.transpose(2, 0, 1)).float()

            # Run refiner with OOM retry
            try:
                refined = refiner.infer_tile(rgb_tensor, trimap_tensor, alpha_tensor, bg_tensor)
                refined_np = refined[0].cpu().numpy()  # (H, W)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM on tile ({tile.x0},{tile.y0}), using backbone alpha")
                torch.cuda.empty_cache()
                refined_np = alpha_crop

            tile_alphas.append((tile, refined_np))

        # Stitch tiles
        a1 = stitch_tiles(tile_alphas, a0prime, band, feather)
        a1_results.append(a1)

        if t % 50 == 0:
            logger.info(f"Pass B: frame {t}/{num_frames}, {len(tiles)} tiles")

    logger.info(f"Pass B complete: {num_frames} frames")
    return a1_results
