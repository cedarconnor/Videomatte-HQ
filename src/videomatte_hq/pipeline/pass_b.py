"""Pass B — Edge refinement with trimap-guided tiling.

Runs the edge refiner on boundary-only tiles at native resolution,
then stitches results via band-scoped logit blending.

Tiles of the same size are batched together for GPU efficiency.
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
    tile_batch_size = getattr(cfg.tiles, 'tile_batch_size', 4)

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

        # ---------------------------------------------------------------
        # Group tiles by size for batched inference
        # ---------------------------------------------------------------
        size_groups: dict[tuple[int, int], list[tuple[int, Tile]]] = {}
        for idx, tile in enumerate(tiles):
            key = (tile.y1 - tile.y0, tile.x1 - tile.x0)
            if key not in size_groups:
                size_groups[key] = []
            size_groups[key].append((idx, tile))

        tile_alphas: list[Optional[tuple[Tile, np.ndarray]]] = [None] * len(tiles)

        for (gh, gw), group_items in size_groups.items():
            # Process each size group in sub-batches
            for batch_start in range(0, len(group_items), tile_batch_size):
                batch_items = group_items[batch_start:batch_start + tile_batch_size]

                rgb_tiles_batch = []
                trimap_tiles_batch = []
                alpha_tiles_batch = []
                bg_tiles_batch = []

                for _, tile in batch_items:
                    rgb_crop = frame[tile.y0:tile.y1, tile.x0:tile.x1]
                    trimap_crop = trimap[tile.y0:tile.y1, tile.x0:tile.x1]
                    alpha_crop = a0prime[tile.y0:tile.y1, tile.x0:tile.x1]

                    rgb_tiles_batch.append(
                        torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float()
                    )
                    trimap_tiles_batch.append(
                        torch.from_numpy(trimap_crop).unsqueeze(0).float()
                    )
                    alpha_tiles_batch.append(
                        torch.from_numpy(alpha_crop).unsqueeze(0).float()
                    )

                    bg_t = None
                    if (
                        bg_plate is not None
                        and bg_confidence is not None
                        and cfg.refine.use_bg_plate
                    ):
                        bg_crop = bg_plate[tile.y0:tile.y1, tile.x0:tile.x1].copy()
                        conf_crop = bg_confidence[tile.y0:tile.y1, tile.x0:tile.x1]
                        bg_crop[conf_crop < cfg.refine.bg_confidence_gate] = 0.0
                        bg_t = torch.from_numpy(bg_crop.transpose(2, 0, 1)).float()
                    bg_tiles_batch.append(bg_t)

                # Batched inference
                try:
                    if t == 0 and batch_start == 0:
                        logger.debug(
                            f"Pass B batch: {len(batch_items)} tiles of {gw}×{gh}"
                        )

                    results = refiner.infer_tile_batch(
                        rgb_tiles_batch,
                        trimap_tiles_batch,
                        alpha_tiles_batch,
                        bg_tiles_batch,
                    )

                    for (orig_idx, tile), refined_tensor in zip(batch_items, results):
                        tile_alphas[orig_idx] = (tile, refined_tensor[0].cpu().numpy())

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"OOM on batch of {len(batch_items)} tiles, "
                        f"using backbone alpha"
                    )
                    torch.cuda.empty_cache()
                    for orig_idx, tile in batch_items:
                        alpha_crop = a0prime[tile.y0:tile.y1, tile.x0:tile.x1]
                        tile_alphas[orig_idx] = (tile, alpha_crop)

                except RuntimeError as e:
                    logger.error(
                        f"RuntimeError in Pass B batch frame {t}: {e}"
                    )
                    raise

        # Filter out any None entries (safety net)
        tile_alphas_clean = [ta for ta in tile_alphas if ta is not None]

        # Stitch tiles
        a1 = stitch_tiles(tile_alphas_clean, a0prime, band, feather)
        a1_results.append(a1)

        if t % 50 == 0:
            logger.info(
                f"Pass B: frame {t}/{num_frames}, "
                f"{len(tiles)} tiles (batched, max_batch={tile_batch_size})"
            )

    logger.info(f"Pass B complete: {num_frames} frames")
    return a1_results
