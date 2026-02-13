"""Pass A′ — Intermediate refinement at 4K.

Bridges the resolution gap between Pass A (2048) and Pass B (8K).
Applies guided-filter delta clamping to constrain corrections to medium frequency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.intermediate.guided_filter import guided_filter
from videomatte_hq.roi.detect import BBox

logger = logging.getLogger(__name__)


def run_pass_a_prime(
    source,
    a0_results: list[np.ndarray],
    rois: list[BBox],
    cfg: VideoMatteConfig,
    cache_dir: Path,
) -> list[np.ndarray]:
    """Run Pass A′: intermediate 4K refinement.

    Args:
        source: FrameSource.
        a0_results: Pass A alpha results at full resolution.
        rois: Per-frame ROIs.
        cfg: Pipeline config.
        cache_dir: Cache directory.

    Returns:
        List of (H, W) float32 alpha arrays (A0prime_8k).
    """
    num_frames = source.num_frames
    full_h, full_w = source.resolution
    long_side = cfg.intermediate.long_side

    # Load refiner model
    model_name = cfg.intermediate.model
    if model_name == "vitmatte":
        from videomatte_hq.models.edge_vitmatte import ViTMatteModel
        refiner = ViTMatteModel(device=cfg.runtime.device, precision=cfg.runtime.precision)
    else:
        raise ValueError(f"Unknown intermediate model: {model_name}")

    refiner.load_weights(cfg.runtime.device)

    results = []

    for t in range(num_frames):
        roi = rois[t]
        frame = source[t]  # (H, W, C)
        a0 = a0_results[t]  # (H, W)

        # Crop to ROI
        rgb_crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
        a0_crop = a0[roi.y0:roi.y1, roi.x0:roi.x1]

        # Resize to 4K working resolution
        crop_h, crop_w = rgb_crop.shape[:2]
        scale = min(long_side / max(crop_h, crop_w), 1.0)
        work_h, work_w = int(crop_h * scale), int(crop_w * scale)
        # Round to multiple of 16
        work_h = ((work_h + 15) // 16) * 16
        work_w = ((work_w + 15) // 16) * 16

        rgb_4k = cv2.resize(rgb_crop, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
        a0_4k = cv2.resize(a0_crop, (work_w, work_h), interpolation=cv2.INTER_LINEAR)

        # Generate trimap from A0 for guidance
        trimap_4k = np.full((work_h, work_w), 0.5, dtype=np.float32)
        trimap_4k[a0_4k > 0.95] = 1.0
        trimap_4k[a0_4k < 0.05] = 0.0

        # Run refiner
        rgb_tensor = torch.from_numpy(rgb_4k.transpose(2, 0, 1)).float()
        trimap_tensor = torch.from_numpy(trimap_4k).unsqueeze(0).float()

        a_prime_raw = refiner.infer_frame(rgb_tensor, trimap_tensor)
        a_prime_raw = a_prime_raw[0].cpu().numpy()  # (H, W)

        # Guided-filter delta clamping: constrain to medium frequencies
        delta_raw = a_prime_raw - a0_4k
        delta_filtered = guided_filter(
            guide=rgb_4k,
            src=delta_raw,
            radius=cfg.intermediate.guide_filter_radius,
            eps=cfg.intermediate.guide_filter_eps,
        )

        # A′ = backbone + medium-frequency corrections only  
        a0prime_4k = np.clip(a0_4k + delta_filtered, 0.0, 1.0)

        # Upscale back to ROI size then paste into full frame
        a0prime_roi = cv2.resize(a0prime_4k, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        a0prime_full = a0_results[t].copy()
        a0prime_full[roi.y0:roi.y1, roi.x0:roi.x1] = a0prime_roi
        results.append(a0prime_full)

        if t % 50 == 0:
            logger.info(f"Pass A′: frame {t}/{num_frames}")

    logger.info(f"Pass A′ complete: {num_frames} frames")
    return results
