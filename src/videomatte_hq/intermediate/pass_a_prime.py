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


def _mean_abs_diff_small(curr: np.ndarray, prev: np.ndarray, max_long_side: int = 320) -> float:
    """Fast motion proxy using downscaled mean absolute difference."""
    if curr.ndim == 2:
        curr = curr[..., np.newaxis]
    if prev.ndim == 2:
        prev = prev[..., np.newaxis]

    h, w = curr.shape[:2]
    if prev.shape[:2] != (h, w):
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR)
        if prev.ndim == 2:
            prev = prev[..., np.newaxis]

    scale = min(max_long_side / max(h, w), 1.0)
    if scale < 1.0:
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        curr = cv2.resize(curr, (sw, sh), interpolation=cv2.INTER_LINEAR)
        prev = cv2.resize(prev, (sw, sh), interpolation=cv2.INTER_LINEAR)
        if curr.ndim == 2:
            curr = curr[..., np.newaxis]
        if prev.ndim == 2:
            prev = prev[..., np.newaxis]

    return float(np.abs(curr.astype(np.float32) - prev.astype(np.float32)).mean())


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
    skipped_frames = 0
    last_refined_idx = -1
    prev_delta_full = np.zeros((full_h, full_w), dtype=np.float32)
    prev_delta_4k = None  # for temporal smoothing of filtered delta

    for t in range(num_frames):
        roi = rois[t]
        frame = source[t]  # (H, W, C)
        a0 = a0_results[t]  # (H, W)

        # Crop to ROI
        rgb_crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
        a0_crop = a0[roi.y0:roi.y1, roi.x0:roi.x1]

        # Skip expensive A′ inference on stable frames, periodically forcing a refresh.
        if cfg.intermediate.selective_enabled and t > 0 and last_refined_idx >= 0:
            prev_frame = source[t - 1]
            prev_rgb_crop = prev_frame[roi.y0:roi.y1, roi.x0:roi.x1]
            prev_a0_crop = a0_results[t - 1][roi.y0:roi.y1, roi.x0:roi.x1]

            rgb_change = _mean_abs_diff_small(rgb_crop, prev_rgb_crop)
            a0_change = _mean_abs_diff_small(a0_crop, prev_a0_crop)
            due_recheck = (
                cfg.intermediate.selective_recheck_every > 0
                and (t % cfg.intermediate.selective_recheck_every == 0)
            )
            can_skip_more = (t - last_refined_idx) <= cfg.intermediate.selective_max_skip

            if (
                not due_recheck
                and can_skip_more
                and rgb_change < cfg.intermediate.selective_rgb_threshold
                and a0_change < cfg.intermediate.selective_a0_threshold
            ):
                prev_delta_full *= cfg.intermediate.selective_delta_decay
                a0prime_full = np.clip(a0 + prev_delta_full, 0.0, 1.0).astype(np.float32)
                results.append(a0prime_full)
                skipped_frames += 1
                if t % 50 == 0:
                    logger.info(
                        f"Pass A′: frame {t}/{num_frames} (skipped, rgb={rgb_change:.4f}, a0={a0_change:.4f})"
                    )
                continue

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
        # Temporal smoothing of filtered delta (§9.5)
        if t > 0 and cfg.intermediate.temporal_smooth != "none":
            smooth_strength = cfg.intermediate.smooth_strength
            if prev_delta_4k is not None and prev_delta_4k.shape == delta_filtered.shape:
                # EMA (used for both 'ema' and 'flow' fallback at this stage)
                delta_filtered = (
                    smooth_strength * prev_delta_4k + (1.0 - smooth_strength) * delta_filtered
                ).astype(np.float32)

        prev_delta_4k = delta_filtered.copy()

        # A′ = backbone + medium-frequency corrections only  
        a0prime_4k = np.clip(a0_4k + delta_filtered, 0.0, 1.0)

        # Upscale back to ROI size then paste into full frame
        a0prime_roi = cv2.resize(a0prime_4k, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        a0prime_full = a0_results[t].copy()
        a0prime_full[roi.y0:roi.y1, roi.x0:roi.x1] = a0prime_roi
        a0prime_full = a0prime_full.astype(np.float32)
        prev_delta_full = a0prime_full - a0
        last_refined_idx = t
        results.append(a0prime_full)

        if t % 50 == 0:
            logger.info(f"Pass A′: frame {t}/{num_frames}")

    if cfg.intermediate.selective_enabled:
        logger.info(f"Pass A′ selective skip: {skipped_frames}/{num_frames} frames reused prior delta")
    logger.info(f"Pass A′ complete: {num_frames} frames")
    return results
