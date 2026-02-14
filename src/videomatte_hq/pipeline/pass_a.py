"""Pass A — Global matte backbone.

Crops to ROI, downscales, runs temporal video matting model in chunks
with logit-space crossfade, then upscales back to full resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.reader import FrameSource
from videomatte_hq.roi.detect import BBox
from videomatte_hq.safe_math import logit_blend

logger = logging.getLogger(__name__)


def _crop_and_resize(frame: np.ndarray, roi: BBox, long_side: int) -> tuple[np.ndarray, float]:
    """Crop frame to ROI and resize so longest side = long_side.

    Returns (resized_crop, scale_factor).
    """
    crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
    h, w = crop.shape[:2]
    scale = long_side / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _upscale_alpha_to_full(
    alpha_roi: np.ndarray,
    roi: BBox,
    full_h: int,
    full_w: int,
) -> np.ndarray:
    """Upscale ROI-scale alpha back to full frame resolution."""
    # Resize alpha to ROI dimensions
    roi_h = roi.y1 - roi.y0
    roi_w = roi.x1 - roi.x0
    alpha_resized = cv2.resize(alpha_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    # Paste into full frame
    full_alpha = np.zeros((full_h, full_w), dtype=np.float32)
    full_alpha[roi.y0:roi.y1, roi.x0:roi.x1] = alpha_resized
    return full_alpha


def _crossfade_weights(overlap_len: int, total_len: int, position: str) -> np.ndarray:
    """Generate linear crossfade weights for chunk overlap."""
    weights = np.ones(total_len, dtype=np.float32)
    if position == "start":
        # Ramp down at end
        weights[-overlap_len:] = np.linspace(1, 0, overlap_len)
    elif position == "end":
        # Ramp up at start
        weights[:overlap_len] = np.linspace(0, 1, overlap_len)
    elif position == "middle":
        # Ramp up at start, ramp down at end
        weights[:overlap_len] = np.linspace(0, 1, overlap_len)
        weights[-overlap_len:] = np.linspace(1, 0, overlap_len)
    return weights


def run_pass_a(
    source,
    rois: list[BBox],
    cfg: VideoMatteConfig,
    cache_dir: Path,
) -> list[np.ndarray]:
    """Run Pass A: global matte backbone.

    Args:
        source: FrameSource with indexed access.
        rois: Per-frame smoothed ROI bounding boxes.
        cfg: Full pipeline config.
        cache_dir: Cache directory for intermediate results.

    Returns:
        List of (H, W) float32 alpha arrays at full resolution (A0_8k).
    """
    num_frames = source.num_frames
    full_h, full_w = source.resolution
    long_side = cfg.global_.long_side
    chunk_len = cfg.global_.chunk_len
    chunk_overlap = cfg.global_.chunk_overlap

    # Load model
    model_name = cfg.global_.model
    if model_name == "rvm":
        from videomatte_hq.models.global_rvm import RVMModel
        model = RVMModel(
            device=cfg.runtime.device,
            precision=cfg.runtime.precision,
        )
    else:
        raise ValueError(f"Unknown global model: {model_name}")

    model.load_weights(cfg.runtime.device)

    # Precompute a SINGLE padded size across ALL frames so recurrent state
    # dimensions stay consistent across chunks.
    global_max_h = 0
    global_max_w = 0
    for t in range(num_frames):
        roi = rois[t]
        crop_h, crop_w = roi.height, roi.width
        scale = long_side / max(crop_h, crop_w)
        h_scaled = int(crop_h * scale)
        w_scaled = int(crop_w * scale)
        global_max_h = max(global_max_h, h_scaled)
        global_max_w = max(global_max_w, w_scaled)
    # Round up to multiple of 16 for model compatibility
    global_max_h = ((global_max_h + 15) // 16) * 16
    global_max_w = ((global_max_w + 15) // 16) * 16
    logger.info(f"Pass A uniform tensor size: {global_max_w}×{global_max_h}")

    # Process in chunks
    a0_results: list[np.ndarray] = [None] * num_frames
    chunk_start = 0
    recurrent_state = None

    while chunk_start < num_frames:
        chunk_end = min(chunk_start + chunk_len, num_frames)
        actual_len = chunk_end - chunk_start

        logger.info(f"Pass A chunk [{chunk_start}:{chunk_end}] ({actual_len} frames)")

        # Load and preprocess chunk frames — pad to global uniform size
        padded_frames = []
        for t in range(chunk_start, chunk_end):
            frame = source[t]
            roi = rois[t]
            crop, _ = _crop_and_resize(frame, roi, long_side)
            # (H, W, C) → (C, H, W)
            tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float()
            _, h, w = tensor.shape
            padded = torch.zeros(3, global_max_h, global_max_w, dtype=torch.float32)
            padded[:, :h, :w] = tensor
            padded_frames.append(padded)

        batch = torch.stack(padded_frames)  # (T, C, H, W)

        # Run model
        alpha_chunk, recurrent_state = model.infer_chunk(batch, recurrent_state)
        # alpha_chunk: (T, 1, H, W)

        # Determine crossfade position
        is_first = chunk_start == 0
        is_last = chunk_end >= num_frames

        for i in range(actual_len):
            t = chunk_start + i
            roi = rois[t]

            # Extract alpha and un-pad
            crop_h = rois[t].height
            crop_w = rois[t].width
            scale = long_side / max(crop_h, crop_w)
            real_h = int(crop_h * scale)
            real_w = int(crop_w * scale)

            alpha_frame = alpha_chunk[i, 0, :real_h, :real_w].cpu().numpy()

            # Upscale to full resolution
            a0_full = _upscale_alpha_to_full(alpha_frame, roi, full_h, full_w)

            if a0_results[t] is None:
                a0_results[t] = a0_full
            else:
                # Overlap region — blend with previous chunk in logit space
                overlap_idx = i  # position within overlap zone
                blend_alpha = overlap_idx / chunk_overlap if chunk_overlap > 0 else 1.0
                blend_alpha = min(blend_alpha, 1.0)
                # Weighted blend: new chunk weight increases linearly
                a0_results[t] = (1.0 - blend_alpha) * a0_results[t] + blend_alpha * a0_full

        # Advance with overlap
        if not is_last:
            chunk_start = chunk_end - chunk_overlap
        else:
            break

    # Ensure no None results (shouldn't happen)
    for t in range(num_frames):
        if a0_results[t] is None:
            logger.warning(f"Pass A: frame {t} has no result, using zeros")
            a0_results[t] = np.zeros((full_h, full_w), dtype=np.float32)

    logger.info(f"Pass A complete: {num_frames} frames processed")
    return a0_results
