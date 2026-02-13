"""Background plate estimation for locked-off shots.

Computes a clean background reference by taking the temporal median across
sampled frames. Includes a deterministic fallback pipeline for persistent
occlusion regions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from videomatte_hq.config import BackgroundConfig
    from videomatte_hq.io.reader import FrameSource

logger = logging.getLogger(__name__)


def _sample_frame_indices(num_frames: int, sample_count: int) -> list[int]:
    """Choose frame indices uniformly across the sequence."""
    if sample_count >= num_frames:
        return list(range(num_frames))
    step = num_frames / sample_count
    return [int(i * step) for i in range(sample_count)]


def estimate_background_plate(
    source: "FrameSource",
    cfg: "BackgroundConfig",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a clean background plate via temporal median.

    Args:
        source: Frame source providing indexed access to frames.
        cfg: Background config section.

    Returns:
        (bg_plate, sampled_frames):
            bg_plate: (H, W, C) float32 — clean background.
            sampled_frames: (N, H, W, C) float32 — the frames used for sampling.
    """
    num_frames = source.num_frames
    indices = _sample_frame_indices(num_frames, cfg.sample_count)
    logger.info(f"Sampling {len(indices)} frames for BG estimation")

    # Load sampled frames into memory
    sampled = []
    for idx in indices:
        frame = source[idx]
        sampled.append(frame)
    sampled_frames = np.stack(sampled, axis=0)  # (N, H, W, C)

    # Temporal median
    bg_plate = np.median(sampled_frames, axis=0).astype(np.float32)  # (H, W, C)
    logger.info(f"BG plate computed: {bg_plate.shape}, range [{bg_plate.min():.3f}, {bg_plate.max():.3f}]")

    return bg_plate, sampled_frames


def apply_occlusion_fallback(
    bg_plate: np.ndarray,
    occlusion_mask: np.ndarray,
    sampled_frames: np.ndarray,
    bg_confidence: np.ndarray,
    method: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Fill in occluded regions of the background plate.

    Fallback pipeline (priority order):
    1. Temporal extremes — find least-occluded sample per pixel
    2. Patch inpainting — spatial patch matching from non-occluded neighbors
    3. AI inpainting — last resort

    Each fallback updates bg_confidence to reflect reliability.

    Args:
        bg_plate: (H, W, C) current BG estimate.
        occlusion_mask: (H, W) bool — True = occluded.
        sampled_frames: (N, H, W, C) sampled frames.
        bg_confidence: (H, W) current confidence map.
        method: 'auto', 'temporal_extremes', 'patch_inpaint', 'ai_inpaint'.

    Returns:
        Updated (bg_plate, bg_confidence).
    """
    if not occlusion_mask.any():
        return bg_plate, bg_confidence

    occluded_pct = occlusion_mask.sum() / occlusion_mask.size * 100
    logger.info(f"Occlusion fallback: {occluded_pct:.1f}% of frame occluded")

    bg_plate = bg_plate.copy()
    bg_confidence = bg_confidence.copy()

    # Method 1: Temporal extremes
    if method in ("auto", "temporal_extremes"):
        from videomatte_hq.io.colorspace import rgb_to_luma

        # For each occluded pixel, find the frame where it's least different from neighbors
        overall_mean = sampled_frames.mean(axis=0)  # (H, W, C)
        luma_diffs = np.zeros((sampled_frames.shape[0],) + occlusion_mask.shape, dtype=np.float32)

        for i, frame in enumerate(sampled_frames):
            luma_diffs[i] = np.abs(rgb_to_luma(frame) - rgb_to_luma(overall_mean))

        # For occluded pixels, pick the frame with minimum diff
        best_frame_idx = np.argmin(luma_diffs, axis=0)  # (H, W)

        oy, ox = np.where(occlusion_mask)
        for y, x in zip(oy, ox):
            best_idx = best_frame_idx[y, x]
            bg_plate[y, x] = sampled_frames[best_idx, y, x]

        # Update confidence for pixels fixed by temporal extremes
        bg_confidence[occlusion_mask] = np.maximum(bg_confidence[occlusion_mask], 0.6)

        # Check if still occluded (heuristic: if best-frame diff is still high)
        still_occluded = occlusion_mask & (luma_diffs.min(axis=0) > 0.15)
        logger.info(f"After temporal extremes: {still_occluded.sum()} pixels still occluded")

        if not still_occluded.any():
            return bg_plate, bg_confidence

        occlusion_mask = still_occluded

    # Method 2: Patch-based inpainting using OpenCV
    if method in ("auto", "patch_inpaint"):
        import cv2

        # Convert to uint8 for inpainting
        bg_u8 = np.clip(bg_plate * 255, 0, 255).astype(np.uint8)
        mask_u8 = occlusion_mask.astype(np.uint8) * 255

        inpainted = cv2.inpaint(bg_u8, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        bg_plate[occlusion_mask] = inpainted[occlusion_mask].astype(np.float32) / 255.0

        bg_confidence[occlusion_mask] = np.maximum(bg_confidence[occlusion_mask], 0.4)
        logger.info("Patch inpainting applied to remaining occluded regions")

    return bg_plate, bg_confidence
