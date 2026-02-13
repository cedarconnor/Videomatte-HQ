"""Temporal stabilization — flow-guided delta stabilization (Pass C).

Stabilizes the high-frequency edge delta while keeping the backbone untouched.
Structural delta gets conservative stabilization; detail gets aggressive.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.temporal.frequency_separation import split_delta, classify_structural_regions

logger = logging.getLogger(__name__)


def run_temporal_stabilization(
    a0prime_results: list[np.ndarray],
    a1_results: list[np.ndarray],
    per_frame_data: list[dict],
    source,
    cfg: VideoMatteConfig,
) -> list[np.ndarray]:
    """Run Pass C: frequency-separation temporal stabilization.

    Args:
        a0prime_results: Pass A′ alpha (backbone) per frame.
        a1_results: Pass B alpha (refined) per frame.
        per_frame_data: Per-frame band/trimap data.
        source: FrameSource for RGB frames.
        cfg: Pipeline config.

    Returns:
        List of (H, W) float32 final alpha per frame.
    """
    num_frames = len(a0prime_results)

    if cfg.temporal.method == "none" or num_frames < 2:
        return [np.clip(a1, 0, 1) for a1 in a1_results]

    # Extract config
    structural_sigma = cfg.temporal.structural_sigma
    structural_threshold = cfg.temporal.structural_threshold
    structural_strength = cfg.temporal.structural_blend_strength
    detail_strength = cfg.temporal.detail_blend_strength
    fallback_threshold = cfg.temporal.fallback_threshold

    # Load flow model if needed
    flow_model = None
    if cfg.temporal.method == "frequency_separation":
        from videomatte_hq.models.flow_raft import RAFTFlowModel
        flow_model = RAFTFlowModel(
            device=cfg.runtime.device,
            precision=cfg.runtime.precision,
        )
        flow_model.load_weights(cfg.runtime.device)

    final_alphas = [None] * num_frames

    # First frame: no stabilization needed
    D0 = a1_results[0] - a0prime_results[0]
    D0_structural, D0_detail = split_delta(D0, structural_sigma)
    final_alphas[0] = np.clip(a0prime_results[0] + D0, 0.0, 1.0)

    prev_D_structural_stable = D0_structural
    prev_D_detail_stable = D0_detail

    for t in range(1, num_frames):
        # Extract delta
        D_t = a1_results[t] - a0prime_results[t]
        D_structural, D_detail = split_delta(D_t, structural_sigma)

        if flow_model is not None and cfg.temporal.method == "frequency_separation":
            # Compute flow
            frame_prev = source[t - 1]
            frame_curr = source[t]

            frame_prev_t = torch.from_numpy(frame_prev.transpose(2, 0, 1)).float().unsqueeze(0)
            frame_curr_t = torch.from_numpy(frame_curr.transpose(2, 0, 1)).float().unsqueeze(0)

            from videomatte_hq.temporal.flow_consistency import compute_flow_confidence

            flow_fwd, flow_bwd = flow_model.compute_bidirectional_flow(frame_prev_t, frame_curr_t)
            flow_confidence = compute_flow_confidence(flow_fwd, flow_bwd, cfg.temporal.flow_consistency_sigma)

            # Convert flow confidence to numpy
            conf = flow_confidence[0, 0].cpu().numpy()  # (H, W)

            # Warp previous stabilized deltas to current frame
            from videomatte_hq.temporal.warp import warp as torch_warp

            prev_struct_t = torch.from_numpy(prev_D_structural_stable).float().unsqueeze(0).unsqueeze(0)
            prev_detail_t = torch.from_numpy(prev_D_detail_stable).float().unsqueeze(0).unsqueeze(0)

            # Resize flow to match alpha resolution if needed
            h, w = D_structural.shape
            if flow_fwd.shape[2] != h or flow_fwd.shape[3] != w:
                flow_fwd = torch.nn.functional.interpolate(flow_fwd, (h, w), mode="bilinear", align_corners=True)
                flow_fwd[:, 0] *= w / flow_fwd.shape[3] if flow_fwd.shape[3] != w else 1
                flow_fwd[:, 1] *= h / flow_fwd.shape[2] if flow_fwd.shape[2] != h else 1
                conf = np.array(
                    torch.nn.functional.interpolate(
                        flow_confidence, (h, w), mode="bilinear", align_corners=True
                    )[0, 0].cpu()
                )

            D_struct_prev_warped = torch_warp(prev_struct_t, flow_fwd)[0, 0].cpu().numpy()
            D_detail_prev_warped = torch_warp(prev_detail_t, flow_fwd)[0, 0].cpu().numpy()

            # Structural: conservative stabilization
            structural_blend = conf * structural_strength
            structural_regions = classify_structural_regions(D_structural, structural_threshold)
            # In structural-correction regions, reduce blend even further
            structural_blend[structural_regions] *= 0.5

            D_structural_stable = (1 - structural_blend) * D_structural + structural_blend * D_struct_prev_warped

            # Detail: aggressive stabilization
            detail_blend = conf * detail_strength
            D_detail_stable = (1 - detail_blend) * D_detail + detail_blend * D_detail_prev_warped

            # Fallback: where flow confidence is very low, use raw delta
            low_conf = conf < fallback_threshold
            D_structural_stable[low_conf] = D_structural[low_conf]
            D_detail_stable[low_conf] = D_detail[low_conf]

        else:
            # No flow model — use raw deltas
            D_structural_stable = D_structural
            D_detail_stable = D_detail

        # Compose final alpha
        D_stable = D_structural_stable + D_detail_stable
        final_alphas[t] = np.clip(a0prime_results[t] + D_stable, 0.0, 1.0)

        prev_D_structural_stable = D_structural_stable
        prev_D_detail_stable = D_detail_stable

        if t % 50 == 0:
            logger.info(f"Pass C: frame {t}/{num_frames}")

    logger.info(f"Pass C complete: {num_frames} frames stabilized")
    return final_alphas
