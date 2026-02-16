"""Memory-stage foreground region prior generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.propagation_assist import propagate_masks_assist

logger = logging.getLogger(__name__)


@dataclass
class MemoryRegionPriorResult:
    """Metadata and dense per-frame priors used to constrain memory pass."""

    priors: list[np.ndarray]  # [H, W] float32 in [0, 1], one per local frame
    guidance_masks: list[np.ndarray] | None  # [H, W] coarse propagated subject mask per local frame
    mode: str
    anchor_local_frame: int
    anchor_absolute_frame: int
    backend_requested: str | None
    backend_used: str | None
    mean_coverage: float
    min_coverage: float
    max_coverage: float
    note: str | None = None


def _source_num_frames(source: Any) -> int:
    if hasattr(source, "num_frames"):
        return int(source.num_frames)
    return len(source)


def _source_resolution(source: Any) -> tuple[int, int]:
    if hasattr(source, "resolution"):
        return tuple(source.resolution)
    first = source[0]
    return int(first.shape[0]), int(first.shape[1])


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=np.float32)
    if out.ndim == 3:
        out = out[..., 0]
    max_val = float(out.max(initial=0.0))
    if max_val > 1.0:
        out = out / max(max_val, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim != 3:
        raise ValueError("Expected RGB-like frame for memory region prior propagation.")
    if rgb.shape[2] > 3:
        rgb = rgb[..., :3]
    out = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        out /= float(np.iinfo(rgb.dtype).max)
    elif float(out.max(initial=0.0)) > 1.0:
        out /= max(float(out.max(initial=0.0)), 1.0)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def _resolve_local_keyframes(
    keyframe_masks: dict[int, np.ndarray],
    num_frames: int,
    frame_start: int,
) -> dict[int, np.ndarray]:
    direct = {k: v for k, v in keyframe_masks.items() if 0 <= k < num_frames}
    shifted = {k - frame_start: v for k, v in keyframe_masks.items() if 0 <= (k - frame_start) < num_frames}
    return shifted if len(shifted) > len(direct) else direct


def _pick_anchor_local_frame(
    local_keyframes: dict[int, np.ndarray],
    cfg: VideoMatteConfig,
) -> int:
    frames = sorted(local_keyframes.keys())
    if not frames:
        raise ValueError("Cannot pick anchor frame without local keyframes.")

    requested = int(getattr(cfg.memory, "region_constraint_anchor_frame", -1))
    if requested < 0:
        return int(frames[0])
    if requested in local_keyframes:
        return int(requested)

    frame_start = max(int(cfg.io.frame_start), 0)
    requested_local = requested - frame_start
    if requested_local in local_keyframes:
        return int(requested_local)

    nearest = min(frames, key=lambda x: abs(int(x) - int(requested_local)))
    logger.warning(
        "Memory region prior: requested anchor frame %d not assigned; using nearest local keyframe %d.",
        requested,
        int(nearest),
    )
    return int(nearest)


def _mask_to_bbox_prior(
    alpha: np.ndarray,
    threshold: float,
    margin_px: int,
    expand_ratio: float,
) -> np.ndarray:
    h, w = alpha.shape[:2]
    fg = alpha >= float(np.clip(threshold, 0.0, 1.0))
    if not fg.any():
        return np.zeros((h, w), dtype=np.float32)

    ys, xs = np.where(fg)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad_x = int(round(max(float(margin_px), bw * float(max(expand_ratio, 0.0)))))
    pad_y = int(round(max(float(margin_px), bh * float(max(expand_ratio, 0.0)))))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)

    prior = np.zeros((h, w), dtype=np.float32)
    prior[y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return prior


def _dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    r = max(0, int(radius_px))
    if r <= 0:
        return mask.astype(np.float32)
    k = max(1, 2 * r + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate((mask > 0.0).astype(np.uint8), kernel, iterations=1)
    return dilated.astype(np.float32)


def _soften_prior(mask: np.ndarray, radius_px: int) -> np.ndarray:
    r = max(0, int(radius_px))
    if r <= 0:
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    k = max(1, 2 * r + 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigmaX=0.0, sigmaY=0.0)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def _nearest_keyframe_mask(frame_idx: int, local_keyframes: dict[int, np.ndarray]) -> np.ndarray:
    anchor = min(local_keyframes.keys(), key=lambda k: abs(int(k) - int(frame_idx)))
    return local_keyframes[int(anchor)]


def build_memory_region_priors(
    source: Any,
    keyframe_masks: dict[int, np.ndarray],
    cfg: VideoMatteConfig,
) -> MemoryRegionPriorResult | None:
    """Build dense per-frame region priors for Stage 2 memory constraint."""

    enabled = bool(getattr(cfg.memory, "region_constraint_enabled", False))
    source_mode = str(getattr(cfg.memory, "region_constraint_source", "none") or "none").strip().lower()
    if not enabled or source_mode == "none":
        return None

    num_frames = _source_num_frames(source)
    frame_start = max(int(cfg.io.frame_start), 0)
    full_h, full_w = _source_resolution(source)
    local_keyframes = _resolve_local_keyframes(
        keyframe_masks=keyframe_masks,
        num_frames=num_frames,
        frame_start=frame_start,
    )
    if not local_keyframes:
        logger.warning("Memory region prior: no keyframe assignments overlap clip range; skipping.")
        return None

    anchor_local = _pick_anchor_local_frame(local_keyframes=local_keyframes, cfg=cfg)
    anchor_abs = int(frame_start + anchor_local)
    anchor_alpha = _normalize_mask(local_keyframes[anchor_local])
    anchor_cov = float(anchor_alpha.mean())
    if anchor_cov <= 1e-6:
        logger.warning(
            "Memory region prior: anchor frame %d has empty mask coverage (%.6f); skipping constraint.",
            anchor_abs,
            anchor_cov,
        )
        return None

    mode = source_mode
    backend_requested: str | None = None
    backend_used: str | None = None
    note: str | None = None
    propagated_masks: dict[int, np.ndarray] = {}

    if mode in {"propagated_bbox", "propagated_mask"}:
        backend_requested = str(getattr(cfg.memory, "region_constraint_backend", "sam2_video_predictor"))
        prop_result = propagate_masks_assist(
            frame_loader=lambda idx: _to_rgb_u8(source[int(idx)]),
            frame_start=0,
            frame_end=max(num_frames - 1, 0),
            anchor_frame=int(anchor_local),
            anchor_mask=anchor_alpha,
            backend=backend_requested,
            fallback_to_flow=bool(getattr(cfg.memory, "region_constraint_fallback_to_flow", True)),
            flow_downscale=float(getattr(cfg.memory, "region_constraint_flow_downscale", 0.5)),
            flow_min_coverage=float(getattr(cfg.memory, "region_constraint_flow_min_coverage", 0.002)),
            flow_max_coverage=float(getattr(cfg.memory, "region_constraint_flow_max_coverage", 0.98)),
            flow_feather_px=max(0, int(getattr(cfg.memory, "region_constraint_flow_feather_px", 1))),
            samurai_model_cfg=str(getattr(cfg.memory, "region_constraint_samurai_model_cfg", "") or ""),
            samurai_checkpoint=str(getattr(cfg.memory, "region_constraint_samurai_checkpoint", "") or ""),
            samurai_offload_video_to_cpu=bool(
                getattr(cfg.memory, "region_constraint_samurai_offload_video_to_cpu", False)
            ),
            samurai_offload_state_to_cpu=bool(
                getattr(cfg.memory, "region_constraint_samurai_offload_state_to_cpu", False)
            ),
            device_hint=str(getattr(cfg.runtime, "device", "cuda") or "cuda"),
        )
        backend_used = prop_result.backend_used
        note = prop_result.note
        propagated_masks = {int(k): _normalize_mask(v) for k, v in prop_result.masks.items()}
        if not propagated_masks:
            logger.warning("Memory region prior: propagation returned no masks; skipping constraint.")
            return None
    elif mode != "nearest_keyframe_bbox":
        logger.warning("Memory region prior: unsupported source '%s'; skipping.", mode)
        return None

    threshold = float(np.clip(getattr(cfg.memory, "region_constraint_threshold", 0.2), 0.0, 1.0))
    margin_px = max(0, int(getattr(cfg.memory, "region_constraint_bbox_margin_px", 96)))
    expand_ratio = float(max(getattr(cfg.memory, "region_constraint_bbox_expand_ratio", 0.15), 0.0))
    dilate_px = max(0, int(getattr(cfg.memory, "region_constraint_dilate_px", 24)))
    soften_px = max(0, int(getattr(cfg.memory, "region_constraint_soften_px", 0)))
    min_cov = float(np.clip(getattr(cfg.memory, "region_constraint_flow_min_coverage", 0.002), 0.0, 1.0))
    max_cov = float(np.clip(getattr(cfg.memory, "region_constraint_flow_max_coverage", 0.98), 0.0, 1.0))

    priors: list[np.ndarray] = []
    guidance_masks: list[np.ndarray] = []
    coverages: list[float] = []
    prev_prior: np.ndarray | None = None

    for t in range(num_frames):
        if mode in {"propagated_bbox", "propagated_mask"}:
            base = propagated_masks.get(int(t))
            if base is None:
                base = propagated_masks.get(int(anchor_local), anchor_alpha)
        else:
            base = _normalize_mask(_nearest_keyframe_mask(t, local_keyframes))

        if base.shape[:2] != (full_h, full_w):
            base = cv2.resize(base, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
            base = np.clip(base, 0.0, 1.0).astype(np.float32)
        guidance_masks.append(np.clip(base, 0.0, 1.0).astype(np.float32))

        if mode in {"propagated_bbox", "nearest_keyframe_bbox"}:
            prior = _mask_to_bbox_prior(
                alpha=base,
                threshold=threshold,
                margin_px=margin_px,
                expand_ratio=expand_ratio,
            )
        else:
            prior = (base >= threshold).astype(np.float32)

        prior = _dilate_binary(prior, radius_px=dilate_px)
        prior = _soften_prior(prior, radius_px=soften_px)

        cov = float(prior.mean())
        if cov < min_cov or cov > max_cov:
            if prev_prior is not None:
                prior = prev_prior.copy()
                cov = float(prior.mean())
            else:
                prior = np.ones((full_h, full_w), dtype=np.float32)
                cov = 1.0

        priors.append(np.clip(prior, 0.0, 1.0).astype(np.float32))
        coverages.append(cov)
        prev_prior = priors[-1]

    return MemoryRegionPriorResult(
        priors=priors,
        guidance_masks=guidance_masks,
        mode=mode,
        anchor_local_frame=int(anchor_local),
        anchor_absolute_frame=int(anchor_abs),
        backend_requested=backend_requested,
        backend_used=backend_used,
        mean_coverage=float(np.mean(coverages)) if coverages else 0.0,
        min_coverage=float(np.min(coverages)) if coverages else 0.0,
        max_coverage=float(np.max(coverages)) if coverages else 0.0,
        note=note,
    )
