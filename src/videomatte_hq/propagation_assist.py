"""Phase 4 long-range keyframe mask propagation assist."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


SUPPORTED_PROPAGATION_BACKENDS = (
    "flow",
    "sam2_video_predictor",
    "cutie",
)


class PropagationUnavailable(RuntimeError):
    """Raised when an optional propagation backend cannot be used."""


@dataclass
class PropagationAssistResult:
    masks: dict[int, np.ndarray]  # local frame index -> alpha float32 [0, 1]
    backend_used: str
    note: str | None = None


def _normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    out = np.asarray(alpha, dtype=np.float32)
    if out.ndim == 3:
        if out.shape[2] == 4:
            out = out[..., 3]
        else:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    if out.size == 0:
        raise ValueError("Anchor mask is empty.")
    if float(out.max(initial=0.0)) > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_gray_small(frame_rgb_u8: np.ndarray, downscale: float) -> np.ndarray:
    if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] < 3:
        raise ValueError("Expected RGB-like frame for propagation assist.")
    gray = cv2.cvtColor(frame_rgb_u8[..., :3], cv2.COLOR_RGB2GRAY)
    if downscale >= 0.999:
        return gray
    h, w = gray.shape[:2]
    new_w = max(16, int(round(w * downscale)))
    new_h = max(16, int(round(h * downscale)))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _warp_mask_with_flow(mask_small: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = mask_small.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    return cv2.remap(
        mask_small.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _postprocess_small_mask(mask_small: np.ndarray, feather_px: int) -> np.ndarray:
    out = np.clip(np.asarray(mask_small, dtype=np.float32), 0.0, 1.0)
    if feather_px > 0:
        k = int(max(1, feather_px)) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.0, sigmaY=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _propagate_flow_backend(
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    anchor_mask: np.ndarray,
    flow_downscale: float,
    flow_min_coverage: float,
    flow_max_coverage: float,
    feather_px: int,
) -> dict[int, np.ndarray]:
    if frame_end < frame_start:
        raise ValueError(f"Invalid propagation range: {frame_start}..{frame_end}")
    if anchor_frame < frame_start or anchor_frame > frame_end:
        raise ValueError(f"Anchor frame {anchor_frame} is outside range {frame_start}..{frame_end}")

    anchor_alpha = _normalize_alpha(anchor_mask)
    h_full, w_full = anchor_alpha.shape[:2]
    downscale = float(np.clip(flow_downscale, 0.15, 1.0))
    small_w = max(16, int(round(w_full * downscale)))
    small_h = max(16, int(round(h_full * downscale)))
    anchor_small = cv2.resize(anchor_alpha, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    anchor_gray_small = _to_gray_small(frame_loader(anchor_frame), downscale=downscale)
    if anchor_gray_small.shape[:2] != (small_h, small_w):
        anchor_gray_small = cv2.resize(anchor_gray_small, (small_w, small_h), interpolation=cv2.INTER_AREA)

    masks_small: dict[int, np.ndarray] = {int(anchor_frame): anchor_small}

    # Forward pass: anchor -> frame_end
    prev_gray = anchor_gray_small
    prev_mask = anchor_small
    for idx in range(anchor_frame + 1, frame_end + 1):
        cur_gray = _to_gray_small(frame_loader(idx), downscale=downscale)
        if cur_gray.shape[:2] != prev_gray.shape[:2]:
            cur_gray = cv2.resize(cur_gray, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=cur_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=35,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        cur_mask = _warp_mask_with_flow(prev_mask, flow)
        cur_mask = _postprocess_small_mask(cur_mask, feather_px=feather_px)
        cov = float(cur_mask.mean())
        if cov < float(flow_min_coverage) or cov > float(flow_max_coverage):
            cur_mask = prev_mask.copy()
        masks_small[int(idx)] = cur_mask
        prev_gray = cur_gray
        prev_mask = cur_mask

    # Backward pass: anchor -> frame_start
    prev_gray = anchor_gray_small
    prev_mask = anchor_small
    for idx in range(anchor_frame - 1, frame_start - 1, -1):
        cur_gray = _to_gray_small(frame_loader(idx), downscale=downscale)
        if cur_gray.shape[:2] != prev_gray.shape[:2]:
            cur_gray = cv2.resize(cur_gray, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=cur_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=35,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        cur_mask = _warp_mask_with_flow(prev_mask, flow)
        cur_mask = _postprocess_small_mask(cur_mask, feather_px=feather_px)
        cov = float(cur_mask.mean())
        if cov < float(flow_min_coverage) or cov > float(flow_max_coverage):
            cur_mask = prev_mask.copy()
        masks_small[int(idx)] = cur_mask
        prev_gray = cur_gray
        prev_mask = cur_mask

    masks: dict[int, np.ndarray] = {}
    for idx, mask_small in masks_small.items():
        full = cv2.resize(mask_small, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
        masks[int(idx)] = np.clip(full, 0.0, 1.0).astype(np.float32)
    return masks


def _raise_backend_unavailable(backend: str) -> None:
    if backend == "sam2_video_predictor":
        raise PropagationUnavailable(
            "SAM2VideoPredictor backend is not wired in this environment. "
            "Use fallback-to-flow or integrate local SAM2 predictor runtime."
        )
    if backend == "cutie":
        raise PropagationUnavailable(
            "Cutie backend is not wired in this environment. "
            "Use fallback-to-flow or integrate local Cutie tracker runtime."
        )
    raise PropagationUnavailable(f"Unsupported propagation backend: {backend}")


def propagate_masks_assist(
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    anchor_mask: np.ndarray,
    backend: str = "flow",
    fallback_to_flow: bool = True,
    flow_downscale: float = 0.5,
    flow_min_coverage: float = 0.002,
    flow_max_coverage: float = 0.98,
    flow_feather_px: int = 1,
) -> PropagationAssistResult:
    """Propagate one keyframe mask across a frame range."""
    backend_norm = str(backend or "flow").strip().lower()
    if backend_norm in {"flow", "farneback_flow", "cv2_flow"}:
        masks = _propagate_flow_backend(
            frame_loader=frame_loader,
            frame_start=frame_start,
            frame_end=frame_end,
            anchor_frame=anchor_frame,
            anchor_mask=anchor_mask,
            flow_downscale=flow_downscale,
            flow_min_coverage=flow_min_coverage,
            flow_max_coverage=flow_max_coverage,
            feather_px=max(0, int(flow_feather_px)),
        )
        return PropagationAssistResult(masks=masks, backend_used="flow", note=None)

    if backend_norm in {"sam2", "sam2_video_predictor", "sam2videopredictor", "sam2_video"}:
        try:
            _raise_backend_unavailable("sam2_video_predictor")
        except Exception as exc:
            if fallback_to_flow:
                masks = _propagate_flow_backend(
                    frame_loader=frame_loader,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    anchor_frame=anchor_frame,
                    anchor_mask=anchor_mask,
                    flow_downscale=flow_downscale,
                    flow_min_coverage=flow_min_coverage,
                    flow_max_coverage=flow_max_coverage,
                    feather_px=max(0, int(flow_feather_px)),
                )
                return PropagationAssistResult(
                    masks=masks,
                    backend_used="flow_fallback",
                    note=f"SAM2 unavailable: {exc}",
                )
            raise

    if backend_norm in {"cutie", "cutie_tracker"}:
        try:
            _raise_backend_unavailable("cutie")
        except Exception as exc:
            if fallback_to_flow:
                masks = _propagate_flow_backend(
                    frame_loader=frame_loader,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    anchor_frame=anchor_frame,
                    anchor_mask=anchor_mask,
                    flow_downscale=flow_downscale,
                    flow_min_coverage=flow_min_coverage,
                    flow_max_coverage=flow_max_coverage,
                    feather_px=max(0, int(flow_feather_px)),
                )
                return PropagationAssistResult(
                    masks=masks,
                    backend_used="flow_fallback",
                    note=f"Cutie unavailable: {exc}",
                )
            raise

    raise ValueError(
        f"Unsupported propagation backend: {backend}. "
        f"Supported: {', '.join(SUPPORTED_PROPAGATION_BACKENDS)}"
    )


def select_propagation_frames(
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    stride: int = 8,
    max_new_keyframes: int = 24,
) -> list[int]:
    """Select local frame indices to persist as propagated correction anchors."""
    if frame_end < frame_start:
        return []
    stride_i = max(1, int(stride))
    candidates = [
        idx
        for idx in range(frame_start, frame_end + 1)
        if idx != int(anchor_frame) and abs(idx - int(anchor_frame)) % stride_i == 0
    ]
    # Keep range boundaries if they are not anchor frames.
    if frame_start != anchor_frame:
        candidates.append(int(frame_start))
    if frame_end != anchor_frame:
        candidates.append(int(frame_end))
    candidates = sorted(set(candidates))
    if max_new_keyframes <= 0 or len(candidates) <= max_new_keyframes:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, num=max_new_keyframes, dtype=int)
    return [candidates[int(p)] for p in sorted(set(positions.tolist()))]
