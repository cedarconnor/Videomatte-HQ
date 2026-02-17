"""Prompt-based mask building across a frame range."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from videomatte_hq.mask_builder import build_prompt_mask_grabcut
from videomatte_hq.sam_builder import DEFAULT_SAM_MODEL_ID, build_prompt_mask_sam
from videomatte_hq.samurai_backend import (
    SAMURAI_BACKEND_CANONICAL,
    SamuraiRuntimeConfig,
    is_samurai_backend,
    propagate_with_samurai_from_prompts,
)


@dataclass
class PromptMaskRangeResult:
    masks: dict[int, np.ndarray]  # local frame idx -> alpha float32 [0, 1]
    backend_used: str
    note: str | None = None


@dataclass
class _PromptState:
    box_xyxy: tuple[float, float, float, float]
    fg_points: list[tuple[float, float]]
    bg_points: list[tuple[float, float]]


def _normalize_box(
    box_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_xyxy
    xa = float(max(0, min(width - 1, int(round(min(x0, x1))))))
    xb = float(max(0, min(width - 1, int(round(max(x0, x1))))))
    ya = float(max(0, min(height - 1, int(round(min(y0, y1))))))
    yb = float(max(0, min(height - 1, int(round(max(y0, y1))))))
    return xa, ya, xb, yb


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        float(max(0.0, min(width - 1.0, float(x)))),
        float(max(0.0, min(height - 1.0, float(y)))),
    )


def _to_gray_small(rgb_u8: np.ndarray, downscale: float) -> np.ndarray:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    d = float(np.clip(downscale, 0.15, 1.0))
    if d >= 0.999:
        return gray
    h, w = gray.shape[:2]
    sw = max(16, int(round(w * d)))
    sh = max(16, int(round(h * d)))
    return cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)


def _sample_flow(flow: np.ndarray, x: float, y: float) -> tuple[float, float]:
    h, w = flow.shape[:2]
    xi = int(max(0, min(w - 1, round(x))))
    yi = int(max(0, min(h - 1, round(y))))
    vec = flow[yi, xi]
    return float(vec[0]), float(vec[1])


def _flow_track_state(
    prev_rgb_u8: np.ndarray,
    cur_rgb_u8: np.ndarray,
    prev_state: _PromptState,
    downscale: float,
    track_bg_points: bool = False,
) -> _PromptState:
    h, w = prev_rgb_u8.shape[:2]
    scale = float(np.clip(downscale, 0.15, 1.0))
    prev_gray = _to_gray_small(prev_rgb_u8, scale)
    cur_gray = _to_gray_small(cur_rgb_u8, scale)
    if prev_gray.shape[:2] != cur_gray.shape[:2]:
        cur_gray = cv2.resize(cur_gray, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_AREA)

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=cur_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=31,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    def track_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        tracked: list[tuple[float, float]] = []
        for px, py in points:
            sx = float(px) * scale
            sy = float(py) * scale
            dx, dy = _sample_flow(flow, sx, sy)
            nx = (sx + dx) / scale
            ny = (sy + dy) / scale
            tracked.append(_clamp_point(nx, ny, w, h))
        return tracked

    x0, y0, x1, y1 = prev_state.box_xyxy
    sx0 = int(max(0, min(flow.shape[1] - 1, round(min(x0, x1) * scale))))
    sx1 = int(max(0, min(flow.shape[1] - 1, round(max(x0, x1) * scale))))
    sy0 = int(max(0, min(flow.shape[0] - 1, round(min(y0, y1) * scale))))
    sy1 = int(max(0, min(flow.shape[0] - 1, round(max(y0, y1) * scale))))
    if sx1 <= sx0:
        sx1 = min(flow.shape[1] - 1, sx0 + 1)
    if sy1 <= sy0:
        sy1 = min(flow.shape[0] - 1, sy0 + 1)
    # Prefer subject-driven motion from FG points; fallback to whole-box flow.
    if prev_state.fg_points:
        fg_vecs = []
        for px, py in prev_state.fg_points:
            sx = float(px) * scale
            sy = float(py) * scale
            fg_vecs.append(_sample_flow(flow, sx, sy))
        box_dx = float(np.median([v[0] for v in fg_vecs]))
        box_dy = float(np.median([v[1] for v in fg_vecs]))
    else:
        box_flow = flow[sy0:sy1, sx0:sx1]
        if box_flow.size == 0:
            box_dx, box_dy = 0.0, 0.0
        else:
            box_dx = float(np.median(box_flow[..., 0]))
            box_dy = float(np.median(box_flow[..., 1]))

    nx0 = (min(x0, x1) * scale + box_dx) / scale
    nx1 = (max(x0, x1) * scale + box_dx) / scale
    ny0 = (min(y0, y1) * scale + box_dy) / scale
    ny1 = (max(y0, y1) * scale + box_dy) / scale
    new_box = _normalize_box((nx0, ny0, nx1, ny1), width=w, height=h)

    return _PromptState(
        box_xyxy=new_box,
        fg_points=track_points(prev_state.fg_points),
        bg_points=track_points(prev_state.bg_points) if track_bg_points else list(prev_state.bg_points),
    )


def _build_single_mask(
    *,
    rgb_u8: np.ndarray,
    state: _PromptState,
    backend_requested: str,
    point_radius: int,
    iter_count: int,
    sam_model_id: str,
    sam_local_files_only: bool,
    sam_fallback_to_grabcut: bool,
    device_hint: str,
) -> tuple[np.ndarray, str, str | None]:
    backend = str(backend_requested or "sam").strip().lower()
    if backend in {"grabcut", "classic"}:
        alpha = build_prompt_mask_grabcut(
            frame_rgb_u8=rgb_u8,
            box_xyxy=state.box_xyxy,
            fg_points=state.fg_points,
            bg_points=state.bg_points,
            point_radius=point_radius,
            iter_count=iter_count,
        )
        return alpha, "grabcut", None

    if backend in {"sam", "sam_hq", "segment_anything"}:
        try:
            alpha = build_prompt_mask_sam(
                frame_rgb_u8=rgb_u8,
                box_xyxy=state.box_xyxy,
                fg_points=state.fg_points,
                bg_points=state.bg_points,
                model_id=sam_model_id or DEFAULT_SAM_MODEL_ID,
                local_files_only=bool(sam_local_files_only),
                device_hint=device_hint,
                point_radius=point_radius,
            )
            return alpha, "sam", None
        except Exception as exc:
            if not sam_fallback_to_grabcut:
                raise
            alpha = build_prompt_mask_grabcut(
                frame_rgb_u8=rgb_u8,
                box_xyxy=state.box_xyxy,
                fg_points=state.fg_points,
                bg_points=state.bg_points,
                point_radius=point_radius,
                iter_count=iter_count,
            )
            return alpha, "grabcut_fallback", f"SAM unavailable: {exc}"

    raise ValueError(f"Unsupported backend for prompt mask range build: {backend_requested}")


def build_prompt_masks_range(
    *,
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    box_xyxy: tuple[float, float, float, float],
    fg_points: Sequence[tuple[float, float]] = (),
    bg_points: Sequence[tuple[float, float]] = (),
    backend: str = "sam",
    point_radius: int = 8,
    iter_count: int = 5,
    sam_model_id: str = DEFAULT_SAM_MODEL_ID,
    sam_local_files_only: bool = True,
    sam_fallback_to_grabcut: bool = True,
    samurai_model_cfg: str = "",
    samurai_checkpoint: str = "",
    samurai_offload_video_to_cpu: bool = False,
    samurai_offload_state_to_cpu: bool = False,
    track_prompts_with_flow: bool = False,
    track_bg_points_with_flow: bool = False,
    flow_downscale: float = 0.5,
    device_hint: str = "cuda",
) -> PromptMaskRangeResult:
    """Build prompt-based masks across a range using SAM/GrabCut."""
    if frame_end < frame_start:
        raise ValueError(f"Invalid frame range: {frame_start}..{frame_end}")
    if anchor_frame < frame_start or anchor_frame > frame_end:
        raise ValueError(f"Anchor frame {anchor_frame} is outside range {frame_start}..{frame_end}")

    backend_requested = str(backend or "sam").strip().lower()
    sam2_aliases = {"sam2", "sam2_video_predictor", "sam2videopredictor", "sam2_video"}
    requested_sam2_alias = backend_requested in sam2_aliases
    normalized_backend = SAMURAI_BACKEND_CANONICAL if requested_sam2_alias else backend_requested

    anchor_rgb = frame_loader(int(anchor_frame))
    if anchor_rgb.ndim != 3 or anchor_rgb.shape[2] < 3:
        raise ValueError("Frame loader must return RGB uint8 frames.")
    h, w = anchor_rgb.shape[:2]
    anchor_state = _PromptState(
        box_xyxy=_normalize_box(box_xyxy, width=w, height=h),
        fg_points=[_clamp_point(px, py, w, h) for px, py in fg_points],
        bg_points=[_clamp_point(px, py, w, h) for px, py in bg_points],
    )

    if is_samurai_backend(normalized_backend):
        masks, note = propagate_with_samurai_from_prompts(
            frame_loader=frame_loader,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            anchor_frame=int(anchor_frame),
            box_xyxy=anchor_state.box_xyxy,
            fg_points=anchor_state.fg_points,
            bg_points=anchor_state.bg_points,
            runtime=SamuraiRuntimeConfig(
                model_cfg=str(samurai_model_cfg or "").strip(),
                checkpoint=str(samurai_checkpoint or "").strip(),
                offload_video_to_cpu=bool(samurai_offload_video_to_cpu),
                offload_state_to_cpu=bool(samurai_offload_state_to_cpu),
            ),
            device_hint=device_hint,
        )
        return PromptMaskRangeResult(
            masks={int(k): np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0).astype(np.float32) for k, v in masks.items()},
            backend_used=("sam2_video_predictor" if requested_sam2_alias else SAMURAI_BACKEND_CANONICAL),
            note=note,
        )

    states: dict[int, _PromptState] = {int(anchor_frame): anchor_state}
    masks: dict[int, np.ndarray] = {}
    backend_hits: set[str] = set()
    first_note: str | None = None

    alpha_anchor, backend_used, note = _build_single_mask(
        rgb_u8=anchor_rgb,
        state=anchor_state,
        backend_requested=normalized_backend,
        point_radius=point_radius,
        iter_count=iter_count,
        sam_model_id=sam_model_id,
        sam_local_files_only=sam_local_files_only,
        sam_fallback_to_grabcut=sam_fallback_to_grabcut,
        device_hint=device_hint,
    )
    masks[int(anchor_frame)] = np.clip(alpha_anchor.astype(np.float32), 0.0, 1.0)
    backend_hits.add(backend_used)
    first_note = note

    # Forward tracking/build.
    prev_idx = int(anchor_frame)
    prev_rgb = anchor_rgb
    prev_state = anchor_state
    for idx in range(int(anchor_frame) + 1, int(frame_end) + 1):
        cur_rgb = frame_loader(idx)
        if track_prompts_with_flow:
            cur_state = _flow_track_state(
                prev_rgb_u8=prev_rgb,
                cur_rgb_u8=cur_rgb,
                prev_state=prev_state,
                downscale=flow_downscale,
                track_bg_points=track_bg_points_with_flow,
            )
        else:
            cur_state = prev_state
        states[idx] = cur_state
        alpha, bu, note = _build_single_mask(
            rgb_u8=cur_rgb,
            state=cur_state,
            backend_requested=normalized_backend,
            point_radius=point_radius,
            iter_count=iter_count,
            sam_model_id=sam_model_id,
            sam_local_files_only=sam_local_files_only,
            sam_fallback_to_grabcut=sam_fallback_to_grabcut,
            device_hint=device_hint,
        )
        masks[idx] = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        backend_hits.add(bu)
        if first_note is None and note:
            first_note = note
        prev_idx = idx
        prev_rgb = cur_rgb
        prev_state = cur_state

    # Backward tracking/build.
    prev_idx = int(anchor_frame)
    prev_rgb = anchor_rgb
    prev_state = anchor_state
    for idx in range(int(anchor_frame) - 1, int(frame_start) - 1, -1):
        cur_rgb = frame_loader(idx)
        if track_prompts_with_flow:
            cur_state = _flow_track_state(
                prev_rgb_u8=prev_rgb,
                cur_rgb_u8=cur_rgb,
                prev_state=prev_state,
                downscale=flow_downscale,
                track_bg_points=track_bg_points_with_flow,
            )
        else:
            cur_state = prev_state
        states[idx] = cur_state
        alpha, bu, note = _build_single_mask(
            rgb_u8=cur_rgb,
            state=cur_state,
            backend_requested=normalized_backend,
            point_radius=point_radius,
            iter_count=iter_count,
            sam_model_id=sam_model_id,
            sam_local_files_only=sam_local_files_only,
            sam_fallback_to_grabcut=sam_fallback_to_grabcut,
            device_hint=device_hint,
        )
        masks[idx] = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        backend_hits.add(bu)
        if first_note is None and note:
            first_note = note
        prev_idx = idx
        prev_rgb = cur_rgb
        prev_state = cur_state

    if backend_hits == {"sam"}:
        backend_final = "sam"
    elif "sam" in backend_hits and "grabcut_fallback" in backend_hits:
        backend_final = "sam_with_grabcut_fallback"
    elif backend_hits == {"grabcut_fallback"}:
        backend_final = "grabcut_fallback"
    elif backend_hits == {"grabcut"}:
        backend_final = "grabcut"
    else:
        backend_final = "+".join(sorted(backend_hits))

    return PromptMaskRangeResult(
        masks=masks,
        backend_used=backend_final,
        note=first_note,
    )
