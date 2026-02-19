"""Option B confidence-gated temporal cleanup outside edge band."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from videomatte_hq.intermediate.guided_filter import guided_filter

logger = logging.getLogger(__name__)


def _to_rgb_float(frame: np.ndarray) -> np.ndarray:
    rgb = frame
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]

    out = rgb.astype(np.float32)
    if np.issubdtype(frame.dtype, np.integer):
        out /= float(np.iinfo(frame.dtype).max)
    elif out.max() > 1.0:
        out /= max(out.max(), 1.0)
    return np.clip(out, 0.0, 1.0)


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    rgb = _to_rgb_float(frame)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        gray = cv2.cvtColor((rgb[..., :3] * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (np.squeeze(rgb) * 255.0).astype(np.uint8)
    return gray


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _estimate_backflow(prev_frame: np.ndarray, curr_frame: np.ndarray, max_side: int) -> np.ndarray:
    prev_gray = _to_gray_u8(prev_frame)
    curr_gray = _to_gray_u8(curr_frame)
    if prev_gray.shape != curr_gray.shape:
        raise ValueError(f"Flow frame shape mismatch: prev={prev_gray.shape} curr={curr_gray.shape}")

    h, w = curr_gray.shape
    limit = max(int(max_side), 0)
    scale = 1.0
    prev_work = prev_gray
    curr_work = curr_gray

    if limit > 0 and max(h, w) > limit:
        scale = float(limit) / float(max(h, w))
        w_work = max(8, int(round(w * scale)))
        h_work = max(8, int(round(h * scale)))
        prev_work = cv2.resize(prev_gray, (w_work, h_work), interpolation=cv2.INTER_AREA)
        curr_work = cv2.resize(curr_gray, (w_work, h_work), interpolation=cv2.INTER_AREA)

    # Backward flow: displacement from current frame pixels into previous frame coordinates.
    backflow = cv2.calcOpticalFlowFarneback(
        curr_work,
        prev_work,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )

    if scale != 1.0:
        backflow = cv2.resize(backflow, (w, h), interpolation=cv2.INTER_LINEAR)
        backflow[..., 0] /= scale
        backflow[..., 1] /= scale

    return backflow.astype(np.float32)


def _build_remap_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return grid_x, grid_y


def _warp_with_backflow(
    src: np.ndarray,
    backflow: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    if src.shape[:2] != backflow.shape[:2]:
        raise ValueError(f"Warp shape mismatch: src={src.shape[:2]} flow={backflow.shape[:2]}")
    if grid_x.shape != src.shape[:2] or grid_y.shape != src.shape[:2]:
        raise ValueError(f"Warp grid shape mismatch: grid={grid_x.shape} src={src.shape[:2]}")

    map_x = grid_x + backflow[..., 0]
    map_y = grid_y + backflow[..., 1]
    return cv2.remap(
        src.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _resolve_local_anchor_frames(
    anchor_frames: set[int] | list[int] | None,
    num_frames: int,
    frame_start: int,
) -> set[int]:
    if not anchor_frames:
        return set()
    anchors = set(int(a) for a in anchor_frames)
    direct = {a for a in anchors if 0 <= a < num_frames}
    shifted = {a - frame_start for a in anchors if 0 <= (a - frame_start) < num_frames}
    if len(shifted) > len(direct):
        return shifted
    return direct


def run_pass_temporal_cleanup(
    source: Any,
    alphas: list[np.ndarray],
    confidences: list[np.ndarray],
    cfg: Any,
    anchor_frames: set[int] | list[int] | None = None,
) -> list[np.ndarray]:
    """Edge-aware temporal cleanup with confidence gating.

    Rules:
    - Never temporal-average inside edge band.
    - Outside edge band, apply EMA where confidence is sufficient.
    - Optionally reset smoothing near new anchors.
    - Optionally run guided edge snap inside band.
    """

    if len(alphas) != len(confidences):
        raise ValueError("alphas/confidences length mismatch")

    if not alphas:
        return []

    if not cfg.temporal_cleanup.enabled or len(alphas) < 2:
        return [np.clip(a, 0.0, 1.0).astype(np.float32) for a in alphas]

    base_ema = float(np.clip(cfg.temporal_cleanup.outside_band_ema, 0.0, 1.0))
    outside_ema_enabled = bool(getattr(cfg.temporal_cleanup, "outside_band_ema_enabled", True))
    min_conf = float(np.clip(cfg.temporal_cleanup.min_confidence, 0.0, 1.0))
    confidence_clamp_enabled = bool(getattr(cfg.temporal_cleanup, "confidence_clamp_enabled", False))
    edge_lo = float(np.clip(cfg.temporal_cleanup.edge_bg_threshold, 0.0, 0.49))
    edge_hi = float(np.clip(cfg.temporal_cleanup.edge_fg_threshold, 0.51, 1.0))
    edge_radius = max(int(cfg.temporal_cleanup.edge_band_radius_px), 0)
    edge_band_ema_enabled = bool(getattr(cfg.temporal_cleanup, "edge_band_ema_enabled", False))
    edge_band_ema = float(np.clip(getattr(cfg.temporal_cleanup, "edge_band_ema", 0.0), 0.0, 1.0))
    edge_band_min_conf = float(
        np.clip(getattr(cfg.temporal_cleanup, "edge_band_min_confidence", min_conf), 0.0, 1.0)
    )
    clamp_delta = float(max(cfg.temporal_cleanup.clamp_delta, 0.0))
    edge_snap_min_conf = float(
        np.clip(getattr(cfg.temporal_cleanup, "edge_snap_min_confidence", 0.0), 0.0, 1.0)
    )
    motion_warp_enabled = bool(getattr(cfg.temporal_cleanup, "motion_warp_enabled", False))
    motion_warp_max_side = max(int(getattr(cfg.temporal_cleanup, "motion_warp_max_side", 960)), 0)

    reset_frames = max(int(cfg.temporal_cleanup.anchor_reset_frames), 1)
    local_anchors = _resolve_local_anchor_frames(
        anchor_frames=anchor_frames,
        num_frames=len(alphas),
        frame_start=max(int(getattr(cfg.io, "frame_start", 0)), 0),
    )
    if 0 not in local_anchors:
        local_anchors.add(0)

    output = [np.clip(alphas[0], 0.0, 1.0).astype(np.float32)]
    last_anchor = 0
    remap_grid_x: np.ndarray | None = None
    remap_grid_y: np.ndarray | None = None
    motion_warp_failures = 0

    for t in range(1, len(alphas)):
        if t in local_anchors:
            last_anchor = t

        curr = np.clip(alphas[t], 0.0, 1.0).astype(np.float32)
        prev = output[t - 1]
        conf_curr = np.clip(confidences[t], 0.0, 1.0).astype(np.float32)
        conf_prev = np.clip(confidences[t - 1], 0.0, 1.0).astype(np.float32)
        prev_for_blend = prev
        conf_prev_for_pair = conf_prev

        if motion_warp_enabled and (outside_ema_enabled or edge_band_ema_enabled):
            try:
                backflow = _estimate_backflow(
                    prev_frame=source[t - 1],
                    curr_frame=source[t],
                    max_side=motion_warp_max_side,
                )
                if backflow.shape[:2] == curr.shape[:2]:
                    if (
                        remap_grid_x is None
                        or remap_grid_y is None
                        or remap_grid_x.shape != curr.shape[:2]
                        or remap_grid_y.shape != curr.shape[:2]
                    ):
                        remap_grid_x, remap_grid_y = _build_remap_grids(curr.shape[0], curr.shape[1])
                    prev_for_blend = _warp_with_backflow(prev_for_blend, backflow, remap_grid_x, remap_grid_y)
                    conf_prev_for_pair = _warp_with_backflow(conf_prev, backflow, remap_grid_x, remap_grid_y)
            except Exception as exc:
                motion_warp_failures += 1
                if motion_warp_failures == 1:
                    logger.warning(
                        "Temporal cleanup: disabling motion warp after flow failure at frame %d: %s",
                        t,
                        exc,
                    )
                motion_warp_enabled = False

        edge_band = (curr > edge_lo) & (curr < edge_hi)
        edge_band = _dilate(edge_band, edge_radius)
        safe_region = ~edge_band

        conf_pair = np.minimum(conf_curr, np.clip(conf_prev_for_pair, 0.0, 1.0))
        conf_scale = np.clip((conf_pair - min_conf) / max(1.0 - min_conf, 1e-6), 0.0, 1.0)
        if outside_ema_enabled:
            ema_map = base_ema * conf_scale
        else:
            ema_map = np.zeros_like(curr, dtype=np.float32)
        edge_conf_scale = np.clip((conf_pair - edge_band_min_conf) / max(1.0 - edge_band_min_conf, 1e-6), 0.0, 1.0)
        if edge_band_ema_enabled:
            edge_ema_map = edge_band_ema * edge_conf_scale
        else:
            edge_ema_map = np.zeros_like(curr, dtype=np.float32)

        if cfg.temporal_cleanup.reset_on_new_anchor:
            dist = t - last_anchor
            ramp = np.clip(dist / float(reset_frames), 0.0, 1.0)
            ema_map *= ramp
            edge_ema_map *= ramp

        if clamp_delta > 0.0:
            clamped_prev = np.clip(prev_for_blend, curr - clamp_delta, curr + clamp_delta)
            if confidence_clamp_enabled:
                prev_for_blend = prev_for_blend.copy()
                clamp_mask = conf_pair >= min_conf
                prev_for_blend[clamp_mask] = clamped_prev[clamp_mask]

        outside_blend = curr * (1.0 - ema_map) + prev_for_blend * ema_map
        edge_blend = curr * (1.0 - edge_ema_map) + prev_for_blend * edge_ema_map
        out = curr.copy()
        out[safe_region] = outside_blend[safe_region]
        if edge_band_ema_enabled and edge_band_ema > 0.0:
            out[edge_band] = edge_blend[edge_band]

        if cfg.temporal_cleanup.edge_snap_enabled:
            rgb = _to_rgb_float(source[t])
            snap = guided_filter(
                guide=rgb,
                src=out,
                radius=max(int(cfg.temporal_cleanup.edge_snap_radius), 1),
                eps=float(max(cfg.temporal_cleanup.edge_snap_eps, 1e-6)),
            )
            snap_mask = edge_band & (conf_pair >= edge_snap_min_conf)
            out[snap_mask] = np.clip(snap[snap_mask], 0.0, 1.0)

        output.append(np.clip(out, 0.0, 1.0).astype(np.float32))

        if t == 1 or (t + 1) % 50 == 0:
            logger.info(
                "Temporal cleanup: frame %d/%d, edge_cov=%.3f motion_warp=%s",
                t + 1,
                len(alphas),
                float(edge_band.mean()),
                "on" if motion_warp_enabled else "off",
            )

    return output
