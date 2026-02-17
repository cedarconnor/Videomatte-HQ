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


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


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

    for t in range(1, len(alphas)):
        if t in local_anchors:
            last_anchor = t

        curr = np.clip(alphas[t], 0.0, 1.0).astype(np.float32)
        prev = output[t - 1]
        conf_curr = np.clip(confidences[t], 0.0, 1.0).astype(np.float32)
        conf_prev = np.clip(confidences[t - 1], 0.0, 1.0).astype(np.float32)

        edge_band = (curr > edge_lo) & (curr < edge_hi)
        edge_band = _dilate(edge_band, edge_radius)
        safe_region = ~edge_band

        conf_pair = np.minimum(conf_curr, conf_prev)
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
            clamped_prev = np.clip(prev, curr - clamp_delta, curr + clamp_delta)
            if confidence_clamp_enabled:
                prev_for_blend = prev.copy()
                clamp_mask = conf_pair >= min_conf
                prev_for_blend[clamp_mask] = clamped_prev[clamp_mask]
            else:
                prev_for_blend = prev
        else:
            prev_for_blend = prev

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
                "Temporal cleanup: frame %d/%d, edge_cov=%.3f",
                t + 1,
                len(alphas),
                float(edge_band.mean()),
            )

    return output
