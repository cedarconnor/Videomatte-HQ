"""Option B memory bank/query coarse alpha generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryAnchor:
    """Encoded memory anchor (feature stats + metadata)."""

    frame: int
    fg_mean: np.ndarray
    fg_std: np.ndarray
    bg_mean: np.ndarray
    bg_std: np.ndarray
    quality: float
    source: str  # "keyframe" or "auto"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    out = mask.astype(np.float32)
    if out.max() > 1.0:
        out = out / max(float(out.max()), 1.0)
    return np.clip(out, 0.0, 1.0)


def _to_rgb_float(frame: np.ndarray) -> np.ndarray:
    rgb = frame
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]

    out = rgb.astype(np.float32)
    if np.issubdtype(frame.dtype, np.integer):
        max_val = np.iinfo(frame.dtype).max
        out /= float(max_val)
    elif out.max() > 1.0:
        out /= max(out.max(), 1.0)
    return np.clip(out, 0.0, 1.0)


def _resize_with_long_side(
    arr: np.ndarray,
    long_side: int,
    interpolation: int,
) -> np.ndarray:
    h, w = arr.shape[:2]
    if long_side <= 0 or max(h, w) <= long_side:
        return arr
    scale = long_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _compute_features(rgb_small: np.ndarray, spatial_weight: float) -> np.ndarray:
    rgb_u8 = (np.clip(rgb_small, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[..., 0] / 255.0
    a = (lab[..., 1] - 128.0) / 127.0
    b = (lab[..., 2] - 128.0) / 127.0

    h, w = rgb_small.shape[:2]
    if spatial_weight > 0.0:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        x_norm = (xs / max(w - 1, 1) - 0.5) * 2.0
        y_norm = (ys / max(h - 1, 1) - 0.5) * 2.0
        return np.stack(
            [l, a, b, x_norm * spatial_weight, y_norm * spatial_weight],
            axis=-1,
        )
    return np.stack([l, a, b], axis=-1)


def _prepare_anchor_mask(mask_small: np.ndarray) -> np.ndarray:
    """Normalize keyframe masks for anchor extraction.

    Keyframe assignments can be binary masks or soft alpha mattes. For anchor
    encoding we need a stable FG/BG split; if amplitude is very low we normalize
    by local max so valid soft assignments still produce anchors.
    """
    m = np.clip(mask_small.astype(np.float32), 0.0, 1.0)
    max_val = float(m.max())
    if max_val <= 1e-6:
        return m
    if max_val < 0.5:
        m = m / max_val
    return np.clip(m, 0.0, 1.0)


def _build_anchor_from_features(
    frame_idx: int,
    features: np.ndarray,
    mask_small: np.ndarray,
    source: str,
    quality: float,
) -> MemoryAnchor | None:
    prepared = _prepare_anchor_mask(mask_small)
    fg = prepared >= 0.5
    if int(fg.sum()) < 16:
        adaptive_thr = float(np.quantile(prepared, 0.85))
        fg = prepared >= adaptive_thr
    bg = ~fg

    if int(fg.sum()) < 16 or int(bg.sum()) < 16:
        return None

    fg_feat = features[fg]
    bg_feat = features[bg]

    fg_mean = fg_feat.mean(axis=0).astype(np.float32)
    bg_mean = bg_feat.mean(axis=0).astype(np.float32)
    fg_std = np.maximum(fg_feat.std(axis=0), 0.02).astype(np.float32)
    bg_std = np.maximum(bg_feat.std(axis=0), 0.02).astype(np.float32)

    return MemoryAnchor(
        frame=frame_idx,
        fg_mean=fg_mean,
        fg_std=fg_std,
        bg_mean=bg_mean,
        bg_std=bg_std,
        quality=float(np.clip(quality, 0.05, 1.0)),
        source=source,
    )


def _query_anchor_prob(
    features: np.ndarray,
    anchor: MemoryAnchor,
    temperature: float,
) -> np.ndarray:
    diff_fg = (features - anchor.fg_mean) / anchor.fg_std
    diff_bg = (features - anchor.bg_mean) / anchor.bg_std
    d_fg = np.sum(diff_fg * diff_fg, axis=-1)
    d_bg = np.sum(diff_bg * diff_bg, axis=-1)
    logits = 0.5 * (d_bg - d_fg) / max(temperature, 1e-6)
    return _sigmoid(logits).astype(np.float32)


def _resolve_local_keyframes(
    keyframe_masks: dict[int, np.ndarray],
    num_frames: int,
    frame_start: int,
) -> dict[int, np.ndarray]:
    direct = {k: v for k, v in keyframe_masks.items() if 0 <= k < num_frames}
    shifted = {k - frame_start: v for k, v in keyframe_masks.items() if 0 <= (k - frame_start) < num_frames}

    if len(shifted) > len(direct):
        logger.info(
            "Memory pass: mapped keyframes using frame_start offset (%d -> local frame index)",
            frame_start,
        )
        return shifted
    return direct


def _enforce_budget(bank: list[MemoryAnchor], budget: int) -> None:
    while len(bank) > budget:
        candidates = [i for i, a in enumerate(bank) if a.source != "keyframe"]
        if candidates:
            drop_idx = min(candidates, key=lambda i: (bank[i].quality, bank[i].frame))
        else:
            drop_idx = min(range(len(bank)), key=lambda i: (bank[i].quality, bank[i].frame))
        del bank[drop_idx]


def _nearest_keyframe(frame_idx: int, keyframes: list[int]) -> int:
    return min(keyframes, key=lambda k: abs(k - frame_idx))


def _run_placeholder_nearest_keyframe(
    num_frames: int,
    keyframe_masks: dict[int, np.ndarray],
    cfg: VideoMatteConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    keyframes = sorted(keyframe_masks.keys())
    window = max(cfg.memory.window, 1)

    coarse_alphas: list[np.ndarray] = []
    confidence_maps: list[np.ndarray] = []

    for t in range(num_frames):
        anchor = _nearest_keyframe(t, keyframes)
        anchor_alpha = keyframe_masks[anchor]
        alpha = np.clip(anchor_alpha, 0.0, 1.0).astype(np.float32)

        dist = abs(t - anchor)
        base_conf = max(0.05, 1.0 - (dist / float(window)))
        conf = np.full_like(alpha, base_conf, dtype=np.float32)

        edge = (alpha > 0.05) & (alpha < 0.95)
        conf[edge] = np.minimum(conf[edge], base_conf * 0.8)

        coarse_alphas.append(alpha)
        confidence_maps.append(conf)

    return coarse_alphas, confidence_maps


def _source_num_frames(source: Any) -> int:
    if hasattr(source, "num_frames"):
        return int(source.num_frames)
    return len(source)


def _source_resolution(source: Any) -> tuple[int, int]:
    if hasattr(source, "resolution"):
        return tuple(source.resolution)
    first = source[0]
    return int(first.shape[0]), int(first.shape[1])


def run_pass_memory(
    source: Any,
    keyframe_masks: dict[int, np.ndarray],
    cfg: VideoMatteConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build coarse alpha/confidence from target-assigned memory anchors."""

    if not keyframe_masks:
        raise ValueError("No keyframe masks available for memory propagation.")

    num_frames = _source_num_frames(source)
    full_h, full_w = _source_resolution(source)
    local_keyframes = _resolve_local_keyframes(
        keyframe_masks=keyframe_masks,
        num_frames=num_frames,
        frame_start=max(cfg.io.frame_start, 0),
    )
    if not local_keyframes:
        raise ValueError("No keyframe assignments overlap the requested frame range.")

    backend = str(cfg.memory.backend).strip().lower()
    if backend == "placeholder_nearest_keyframe":
        return _run_placeholder_nearest_keyframe(
            num_frames=num_frames,
            keyframe_masks=local_keyframes,
            cfg=cfg,
        )

    if backend not in {"appearance_memory_bank", "memory_bank_v1"}:
        raise ValueError(f"Unsupported memory backend: {cfg.memory.backend}")

    query_long_side = int(getattr(cfg.memory, "query_long_side", 960))
    spatial_weight = float(getattr(cfg.memory, "spatial_weight", 0.1))
    temperature = float(getattr(cfg.memory, "temperature", 1.0))
    budget = max(1, min(int(cfg.memory.memory_frames), int(cfg.memory.max_anchors)))
    min_gap_cfg = int(getattr(cfg.memory, "auto_anchor_min_gap", 0))
    min_anchor_gap = min_gap_cfg if min_gap_cfg > 0 else max(1, cfg.memory.window // max(budget, 1))
    reanchor_threshold = float(np.clip(cfg.memory.confidence_reanchor_threshold, 0.0, 1.0))
    window = max(int(cfg.memory.window), 1)

    memory_bank: list[MemoryAnchor] = []
    keyframes_sorted = sorted(local_keyframes.keys())
    for frame_idx in keyframes_sorted:
        rgb = _to_rgb_float(source[frame_idx])
        rgb_small = _resize_with_long_side(rgb, query_long_side, interpolation=cv2.INTER_LINEAR)
        features = _compute_features(rgb_small, spatial_weight=spatial_weight)
        mask = _normalize_mask(local_keyframes[frame_idx])
        mask_small = cv2.resize(mask, (rgb_small.shape[1], rgb_small.shape[0]), interpolation=cv2.INTER_NEAREST)
        anchor = _build_anchor_from_features(
            frame_idx=frame_idx,
            features=features,
            mask_small=mask_small,
            source="keyframe",
            quality=1.0,
        )
        if anchor is not None:
            memory_bank.append(anchor)
        else:
            logger.warning(
                "Memory pass: skipped keyframe %d due to degenerate mask (coverage=%.5f, max=%.5f)",
                frame_idx,
                float(mask_small.mean()),
                float(mask_small.max()),
            )

    if not memory_bank:
        logger.warning(
            "Memory pass: no valid memory anchors could be built; falling back to placeholder backend."
        )
        return _run_placeholder_nearest_keyframe(
            num_frames=num_frames,
            keyframe_masks=local_keyframes,
            cfg=cfg,
        )

    _enforce_budget(memory_bank, budget)
    last_anchor_frame = max(a.frame for a in memory_bank)

    coarse_alphas: list[np.ndarray] = []
    confidence_maps: list[np.ndarray] = []

    for t in range(num_frames):
        rgb = _to_rgb_float(source[t])
        rgb_small = _resize_with_long_side(rgb, query_long_side, interpolation=cv2.INTER_LINEAR)
        features = _compute_features(rgb_small, spatial_weight=spatial_weight)

        probs: list[np.ndarray] = []
        weights: list[float] = []
        for anchor in memory_bank:
            prob = _query_anchor_prob(features, anchor, temperature=temperature)
            temporal_weight = float(np.exp(-abs(t - anchor.frame) / float(window)))
            weight = max(1e-4, temporal_weight * anchor.quality)
            probs.append(prob)
            weights.append(weight)

        stacked = np.stack(probs, axis=0)
        alpha_small = np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float32)).astype(np.float32)

        variance = np.average(
            (stacked - alpha_small[None, ...]) ** 2,
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        )
        agreement = np.clip(1.0 - np.sqrt(np.maximum(variance, 0.0)) * 2.0, 0.0, 1.0)
        certainty = np.abs(alpha_small - 0.5) * 2.0
        conf_small = np.clip(0.5 * agreement + 0.5 * certainty, 0.0, 1.0).astype(np.float32)

        alpha = cv2.resize(alpha_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        conf = cv2.resize(conf_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        coarse_alphas.append(np.clip(alpha, 0.0, 1.0))
        confidence_maps.append(np.clip(conf, 0.0, 1.0))

        mean_conf = float(conf_small.mean())
        if reanchor_threshold > 0.0 and mean_conf < reanchor_threshold and (t - last_anchor_frame) >= min_anchor_gap:
            mask_small = (alpha_small >= 0.5).astype(np.float32)
            coverage = float(mask_small.mean())
            if 0.002 <= coverage <= 0.98:
                anchor = _build_anchor_from_features(
                    frame_idx=t,
                    features=features,
                    mask_small=mask_small,
                    source="auto",
                    quality=max(0.2, mean_conf),
                )
                if anchor is not None:
                    memory_bank.append(anchor)
                    _enforce_budget(memory_bank, budget)
                    last_anchor_frame = t
                    logger.info(
                        "Memory pass: added auto anchor at frame %d (coverage=%.3f conf=%.3f, bank=%d)",
                        t,
                        coverage,
                        mean_conf,
                        len(memory_bank),
                    )

        if t == 0 or (t + 1) % 50 == 0:
            logger.info(
                "Memory pass: frame %d/%d, anchors=%d, mean_conf=%.3f",
                t + 1,
                num_frames,
                len(memory_bank),
                mean_conf,
            )

    return coarse_alphas, confidence_maps
