"""Option B high-resolution edge refinement from memory coarse alpha/confidence."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from videomatte_hq.band.feather import compute_feather_mask
from videomatte_hq.intermediate.guided_filter import guided_filter
from videomatte_hq.tiling.windows import hann_2d

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


def _build_unknown_band(alpha: np.ndarray, conf: np.ndarray, cfg: Any) -> np.ndarray:
    low = float(np.clip(cfg.alpha_bg_threshold, 0.0, 0.49))
    high = float(np.clip(cfg.alpha_fg_threshold, 0.51, 1.0))
    if high <= low:
        high = min(low + 0.1, 0.99)

    radius = max(int(cfg.unknown_band_px), 0)
    boundary = (alpha > low) & (alpha < high)
    boundary = _dilate(boundary, radius)

    min_conf = float(np.clip(cfg.min_confidence, 0.0, 1.0))
    low_conf = conf < min_conf
    # Prevent full-frame low-confidence regions from exploding refinement.
    support = _dilate(boundary, max(radius, 1))
    low_conf = low_conf & support

    band = boundary | low_conf
    return band


def _iter_tiles(h: int, w: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    step = max(1, tile_size - overlap)
    tiles: list[tuple[int, int, int, int]] = []

    y = 0
    while y < h:
        x = 0
        y1 = min(y + tile_size, h)
        while x < w:
            x1 = min(x + tile_size, w)
            tiles.append((x, y, x1, y1))
            if x1 >= w:
                break
            x += step
        if y1 >= h:
            break
        y += step

    return tiles


def _refine_tile(
    rgb_tile: np.ndarray,
    alpha_tile: np.ndarray,
    conf_tile: np.ndarray,
    tile_band: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    radius = max(int(cfg.guided_radius), 1)
    eps = float(max(cfg.guided_eps, 1e-6))
    guided = guided_filter(rgb_tile, alpha_tile, radius=radius, eps=eps)

    luma = (
        0.2126 * rgb_tile[..., 0]
        + 0.7152 * rgb_tile[..., 1]
        + 0.0722 * rgb_tile[..., 2]
    ).astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    p95 = float(np.percentile(edge, 95.0))
    if p95 > 1e-6:
        edge_norm = np.clip(edge / p95, 0.0, 1.0)
    else:
        edge_norm = np.zeros_like(edge, dtype=np.float32)

    edge_boost = float(max(cfg.edge_boost, 0.0))
    sharpen = alpha_tile + edge_boost * edge_norm * np.tanh((alpha_tile - 0.5) * 4.0)
    candidate = 0.5 * guided + 0.5 * sharpen

    min_conf = float(np.clip(cfg.min_confidence, 1e-3, 1.0))
    conf_gate = np.clip((min_conf - conf_tile) / min_conf, 0.0, 1.0)
    confidence_gain = float(max(cfg.confidence_gain, 0.0))
    blend = tile_band.astype(np.float32) * np.clip((0.25 + 0.75 * conf_gate) * confidence_gain, 0.0, 1.0)

    out = alpha_tile * (1.0 - blend) + candidate * blend
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def run_pass_refine(
    source: Any,
    coarse_alphas: list[np.ndarray],
    confidence_maps: list[np.ndarray],
    cfg: Any,
) -> list[np.ndarray]:
    """Run boundary-band refinement at full resolution.

    This stage refines only uncertain regions from coarse alpha, with confidence
    gating and feathered blending to avoid whole-frame softening.
    """

    if len(coarse_alphas) != len(confidence_maps):
        raise ValueError("coarse_alphas and confidence_maps must have same length")

    if not cfg.refine.enabled:
        return [np.clip(a, 0.0, 1.0).astype(np.float32) for a in coarse_alphas]

    backend = str(cfg.refine.backend).strip().lower()
    if backend not in {"guided_band", "guided_band_v1"}:
        raise ValueError(f"Unsupported refine backend: {cfg.refine.backend}")

    tile_size = max(int(cfg.refine.tile_size), 64)
    overlap = int(np.clip(cfg.refine.overlap, 0, tile_size - 1))
    min_cov = float(np.clip(cfg.refine.tile_min_coverage, 0.0, 1.0))

    outputs: list[np.ndarray] = []
    for t in range(len(coarse_alphas)):
        alpha = np.clip(coarse_alphas[t], 0.0, 1.0).astype(np.float32)
        conf = np.clip(confidence_maps[t], 0.0, 1.0).astype(np.float32)
        rgb = _to_rgb_float(source[t])

        if alpha.shape != conf.shape:
            raise ValueError(f"Alpha/confidence shape mismatch at frame {t}: {alpha.shape} vs {conf.shape}")

        band = _build_unknown_band(alpha, conf, cfg.refine)
        if not band.any():
            outputs.append(alpha)
            continue

        fg_lock = alpha >= float(np.clip(cfg.refine.alpha_fg_threshold, 0.0, 1.0))
        bg_lock = alpha <= float(np.clip(cfg.refine.alpha_bg_threshold, 0.0, 1.0))

        h, w = alpha.shape
        acc_num = np.zeros((h, w), dtype=np.float64)
        acc_den = np.zeros((h, w), dtype=np.float64)

        for x0, y0, x1, y1 in _iter_tiles(h, w, tile_size=tile_size, overlap=overlap):
            tile_band = band[y0:y1, x0:x1]
            coverage = float(tile_band.mean())
            if coverage < min_cov:
                continue

            rgb_tile = rgb[y0:y1, x0:x1]
            alpha_tile = alpha[y0:y1, x0:x1]
            conf_tile = conf[y0:y1, x0:x1]

            refined_tile = _refine_tile(
                rgb_tile=rgb_tile,
                alpha_tile=alpha_tile,
                conf_tile=conf_tile,
                tile_band=tile_band,
                cfg=cfg.refine,
            )

            th, tw = refined_tile.shape
            window = np.maximum(hann_2d(th, tw), 1e-3)
            weight = window * tile_band.astype(np.float32)

            acc_num[y0:y1, x0:x1] += weight * refined_tile
            acc_den[y0:y1, x0:x1] += weight

        stitched = alpha.copy()
        has = acc_den > 1e-8
        stitched[has] = (acc_num[has] / acc_den[has]).astype(np.float32)

        feather_px = max(1, overlap // 2 if overlap > 0 else max(int(cfg.refine.unknown_band_px) // 2, 1))
        feather = compute_feather_mask(band, feather_px=feather_px)
        mix = feather * np.clip((1.0 - conf) * float(max(cfg.refine.confidence_gain, 0.0)), 0.0, 1.0)

        refined = alpha * (1.0 - mix) + stitched * mix
        refined[fg_lock] = 1.0
        refined[bg_lock] = 0.0

        outputs.append(np.clip(refined, 0.0, 1.0).astype(np.float32))

        if t == 0 or (t + 1) % 50 == 0:
            logger.info(
                "Refine pass: frame %d/%d, band_coverage=%.3f",
                t + 1,
                len(coarse_alphas),
                float(band.mean()),
            )

    return outputs

