"""Option B high-resolution edge refinement from memory coarse alpha/confidence."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

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


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _cleanup_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out.astype(bool)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    fg = mask.astype(np.uint8)
    if int(fg.sum()) <= 0:
        return mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(1 + np.argmax(areas))
    return labels == largest_label


def _build_region_trimap_masks(region_guidance: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    thr = float(np.clip(getattr(cfg, "region_trimap_threshold", 0.5), 0.0, 1.0))
    clean_px = max(0, int(getattr(cfg, "region_trimap_cleanup_px", 1)))
    keep_largest = bool(getattr(cfg, "region_trimap_keep_largest", True))
    min_cov = float(np.clip(getattr(cfg, "region_trimap_min_coverage", 0.002), 0.0, 1.0))
    max_cov = float(np.clip(getattr(cfg, "region_trimap_max_coverage", 0.98), 0.0, 1.0))
    fg_erode = max(0, int(getattr(cfg, "region_trimap_fg_erode_px", 3)))
    bg_dilate = max(0, int(getattr(cfg, "region_trimap_bg_dilate_px", 16)))

    mask = np.asarray(region_guidance, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask >= thr
    if clean_px > 0:
        binary = _cleanup_binary_mask(binary, radius=clean_px)
    if keep_largest:
        binary = _keep_largest_component(binary)
    if not binary.any():
        return None

    cov = float(binary.mean())
    if cov < min_cov or cov > max_cov:
        return None

    fg_conf = _erode(binary, radius=fg_erode)
    if not fg_conf.any():
        fg_conf = binary.copy()
    fg_loose = _dilate(binary, radius=bg_dilate)
    if not fg_loose.any():
        fg_loose = binary.copy()

    unknown = fg_loose & (~fg_conf)
    if not unknown.any():
        shell = _dilate(binary, radius=1) & (~_erode(binary, radius=1))
        unknown = shell if shell.any() else fg_loose.copy()
    return fg_conf.astype(bool), unknown.astype(bool), fg_loose.astype(bool)


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


def _blend_refined_candidate(
    alpha_tile: np.ndarray,
    candidate: np.ndarray,
    conf_tile: np.ndarray,
    tile_band: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    min_conf = float(np.clip(cfg.min_confidence, 1e-3, 1.0))
    conf_gate = np.clip((min_conf - conf_tile) / min_conf, 0.0, 1.0)
    confidence_gain = float(max(cfg.confidence_gain, 0.0))
    blend = tile_band.astype(np.float32) * np.clip((0.25 + 0.75 * conf_gate) * confidence_gain, 0.0, 1.0)
    out = alpha_tile * (1.0 - blend) + candidate * blend
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _build_refine_trimap(alpha: np.ndarray, band: np.ndarray, fg_lock: np.ndarray, bg_lock: np.ndarray) -> np.ndarray:
    trimap = (alpha >= 0.5).astype(np.float32)
    trimap[band] = 0.5
    trimap[fg_lock] = 1.0
    trimap[bg_lock] = 0.0
    return trimap.astype(np.float32)


def _load_mematte_refiner(cfg: Any):
    from videomatte_hq.models.edge_mematte import MEMatteModel

    refiner = MEMatteModel(
        repo_dir=str(getattr(cfg.refine, "mematte_repo_dir", "third_party/MEMatte")),
        checkpoint_path=str(
            getattr(
                cfg.refine,
                "mematte_checkpoint",
                "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth",
            )
        ),
        device=str(getattr(cfg.runtime, "device", "cuda")),
        precision=str(getattr(cfg.runtime, "precision", "fp16")),
        max_number_token=int(getattr(cfg.refine, "mematte_max_number_token", 18500)),
        patch_decoder=bool(getattr(cfg.refine, "mematte_patch_decoder", True)),
    )
    refiner.load_weights(str(getattr(cfg.runtime, "device", "cuda")))
    return refiner


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
    return _blend_refined_candidate(alpha_tile, candidate, conf_tile, tile_band, cfg)


def run_pass_refine(
    source: Any,
    coarse_alphas: list[np.ndarray],
    confidence_maps: list[np.ndarray],
    cfg: Any,
    region_guidance_masks: list[np.ndarray] | None = None,
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
    use_mematte = backend in {"mematte", "mematte_band", "mematte_tiled"}
    if backend not in {"guided_band", "guided_band_v1", "mematte", "mematte_band", "mematte_tiled"}:
        raise ValueError(f"Unsupported refine backend: {cfg.refine.backend}")

    tile_size = max(int(cfg.refine.tile_size), 64)
    overlap = int(np.clip(cfg.refine.overlap, 0, tile_size - 1))
    min_cov = float(np.clip(cfg.refine.tile_min_coverage, 0.0, 1.0))
    mematte_refiner = _load_mematte_refiner(cfg) if use_mematte else None

    outputs: list[np.ndarray] = []
    for t in range(len(coarse_alphas)):
        alpha = np.clip(coarse_alphas[t], 0.0, 1.0).astype(np.float32)
        conf = np.clip(confidence_maps[t], 0.0, 1.0).astype(np.float32)
        rgb = _to_rgb_float(source[t])

        if alpha.shape != conf.shape:
            raise ValueError(f"Alpha/confidence shape mismatch at frame {t}: {alpha.shape} vs {conf.shape}")

        fg_lock = alpha >= float(np.clip(cfg.refine.alpha_fg_threshold, 0.0, 1.0))
        bg_lock = alpha <= float(np.clip(cfg.refine.alpha_bg_threshold, 0.0, 1.0))
        band = _build_unknown_band(alpha, conf, cfg.refine)

        guidance_cov = -1.0
        if (
            bool(getattr(cfg.refine, "region_trimap_enabled", True))
            and region_guidance_masks is not None
            and t < len(region_guidance_masks)
        ):
            guidance = np.asarray(region_guidance_masks[t], dtype=np.float32)
            if guidance.shape[:2] != alpha.shape[:2]:
                guidance = cv2.resize(guidance, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
            trimap_masks = _build_region_trimap_masks(guidance, cfg.refine)
            if trimap_masks is not None:
                fg_conf, unknown, fg_loose = trimap_masks
                fg_lock = fg_lock | fg_conf
                bg_lock = bg_lock | (~fg_loose)
                band = (band & fg_loose) | unknown
                guidance_cov = float(fg_loose.mean())

        if not band.any():
            clamped = alpha.copy()
            clamped[fg_lock] = 1.0
            clamped[bg_lock] = 0.0
            outputs.append(np.clip(clamped, 0.0, 1.0).astype(np.float32))
            continue

        trimap_frame = _build_refine_trimap(alpha, band, fg_lock, bg_lock) if use_mematte else None

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

            if use_mematte:
                assert mematte_refiner is not None
                assert trimap_frame is not None
                rgb_t = torch.from_numpy(np.transpose(rgb_tile, (2, 0, 1)).copy()).float()
                trimap_t = torch.from_numpy(trimap_frame[y0:y1, x0:x1][None, ...].copy()).float()
                alpha_prior_t = torch.from_numpy(alpha_tile[None, ...].copy()).float()
                try:
                    candidate_t = mematte_refiner.infer_tile(
                        rgb_tile=rgb_t,
                        trimap_tile=trimap_t,
                        alpha_prior=alpha_prior_t,
                        bg_tile=None,
                    )
                    candidate = candidate_t[0].detach().cpu().numpy().astype(np.float32)
                    refined_tile = _blend_refined_candidate(
                        alpha_tile=alpha_tile,
                        candidate=candidate,
                        conf_tile=conf_tile,
                        tile_band=tile_band,
                        cfg=cfg.refine,
                    )
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    refined_tile = alpha_tile
            else:
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
            if guidance_cov >= 0.0:
                logger.info(
                    "Refine pass: frame %d/%d, band_coverage=%.3f, guidance_coverage=%.3f",
                    t + 1,
                    len(coarse_alphas),
                    float(band.mean()),
                    guidance_cov,
                )
            else:
                logger.info(
                    "Refine pass: frame %d/%d, band_coverage=%.3f",
                    t + 1,
                    len(coarse_alphas),
                    float(band.mean()),
                )

    return outputs
