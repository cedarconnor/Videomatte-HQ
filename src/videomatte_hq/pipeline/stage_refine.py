"""Stage-2 high-resolution refinement with tiled unknown-band compute."""

from __future__ import annotations

from collections.abc import Callable
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from videomatte_hq.pipeline.stage_qc import compute_boundary_iou, compute_iou
from videomatte_hq.pipeline.stage_trimap import (
    build_trimap_hybrid,
    build_trimap_from_logits,
    build_trimap_morphological,
    resize_binary_mask,
    resize_logits,
    sigmoid_logits,
)
from videomatte_hq.protocols import BatchedEdgeRefiner, EdgeRefiner, FrameSourceLike
from videomatte_hq.tiling.windows import hann_2d

logger = logging.getLogger(__name__)


def _to_rgb_float(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim != 3:
        raise ValueError(f"RGB frame must be HxWxC, got shape={rgb.shape}")
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    elif rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] < 3:
        raise ValueError(f"Unsupported channel count for RGB conversion: {rgb.shape[2]}")
    rgb = rgb[..., :3]

    if rgb.dtype == np.uint8:
        out = rgb.astype(np.float32) / 255.0
    elif np.issubdtype(rgb.dtype, np.integer):
        out = rgb.astype(np.float32) / float(np.iinfo(rgb.dtype).max)
    else:
        out = rgb.astype(np.float32)
        max_val = float(out.max()) if out.size else 1.0
        if max_val > 1.0:
            out = out / max(max_val, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _resolution_scale(shape: tuple[int, int], reference_long_side: int = 1920) -> float:
    long_side = max(int(shape[0]), int(shape[1]))
    return max(1.0, long_side / float(reference_long_side))


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def _iter_tiles(h: int, w: int, tile_size: int, overlap: int):
    step = max(1, int(tile_size) - int(overlap))
    y = 0
    while y < h:
        y1 = min(y + tile_size, h)
        x = 0
        while x < w:
            x1 = min(x + tile_size, w)
            yield x, y, x1, y1
            if x1 >= w:
                break
            x += step
        if y1 >= h:
            break
        y += step


@dataclass(slots=True)
class RefineStageConfig:
    refine_enabled: bool = True
    mematte_repo_dir: str = "third_party/MEMatte"
    mematte_checkpoint: str = "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth"
    mematte_max_tokens: int = 18500
    mematte_patch_decoder: bool = True
    tile_size: int = 1536
    tile_overlap: int = 96
    tile_batch_size: int = 1
    tile_min_unknown_coverage: float = 0.001
    trimap_mode: str = "morphological"
    trimap_erosion_px: int = 20
    trimap_dilation_px: int = 10
    trimap_fg_threshold: float = 0.9
    trimap_bg_threshold: float = 0.1
    trimap_fallback_band_px: int = 1
    unknown_edge_blend_px: int = 4
    skip_iou_threshold: float = 0.97
    preview_solidify_mask: bool = True
    preview_fg_floor: float = 0.9
    preview_bg_ceiling: float = 0.1
    preview_close_ratio: float = 0.006
    preview_min_close_px: int = 5
    preview_overlap_dilate_ratio: float = 0.015
    preview_min_overlap_dilate_px: int = 3
    device: str = "cuda"
    precision: str = "fp16"


@dataclass(slots=True)
class RefineSequenceResult:
    alphas: list[np.ndarray]
    reused_frames: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _RefinerCallCounter(BatchedEdgeRefiner):
    """Wrap an EdgeRefiner and count actual tile-level refine invocations."""

    inner: EdgeRefiner
    calls: int = 0

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.inner.refine(rgb, trimap)

    def refine_batch(self, rgb_tiles: list[np.ndarray], trimap_tiles: list[np.ndarray]) -> list[np.ndarray]:
        if len(rgb_tiles) != len(trimap_tiles):
            raise ValueError("rgb_tiles and trimap_tiles must have matching lengths.")
        if not rgb_tiles:
            return []
        self.calls += len(rgb_tiles)
        if isinstance(self.inner, BatchedEdgeRefiner):
            return self.inner.refine_batch(rgb_tiles, trimap_tiles)
        return [self.inner.refine(rgb, trimap) for rgb, trimap in zip(rgb_tiles, trimap_tiles)]


@dataclass(slots=True)
class MEMatteEdgeRefiner(BatchedEdgeRefiner):
    """EdgeRefiner adapter backed by the detectron2-free MEMatte wrapper."""

    repo_dir: str = "third_party/MEMatte"
    checkpoint_path: str = "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth"
    device: str = "cuda"
    precision: str = "fp16"
    max_number_token: int = 18500
    patch_decoder: bool = True
    _model: object | None = None

    def _ensure_loaded(self) -> object:
        if self._model is not None:
            return self._model
        from videomatte_hq.models.edge_mematte import MEMatteModel

        model = MEMatteModel(
            repo_dir=self.repo_dir,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            precision=self.precision,
            max_number_token=self.max_number_token,
            patch_decoder=self.patch_decoder,
        )
        model.load_weights(self.device)
        self._model = model
        return model

    @staticmethod
    def _prepare_mematte_inputs(rgb: np.ndarray, trimap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_f = _to_rgb_float(rgb)
        tri_f = np.asarray(trimap, dtype=np.float32)
        if tri_f.ndim == 3:
            tri_f = tri_f[..., 0]
        tri_f = np.clip(tri_f, 0.0, 1.0)
        # Strict > 0.5 so unknown pixels (exactly 0.5) get alpha_prior=0
        # rather than being biased toward foreground.
        alpha_prior = (tri_f > 0.5).astype(np.float32)
        return rgb_f, tri_f, alpha_prior

    @staticmethod
    def _alpha_to_numpy(alpha_t: object) -> np.ndarray:
        if hasattr(alpha_t, "detach"):
            alpha_np = alpha_t.detach().cpu().numpy()
        else:
            alpha_np = np.asarray(alpha_t)
        if alpha_np.ndim == 3 and alpha_np.shape[0] == 1:
            alpha_np = alpha_np[0]
        return np.clip(alpha_np.astype(np.float32), 0.0, 1.0)

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        model = self._ensure_loaded()
        import torch

        rgb_f, tri_f, alpha_prior = self._prepare_mematte_inputs(rgb, trimap)

        rgb_t = torch.from_numpy(np.transpose(rgb_f, (2, 0, 1)).copy()).float()
        trimap_t = torch.from_numpy(tri_f[None, ...].copy()).float()
        alpha_prior_t = torch.from_numpy(alpha_prior[None, ...].copy()).float()

        alpha_t = model.infer_tile(
            rgb_tile=rgb_t,
            trimap_tile=trimap_t,
            alpha_prior=alpha_prior_t,
            bg_tile=None,
        )
        return self._alpha_to_numpy(alpha_t)

    def refine_batch(self, rgb_tiles: list[np.ndarray], trimap_tiles: list[np.ndarray]) -> list[np.ndarray]:
        if len(rgb_tiles) != len(trimap_tiles):
            raise ValueError("rgb_tiles and trimap_tiles must have matching lengths.")
        if not rgb_tiles:
            return []

        model = self._ensure_loaded()
        import torch

        rgb_tensors = []
        trimap_tensors = []
        alpha_prior_tensors = []
        for rgb, trimap in zip(rgb_tiles, trimap_tiles):
            rgb_f, tri_f, alpha_prior = self._prepare_mematte_inputs(rgb, trimap)
            rgb_tensors.append(torch.from_numpy(np.transpose(rgb_f, (2, 0, 1)).copy()).float())
            trimap_tensors.append(torch.from_numpy(tri_f[None, ...].copy()).float())
            alpha_prior_tensors.append(torch.from_numpy(alpha_prior[None, ...].copy()).float())

        alpha_tiles = model.infer_tile_batch(
            rgb_tiles=rgb_tensors,
            trimap_tiles=trimap_tensors,
            alpha_priors=alpha_prior_tensors,
            bg_tiles=None,
        )
        if len(alpha_tiles) != len(rgb_tiles):
            raise RuntimeError(
                "MEMatte batched refine returned an unexpected number of tiles: "
                f"expected={len(rgb_tiles)} got={len(alpha_tiles)}"
            )
        return [self._alpha_to_numpy(alpha_t) for alpha_t in alpha_tiles]


def _largest_component(binary: np.ndarray) -> np.ndarray:
    fg = (np.asarray(binary) > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(1 + np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    fg = (np.asarray(binary) > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg
    bg = (fg == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    out = fg.copy()
    h, w = fg.shape
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        ww = int(stats[label_id, cv2.CC_STAT_WIDTH])
        hh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        touches_border = x <= 0 or y <= 0 or (x + ww) >= w or (y + hh) >= h
        if not touches_border:
            out[labels == label_id] = 1
    return out.astype(np.uint8)


def _filter_component_by_previous(
    binary: np.ndarray,
    previous_mask: np.ndarray | None,
    *,
    overlap_dilate_ratio: float,
    min_overlap_dilate_px: int,
) -> np.ndarray:
    fg = (np.asarray(binary) > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg
    if previous_mask is None:
        return _largest_component(fg)

    prev = (np.asarray(previous_mask, dtype=np.float32) >= 0.5).astype(np.uint8)
    if int(prev.sum()) == 0:
        return _largest_component(fg)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 2:
        return fg

    k = int(round(max(fg.shape) * float(overlap_dilate_ratio)))
    k = max(int(min_overlap_dilate_px), k)
    k = int(np.clip(k, 3, 301))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    prev_dilated = cv2.dilate(prev, kernel, iterations=1) > 0

    keep_ids: list[int] = []
    overlaps: list[tuple[int, int, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == label_id
        overlap = int(np.logical_and(comp, prev_dilated).sum())
        overlaps.append((label_id, overlap, area))
        if overlap > 0:
            keep_ids.append(label_id)

    if not keep_ids:
        prev_coords = np.argwhere(prev > 0)
        prev_center = prev_coords.mean(axis=0) if prev_coords.size else np.array([0.0, 0.0], dtype=np.float32)
        best_id = 1
        best_dist = float("inf")
        best_area = -1
        for label_id, _, area in overlaps:
            cx = float(centroids[label_id][0])
            cy = float(centroids[label_id][1])
            dist = float((cx - prev_center[1]) ** 2 + (cy - prev_center[0]) ** 2)
            if dist < best_dist or (dist == best_dist and area > best_area):
                best_dist = dist
                best_area = area
                best_id = label_id
        keep_ids = [best_id]

    keep = np.isin(labels, np.asarray(keep_ids, dtype=np.int32))
    return keep.astype(np.uint8)


def _cleanup_preview_mask(mask: np.ndarray, previous_mask: np.ndarray | None, cfg: RefineStageConfig) -> np.ndarray:
    fg = (np.asarray(mask, dtype=np.float32) >= 0.5).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg.astype(np.float32)

    fg = _filter_component_by_previous(
        fg,
        previous_mask=previous_mask,
        overlap_dilate_ratio=float(cfg.preview_overlap_dilate_ratio),
        min_overlap_dilate_px=int(cfg.preview_min_overlap_dilate_px),
    )

    k = int(round(max(fg.shape) * float(cfg.preview_close_ratio)))
    k = max(int(cfg.preview_min_close_px), k)
    k = int(np.clip(k, 3, 41))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    fg = _fill_holes(fg)
    fg = _largest_component(fg)
    return fg.astype(np.float32)


def _run_refine_tiles(
    refiner: EdgeRefiner,
    rgb_tiles: list[np.ndarray],
    trimap_tiles: list[np.ndarray],
) -> list[np.ndarray]:
    if len(rgb_tiles) != len(trimap_tiles):
        raise ValueError("rgb_tiles and trimap_tiles must have matching lengths.")
    if not rgb_tiles:
        return []
    if len(rgb_tiles) == 1:
        return [np.asarray(refiner.refine(rgb_tiles[0], trimap_tiles[0]), dtype=np.float32)]
    if isinstance(refiner, BatchedEdgeRefiner):
        outputs = refiner.refine_batch(rgb_tiles, trimap_tiles)
        if len(outputs) != len(rgb_tiles):
            raise RuntimeError(
                "Batched edge refiner returned an unexpected number of tiles: "
                f"expected={len(rgb_tiles)} got={len(outputs)}"
            )
        return [np.asarray(alpha, dtype=np.float32) for alpha in outputs]
    return [np.asarray(refiner.refine(rgb, trimap), dtype=np.float32) for rgb, trimap in zip(rgb_tiles, trimap_tiles)]


def _accumulate_tile_alpha(
    acc_num: np.ndarray,
    acc_den: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    unknown_tile: np.ndarray,
    tile_alpha: np.ndarray,
) -> None:
    tile_h = y1 - y0
    tile_w = x1 - x0
    if tile_alpha.shape != (tile_h, tile_w):
        tile_alpha = cv2.resize(
            tile_alpha,
            (tile_w, tile_h),
            interpolation=cv2.INTER_LINEAR,
        )
    window = hann_2d(tile_h, tile_w)
    weight = window * unknown_tile.astype(np.float32)
    acc_num[y0:y1, x0:x1] += weight * np.clip(tile_alpha, 0.0, 1.0)
    acc_den[y0:y1, x0:x1] += weight


def _refine_frame_tiled(
    rgb: np.ndarray,
    trimap: np.ndarray,
    coarse_prob: np.ndarray,
    refiner: EdgeRefiner,
    cfg: RefineStageConfig,
) -> np.ndarray:
    h, w = trimap.shape
    unknown = trimap == 0.5

    alpha = np.asarray(coarse_prob, dtype=np.float32).copy()
    alpha[trimap >= 1.0] = 1.0
    alpha[trimap <= 0.0] = 0.0

    if not bool(unknown.any()):
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    acc_num = np.zeros((h, w), dtype=np.float32)
    acc_den = np.zeros((h, w), dtype=np.float32)

    tile_size = max(64, int(cfg.tile_size))
    overlap = int(np.clip(cfg.tile_overlap, 0, tile_size - 1))
    tile_batch_size = max(1, int(cfg.tile_batch_size))
    min_cov = float(np.clip(cfg.tile_min_unknown_coverage, 0.0, 1.0))

    active_tiles: list[tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for x0, y0, x1, y1 in _iter_tiles(h, w, tile_size=tile_size, overlap=overlap):
        unknown_tile = unknown[y0:y1, x0:x1]
        coverage = float(unknown_tile.mean())
        if coverage <= min_cov:
            continue
        active_tiles.append(
            (
                x0,
                y0,
                x1,
                y1,
                unknown_tile,
                rgb[y0:y1, x0:x1],
                trimap[y0:y1, x0:x1],
            )
        )

    shape_groups: dict[tuple[int, int], list[tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for tile in active_tiles:
        x0, y0, x1, y1, _, _, _ = tile
        shape_groups.setdefault((y1 - y0, x1 - x0), []).append(tile)

    for group_tiles in shape_groups.values():
        for start in range(0, len(group_tiles), tile_batch_size):
            chunk_tiles = group_tiles[start : start + tile_batch_size]
            rgb_tiles = [tile[5] for tile in chunk_tiles]
            trimap_tiles = [tile[6] for tile in chunk_tiles]
            tile_alphas = _run_refine_tiles(refiner, rgb_tiles, trimap_tiles)
            for tile, tile_alpha in zip(chunk_tiles, tile_alphas):
                x0, y0, x1, y1, unknown_tile, _, _ = tile
                _accumulate_tile_alpha(
                    acc_num,
                    acc_den,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    unknown_tile=unknown_tile,
                    tile_alpha=tile_alpha,
                )

    has = acc_den > 1e-8
    alpha[has] = (acc_num[has] / acc_den[has]).astype(np.float32)
    alpha = _apply_unknown_edge_blend(alpha, coarse_prob, trimap, cfg)
    alpha[trimap >= 1.0] = 1.0
    alpha[trimap <= 0.0] = 0.0
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _apply_unknown_edge_blend(
    refined_alpha: np.ndarray,
    coarse_prob: np.ndarray,
    trimap: np.ndarray,
    cfg: RefineStageConfig,
) -> np.ndarray:
    blend_px = max(0, int(cfg.unknown_edge_blend_px))
    if blend_px <= 0:
        return np.clip(np.asarray(refined_alpha, dtype=np.float32), 0.0, 1.0).astype(np.float32)

    unknown = trimap == 0.5
    if not bool(unknown.any()):
        return np.clip(np.asarray(refined_alpha, dtype=np.float32), 0.0, 1.0).astype(np.float32)

    radius = max(1, int(round(float(blend_px) * _resolution_scale(trimap.shape))))
    dist = cv2.distanceTransform(unknown.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    refine_weight = _smoothstep01(dist / float(radius))

    refined = np.clip(np.asarray(refined_alpha, dtype=np.float32), 0.0, 1.0)
    prior = np.clip(np.asarray(coarse_prob, dtype=np.float32), 0.0, 1.0)
    out = refined.copy()
    out[unknown] = (
        prior[unknown] * (1.0 - refine_weight[unknown]) + refined[unknown] * refine_weight[unknown]
    ).astype(np.float32)
    out[trimap >= 1.0] = 1.0
    out[trimap <= 0.0] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _build_frame_trimap_and_prior(
    mask_up: np.ndarray,
    coarse_logit: np.ndarray,
    shape: tuple[int, int],
    cfg: RefineStageConfig,
) -> tuple[np.ndarray, np.ndarray]:
    logits_up = resize_logits(coarse_logit, shape)
    if cfg.trimap_mode == "morphological":
        trimap = build_trimap_morphological(
            mask_up,
            erosion_px=cfg.trimap_erosion_px,
            dilation_px=cfg.trimap_dilation_px,
        )
        coarse_prob = sigmoid_logits(logits_up)
    elif cfg.trimap_mode == "hybrid":
        trimap = build_trimap_hybrid(
            mask_up,
            logits_up,
            erosion_px=cfg.trimap_erosion_px,
            dilation_px=cfg.trimap_dilation_px,
            fg_threshold=cfg.trimap_fg_threshold,
            bg_threshold=cfg.trimap_bg_threshold,
            fallback_band_px=cfg.trimap_fallback_band_px,
        )
        coarse_prob = sigmoid_logits(logits_up)
    else:
        trimap = build_trimap_from_logits(
            logits_up,
            fg_threshold=cfg.trimap_fg_threshold,
            bg_threshold=cfg.trimap_bg_threshold,
            fallback_band_px=cfg.trimap_fallback_band_px,
        )
        coarse_prob = sigmoid_logits(logits_up)
    return trimap.astype(np.float32), np.asarray(coarse_prob, dtype=np.float32)


def build_edge_refiner(cfg: RefineStageConfig) -> EdgeRefiner:
    return MEMatteEdgeRefiner(
        repo_dir=cfg.mematte_repo_dir,
        checkpoint_path=cfg.mematte_checkpoint,
        device=cfg.device,
        precision=cfg.precision,
        max_number_token=cfg.mematte_max_tokens,
        patch_decoder=cfg.mematte_patch_decoder,
    )


def refine_sequence(
    source: FrameSourceLike,
    coarse_masks: list[np.ndarray],
    coarse_logits: list[np.ndarray],
    cfg: RefineStageConfig,
    refiner: EdgeRefiner | None = None,
    trimap_callback: Callable[[int, np.ndarray], None] | None = None,
) -> RefineSequenceResult:
    """Run stage-2 refinement for an entire sequence."""
    num_frames = int(len(source))
    if len(coarse_masks) != num_frames or len(coarse_logits) != num_frames:
        raise ValueError(
            "coarse_masks and coarse_logits must match source frame count. "
            f"got masks={len(coarse_masks)} logits={len(coarse_logits)} source={num_frames}"
        )

    use_refiner = bool(cfg.refine_enabled)
    source_shape = tuple(int(v) for v in source.resolution)
    if not use_refiner:
        logger.info("MEMatte refinement disabled — generating SAM-only preview alphas.")
        out_alphas: list[np.ndarray] = []
        prev_mask: np.ndarray | None = None
        for frame_idx in range(num_frames):
            mask_up = resize_binary_mask(coarse_masks[frame_idx], source_shape, threshold=0.5)
            if trimap_callback is not None:
                trimap, _ = _build_frame_trimap_and_prior(
                    mask_up,
                    coarse_logits[frame_idx],
                    source_shape,
                    cfg,
                )
                trimap_callback(frame_idx, trimap)
            alpha = _cleanup_preview_mask(mask_up, prev_mask, cfg)
            out_alphas.append(np.clip(alpha, 0.0, 1.0).astype(np.float32))
            prev_mask = mask_up
        return RefineSequenceResult(alphas=out_alphas, reused_frames=[])

    active_refiner = refiner
    if active_refiner is None:
        active_refiner = build_edge_refiner(cfg)
    if active_refiner is None:
        raise RuntimeError("MEMatte refinement is required, but no refiner instance is available.")
    counted_refiner = _RefinerCallCounter(active_refiner)

    reused_frames: list[int] = []
    out_alphas: list[np.ndarray] = []
    prev_mask: np.ndarray | None = None
    prev_alpha: np.ndarray | None = None

    for frame_idx in range(num_frames):
        mask_up = resize_binary_mask(coarse_masks[frame_idx], source_shape, threshold=0.5)
        should_reuse = bool(
            prev_mask is not None
            and prev_alpha is not None
            and compute_iou(mask_up, prev_mask) > float(cfg.skip_iou_threshold)
            and compute_boundary_iou(mask_up, prev_mask) > float(cfg.skip_iou_threshold) * 0.9
        )
        trimap: np.ndarray | None = None
        coarse_prob: np.ndarray | None = None
        if not should_reuse or trimap_callback is not None:
            trimap, coarse_prob = _build_frame_trimap_and_prior(
                mask_up,
                coarse_logits[frame_idx],
                source_shape,
                cfg,
            )
            if trimap_callback is not None:
                trimap_callback(frame_idx, trimap)

        if should_reuse:
            reused_alpha = prev_alpha.copy()
            out_alphas.append(reused_alpha)
            prev_alpha = reused_alpha
            reused_frames.append(frame_idx)
            prev_mask = mask_up
            continue

        # Decode the full-resolution frame only when MEMatte actually needs to run.
        rgb = _to_rgb_float(source[frame_idx])
        if trimap is None or coarse_prob is None:
            trimap, coarse_prob = _build_frame_trimap_and_prior(
                mask_up,
                coarse_logits[frame_idx],
                source_shape,
                cfg,
            )

        alpha = _refine_frame_tiled(
            rgb=rgb,
            trimap=trimap,
            coarse_prob=coarse_prob,
            refiner=counted_refiner,
            cfg=cfg,
        )

        out_alphas.append(alpha)
        prev_mask = mask_up
        prev_alpha = alpha

        if frame_idx == 0 or (frame_idx + 1) % 50 == 0:
            logger.info(
                "Refine frame %d/%d complete (reused=%d).",
                frame_idx + 1,
                num_frames,
                len(reused_frames),
            )

    if num_frames > 0 and counted_refiner.calls <= 0:
        raise RuntimeError(
            "MEMatte did not execute on any tiles (trimap unknown band was empty for all processed frames). "
            "Widen the trimap unknown band (adjust trimap_fg_threshold/trimap_bg_threshold) and retry."
        )

    return RefineSequenceResult(alphas=out_alphas, reused_frames=reused_frames)
