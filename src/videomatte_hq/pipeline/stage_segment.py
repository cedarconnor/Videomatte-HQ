"""Stage-1 segmentation and temporal chunk orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from videomatte_hq.pipeline.stage_qc import check_drift
from videomatte_hq.pipeline.stage_trimap import probability_to_logits, resize_logits, sigmoid_logits
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter
from videomatte_hq.protocols import FrameSourceLike, PromptAdapter, SegmentPrompt, SegmentResult, Segmenter

logger = logging.getLogger(__name__)

SAM_MODEL_QUALITY_ORDER: tuple[str, ...] = (
    "sam2_l.pt",
    "sam2_b.pt",
    "sam2_s.pt",
    "sam2_t.pt",
    "sam_h.pt",
    "sam_l.pt",
    "sam_b.pt",
    "mobile_sam.pt",
)


class SegmentBackend(Protocol):
    """Low-level per-chunk segmentation backend."""

    def segment_chunk(
        self,
        frames: list[np.ndarray],
        prompt: SegmentPrompt,
        anchor_frame_index: int = 0,
        reference_anchor_area: float | None = None,
    ) -> list[np.ndarray]:
        """Return per-frame logits for the chunk."""


class VideoSegmentBackend(Protocol):
    """Optional low-level full-video segmentation backend."""

    def segment_video_sequence(
        self,
        video_path: str | Path,
        prompt: SegmentPrompt,
        *,
        start_frame: int = 0,
        num_frames: int,
        frame_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        """Return per-frame logits for a video source."""


def _resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target = int(max(1, long_side))
    if max(h, w) == target:
        return frame
    scale = target / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    src = np.asarray(mask, dtype=np.float32)
    if src.ndim == 3:
        src = src[..., 0]
    if src.shape != (h, w):
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    return (src >= float(threshold)).astype(np.float32)


def _expand_bbox(
    bbox: tuple[float, float, float, float] | None,
    frame_shape: tuple[int, int],
    *,
    expand_ratio: float,
    min_expand_px: int,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    expand = max(float(min_expand_px), max(bw, bh) * float(expand_ratio))
    return (
        max(0.0, x0 - expand),
        max(0.0, y0 - expand),
        min(float(w), x1 + expand),
        min(float(h), y1 + expand),
    )


def _bbox_from_mask(
    mask: np.ndarray,
    *,
    expand_ratio: float = 0.0,
    min_expand_px: int = 0,
) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(np.asarray(mask, dtype=np.float32) >= 0.5)
    if ys.size == 0:
        return None
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    if expand_ratio > 0.0 or min_expand_px > 0:
        return _expand_bbox(
            bbox,
            frame_shape=np.asarray(mask).shape[:2],
            expand_ratio=expand_ratio,
            min_expand_px=min_expand_px,
        )
    return bbox


def _scale_prompt(prompt: SegmentPrompt, from_shape: tuple[int, int], to_shape: tuple[int, int]) -> SegmentPrompt:
    if from_shape == to_shape:
        return prompt
    sy = float(to_shape[0]) / float(max(from_shape[0], 1))
    sx = float(to_shape[1]) / float(max(from_shape[1], 1))

    bbox = None
    if prompt.bbox is not None:
        x0, y0, x1, y1 = prompt.bbox
        bbox = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

    pos = [(float(x * sx), float(y * sy)) for x, y in prompt.positive_points]
    neg = [(float(x * sx), float(y * sy)) for x, y in prompt.negative_points]

    mask = None
    if prompt.mask is not None:
        m = np.asarray(prompt.mask, dtype=np.float32)
        mask = cv2.resize(m, (to_shape[1], to_shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return SegmentPrompt(bbox=bbox, positive_points=pos, negative_points=neg, mask=mask)


def _as_probabilities(logits_or_prob: np.ndarray) -> np.ndarray:
    x = np.asarray(logits_or_prob, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if lo >= 0.0 and hi <= 1.0:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return sigmoid_logits(x)


def _ordered_sam_model_candidates(model_name: str) -> list[str]:
    requested = str(model_name).strip() or "sam2_l.pt"
    candidates: list[str] = [requested]

    req_path = Path(requested)
    looks_like_path = req_path.is_absolute() or any(sep in requested for sep in ("/", "\\"))
    if looks_like_path:
        return candidates

    local_models = [m for m in SAM_MODEL_QUALITY_ORDER if m != requested and Path(m).exists()]
    remote_models = [m for m in SAM_MODEL_QUALITY_ORDER if m != requested and m not in local_models]
    candidates.extend(local_models)
    candidates.extend(remote_models)
    return candidates


def _to_mask(logits_or_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = _as_probabilities(logits_or_prob)
    return (probs >= float(threshold)).astype(np.float32)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
    a = np.asarray(mask_a, dtype=np.float32) >= float(threshold)
    b = np.asarray(mask_b, dtype=np.float32) >= float(threshold)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _select_mask_candidate(arr: np.ndarray, prompt: SegmentPrompt | None) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[0] == 0:
        return np.zeros(data.shape[1:], dtype=np.float32)
    if data.ndim != 3 or data.shape[0] <= 0:
        raise ValueError(f"Unexpected mask candidate shape: {data.shape}")
    if prompt is None:
        areas = data.reshape(data.shape[0], -1).sum(axis=1)
        return data[int(np.argmax(areas))]

    prompt_mask = None
    prompt_mask_area = 0.0
    if prompt.mask is not None:
        prompt_mask = np.asarray(prompt.mask, dtype=np.float32)
        if prompt_mask.shape != data.shape[1:]:
            prompt_mask = cv2.resize(prompt_mask, (data.shape[2], data.shape[1]), interpolation=cv2.INTER_LINEAR)
        prompt_mask_area = float((prompt_mask >= 0.5).sum())

    best_idx = 0
    best_score = -1e18
    for idx in range(data.shape[0]):
        cand = data[idx]
        cand_bin = cand >= 0.5
        score = 0.0

        if prompt.positive_points:
            hits = 0
            for x, y in prompt.positive_points:
                xi = int(np.clip(round(x), 0, cand.shape[1] - 1))
                yi = int(np.clip(round(y), 0, cand.shape[0] - 1))
                if cand_bin[yi, xi]:
                    hits += 1
            score += 3.0 * float(hits)
            score -= 2.0 * float(len(prompt.positive_points) - hits)

        if prompt.negative_points:
            ok = 0
            for x, y in prompt.negative_points:
                xi = int(np.clip(round(x), 0, cand.shape[1] - 1))
                yi = int(np.clip(round(y), 0, cand.shape[0] - 1))
                if not cand_bin[yi, xi]:
                    ok += 1
            score += 2.0 * float(ok)
            score -= 2.0 * float(len(prompt.negative_points) - ok)

        if prompt_mask is not None:
            iou = _mask_iou(cand, prompt_mask, threshold=0.5)
            score += 8.0 * float(iou)
            cand_area = float(cand_bin.sum())
            denom = max(prompt_mask_area, 1.0)
            area_penalty = abs(cand_area - prompt_mask_area) / denom
            score -= 2.0 * float(area_penalty)

        if prompt.bbox is not None:
            x0, y0, x1, y1 = prompt.bbox
            ix0 = int(np.clip(np.floor(x0), 0, cand.shape[1]))
            iy0 = int(np.clip(np.floor(y0), 0, cand.shape[0]))
            ix1 = int(np.clip(np.ceil(x1), 0, cand.shape[1]))
            iy1 = int(np.clip(np.ceil(y1), 0, cand.shape[0]))
            if ix1 > ix0 and iy1 > iy0:
                inside = float(cand_bin[iy0:iy1, ix0:ix1].sum())
                outside = float(cand_bin.sum()) - inside
                score += 1.5 * inside / max((ix1 - ix0) * (iy1 - iy0), 1.0)
                score -= 1.0 * outside / max(float(cand_bin.size), 1.0)

        if score > best_score:
            best_score = score
            best_idx = idx

    return data[int(best_idx)]


def _apply_temporal_area_guard(
    probability: np.ndarray,
    previous_mask: np.ndarray | None,
    *,
    threshold: float = 0.5,
    max_area_ratio: float = 3.0,
    min_iou: float = 0.20,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if previous_mask is None:
        return prob
    prev = (np.asarray(previous_mask, dtype=np.float32) >= float(threshold)).astype(np.float32)
    if not bool((prev >= 0.5).any()):
        return prob

    cur = (prob >= float(threshold)).astype(np.float32)
    prev_area = float((prev >= 0.5).sum())
    cur_area = float((cur >= 0.5).sum())
    if prev_area <= 0.0:
        return prob
    area_ratio = cur_area / max(prev_area, 1.0)
    area_ratio_limit = max(float(max_area_ratio), 1.01)
    low_ratio_limit = 1.0 / area_ratio_limit
    iou = _mask_iou(cur, prev, threshold=0.5)
    if iou < float(min_iou) and (area_ratio > area_ratio_limit or area_ratio < low_ratio_limit):
        return prev.astype(np.float32)
    return prob


def _apply_anchor_reference_area_guard(
    probability: np.ndarray,
    previous_mask: np.ndarray | None,
    *,
    reference_area: float | None,
    threshold: float = 0.5,
    max_area_ratio: float = 4.0,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if previous_mask is None or reference_area is None:
        return prob
    ref = float(reference_area)
    if ref <= 0.0:
        return prob
    ratio_limit = max(float(max_area_ratio), 1.01)
    low_ratio_limit = 1.0 / ratio_limit

    prev = (np.asarray(previous_mask, dtype=np.float32) >= float(threshold)).astype(np.float32)
    cur = (prob >= float(threshold)).astype(np.float32)
    cur_area = float((cur >= 0.5).sum())
    ratio = cur_area / max(ref, 1.0)
    if ratio > ratio_limit or ratio < low_ratio_limit:
        return prev.astype(np.float32)
    return prob


def _blend_overlap(existing: np.ndarray, incoming: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - a) * np.asarray(existing, dtype=np.float32) + a * np.asarray(incoming, dtype=np.float32)
    return out.astype(np.float32)


def _gate_probability_by_bbox(
    probability: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if bbox is None:
        return prob

    h, w = prob.shape[:2]
    x0, y0, x1, y1 = bbox
    ix0 = int(np.clip(np.floor(x0), 0, w))
    iy0 = int(np.clip(np.floor(y0), 0, h))
    ix1 = int(np.clip(np.ceil(x1), 0, w))
    iy1 = int(np.clip(np.ceil(y1), 0, h))
    if ix1 <= ix0 or iy1 <= iy0:
        return np.zeros_like(prob, dtype=np.float32)

    gated = np.zeros_like(prob, dtype=np.float32)
    gated[iy0:iy1, ix0:ix1] = prob[iy0:iy1, ix0:ix1]
    return gated


def _filter_probability_by_prev_component(
    probability: np.ndarray,
    previous_mask: np.ndarray | None,
    *,
    threshold: float = 0.5,
    overlap_dilate_ratio: float = 0.015,
    min_overlap_dilate_px: int = 3,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if previous_mask is None:
        return prob

    prev = np.asarray(previous_mask, dtype=np.float32) >= float(threshold)
    if not bool(prev.any()):
        return prob

    cur = prob >= float(threshold)
    if not bool(cur.any()):
        return prob

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cur.astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return prob

    k = int(round(max(cur.shape) * float(overlap_dilate_ratio)))
    k = max(int(min_overlap_dilate_px), k)
    k = int(np.clip(k, 3, 401))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    prev_dilated = cv2.dilate(prev.astype(np.uint8), kernel, iterations=1) > 0

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
        prev_coords = np.argwhere(prev)
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
    return np.where(keep, prob, 0.0).astype(np.float32)


def _apply_strict_background_suppression(
    probability: np.ndarray,
    previous_mask: np.ndarray | None,
    *,
    threshold: float = 0.5,
    bbox_expand_ratio: float = 0.10,
    min_bbox_expand_px: int = 24,
    overlap_dilate_ratio: float = 0.03,
    min_overlap_dilate_px: int = 12,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if previous_mask is None:
        return prob

    prev = np.asarray(previous_mask, dtype=np.float32)
    if not bool((prev >= float(threshold)).any()):
        return prob

    strict_bbox = _bbox_from_mask(
        prev,
        expand_ratio=float(bbox_expand_ratio),
        min_expand_px=int(min_bbox_expand_px),
    )
    gated = _gate_probability_by_bbox(prob, strict_bbox)
    return _filter_probability_by_prev_component(
        gated,
        previous_mask=prev,
        threshold=threshold,
        overlap_dilate_ratio=float(overlap_dilate_ratio),
        min_overlap_dilate_px=int(min_overlap_dilate_px),
    )


def _to_bgr_u8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] < 3:
        raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
    arr = arr[..., :3]

    if arr.dtype == np.uint8:
        rgb = arr
    else:
        rgb = arr.astype(np.float32)
        if np.issubdtype(arr.dtype, np.integer):
            rgb = rgb / float(np.iinfo(arr.dtype).max)
        elif (float(rgb.max()) if rgb.size else 0.0) > 1.0:
            rgb = rgb / max(float(rgb.max()) if rgb.size else 1.0, 1.0)
        rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _normalize_precision(precision: str) -> str:
    p = str(precision).strip().lower()
    if p in {"fp16", "float16", "half"}:
        return "fp16"
    if p in {"bf16", "bfloat16"}:
        return "bf16"
    return "fp32"


def _is_unexpected_half_kwarg_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "unexpected keyword" in text and "half" in text


@dataclass(slots=True)
class SegmentStageConfig:
    backend: str = "ultralytics_sam3"
    sam3_model: str = "sam2_l.pt"
    processing_long_side: int = 960
    chunk_size: int = 100
    chunk_overlap: int = 5
    device: str = "cuda"
    precision: str = "fp16"
    mask_threshold: float = 0.5
    bbox_expand_ratio: float = 0.08
    min_bbox_expand_px: int = 12
    temporal_component_filter: bool = False
    strict_background_suppression: bool = False
    strict_bbox_expand_ratio: float = 0.10
    strict_min_bbox_expand_px: int = 24
    strict_overlap_dilate_ratio: float = 0.03
    strict_min_overlap_dilate_px: int = 12
    strict_temporal_guard: bool = True
    strict_max_area_ratio: float = 3.0
    strict_min_iou: float = 0.20
    strict_anchor_max_area_ratio: float = 4.0
    drift_iou_threshold: float = 0.70
    drift_area_threshold: float = 0.40
    max_reanchors_per_chunk: int = 2


@dataclass(slots=True)
class StaticMaskSegmentBackend:
    """Deterministic backend that propagates the anchor mask to all frames."""

    mask_threshold: float = 0.5
    confidence: float = 0.995

    def segment_chunk(
        self,
        frames: list[np.ndarray],
        prompt: SegmentPrompt,
        anchor_frame_index: int = 0,
    ) -> list[np.ndarray]:
        if not frames:
            return []
        shape = frames[0].shape[:2]

        if prompt.mask is not None:
            mask = _resize_mask(prompt.mask, shape, threshold=self.mask_threshold)
        elif prompt.bbox is not None:
            x0, y0, x1, y1 = prompt.bbox
            h, w = shape
            mask = np.zeros((h, w), dtype=np.float32)
            ix0 = int(np.clip(round(x0), 0, w))
            iy0 = int(np.clip(round(y0), 0, h))
            ix1 = int(np.clip(round(x1), 0, w))
            iy1 = int(np.clip(round(y1), 0, h))
            if ix1 > ix0 and iy1 > iy0:
                mask[iy0:iy1, ix0:ix1] = 1.0
        else:
            h, w = shape
            mask = np.zeros((h, w), dtype=np.float32)

        prob_fg = float(np.clip(self.confidence, 0.5, 1.0 - 1e-6))
        prob_bg = 1.0 - prob_fg
        prob = np.where(mask >= 0.5, prob_fg, prob_bg).astype(np.float32)
        logit = probability_to_logits(prob)
        return [logit.copy() for _ in frames]


@dataclass(slots=True)
class UltralyticsSAM3SegmentBackend:
    """Best-effort Ultralytics SAM backend adapter.

    This wrapper runs per-frame prompting for compatibility across Ultralytics
    versions and returns logits derived from the predicted mask probability map.
    """

    model_name: str = "sam2_l.pt"
    device: str = "cuda"
    precision: str = "fp16"
    enable_tf32: bool = True
    processing_long_side: int = 960
    mask_threshold: float = 0.5
    temporal_component_filter: bool = False
    strict_background_suppression: bool = False
    strict_bbox_expand_ratio: float = 0.10
    strict_min_bbox_expand_px: int = 24
    strict_overlap_dilate_ratio: float = 0.03
    strict_min_overlap_dilate_px: int = 12
    strict_temporal_guard: bool = True
    strict_max_area_ratio: float = 3.0
    strict_min_iou: float = 0.20
    strict_anchor_max_area_ratio: float = 4.0
    prompt_adapter: PromptAdapter = field(default_factory=MaskPromptAdapter)
    _model: object | None = None
    _runtime_configured: bool = False
    _half_kwarg_supported: bool | None = None
    _preferred_prompt_variant: str | None = None
    _resolved_model_name: str | None = None

    def _video_predictor_class(self):
        name = str(self._resolved_model_name or self.model_name).lower()
        try:
            from ultralytics.models.sam.predict import SAM2VideoPredictor, SAM3VideoPredictor
        except Exception:
            return None
        if "sam3" in name:
            return SAM3VideoPredictor
        if "sam2" in name:
            return SAM2VideoPredictor
        return None

    def _is_cuda(self) -> bool:
        return str(self.device).strip().lower().startswith("cuda")

    def _configure_runtime(self) -> None:
        if self._runtime_configured:
            return
        self._runtime_configured = True
        if not self._is_cuda():
            return

        try:
            import torch
        except Exception:
            return

        try:
            if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "benchmark"):
                torch.backends.cudnn.benchmark = True
            if self.enable_tf32:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
        except Exception:
            # Runtime tuning is opportunistic; inference should continue even if this fails.
            pass

    def _build_infer_kwargs(self, variant: dict[str, object]) -> dict[str, object]:
        kwargs: dict[str, object] = {"verbose": False, "device": self.device, **variant}
        use_half = self._is_cuda() and _normalize_precision(self.precision) == "fp16"
        if use_half and self._half_kwarg_supported is not False:
            kwargs["half"] = True
        return kwargs

    def _build_video_kwargs(
        self,
        prompt: SegmentPrompt,
        *,
        predictor_class: object,
        video_path: str | Path,
    ) -> dict[str, object]:
        points_flat = [list(p) for p in prompt.positive_points + prompt.negative_points]
        labels_flat = [1] * len(prompt.positive_points) + [0] * len(prompt.negative_points)
        points = [points_flat] if points_flat else None
        labels = [labels_flat] if labels_flat else None
        bboxes = [list(prompt.bbox)] if prompt.bbox is not None else None

        kwargs: dict[str, object] = {
            "source": str(video_path),
            "stream": True,
            "predictor": predictor_class,
            "verbose": False,
            "device": self.device,
            "imgsz": int(max(32, self.processing_long_side)),
        }
        if bboxes is not None:
            kwargs["bboxes"] = bboxes
        if points is not None:
            kwargs["points"] = points
            kwargs["labels"] = labels
        if self._is_cuda() and _normalize_precision(self.precision) == "fp16" and self._half_kwarg_supported is not False:
            kwargs["half"] = True
        return kwargs

    def _prepare_video_predictor_for_stream(self, predictor: object, *, start_frame: int) -> None:
        """Reset predictor state and optionally pre-seek its video dataset before streaming."""
        try:
            if hasattr(predictor, "reset_prompts"):
                predictor.reset_prompts()
        except Exception:
            # Prompt reset is best-effort; runtime can proceed without it.
            pass

        if hasattr(predictor, "inference_state"):
            reset_done = False
            try:
                state = getattr(predictor, "inference_state")
            except Exception as exc:
                logger.warning("Failed to access Ultralytics predictor inference_state during reset: %s", exc)
                state = None

            if state is not None:
                clear_fn = getattr(state, "clear", None)
                if callable(clear_fn):
                    try:
                        clear_fn()
                        reset_done = True
                    except Exception as exc:
                        logger.warning(
                            "Failed to clear Ultralytics predictor inference_state in place; falling back to replacement: %s",
                            exc,
                        )

            if not reset_done:
                try:
                    predictor.inference_state = {}
                except Exception as exc:
                    logger.warning("Failed to replace Ultralytics predictor inference_state during reset: %s", exc)

        start = max(0, int(start_frame))
        if start <= 0:
            return

        dataset = getattr(predictor, "dataset", None)
        if dataset is None:
            raise RuntimeError("Ultralytics predictor dataset is not initialized for offset video fast path.")
        if str(getattr(dataset, "mode", "")).lower() != "video":
            raise RuntimeError("Ultralytics predictor dataset is not in video mode for offset video fast path.")

        total_frames = int(getattr(dataset, "frames", 0) or 0)
        if total_frames > 0 and start >= total_frames:
            raise RuntimeError(
                f"Requested start frame {start} is outside video length {total_frames} for offset video fast path."
            )

        cap = getattr(dataset, "cap", None)
        if cap is None or not hasattr(cap, "set"):
            raise RuntimeError("Ultralytics predictor video capture is unavailable for offset video fast path.")

        ok = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, start))
        pos = None
        if hasattr(cap, "get"):
            try:
                pos = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
            except Exception:
                pos = None

        # OpenCV seek reports can be codec-dependent, but a clear position mismatch
        # should fail regardless of the boolean return value from cap.set().
        if pos is not None and abs(pos - float(start)) > 2.0:
            raise RuntimeError(
                f"Failed to seek Ultralytics video predictor to start frame {start} (reported position {pos:.1f})."
            )

        if hasattr(dataset, "frame"):
            # Loader increments `dataset.frame` after retrieve; preserve Ultralytics 1-based frame indexing semantics.
            dataset.frame = start

    def _prompt_variants(self, prompt: SegmentPrompt) -> list[tuple[str, dict[str, object]]]:
        points_flat = [list(p) for p in prompt.positive_points + prompt.negative_points]
        labels_flat = [1] * len(prompt.positive_points) + [0] * len(prompt.negative_points)
        # Ultralytics SAM expects one object prompt as (1, K, 2) and (1, K).
        points = [points_flat] if points_flat else None
        labels = [labels_flat] if labels_flat else None
        bbox = [list(prompt.bbox)] if prompt.bbox is not None else None

        variants: list[tuple[str, dict[str, object]]] = []
        # SAM prompt tensor shapes vary across versions.
        # Prefer combined prompts to avoid overly tight point-only masks.
        if points is not None and labels is not None and bbox is not None:
            variants.append(("bbox_points", {"points": points, "labels": labels, "bboxes": bbox}))
        if bbox is not None:
            variants.append(("bbox", {"bboxes": bbox}))
        if points is not None and labels is not None:
            variants.append(("points", {"points": points, "labels": labels}))
        if not variants:
            variants.append(("none", {}))
        return variants

    def _ordered_prompt_variants(
        self,
        variants: list[tuple[str, dict[str, object]]],
    ) -> list[tuple[str, dict[str, object]]]:
        preferred = self._preferred_prompt_variant
        if not preferred:
            return variants
        for idx, (name, _) in enumerate(variants):
            if name == preferred:
                return [variants[idx], *variants[:idx], *variants[idx + 1 :]]
        return variants

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import SAM
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics SAM backend requested but ultralytics is unavailable. "
                "Install with: pip install ultralytics"
            ) from exc

        errors: list[str] = []
        for candidate in _ordered_sam_model_candidates(self.model_name):
            try:
                self._model = SAM(candidate)
                if candidate != self.model_name:
                    logger.warning(
                        "Requested SAM model '%s' unavailable; falling back to '%s'.",
                        self.model_name,
                        candidate,
                    )
                self._resolved_model_name = str(candidate)
                return self._model
            except Exception as exc:
                errors.append(f"{candidate}: {exc.__class__.__name__}: {exc}")
                continue

        detail = "; ".join(errors[:5]) if errors else "unknown error"
        raise RuntimeError(f"Failed to load any SAM model candidates: {detail}")

    def _result_to_probability(
        self,
        result: object,
        shape: tuple[int, int],
        prompt: SegmentPrompt | None = None,
    ) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        prob = np.zeros((h, w), dtype=np.float32)

        masks_obj = getattr(result, "masks", None)
        data = getattr(masks_obj, "data", None) if masks_obj is not None else None
        if data is not None:
            if hasattr(data, "detach"):
                arr = data.detach().cpu().numpy()
            else:
                arr = np.asarray(data)
            if arr.ndim in {2, 3}:
                selected = _select_mask_candidate(arr, prompt=prompt)
                prob = selected.astype(np.float32)

        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        if float(prob.max()) > 1.0 or float(prob.min()) < 0.0:
            prob = sigmoid_logits(prob)
        return np.clip(prob, 0.0, 1.0).astype(np.float32)

    def _infer_single(self, model: object, frame: np.ndarray, prompt: SegmentPrompt) -> np.ndarray:
        variants = self._ordered_prompt_variants(self._prompt_variants(prompt))

        errors: list[str] = []
        for variant_name, variant in variants:
            kwargs = self._build_infer_kwargs(variant)
            try:
                if hasattr(model, "predict"):
                    results = model.predict(source=frame, **kwargs)
                else:
                    results = model(frame, **kwargs)
                if kwargs.get("half", False):
                    self._half_kwarg_supported = True
            except TypeError as exc:
                if kwargs.get("half", False) and _is_unexpected_half_kwarg_error(exc):
                    if self._half_kwarg_supported is not False:
                        logger.warning(
                            "Ultralytics SAM backend does not accept `half` kwarg; "
                            "disabling fp16 inference hint for this runtime."
                        )
                    self._half_kwarg_supported = False
                    no_half_kwargs = dict(kwargs)
                    no_half_kwargs.pop("half", None)
                    try:
                        if hasattr(model, "predict"):
                            results = model.predict(source=frame, **no_half_kwargs)
                        else:
                            results = model(frame, **no_half_kwargs)
                    except Exception as retry_exc:
                        errors.append(
                            f"{variant_name}: "
                            f"{retry_exc.__class__.__name__}: {retry_exc}"
                        )
                        continue
                else:
                    errors.append(f"{variant_name}: {exc.__class__.__name__}: {exc}")
                    continue
            except Exception as exc:
                errors.append(f"{variant_name}: {exc.__class__.__name__}: {exc}")
                continue

            self._preferred_prompt_variant = variant_name
            result = results[0] if isinstance(results, (list, tuple)) else results
            prob = self._result_to_probability(result, frame.shape[:2], prompt=prompt)
            return probability_to_logits(prob)

        detail = "; ".join(errors[:3]) if errors else "unknown error"
        raise RuntimeError(f"Ultralytics SAM inference failed for all prompt variants: {detail}")

    def segment_video_sequence(
        self,
        video_path: str | Path,
        prompt: SegmentPrompt,
        *,
        start_frame: int = 0,
        num_frames: int,
        frame_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        total = max(0, int(num_frames))
        if total == 0:
            return []
        start = max(0, int(start_frame))

        predictor_cls = self._video_predictor_class()
        if predictor_cls is None:
            raise RuntimeError("Video predictor is not available for the configured SAM model.")

        model = self._load_model()
        self._configure_runtime()
        kwargs = self._build_video_kwargs(prompt, predictor_class=predictor_cls, video_path=video_path)

        try:
            stream = model.predict(**kwargs)
        except TypeError as exc:
            if kwargs.get("half", False) and _is_unexpected_half_kwarg_error(exc):
                self._half_kwarg_supported = False
                kwargs.pop("half", None)
                stream = model.predict(**kwargs)
            else:
                raise
        else:
            if kwargs.get("half", False):
                self._half_kwarg_supported = True

        predictor = getattr(model, "predictor", None)
        on_predict_start_callbacks: list[object] | None = None
        predictor_init_callback = None
        callbacks_map = getattr(predictor, "callbacks", None) if predictor is not None else None
        callbacks_list = callbacks_map.get("on_predict_start") if isinstance(callbacks_map, dict) else None
        if isinstance(callbacks_list, list):
            def _videomatte_prepare_callback(pred):
                self._prepare_video_predictor_for_stream(pred, start_frame=start)

            callbacks_list.insert(0, _videomatte_prepare_callback)
            on_predict_start_callbacks = callbacks_list
            predictor_init_callback = _videomatte_prepare_callback

        logits: list[np.ndarray] = []
        current_prompt = prompt
        prev_mask: np.ndarray | None = None
        anchor_area: float | None = None
        if prompt.mask is not None:
            anchor_prompt_mask = _resize_mask(prompt.mask, frame_shape, threshold=self.mask_threshold)
            anchor_area = float((anchor_prompt_mask >= 0.5).sum())
            if anchor_area <= 0.0:
                anchor_area = None

        try:
            for result in stream:
                prob = self._result_to_probability(result, frame_shape, prompt=current_prompt)
                frame_logit = probability_to_logits(prob)
                if prev_mask is not None and (self.temporal_component_filter or self.strict_background_suppression):
                    prob = _as_probabilities(frame_logit)
                    if self.strict_background_suppression:
                        prob = _apply_strict_background_suppression(
                            prob,
                            prev_mask,
                            threshold=self.mask_threshold,
                            bbox_expand_ratio=self.strict_bbox_expand_ratio,
                            min_bbox_expand_px=self.strict_min_bbox_expand_px,
                            overlap_dilate_ratio=self.strict_overlap_dilate_ratio,
                            min_overlap_dilate_px=self.strict_min_overlap_dilate_px,
                        )
                        if self.strict_temporal_guard:
                            prob = _apply_temporal_area_guard(
                                prob,
                                prev_mask,
                                threshold=self.mask_threshold,
                                max_area_ratio=self.strict_max_area_ratio,
                                min_iou=self.strict_min_iou,
                            )
                            prob = _apply_anchor_reference_area_guard(
                                prob,
                                prev_mask,
                                reference_area=anchor_area,
                                threshold=self.mask_threshold,
                                max_area_ratio=self.strict_anchor_max_area_ratio,
                            )
                    else:
                        prob = _filter_probability_by_prev_component(
                            prob,
                            prev_mask,
                            threshold=self.mask_threshold,
                        )
                    frame_logit = probability_to_logits(prob)

                logits.append(frame_logit.astype(np.float32))
                prev_mask = _to_mask(frame_logit, threshold=self.mask_threshold)
                if anchor_area is None:
                    area = float((prev_mask >= 0.5).sum())
                    if area > 0.0:
                        anchor_area = area
                current_prompt = self.prompt_adapter.adapt(prev_mask, frame_shape)

                if len(logits) >= total:
                    break
        finally:
            if on_predict_start_callbacks is not None and predictor_init_callback is not None:
                try:
                    on_predict_start_callbacks.remove(predictor_init_callback)
                except ValueError:
                    pass

        if len(logits) != total:
            raise RuntimeError(
                f"Video predictor returned {len(logits)} frames; expected {total}."
            )
        return logits

    def segment_chunk(
        self,
        frames: list[np.ndarray],
        prompt: SegmentPrompt,
        anchor_frame_index: int = 0,
    ) -> list[np.ndarray]:
        if not frames:
            return []
        model = self._load_model()
        self._configure_runtime()

        logits: list[np.ndarray] = []
        current_prompt = prompt
        prev_mask: np.ndarray | None = None
        anchor_area: float | None = None
        if reference_anchor_area is not None and float(reference_anchor_area) > 0.0:
            anchor_area = float(reference_anchor_area)
        elif prompt.mask is not None:
            anchor_prompt_mask = _resize_mask(prompt.mask, frames[0].shape[:2], threshold=self.mask_threshold)
            anchor_area = float((anchor_prompt_mask >= 0.5).sum())
            if anchor_area <= 0.0:
                anchor_area = None

        for frame in frames:
            if prev_mask is not None:
                current_prompt = self.prompt_adapter.adapt(prev_mask, frame.shape[:2])
            frame_bgr = _to_bgr_u8(frame)
            frame_logit = self._infer_single(model, frame_bgr, current_prompt)
            if prev_mask is not None and (self.temporal_component_filter or self.strict_background_suppression):
                prob = _as_probabilities(frame_logit)
                if self.strict_background_suppression:
                    prob = _apply_strict_background_suppression(
                        prob,
                        prev_mask,
                        threshold=self.mask_threshold,
                        bbox_expand_ratio=self.strict_bbox_expand_ratio,
                        min_bbox_expand_px=self.strict_min_bbox_expand_px,
                        overlap_dilate_ratio=self.strict_overlap_dilate_ratio,
                        min_overlap_dilate_px=self.strict_min_overlap_dilate_px,
                    )
                    if self.strict_temporal_guard:
                        prob = _apply_temporal_area_guard(
                            prob,
                            prev_mask,
                            threshold=self.mask_threshold,
                            max_area_ratio=self.strict_max_area_ratio,
                            min_iou=self.strict_min_iou,
                        )
                        prob = _apply_anchor_reference_area_guard(
                            prob,
                            prev_mask,
                            reference_area=anchor_area,
                            threshold=self.mask_threshold,
                            max_area_ratio=self.strict_anchor_max_area_ratio,
                        )
                else:
                    prob = _filter_probability_by_prev_component(
                        prob,
                        prev_mask,
                        threshold=self.mask_threshold,
                    )
                frame_logit = probability_to_logits(prob)
            logits.append(frame_logit.astype(np.float32))
            prev_mask = _to_mask(frame_logit, threshold=self.mask_threshold)
            if anchor_area is None:
                area = float((prev_mask >= 0.5).sum())
                if area > 0.0:
                    anchor_area = area
            current_prompt = self.prompt_adapter.adapt(prev_mask, frame.shape[:2])
        return logits


@dataclass(slots=True)
class ChunkedSegmenter(Segmenter):
    """Chunked sequence segmenter with overlap blending and drift re-anchoring."""

    backend: SegmentBackend
    processing_long_side: int = 960
    mask_threshold: float = 0.5
    drift_iou_threshold: float = 0.70
    drift_area_threshold: float = 0.40
    max_reanchors_per_chunk: int = 2
    prompt_adapter: PromptAdapter = field(default_factory=MaskPromptAdapter)

    def segment_sequence(
        self,
        source: FrameSourceLike,
        prompt: SegmentPrompt,
        anchor_frame: int = 0,
        chunk_size: int = 100,
        chunk_overlap: int = 5,
    ) -> SegmentResult:
        num_frames = int(len(source))
        if num_frames <= 0:
            return SegmentResult(masks=[], logits=[], anchored_frames=[])
        if anchor_frame != 0:
            raise NotImplementedError("v2 scaffold currently supports anchor_frame=0 only.")

        chunk_size = max(int(chunk_size), 1)
        chunk_overlap = int(np.clip(chunk_overlap, 0, chunk_size - 1))
        step = max(1, chunk_size - chunk_overlap)

        src_shape = tuple(int(v) for v in source.resolution)
        logits_by_frame: list[np.ndarray | None] = [None] * num_frames
        anchored_frames: list[int] = [0]
        anchor_reference_area_proc: float | None = None

        maybe_video_backend = self.backend if hasattr(self.backend, "segment_video_sequence") else None
        source_is_video = bool(getattr(source, "is_video", False))
        source_video_path = getattr(source, "video_path", None)
        source_video_start = int(getattr(source, "video_frame_start", 0))
        if (
            maybe_video_backend is not None
            and source_is_video
            and source_video_path is not None
        ):
            try:
                logits = maybe_video_backend.segment_video_sequence(
                    source_video_path,
                    prompt=prompt,
                    start_frame=source_video_start,
                    num_frames=num_frames,
                    frame_shape=src_shape,
                )
                if len(logits) == num_frames:
                    masks = [_to_mask(l, threshold=self.mask_threshold) for l in logits]
                    return SegmentResult(masks=masks, logits=[np.asarray(l, dtype=np.float32) for l in logits], anchored_frames=[0])
            except Exception as exc:
                logger.warning("Video fast path failed; falling back to chunked per-frame path: %s", exc)

        prev_tail_mask_proc: np.ndarray | None = None

        for chunk_start in range(0, num_frames, step):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            frames_proc = [
                _resize_long_side(np.asarray(source[i]), self.processing_long_side)
                for i in range(chunk_start, chunk_end)
            ]
            if not frames_proc:
                continue
            proc_shape = frames_proc[0].shape[:2]

            if chunk_start == 0:
                chunk_prompt = _scale_prompt(prompt, src_shape, proc_shape)
                if anchor_reference_area_proc is None and chunk_prompt.mask is not None:
                    anchor_mask_proc = _resize_mask(chunk_prompt.mask, proc_shape, threshold=self.mask_threshold)
                    area = float((anchor_mask_proc >= 0.5).sum())
                    if area > 0.0:
                        anchor_reference_area_proc = area
            else:
                if prev_tail_mask_proc is not None:
                    chunk_prompt = self.prompt_adapter.adapt(prev_tail_mask_proc, proc_shape)
                else:
                    chunk_prompt = _scale_prompt(prompt, src_shape, proc_shape)
                anchored_frames.append(chunk_start)

            if isinstance(self.backend, UltralyticsSAM3SegmentBackend):
                chunk_logits = self.backend.segment_chunk(
                    frames_proc,
                    chunk_prompt,
                    anchor_frame_index=0,
                    reference_anchor_area=anchor_reference_area_proc,
                )
            else:
                chunk_logits = self.backend.segment_chunk(frames_proc, chunk_prompt, anchor_frame_index=0)
            if len(chunk_logits) != len(frames_proc):
                raise RuntimeError(
                    f"Segment backend returned {len(chunk_logits)} frames for chunk size {len(frames_proc)}."
                )
            chunk_masks = [_to_mask(l, threshold=self.mask_threshold) for l in chunk_logits]

            # Intra-chunk drift detection + re-anchor.
            reanchors = 0
            local_idx = 1
            while local_idx < len(chunk_masks):
                drift = check_drift(
                    current_mask=chunk_masks[local_idx],
                    previous_mask=chunk_masks[local_idx - 1],
                    iou_threshold=self.drift_iou_threshold,
                    area_change_threshold=self.drift_area_threshold,
                )
                if drift.drift and reanchors < self.max_reanchors_per_chunk:
                    reprompt = self.prompt_adapter.adapt(chunk_masks[local_idx - 1], proc_shape)
                    if isinstance(self.backend, UltralyticsSAM3SegmentBackend):
                        replacement_logits = self.backend.segment_chunk(
                            frames_proc[local_idx:],
                            reprompt,
                            anchor_frame_index=0,
                            reference_anchor_area=anchor_reference_area_proc,
                        )
                    else:
                        replacement_logits = self.backend.segment_chunk(
                            frames_proc[local_idx:],
                            reprompt,
                            anchor_frame_index=0,
                        )
                    if len(replacement_logits) == (len(frames_proc) - local_idx):
                        chunk_logits[local_idx:] = replacement_logits
                        chunk_masks[local_idx:] = [
                            _to_mask(l, threshold=self.mask_threshold)
                            for l in replacement_logits
                        ]
                        anchored_frames.append(chunk_start + local_idx)
                        reanchors += 1
                        continue
                local_idx += 1

            # Blend overlap against already-filled frames.
            overlap_len = min(chunk_overlap, len(chunk_logits))
            for local_idx, logit_proc in enumerate(chunk_logits):
                global_idx = chunk_start + local_idx
                if global_idx >= num_frames:
                    break
                logit_full = resize_logits(logit_proc, src_shape).astype(np.float32)
                existing = logits_by_frame[global_idx]
                if existing is None:
                    logits_by_frame[global_idx] = logit_full
                    continue

                if overlap_len <= 0 or local_idx >= overlap_len:
                    alpha = 1.0
                else:
                    # Use an open-interval crossfade so neither chunk endpoint is fully discarded.
                    alpha = float((local_idx + 1) / float(overlap_len + 1))
                logits_by_frame[global_idx] = _blend_overlap(existing, logit_full, alpha)

            prev_tail_mask_proc = chunk_masks[-1]
            logger.info(
                "Segment chunk %d..%d processed (reanchors=%d).",
                chunk_start,
                chunk_end - 1,
                reanchors,
            )

        logits: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for idx, maybe_logit in enumerate(logits_by_frame):
            if maybe_logit is None:
                maybe_logit = np.full(src_shape, -6.0, dtype=np.float32)
                logger.warning("Missing segmentation output for frame %d. Filled with background.", idx)
            logit = np.asarray(maybe_logit, dtype=np.float32)
            logits.append(logit)
            masks.append(_to_mask(logit, threshold=self.mask_threshold))

        anchored_sorted = sorted(set(int(i) for i in anchored_frames if 0 <= i < num_frames))
        return SegmentResult(masks=masks, logits=logits, anchored_frames=anchored_sorted)


def build_segmenter(cfg: SegmentStageConfig, *, prompt_adapter: PromptAdapter | None = None) -> ChunkedSegmenter:
    if prompt_adapter is None:
        active_prompt_adapter: PromptAdapter = MaskPromptAdapter(
            bbox_expand_ratio=cfg.bbox_expand_ratio,
            min_bbox_expand_px=cfg.min_bbox_expand_px,
        )
    elif isinstance(prompt_adapter, MaskPromptAdapter):
        prompt_adapter.bbox_expand_ratio = float(cfg.bbox_expand_ratio)
        prompt_adapter.min_bbox_expand_px = int(cfg.min_bbox_expand_px)
        active_prompt_adapter = prompt_adapter
    else:
        active_prompt_adapter = prompt_adapter

    backend_name = str(cfg.backend).strip().lower()
    if backend_name in {"static", "dummy", "mask_static"}:
        backend: SegmentBackend = StaticMaskSegmentBackend(mask_threshold=cfg.mask_threshold)
    elif backend_name in {"ultralytics", "ultralytics_sam3", "sam3"}:
        backend = UltralyticsSAM3SegmentBackend(
            model_name=cfg.sam3_model,
            device=cfg.device,
            precision=cfg.precision,
            processing_long_side=cfg.processing_long_side,
            mask_threshold=cfg.mask_threshold,
            temporal_component_filter=cfg.temporal_component_filter,
            strict_background_suppression=cfg.strict_background_suppression,
            strict_bbox_expand_ratio=cfg.strict_bbox_expand_ratio,
            strict_min_bbox_expand_px=cfg.strict_min_bbox_expand_px,
            strict_overlap_dilate_ratio=cfg.strict_overlap_dilate_ratio,
            strict_min_overlap_dilate_px=cfg.strict_min_overlap_dilate_px,
            strict_temporal_guard=cfg.strict_temporal_guard,
            strict_max_area_ratio=cfg.strict_max_area_ratio,
            strict_min_iou=cfg.strict_min_iou,
            strict_anchor_max_area_ratio=cfg.strict_anchor_max_area_ratio,
            prompt_adapter=active_prompt_adapter,
        )
    else:
        raise ValueError(f"Unsupported segment backend: {cfg.backend}")

    return ChunkedSegmenter(
        backend=backend,
        processing_long_side=cfg.processing_long_side,
        mask_threshold=cfg.mask_threshold,
        drift_iou_threshold=cfg.drift_iou_threshold,
        drift_area_threshold=cfg.drift_area_threshold,
        max_reanchors_per_chunk=cfg.max_reanchors_per_chunk,
        prompt_adapter=active_prompt_adapter,
    )
