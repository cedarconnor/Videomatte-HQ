"""Automatic anchor mask generation helpers for video inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class AutoAnchorResult:
    mask_path: Path
    method: str
    probe_frame: int


def _largest_component(mask: np.ndarray) -> np.ndarray:
    fg = (np.asarray(mask) > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(1 + np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def _extract_person_mask_from_result(result: Any, frame_shape: tuple[int, int]) -> np.ndarray | None:
    h, w = (int(frame_shape[0]), int(frame_shape[1]))

    masks_obj = getattr(result, "masks", None)
    mask_data = getattr(masks_obj, "data", None) if masks_obj is not None else None
    if mask_data is not None:
        arr = mask_data.detach().cpu().numpy() if hasattr(mask_data, "detach") else np.asarray(mask_data)
        if arr.ndim == 3 and arr.shape[0] > 0:
            areas = arr.reshape(arr.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))
            m = arr[idx].astype(np.float32)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            m = (m >= 0.5).astype(np.uint8)
            if int(m.sum()) > 0:
                return (m * 255).astype(np.uint8)

    boxes_obj = getattr(result, "boxes", None)
    xyxy = getattr(boxes_obj, "xyxy", None) if boxes_obj is not None else None
    cls = getattr(boxes_obj, "cls", None) if boxes_obj is not None else None
    conf = getattr(boxes_obj, "conf", None) if boxes_obj is not None else None
    if xyxy is None:
        return None

    boxes = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    classes = cls.detach().cpu().numpy() if hasattr(cls, "detach") else (np.asarray(cls) if cls is not None else None)
    scores = conf.detach().cpu().numpy() if hasattr(conf, "detach") else (np.asarray(conf) if conf is not None else None)
    if boxes.ndim != 2 or boxes.shape[1] < 4 or boxes.shape[0] == 0:
        return None

    best_idx = -1
    best_area = -1.0
    for i in range(boxes.shape[0]):
        if classes is not None and int(classes[i]) != 0:
            continue
        if scores is not None and float(scores[i]) < 0.2:
            continue
        x0, y0, x1, y1 = boxes[i, :4].tolist()
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area > best_area:
            best_area = area
            best_idx = i
    if best_idx < 0:
        return None

    x0, y0, x1, y1 = boxes[best_idx, :4].tolist()
    ix0 = int(np.clip(np.floor(x0), 0, w))
    iy0 = int(np.clip(np.floor(y0), 0, h))
    ix1 = int(np.clip(np.ceil(x1), 0, w))
    iy1 = int(np.clip(np.ceil(y1), 0, h))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[iy0:iy1, ix0:ix1] = 255
    return mask


def _detect_person_anchor_mask(frame_bgr: np.ndarray, device: str = "cpu") -> np.ndarray | None:
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    candidates = (
        "yolo11x-seg.pt",
        "yolo11l-seg.pt",
        "yolo11m-seg.pt",
        "yolo11s-seg.pt",
        "yolo11n-seg.pt",
        "yolov8x-seg.pt",
        "yolov8l-seg.pt",
        "yolov8m-seg.pt",
        "yolov8s-seg.pt",
        "yolov8n-seg.pt",
        "yolo11x.pt",
        "yolo11l.pt",
        "yolo11m.pt",
        "yolo11s.pt",
        "yolo11n.pt",
        "yolov8x.pt",
        "yolov8l.pt",
        "yolov8m.pt",
        "yolov8s.pt",
        "yolov8n.pt",
    )
    for model_name in candidates:
        try:
            model = YOLO(model_name)
            results = model.predict(
                source=frame_bgr,
                device=device,
                classes=[0],
                conf=0.2,
                verbose=False,
            )
            if not results:
                continue
            mask = _extract_person_mask_from_result(results[0], frame_bgr.shape[:2])
            if mask is not None and int(mask.sum()) > 0:
                return mask
        except Exception:
            continue
    return None


def _postprocess_anchor_mask(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape[:2]
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    mask = _largest_component(mask)

    ys, xs = np.where(mask > 0)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    ref = max(bw, bh)

    close_k = int(round(ref * 0.015))
    close_k = int(np.clip(close_k, 3, 41))
    if close_k % 2 == 0:
        close_k += 1
    dilate_k = int(round(ref * 0.05))
    dilate_k = int(np.clip(dilate_k, 5, 81))
    if dilate_k % 2 == 0:
        dilate_k += 1

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # YOLO seg masks clip at their detection bbox, producing hard horizontal
    # cuts at the top/bottom.  Add vertical padding to recover head/feet that
    # the detection box missed.  The padding fills a rectangle at the top and
    # bottom using the mask's horizontal extent so the silhouette stays plausible.
    ys2, xs2 = np.where(mask > 0)
    if ys2.size > 0:
        y_min, y_max = int(ys2.min()), int(ys2.max())
        x_min, x_max = int(xs2.min()), int(xs2.max())
        v_pad = max(int(round(bh * 0.12)), 30)
        pad_top = max(0, y_min - v_pad)
        pad_bot = min(h, y_max + 1 + v_pad)
        # Narrow the horizontal extent slightly so the padding tapers
        x_inset = max(0, int(round((x_max - x_min) * 0.10)))
        mask[pad_top:y_min, x_min + x_inset : x_max + 1 - x_inset] = 1
        mask[y_max + 1 : pad_bot, x_min + x_inset : x_max + 1 - x_inset] = 1

    mask = _largest_component(mask)
    return (mask * 255).astype(np.uint8)


def build_auto_anchor_mask_for_video(
    video_path: str | Path,
    out_path: str | Path,
    *,
    device: str = "cpu",
    frame_start: int = 0,
    max_probe_frames: int = 10,
    black_frame_mean_threshold: float = 8.0,
) -> AutoAnchorResult:
    """Generate an initial anchor mask for a video by probing early frames and detecting a person."""
    video_path = Path(video_path)
    out_path = Path(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for auto-anchor mask: {video_path}")

    frame: np.ndarray | None = None
    selected_frame_index = int(frame_start)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_limit = min(int(frame_start) + max(1, int(max_probe_frames)), max(total_frames, int(frame_start) + 1))
    for probe_idx in range(int(frame_start), scan_limit):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(probe_idx))
        ok, candidate = cap.read()
        if not ok or candidate is None:
            continue
        if float(np.mean(candidate)) < float(black_frame_mean_threshold):
            continue
        frame = candidate
        selected_frame_index = int(probe_idx)
        break
    cap.release()

    if frame is None:
        raise RuntimeError(f"Failed to read a usable (non-black) frame for auto-anchor: {video_path}")

    h, w = frame.shape[:2]
    mask = _detect_person_anchor_mask(frame_bgr=frame, device=str(device))
    method = "ultralytics_person_detect"
    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
        bw = int(w * 0.30)
        bh = int(h * 0.70)
        x0 = (w - bw) // 2
        y0 = int(h * 0.15)
        x1 = x0 + bw
        y1 = min(h, y0 + bh)
        mask[y0:y1, x0:x1] = 255
        method = "fallback_center_box"
    else:
        mask = _postprocess_anchor_mask(mask)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), mask):
        raise RuntimeError(f"Failed to write auto-anchor mask: {out_path}")
    return AutoAnchorResult(mask_path=out_path, method=method, probe_frame=int(selected_frame_index))


__all__ = ["AutoAnchorResult", "build_auto_anchor_mask_for_video"]
