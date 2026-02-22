"""Run a short V2 pipeline smoke test on a video clip.

Creates incrementing output folders:
  output_tests/run_0001
  output_tests/run_0002
  ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.orchestrator import run_pipeline


def _next_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    next_idx = 1
    if existing:
        nums = []
        for p in existing:
            try:
                nums.append(int(p.name.split("_", 1)[1]))
            except Exception:
                continue
        if nums:
            next_idx = max(nums) + 1
    run_dir = root / f"run_{next_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _largest_component(mask: np.ndarray) -> np.ndarray:
    fg = (mask > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(1 + np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def _extract_person_mask_from_result(result: Any, frame_shape: tuple[int, int]) -> np.ndarray | None:
    h, w = frame_shape

    # Prefer segmentation masks when available.
    masks_obj = getattr(result, "masks", None)
    mask_data = getattr(masks_obj, "data", None) if masks_obj is not None else None
    if mask_data is not None:
        arr = mask_data.detach().cpu().numpy() if hasattr(mask_data, "detach") else np.asarray(mask_data)
        if arr.ndim == 3 and arr.shape[0] > 0:
            # choose largest predicted mask
            areas = arr.reshape(arr.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))
            m = arr[idx].astype(np.float32)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            m = (m >= 0.5).astype(np.uint8)
            if int(m.sum()) > 0:
                return (m * 255).astype(np.uint8)

    # Fallback to person bbox from detector output.
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

    mask = np.zeros((h, w), dtype=np.uint8)
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
    mask[iy0:iy1, ix0:ix1] = 255
    return mask


def _detect_person_anchor_mask(frame_bgr: np.ndarray, device: str = "cpu") -> np.ndarray | None:
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    # Highest-quality to lowest-quality fallback order.
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
    dilate_k = int(round(ref * 0.02))
    dilate_k = int(np.clip(dilate_k, 3, 61))
    if dilate_k % 2 == 0:
        dilate_k += 1

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    mask = _largest_component(mask)
    return (mask * 255).astype(np.uint8)


def _build_auto_anchor_mask(
    video_path: Path, out_path: Path, device: str = "cpu", frame_start: int = 0
) -> tuple[Path, str, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for anchor mask: {video_path}")

    # Scan up to 10 frames starting at frame_start to find a non-black frame.
    frame: np.ndarray | None = None
    selected_frame_index = int(frame_start)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_limit = min(frame_start + 10, max(total_frames, frame_start + 1))
    for probe_idx in range(frame_start, scan_limit):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(probe_idx))
        ok, candidate = cap.read()
        if not ok or candidate is None:
            continue
        if float(np.mean(candidate)) < 8.0:
            # Frame is too dark/black; try the next one.
            continue
        frame = candidate
        selected_frame_index = int(probe_idx)
        break
    cap.release()

    if frame is None:
        raise RuntimeError(f"Failed to read a usable (non-black) frame for anchor mask: {video_path}")

    h, w = frame.shape[:2]

    # Prefer explicit person prompt detection.
    mask = _detect_person_anchor_mask(frame_bgr=frame, device=device)
    method = "ultralytics_person_detect"

    # Last-resort fallback for environments where detector fails.
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
        raise RuntimeError(f"Failed to write anchor mask: {out_path}")
    return out_path, method, int(selected_frame_index)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a short V2 clip smoke test")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output-root", default="output_tests", help="Root folder for run_XXXX outputs")
    p.add_argument("--frame-start", type=int, default=0, help="Start frame index")
    p.add_argument("--num-frames", type=int, default=30, help="Number of frames to process")
    p.add_argument("--segment-backend", default="ultralytics_sam3", help="ultralytics_sam3 or static")
    p.add_argument("--sam3-model", default="sam2_l.pt", help="Ultralytics SAM model/checkpoint")
    p.add_argument("--chunk-size", type=int, default=30, help="Segmentation chunk size")
    p.add_argument("--chunk-overlap", type=int, default=5, help="Segmentation chunk overlap")
    p.add_argument("--bbox-expand-ratio", type=float, default=0.08, help="Prompt bbox expansion ratio")
    p.add_argument("--min-bbox-expand-px", type=int, default=12, help="Minimum bbox expansion in pixels")
    p.add_argument(
        "--enable-temporal-component-filter",
        action="store_true",
        help="Enable temporal connected-component filtering in Stage 1",
    )
    p.add_argument(
        "--strict-background-suppression",
        action="store_true",
        help="Enable strict background suppression (more aggressive, may clip fast motion)",
    )
    p.add_argument(
        "--strict-bbox-expand-ratio",
        type=float,
        default=0.10,
        help="Strict mode: bbox expansion ratio around previous mask",
    )
    p.add_argument(
        "--strict-min-bbox-expand-px",
        type=int,
        default=24,
        help="Strict mode: minimum bbox expansion in pixels",
    )
    p.add_argument(
        "--strict-overlap-dilate-ratio",
        type=float,
        default=0.03,
        help="Strict mode: dilation ratio for overlap-connected components",
    )
    p.add_argument(
        "--strict-min-overlap-dilate-px",
        type=int,
        default=12,
        help="Strict mode: minimum dilation pixels for overlap-connected components",
    )
    p.add_argument(
        "--strict-temporal-guard",
        dest="strict_temporal_guard",
        action="store_true",
        help="Strict mode: reject low-IoU area jumps by holding previous mask",
    )
    p.add_argument(
        "--no-strict-temporal-guard",
        dest="strict_temporal_guard",
        action="store_false",
        help="Disable strict mode temporal area/IoU guard",
    )
    p.set_defaults(strict_temporal_guard=True)
    p.add_argument(
        "--strict-max-area-ratio",
        type=float,
        default=3.0,
        help="Strict mode: max per-frame area ratio change before rejection",
    )
    p.add_argument(
        "--strict-min-iou",
        type=float,
        default=0.20,
        help="Strict mode: minimum IoU required to allow large area jumps",
    )
    p.add_argument(
        "--strict-anchor-max-area-ratio",
        type=float,
        default=4.0,
        help="Strict mode: max area ratio vs initial anchor before fallback to previous mask",
    )
    p.add_argument("--anchor-mask", default="", help="Optional pre-drawn anchor mask")
    p.add_argument("--refine", action="store_true", help="Enable MEMatte refinement (off by default)")
    p.add_argument("--device", default="cuda", help="cuda/cpu")
    p.add_argument("--precision", default="fp16", help="fp16/fp32 precision hint for runtime")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    requested_frame_start = int(args.frame_start)
    effective_frame_start = int(requested_frame_start)
    run_dir = _next_run_dir(Path(args.output_root))
    anchor_probe_frame = int(requested_frame_start)

    anchor_method = "provided_mask"
    if str(args.anchor_mask).strip():
        anchor_mask = Path(args.anchor_mask).expanduser().resolve()
    else:
        anchor_mask, anchor_method, anchor_probe_frame = _build_auto_anchor_mask(
            input_path,
            run_dir / "anchor_mask.png",
            device=str(args.device),
            frame_start=int(requested_frame_start),
        )
        effective_frame_start = max(int(requested_frame_start), int(anchor_probe_frame))

    frame_end = int(effective_frame_start) + max(1, int(args.num_frames)) - 1

    cfg = VideoMatteConfig(
        input=str(input_path),
        output_dir=str(run_dir),
        output_alpha="alpha/%06d.png",
        frame_start=int(effective_frame_start),
        frame_end=int(frame_end),
        anchor_mask=str(anchor_mask),
        segment_backend=str(args.segment_backend),
        sam3_model=str(args.sam3_model),
        chunk_size=max(1, int(args.chunk_size)),
        chunk_overlap=max(0, int(args.chunk_overlap)),
        bbox_expand_ratio=float(args.bbox_expand_ratio),
        min_bbox_expand_px=max(0, int(args.min_bbox_expand_px)),
        temporal_component_filter=bool(args.enable_temporal_component_filter),
        strict_background_suppression=bool(args.strict_background_suppression),
        strict_bbox_expand_ratio=float(args.strict_bbox_expand_ratio),
        strict_min_bbox_expand_px=max(0, int(args.strict_min_bbox_expand_px)),
        strict_overlap_dilate_ratio=float(args.strict_overlap_dilate_ratio),
        strict_min_overlap_dilate_px=max(0, int(args.strict_min_overlap_dilate_px)),
        strict_temporal_guard=bool(args.strict_temporal_guard),
        strict_max_area_ratio=max(1.01, float(args.strict_max_area_ratio)),
        strict_min_iou=float(args.strict_min_iou),
        strict_anchor_max_area_ratio=max(1.01, float(args.strict_anchor_max_area_ratio)),
        refine_enabled=bool(args.refine),
        device=str(args.device),
        precision=str(args.precision),
        workers_io=2,
    )

    config_path = run_dir / "config_used.json"
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    run_pipeline(cfg)

    summary = {
        "run_dir": str(run_dir.resolve()),
        "input": str(input_path),
        "requested_frame_start": int(requested_frame_start),
        "frame_start": int(effective_frame_start),
        "frame_end": int(frame_end),
        "anchor_probe_frame": int(anchor_probe_frame),
        "segment_backend": str(args.segment_backend),
        "strict_background_suppression": bool(args.strict_background_suppression),
        "strict_temporal_guard": bool(args.strict_temporal_guard),
        "refine_enabled": bool(args.refine),
        "precision": str(args.precision),
        "anchor_mask": str(anchor_mask),
        "anchor_method": anchor_method,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
