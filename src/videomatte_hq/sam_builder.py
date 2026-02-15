"""Optional SAM-based prompt mask builder."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


DEFAULT_SAM_MODEL_ID = "facebook/sam-vit-base"


def _resolve_device(device_hint: str | None) -> torch.device:
    hint = str(device_hint or "").lower().strip()
    if hint.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    xi = max(0, min(width - 1, xi))
    yi = max(0, min(height - 1, yi))
    return xi, yi


def _normalize_box(
    box_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_xyxy
    xa = max(0, min(width - 1, int(round(min(x0, x1)))))
    xb = max(0, min(width - 1, int(round(max(x0, x1)))))
    ya = max(0, min(height - 1, int(round(min(y0, y1)))))
    yb = max(0, min(height - 1, int(round(max(y0, y1)))))
    if xb - xa < 2 or yb - ya < 2:
        raise ValueError("Selection box is too small. Draw a larger box around the subject.")
    return xa, ya, xb, yb


@lru_cache(maxsize=2)
def _load_sam_components(
    model_id: str,
    device_str: str,
    local_files_only: bool,
) -> tuple[SamProcessor, SamModel]:
    processor = SamProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = SamModel.from_pretrained(model_id, local_files_only=local_files_only)
    model.to(device_str)
    model.eval()
    return processor, model


def build_prompt_mask_sam(
    frame_rgb_u8: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    fg_points: Sequence[tuple[float, float]] = (),
    bg_points: Sequence[tuple[float, float]] = (),
    model_id: str = DEFAULT_SAM_MODEL_ID,
    local_files_only: bool = True,
    device_hint: str = "cuda",
    point_radius: int = 8,
) -> np.ndarray:
    """Build an initial binary alpha mask from prompts using SAM."""
    if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
        raise ValueError("Expected RGB frame with shape (H, W, 3).")
    if frame_rgb_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB frame for SAM mask building.")

    height, width = frame_rgb_u8.shape[:2]
    x0, y0, x1, y1 = _normalize_box(box_xyxy, width=width, height=height)
    device = _resolve_device(device_hint)
    model_name = str(model_id or DEFAULT_SAM_MODEL_ID).strip()
    if not model_name:
        model_name = DEFAULT_SAM_MODEL_ID

    processor, model = _load_sam_components(
        model_id=model_name,
        device_str=str(device),
        local_files_only=bool(local_files_only),
    )

    points: list[list[int]] = []
    labels: list[int] = []
    for px, py in fg_points:
        cx, cy = _clamp_point(px, py, width=width, height=height)
        points.append([cx, cy])
        labels.append(1)
    for px, py in bg_points:
        cx, cy = _clamp_point(px, py, width=width, height=height)
        points.append([cx, cy])
        labels.append(0)
    if not points:
        # Ensure at least one positive prompt in box center.
        points.append([int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5))])
        labels.append(1)

    image = Image.fromarray(frame_rgb_u8, mode="RGB")
    inputs = processor(
        image,
        input_boxes=[[[x0, y0, x1, y1]]],
        input_points=[[[points_i for points_i in points]]],
        input_labels=[[[int(label) for label in labels]]],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    mask_np = np.asarray(masks[0])
    while mask_np.ndim > 3:
        mask_np = mask_np[0]
    if mask_np.ndim == 2:
        mask_stack = mask_np[None, ...]
    elif mask_np.ndim == 3:
        mask_stack = mask_np
    else:
        raise ValueError("Unexpected SAM mask tensor shape.")

    iou_scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
    best_idx = int(np.argmax(iou_scores)) if iou_scores.size else 0
    best_idx = max(0, min(best_idx, mask_stack.shape[0] - 1))
    alpha = mask_stack[best_idx].astype(np.float32)
    alpha = np.where(alpha > 0.0, 1.0, 0.0).astype(np.float32)

    rad = max(1, int(point_radius))
    for px, py in fg_points:
        cx, cy = _clamp_point(px, py, width=width, height=height)
        cv2.circle(alpha, (cx, cy), rad, 1.0, thickness=-1)
    for px, py in bg_points:
        cx, cy = _clamp_point(px, py, width=width, height=height)
        cv2.circle(alpha, (cx, cy), rad, 0.0, thickness=-1)

    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_u8 = cv2.medianBlur(alpha_u8, 3)
    return (alpha_u8.astype(np.float32) / 255.0).astype(np.float32)

