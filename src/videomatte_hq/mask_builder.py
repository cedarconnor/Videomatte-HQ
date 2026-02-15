"""Prompt-based initial mask builder utilities."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


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


def build_prompt_mask_grabcut(
    frame_rgb_u8: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    fg_points: Sequence[tuple[float, float]] = (),
    bg_points: Sequence[tuple[float, float]] = (),
    point_radius: int = 8,
    iter_count: int = 5,
) -> np.ndarray:
    """Build an initial binary alpha from box + FG/BG point prompts.

    Returns:
        float32 alpha in [0, 1], shape (H, W)
    """
    if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
        raise ValueError("Expected RGB frame with shape (H, W, 3).")
    if frame_rgb_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB frame for mask building.")

    height, width = frame_rgb_u8.shape[:2]
    x0, y0, x1, y1 = _normalize_box(box_xyxy, width=width, height=height)
    rect = (x0, y0, x1 - x0, y1 - y0)

    img_bgr = cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2BGR)
    gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            img_bgr,
            gc_mask,
            rect,
            bg_model,
            fg_model,
            iterCount=max(1, int(iter_count)),
            mode=cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        raise ValueError(f"GrabCut failed to initialize with rectangle: {exc}") from exc

    rad = max(1, int(point_radius))
    for px, py in fg_points:
        cx, cy = _clamp_point(px, py, width=width, height=height)
        cv2.circle(gc_mask, (cx, cy), rad, cv2.GC_FGD, thickness=-1)
    for px, py in bg_points:
        cx, cy = _clamp_point(px, py, width=width, height=height)
        cv2.circle(gc_mask, (cx, cy), rad, cv2.GC_BGD, thickness=-1)

    if fg_points or bg_points:
        try:
            cv2.grabCut(
                img_bgr,
                gc_mask,
                None,
                bg_model,
                fg_model,
                iterCount=max(1, int(iter_count // 2) or 1),
                mode=cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as exc:
            raise ValueError(f"GrabCut failed to refine with point prompts: {exc}") from exc

    alpha = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        1.0,
        0.0,
    ).astype(np.float32)

    if float(alpha.max()) < 0.5:
        if fg_points:
            for px, py in fg_points:
                cx, cy = _clamp_point(px, py, width=width, height=height)
                cv2.circle(alpha, (cx, cy), rad, 1.0, thickness=-1)
        else:
            alpha[y0:y1, x0:x1] = 1.0

    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_u8 = cv2.medianBlur(alpha_u8, 3)
    return (alpha_u8.astype(np.float32) / 255.0).astype(np.float32)

