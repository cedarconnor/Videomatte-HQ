"""Option B final matte tuning pass (shrink/grow, feather, and offset)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _morph(alpha: np.ndarray, amount_px: int) -> np.ndarray:
    radius = abs(int(amount_px))
    if radius <= 0:
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    if amount_px > 0:
        return cv2.dilate(alpha, kernel)
    return cv2.erode(alpha, kernel)


def _offset(alpha: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    if offset_x == 0 and offset_y == 0:
        return alpha
    h, w = alpha.shape
    transform = np.array([[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)]], dtype=np.float32)
    return cv2.warpAffine(
        alpha,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _feather(alpha: np.ndarray, feather_px: int) -> np.ndarray:
    radius = max(int(feather_px), 0)
    if radius <= 0:
        return alpha
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(alpha, (ksize, ksize), sigmaX=0.0, sigmaY=0.0)


def run_pass_matte_tuning(
    alphas: list[np.ndarray],
    cfg: Any,
) -> list[np.ndarray]:
    """Run final matte tuning controls on per-frame alpha."""

    if not alphas:
        return []

    tuning = cfg.matte_tuning
    if not tuning.enabled:
        return [np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) for a in alphas]

    shrink_grow_px = int(tuning.shrink_grow_px)
    feather_px = int(tuning.feather_px)
    offset_x_px = int(tuning.offset_x_px)
    offset_y_px = int(tuning.offset_y_px)

    if shrink_grow_px == 0 and feather_px == 0 and offset_x_px == 0 and offset_y_px == 0:
        return [np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) for a in alphas]

    outputs: list[np.ndarray] = []
    for t, alpha_in in enumerate(alphas):
        alpha = np.clip(np.asarray(alpha_in, dtype=np.float32), 0.0, 1.0)
        alpha = _morph(alpha, shrink_grow_px)
        alpha = _offset(alpha, offset_x_px, offset_y_px)
        alpha = _feather(alpha, feather_px)
        outputs.append(np.clip(alpha, 0.0, 1.0).astype(np.float32))

        if t == 0 or (t + 1) % 100 == 0:
            logger.info("Matte tuning: frame %d/%d", t + 1, len(alphas))

    return outputs
