"""Optional matte shape tuning (shrink/grow, feather, offset)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MatteTuningConfig:
    enabled: bool = True
    shrink_grow_px: int = 0
    feather_px: int = 0
    offset_x_px: int = 0
    offset_y_px: int = 0


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


def apply_matte_tuning(alphas: list[np.ndarray], cfg: MatteTuningConfig) -> list[np.ndarray]:
    """Apply artist-facing matte shape controls on a sequence of alpha maps."""
    if not alphas:
        return []
    if not bool(cfg.enabled):
        return [np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) for a in alphas]

    shrink_grow_px = int(cfg.shrink_grow_px)
    feather_px = int(cfg.feather_px)
    offset_x_px = int(cfg.offset_x_px)
    offset_y_px = int(cfg.offset_y_px)
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
            logger.info("Matte tuning frame %d/%d complete.", t + 1, len(alphas))
    return outputs


def run_pass_matte_tuning(alphas: list[np.ndarray], cfg: Any) -> list[np.ndarray]:
    """Backward-compatible wrapper for legacy call sites."""
    tuning = getattr(cfg, "matte_tuning", cfg)
    if isinstance(tuning, MatteTuningConfig):
        return apply_matte_tuning(alphas, tuning)
    return apply_matte_tuning(
        alphas,
        MatteTuningConfig(
            enabled=bool(getattr(tuning, "enabled", True)),
            shrink_grow_px=int(getattr(tuning, "shrink_grow_px", 0)),
            feather_px=int(getattr(tuning, "feather_px", 0)),
            offset_x_px=int(getattr(tuning, "offset_x_px", 0)),
            offset_y_px=int(getattr(tuning, "offset_y_px", 0)),
        ),
    )
