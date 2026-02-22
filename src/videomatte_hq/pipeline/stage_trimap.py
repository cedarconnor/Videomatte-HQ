"""Stage-2 trimap generation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def sigmoid_logits(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for raw logits."""
    x = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def probability_to_logits(probability: np.ndarray) -> np.ndarray:
    """Convert [0, 1] probabilities to logits."""
    p = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def build_trimap_from_logits(
    logits: np.ndarray,
    fg_threshold: float = 0.9,
    bg_threshold: float = 0.1,
) -> np.ndarray:
    """Build confidence-aware trimap from soft logits."""
    probs = sigmoid_logits(logits)
    trimap = np.full_like(probs, 0.5, dtype=np.float32)
    trimap[probs >= float(fg_threshold)] = 1.0
    trimap[probs <= float(bg_threshold)] = 0.0
    return trimap.astype(np.float32)


def resize_logits(logits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize logits with bilinear interpolation."""
    h, w = int(shape[0]), int(shape[1])
    src = np.asarray(logits, dtype=np.float32)
    if src.shape == (h, w):
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def resize_binary_mask(mask: np.ndarray, shape: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    """Resize binary mask with bilinear interpolation and re-threshold."""
    h, w = int(shape[0]), int(shape[1])
    src = np.asarray(mask, dtype=np.float32)
    if src.shape == (h, w):
        out = src
    else:
        out = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return (out >= float(threshold)).astype(np.float32)
