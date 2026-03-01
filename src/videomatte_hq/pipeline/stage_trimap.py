"""Stage-2 trimap generation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def _resolution_scale(shape: tuple[int, ...], reference_long_side: int = 1920) -> float:
    """Compute scale factor for resolution-aware morphological operations.

    Pixel values are specified relative to 1080p (long side 1920).  For higher
    resolutions the effective kernel sizes are scaled up so that the unknown
    band stays proportional to the image.
    """
    long_side = max(int(shape[0]), int(shape[1]))
    return max(1.0, long_side / float(reference_long_side))


def _fallback_trimap_from_binary_band(probability: np.ndarray, band_px: int) -> np.ndarray:
    """Create a narrow unknown band around a hard mask edge when threshold trimap is empty."""
    p = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if p.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape={p.shape}")

    scale = _resolution_scale(p.shape)
    radius = max(1, int(round(int(band_px) * scale)))
    k = int(2 * radius + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = (p >= 0.5).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return np.zeros_like(p, dtype=np.float32)
    if int(mask.sum()) >= int(mask.size):
        return np.ones_like(p, dtype=np.float32)

    eroded_fg = cv2.erode(mask, kernel, iterations=1) > 0
    dilated_fg = cv2.dilate(mask, kernel, iterations=1) > 0

    trimap = np.full(mask.shape, 0.5, dtype=np.float32)
    trimap[eroded_fg] = 1.0
    trimap[~dilated_fg] = 0.0
    return trimap.astype(np.float32)


def build_trimap_morphological(
    binary_mask: np.ndarray,
    erosion_px: int = 20,
    dilation_px: int = 10,
) -> np.ndarray:
    """Build trimap from binary mask using morphological erosion/dilation.

    Erosion moves "definite FG" inward from the SAM edge; dilation moves
    "definite BG" outward. The gap between = unknown band where MEMatte
    refines.

    Pixel values are specified relative to 1080p (long side 1920px).  For
    higher resolutions the kernel sizes scale proportionally so that the
    unknown band stays a consistent fraction of the image.
    """
    mask = np.asarray(binary_mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape={mask.shape}")

    fg = (mask >= 0.5).astype(np.uint8)

    # Edge cases
    if int(fg.sum()) <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    if int(fg.sum()) >= int(fg.size):
        return np.ones_like(mask, dtype=np.float32)

    # Scale pixel values for high-resolution inputs (reference: 1080p / 1920 long side)
    scale = _resolution_scale(mask.shape)
    erosion_px = max(1, int(round(int(erosion_px) * scale)))
    dilation_px = max(1, int(round(int(dilation_px) * scale)))

    ek = int(2 * erosion_px + 1)
    dk = int(2 * dilation_px + 1)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))

    eroded_fg = cv2.erode(fg, erosion_kernel, iterations=1) > 0
    dilated_fg = cv2.dilate(fg, dilation_kernel, iterations=1) > 0

    trimap = np.full(mask.shape, 0.5, dtype=np.float32)
    trimap[eroded_fg] = 1.0
    trimap[~dilated_fg] = 0.0
    return trimap


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
    fallback_band_px: int = 0,
) -> np.ndarray:
    """Build confidence-aware trimap from soft logits."""
    probs = sigmoid_logits(logits)
    trimap = np.full_like(probs, 0.5, dtype=np.float32)
    trimap[probs >= float(fg_threshold)] = 1.0
    trimap[probs <= float(bg_threshold)] = 0.0
    unknown_coverage = float((trimap == 0.5).sum()) / max(trimap.size, 1)
    if int(fallback_band_px) > 0 and unknown_coverage < 0.001:
        trimap = _fallback_trimap_from_binary_band(probs, band_px=int(fallback_band_px))
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
