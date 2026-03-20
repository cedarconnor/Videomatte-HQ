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


def build_trimap_hybrid(
    binary_mask: np.ndarray,
    logits: np.ndarray,
    *,
    erosion_px: int = 20,
    dilation_px: int = 10,
    fg_threshold: float = 0.9,
    bg_threshold: float = 0.1,
    fallback_band_px: int = 0,
) -> np.ndarray:
    """Build a trimap that preserves the morphological band and logit uncertainty.

    The morphological band provides a stable spatial prior. Logit confidence is
    then used as a veto: any region that is not confidently foreground or
    background remains unknown, even if the binary mask morphology would have
    marked it definite.
    """
    morph = build_trimap_morphological(
        binary_mask,
        erosion_px=erosion_px,
        dilation_px=dilation_px,
    ).astype(np.float32)
    logit_trimap = build_trimap_from_logits(
        logits,
        fg_threshold=fg_threshold,
        bg_threshold=bg_threshold,
        fallback_band_px=fallback_band_px,
    ).astype(np.float32)
    probs = sigmoid_logits(logits)

    trimap = morph.copy()
    trimap[logit_trimap == 0.5] = 0.5
    trimap[np.logical_and(morph >= 1.0, probs < float(fg_threshold))] = 0.5
    trimap[np.logical_and(morph <= 0.0, probs > float(bg_threshold))] = 0.5
    return trimap.astype(np.float32)


def build_trimap_gradient_adaptive(
    frame_rgb: np.ndarray,
    alpha_upscaled: np.ndarray,
    base_kernel: int = 7,
    max_extra: int = 20,
    fg_thresh: float = 0.95,
    bg_thresh: float = 0.05,
    gradient_scale: float = 0.5,
) -> np.ndarray:
    """Build gradient-adaptive trimap from upscaled alpha and source frame.

    The unknown band width is modulated by image gradient magnitude — wider
    around complex edges (hair, fine detail) and narrower around clean edges
    (clothing, limbs).

    Output format: 0.0 (definite BG), 0.5 (unknown), 1.0 (definite FG).
    """
    alpha = np.asarray(alpha_upscaled, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError(f"Expected 2D alpha, got shape={alpha.shape}")
    alpha = np.clip(alpha, 0.0, 1.0)

    # Scale kernel sizes for high-resolution inputs
    scale = _resolution_scale(alpha.shape)
    base_kernel = max(3, int(round(int(base_kernel) * scale)))
    max_extra = max(1, int(round(int(max_extra) * scale)))
    # Ensure base_kernel is odd
    if base_kernel % 2 == 0:
        base_kernel += 1

    # Compute image gradient magnitude (Sobel)
    rgb = np.asarray(frame_rgb)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        gray = cv2.cvtColor(rgb[..., :3].astype(np.uint8) if rgb.dtype == np.uint8
                            else (np.clip(rgb[..., :3], 0, 1) * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY)
    elif rgb.ndim == 2:
        gray = rgb.astype(np.uint8) if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    else:
        gray = np.zeros(alpha.shape, dtype=np.uint8)

    if gray.shape[:2] != alpha.shape:
        gray = cv2.resize(gray, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_max = float(grad_mag.max())
    if grad_max > 0:
        grad_mag = grad_mag / grad_max
    grad_mag = grad_mag.astype(np.float32)

    # Binarize the upscaled alpha to create a hard mask edge.
    # Bicubic upscaling creates wide soft halos; thresholding at 0.5
    # ensures FG/BG zones match the actual MA2 mask boundary.
    fg = (alpha >= 0.5).astype(np.uint8)

    # Edge cases: empty or full mask
    if int(fg.sum()) <= 0:
        return np.zeros_like(alpha, dtype=np.float32)
    if int(fg.sum()) >= int(fg.size):
        return np.ones_like(alpha, dtype=np.float32)

    # Build definite FG via erosion, definite BG via dilation (same
    # principle as morphological trimap in v1, but with gradient-adaptive
    # unknown band width).
    ek = int(2 * base_kernel + 1)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
    eroded_fg = cv2.erode(fg, erosion_kernel, iterations=1) > 0

    dk = int(2 * base_kernel + 1)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    dilated_fg = cv2.dilate(fg, dilation_kernel, iterations=1) > 0

    trimap = np.full(alpha.shape, 0.5, dtype=np.float32)
    trimap[eroded_fg] = 1.0
    trimap[~dilated_fg] = 0.0

    # Expand the unknown band further in high-gradient regions.
    # The boundary is the zone between eroded FG and dilated FG.
    boundary = (~eroded_fg & dilated_fg).astype(np.uint8)

    gradient_scale_f = float(gradient_scale)
    extra_sizes = [
        max(1, max_extra // 3),
        max(1, 2 * max_extra // 3),
        max_extra,
    ]
    for extra in extra_sizes:
        k_size = base_kernel + extra
        if k_size % 2 == 0:
            k_size += 1
        k_size = max(3, k_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(boundary, kernel, iterations=1)

        # Only apply wider kernels where gradient is high enough
        if max_extra > 0:
            grad_threshold = float(extra) / float(max_extra)
            high_grad = (grad_mag * gradient_scale_f) > (grad_threshold * gradient_scale_f)
            expansion_mask = (dilated > 0) & high_grad
        else:
            expansion_mask = dilated > 0

        trimap[expansion_mask] = 0.5

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
