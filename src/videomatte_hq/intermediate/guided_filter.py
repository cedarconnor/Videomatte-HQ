"""Fast guided filter implementation for delta clamping.

Smooths a source signal using an RGB guide image, preserving edges
aligned with image structure while removing per-pixel noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Apply guided filter to source using guide image.

    This is a box-filter variant for efficiency at large radii.

    Args:
        guide: (H, W, C) or (H, W) float32 guide image (typically RGB).
        src: (H, W) float32 source to filter (typically delta).
        radius: Filter radius in pixels.
        eps: Regularization parameter (higher = smoother).

    Returns:
        (H, W) float32 filtered result.
    """
    if guide.ndim == 3:
        # Use luminance channel for scalar guide
        guide_gray = 0.2126 * guide[..., 0] + 0.7152 * guide[..., 1] + 0.0722 * guide[..., 2]
    else:
        guide_gray = guide

    guide_gray = guide_gray.astype(np.float32)
    src = src.astype(np.float32)

    ksize = 2 * radius + 1

    # Box filter means
    mean_I = cv2.boxFilter(guide_gray, -1, (ksize, ksize))
    mean_p = cv2.boxFilter(src, -1, (ksize, ksize))
    mean_Ip = cv2.boxFilter(guide_gray * src, -1, (ksize, ksize))
    mean_II = cv2.boxFilter(guide_gray * guide_gray, -1, (ksize, ksize))

    # Covariance and variance
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Average coefficients
    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))

    # Output
    return (mean_a * guide_gray + mean_b).astype(np.float32)


def guided_filter_color(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """Full-color guided filter (uses all 3 channels of guide).

    More accurate at edges but slower than the grayscale version.

    Args:
        guide: (H, W, 3) float32 RGB guide.
        src: (H, W) float32 source.
        radius: Filter radius.
        eps: Regularization.

    Returns:
        (H, W) float32 filtered result.
    """
    h, w = src.shape
    ksize = 2 * radius + 1

    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    mean_I_r = cv2.boxFilter(guide[..., 0], -1, (ksize, ksize))
    mean_I_g = cv2.boxFilter(guide[..., 1], -1, (ksize, ksize))
    mean_I_b = cv2.boxFilter(guide[..., 2], -1, (ksize, ksize))
    mean_p = cv2.boxFilter(src, -1, (ksize, ksize))

    mean_Ip_r = cv2.boxFilter(guide[..., 0] * src, -1, (ksize, ksize))
    mean_Ip_g = cv2.boxFilter(guide[..., 1] * src, -1, (ksize, ksize))
    mean_Ip_b = cv2.boxFilter(guide[..., 2] * src, -1, (ksize, ksize))

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p

    # Variance matrix (3x3 per pixel — compute element by element)
    var_rr = cv2.boxFilter(guide[..., 0] ** 2, -1, (ksize, ksize)) - mean_I_r ** 2 + eps
    var_rg = cv2.boxFilter(guide[..., 0] * guide[..., 1], -1, (ksize, ksize)) - mean_I_r * mean_I_g
    var_rb = cv2.boxFilter(guide[..., 0] * guide[..., 2], -1, (ksize, ksize)) - mean_I_r * mean_I_b
    var_gg = cv2.boxFilter(guide[..., 1] ** 2, -1, (ksize, ksize)) - mean_I_g ** 2 + eps
    var_gb = cv2.boxFilter(guide[..., 1] * guide[..., 2], -1, (ksize, ksize)) - mean_I_g * mean_I_b
    var_bb = cv2.boxFilter(guide[..., 2] ** 2, -1, (ksize, ksize)) - mean_I_b ** 2 + eps

    # Solve 3x3 linear system per pixel (vectorized via inverse)
    det = (var_rr * (var_gg * var_bb - var_gb ** 2)
           - var_rg * (var_rg * var_bb - var_gb * var_rb)
           + var_rb * (var_rg * var_gb - var_gg * var_rb))
    det = np.maximum(det, 1e-10)

    inv_rr = (var_gg * var_bb - var_gb ** 2) / det
    inv_rg = (var_rb * var_gb - var_rg * var_bb) / det
    inv_rb = (var_rg * var_gb - var_gg * var_rb) / det
    inv_gg = (var_rr * var_bb - var_rb ** 2) / det
    inv_gb = (var_rb * var_rg - var_rr * var_gb) / det
    inv_bb = (var_rr * var_gg - var_rg ** 2) / det

    a_r = inv_rr * cov_Ip_r + inv_rg * cov_Ip_g + inv_rb * cov_Ip_b
    a_g = inv_rg * cov_Ip_r + inv_gg * cov_Ip_g + inv_gb * cov_Ip_b
    a_b = inv_rb * cov_Ip_r + inv_gb * cov_Ip_g + inv_bb * cov_Ip_b
    b = mean_p - a_r * mean_I_r - a_g * mean_I_g - a_b * mean_I_b

    # Average coefficients
    mean_a_r = cv2.boxFilter(a_r, -1, (ksize, ksize))
    mean_a_g = cv2.boxFilter(a_g, -1, (ksize, ksize))
    mean_a_b = cv2.boxFilter(a_b, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))

    return (mean_a_r * guide[..., 0] + mean_a_g * guide[..., 1] + mean_a_b * guide[..., 2] + mean_b).astype(np.float32)
