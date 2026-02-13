"""Color space conversion utilities.

Handles sRGB ↔ linear, Rec.709, and ACEScg conversions.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB (inverse gamma)."""
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    )
    return linear.astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB (apply gamma)."""
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def rgb_to_luma(img: np.ndarray) -> np.ndarray:
    """Convert RGB to luma using Rec.709 coefficients.

    Args:
        img: (..., 3) float32 RGB.

    Returns:
        (...,) float32 luminance.
    """
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(
        np.float32
    )


# ACEScg ↔ sRGB matrices (D60/D65 adaptation included)
_ACES_TO_SRGB = np.array(
    [
        [1.70505, -0.62179, -0.08326],
        [-0.13026, 1.14080, -0.01055],
        [-0.02400, -0.12897, 1.15297],
    ],
    dtype=np.float32,
)

_SRGB_TO_ACES = np.array(
    [
        [0.61319, 0.33951, 0.04737],
        [0.07012, 0.91637, 0.01345],
        [0.02058, 0.10958, 0.86985],
    ],
    dtype=np.float32,
)


def acescg_to_srgb_linear(img: np.ndarray) -> np.ndarray:
    """ACEScg → linear sRGB (matrix multiply, no gamma)."""
    return np.einsum("...c,rc->...r", img, _ACES_TO_SRGB).astype(np.float32)


def srgb_linear_to_acescg(img: np.ndarray) -> np.ndarray:
    """Linear sRGB → ACEScg."""
    return np.einsum("...c,rc->...r", img, _SRGB_TO_ACES).astype(np.float32)


def auto_detect_colorspace(img: np.ndarray) -> str:
    """Heuristic colorspace detection.

    Returns 'srgb' for typical 8/16-bit imagery, 'linear' for float/EXR data.
    """
    if img.dtype in (np.float32, np.float64):
        # Check if values exceed 1.0 (likely linear/HDR)
        if img.max() > 1.5:
            return "linear"
    return "srgb"
