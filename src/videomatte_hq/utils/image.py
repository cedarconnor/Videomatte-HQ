"""Image conversion helpers shared across CLI, web APIs, and pipeline backends."""

from __future__ import annotations

import numpy as np


def frame_to_rgb_u8(frame: np.ndarray, *, error_context: str = "frame IO") -> np.ndarray:
    """Convert grayscale/RGB/RGBA image-like arrays to RGB uint8."""
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim != 3:
        raise ValueError(f"Input frame must be RGB-like for {error_context}.")

    channels = int(rgb.shape[2])
    if channels == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif channels >= 3:
        rgb = rgb[..., :3]
    else:
        raise ValueError(f"Input frame must have 1, 3, or 4 channels for {error_context}.")

    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)

    out = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        out = out / float(np.iinfo(rgb.dtype).max)
    elif (float(out.max()) if out.size else 0.0) > 1.0:
        out = out / max(float(out.max()) if out.size else 1.0, 1.0)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
