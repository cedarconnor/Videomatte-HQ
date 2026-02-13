"""Preview compositor — 2×2 mosaic with checkerboard, alpha, white, flicker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig

logger = logging.getLogger(__name__)


def _make_checkerboard(h: int, w: int, size: int = 64) -> np.ndarray:
    """Generate a checkerboard pattern."""
    checker = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if ((y // size) + (x // size)) % 2 == 0:
                checker[y, x] = 0.75
            else:
                checker[y, x] = 0.5
    return checker


def _make_checkerboard_fast(h: int, w: int, size: int = 64) -> np.ndarray:
    """Vectorized checkerboard."""
    ys = np.arange(h) // size
    xs = np.arange(w) // size
    checker = ((ys[:, None] + xs[None, :]) % 2).astype(np.float32)
    return checker * 0.25 + 0.5


def composite_over_checker(rgb: np.ndarray, alpha: np.ndarray, checker_size: int = 64) -> np.ndarray:
    """Composite RGB over checkerboard background."""
    h, w = alpha.shape
    checker = _make_checkerboard_fast(h, w, checker_size)
    checker_rgb = np.stack([checker] * 3, axis=-1)
    a = alpha[..., np.newaxis]
    return rgb * a + checker_rgb * (1 - a)


def composite_over_color(rgb: np.ndarray, alpha: np.ndarray, color: tuple = (1, 1, 1)) -> np.ndarray:
    """Composite over solid color."""
    bg = np.ones_like(rgb) * np.array(color, dtype=np.float32)
    a = alpha[..., np.newaxis]
    return rgb * a + bg * (1 - a)


def alpha_to_rgb(alpha: np.ndarray) -> np.ndarray:
    """Alpha grayscale to RGB."""
    return np.stack([alpha] * 3, axis=-1)


def flicker_heatmap(delta: np.ndarray) -> np.ndarray:
    """Delta magnitude as hot colormap."""
    mag = np.abs(delta)
    mag_norm = np.clip(mag / 0.1, 0, 1)  # normalize to [0, 1]
    # Simple hot colormap: black → red → yellow → white
    r = np.clip(mag_norm * 3, 0, 1)
    g = np.clip(mag_norm * 3 - 1, 0, 1)
    b = np.clip(mag_norm * 3 - 2, 0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def generate_preview(
    source,
    final_alphas: list[np.ndarray],
    a0_results: list[np.ndarray],
    a0prime_results: list[np.ndarray],
    a1_results: list[np.ndarray],
    per_frame_data: list[dict],
    cfg: VideoMatteConfig,
) -> None:
    """Generate live preview video with configurable mosaic layout.

    Args:
        source: FrameSource.
        final_alphas: Final alpha per frame.
        a0_results: Pass A results.
        a0prime_results: Pass A′ results.
        a1_results: Pass B results.
        per_frame_data: Per-frame data with bands etc.
        cfg: Pipeline config.
    """
    if not cfg.preview.enabled:
        return

    output_path = Path(cfg.io.output_preview)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = source.resolution
    scale = min(cfg.preview.scale / max(h, w), 1.0)
    preview_h = int(h * scale)
    preview_w = int(w * scale)

    modes = cfg.preview.modes
    num_panels = min(len(modes), 4)

    # Determine mosaic layout
    if num_panels <= 1:
        mosaic_h, mosaic_w = preview_h, preview_w
    elif num_panels <= 2:
        mosaic_h, mosaic_w = preview_h, preview_w * 2
    else:
        mosaic_h, mosaic_w = preview_h * 2, preview_w * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, source.fps, (mosaic_w, mosaic_h))

    for t in range(0, source.num_frames, cfg.preview.every):
        frame = source[t]
        alpha = final_alphas[t]

        # Downscale
        if scale < 1.0:
            frame_s = cv2.resize(frame, (preview_w, preview_h))
            alpha_s = cv2.resize(alpha, (preview_w, preview_h))
        else:
            frame_s = frame
            alpha_s = alpha

        panels = []
        for mode in modes[:4]:
            if mode == "checker":
                panel = composite_over_checker(frame_s, alpha_s, cfg.preview.checker_size_px)
            elif mode == "alpha":
                panel = alpha_to_rgb(alpha_s)
            elif mode == "white":
                panel = composite_over_color(frame_s, alpha_s, (1, 1, 1))
            elif mode == "black":
                panel = composite_over_color(frame_s, alpha_s, (0, 0, 0))
            elif mode == "flicker":
                if t > 0:
                    prev_alpha = final_alphas[max(0, t - cfg.preview.every)]
                    if scale < 1.0:
                        prev_s = cv2.resize(prev_alpha, (preview_w, preview_h))
                    else:
                        prev_s = prev_alpha
                    delta = alpha_s - prev_s
                    panel = flicker_heatmap(delta)
                else:
                    panel = np.zeros((preview_h, preview_w, 3), dtype=np.float32)
            else:
                panel = alpha_to_rgb(alpha_s)
            panels.append(panel)

        # Pad to 4 panels if needed
        while len(panels) < 4:
            panels.append(np.zeros((preview_h, preview_w, 3), dtype=np.float32))

        # Assemble 2×2 mosaic
        top = np.concatenate([panels[0], panels[1]], axis=1)
        bottom = np.concatenate([panels[2], panels[3]], axis=1)
        mosaic = np.concatenate([top, bottom], axis=0)

        # Write
        mosaic_u8 = np.clip(mosaic * 255, 0, 255).astype(np.uint8)
        mosaic_bgr = cv2.cvtColor(mosaic_u8, cv2.COLOR_RGB2BGR)
        writer.write(mosaic_bgr)

    writer.release()
    logger.info(f"Preview written to {output_path}")
