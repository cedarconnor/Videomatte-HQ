"""Premultiplied foreground extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.writer import write_rgb_frame

logger = logging.getLogger(__name__)


def extract_foreground(
    source,
    alphas: list[np.ndarray],
    cfg: VideoMatteConfig,
    bg_plate: Optional[np.ndarray] = None,
    bg_confidence: Optional[np.ndarray] = None,
) -> None:
    """Extract and write premultiplied foreground frames.

    Args:
        source: FrameSource.
        alphas: Final alpha per frame.
        cfg: Pipeline config.
        bg_plate: Optional BG plate for despill.
        bg_confidence: Optional BG confidence.
    """
    if not cfg.postprocess.fg_output.enabled:
        return

    output_pattern = cfg.io.output_fg
    if output_pattern is None:
        output_pattern = cfg.io.output_alpha.replace("alpha", "fg")

    for t in range(len(alphas)):
        frame = source[t]  # (H, W, 3)
        alpha = alphas[t]  # (H, W)

        if cfg.postprocess.fg_output.despilled and bg_plate is not None and bg_confidence is not None:
            from videomatte_hq.postprocess.despill import despill_frame
            fg = despill_frame(
                frame, alpha, bg_plate, bg_confidence,
                cfg.postprocess.despill.strength,
                cfg.postprocess.despill.min_bg_confidence,
            )
        else:
            fg = frame

        if cfg.postprocess.fg_output.premultiplied:
            fg = fg * alpha[..., np.newaxis]

        # Write
        try:
            path = Path(output_pattern % t)
        except TypeError:
            path = Path(output_pattern.format(t))

        write_rgb_frame(path, fg, depth=16)

        if t % 100 == 0:
            logger.info(f"FG extract: frame {t}")
