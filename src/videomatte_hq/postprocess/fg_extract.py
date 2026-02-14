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

    # Fix: output_fg does not exist in IOConfig. Derive from output_alpha.
    # We replace "alpha" directory with "fg" in the pattern.
    # e.g. "out/alpha/%06d.png" -> "out/fg/%06d.png"
    if hasattr(cfg.io, "output_fg"):
        output_pattern = cfg.io.output_fg
    else:
        output_pattern = cfg.io.output_alpha.replace("alpha", "fg")

    # Ensure parent dir exists
    # We can't easily guess parent from pattern string without parsing, 
    # but write_rgb_frame might handle usage? 
    # Actually we should create directory for first frame to be safe.
    try:
        first_path = Path(output_pattern % 0)
        first_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    for t in range(len(alphas)):
        frame = source[t]  # (H, W, 3)
        alpha = alphas[t]  # (H, W)

        # Fix: FGOutputConfig lacks 'despilled'. Use global despill.enabled.
        # Fix: DespillConfig lacks 'strength' and 'min_bg_confidence'. Use defaults.
        if cfg.postprocess.despill.enabled and bg_plate is not None and bg_confidence is not None:
            from videomatte_hq.postprocess.despill import despill_frame
            # Use defaults 1.0 and 0.4 as they are missing from config
            fg = despill_frame(
                frame, alpha, bg_plate, bg_confidence,
                strength=1.0, 
                min_confidence=0.4,
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
