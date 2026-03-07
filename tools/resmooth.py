"""Re-apply temporal smoothing to existing alpha frames and regenerate preview MP4.

Usage:
    python tools/resmooth.py D:\Videomatte-HQ2\output_ui\V1 --strength 0.7 --threshold 0.04
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from videomatte_hq.postprocess.temporal_smooth import TemporalSmoothConfig, apply_temporal_smooth
from videomatte_hq.io.preview_mp4 import generate_alpha_preview_mp4

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("resmooth")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-smooth existing alpha frames")
    parser.add_argument("output_dir", type=Path, help="Job output directory containing alpha/")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.04, help="Motion threshold (0-1)")
    parser.add_argument("--fps", type=float, default=25.0, help="Preview MP4 framerate")
    parser.add_argument("--no-preview", action="store_true", help="Skip MP4 generation")
    args = parser.parse_args()

    alpha_dir = args.output_dir / "alpha"
    if not alpha_dir.is_dir():
        logger.error("No alpha/ directory found in %s", args.output_dir)
        sys.exit(1)

    frames_sorted = sorted(alpha_dir.glob("*.png"))
    if not frames_sorted:
        logger.error("No PNG files in %s", alpha_dir)
        sys.exit(1)

    frame_start = int(frames_sorted[0].stem)
    logger.info("Loading %d alpha frames from %s ...", len(frames_sorted), alpha_dir)

    alphas: list[np.ndarray] = []
    for p in frames_sorted:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Skipping unreadable frame: %s", p)
            continue
        if img.dtype == np.uint16:
            alphas.append(img.astype(np.float32) / 65535.0)
        elif img.dtype == np.uint8:
            alphas.append(img.astype(np.float32) / 255.0)
        else:
            alphas.append(img.astype(np.float32))

    logger.info("Loaded %d frames, applying temporal smooth (strength=%.2f, threshold=%.3f) ...",
                len(alphas), args.strength, args.threshold)

    cfg = TemporalSmoothConfig(enabled=True, strength=args.strength, motion_threshold=args.threshold)
    alphas = apply_temporal_smooth(alphas, cfg)

    logger.info("Writing smoothed frames back to %s ...", alpha_dir)
    for i, p in enumerate(frames_sorted):
        if i < len(alphas):
            out = np.clip(alphas[i] * 65535.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(p), out)
    logger.info("Wrote %d smoothed frames.", len(alphas))

    if not args.no_preview:
        logger.info("Generating preview MP4 ...")
        try:
            mp4 = generate_alpha_preview_mp4(
                output_dir=args.output_dir,
                alpha_pattern="alpha/%06d.png",
                frame_start=frame_start,
                frame_count=len(alphas),
                fps=args.fps,
            )
            logger.info("Preview: %s", mp4)
        except Exception:
            logger.warning("Preview MP4 generation failed.", exc_info=True)


if __name__ == "__main__":
    main()
