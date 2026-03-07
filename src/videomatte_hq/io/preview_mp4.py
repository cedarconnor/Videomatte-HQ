"""Generate an H.264 MP4 preview from alpha frame sequences."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def generate_alpha_preview_mp4(
    output_dir: Path,
    alpha_pattern: str,
    frame_start: int,
    frame_count: int,
    fps: float,
    max_long_side: int = 3840,
) -> Path:
    """Build an H.264 MP4 preview video from the alpha PNG/EXR sequence.

    Parameters
    ----------
    output_dir : Path
        Root output directory containing the alpha subfolder.
    alpha_pattern : str
        printf-style pattern relative to *output_dir*, e.g. ``"alpha/%06d.png"``.
    frame_start : int
        First frame number in the sequence.
    frame_count : int
        Total number of frames written.
    fps : float
        Playback framerate.
    max_long_side : int
        If the source resolution exceeds this on the long side, scale down
        proportionally (keeping even dimensions).

    Returns
    -------
    Path
        Absolute path to the generated ``alpha_preview.mp4``.
    """
    if frame_count < 1:
        raise ValueError("No frames to encode.")

    # Resolve the first alpha frame to probe resolution.
    first_frame_path = output_dir / (alpha_pattern % frame_start)
    if not first_frame_path.exists():
        raise FileNotFoundError(f"First alpha frame not found: {first_frame_path}")

    img = cv2.imread(str(first_frame_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read alpha frame: {first_frame_path}")
    h, w = img.shape[:2]

    # Build ffmpeg command.
    input_path = str(output_dir / alpha_pattern)
    out_mp4 = output_dir / "alpha_preview.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", str(frame_start),
        "-i", input_path,
    ]

    long_side = max(h, w)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        out_w = int(w * scale) // 2 * 2  # round to even
        out_h = int(h * scale) // 2 * 2
        cmd += ["-vf", f"scale={out_w}:{out_h}"]

    cmd += [
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]

    logger.info("Running ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    logger.info("Preview MP4 written: %s (%.1f MB)", out_mp4, out_mp4.stat().st_size / (1024 * 1024))
    return out_mp4
