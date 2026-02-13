"""ROI QC preview — downscaled video with ROI overlays."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from videomatte_hq.roi.detect import BBox

if TYPE_CHECKING:
    from videomatte_hq.io.reader import FrameSource

logger = logging.getLogger(__name__)


def generate_roi_preview(
    source: "FrameSource",
    rois: list[BBox],
    output_path: Path,
    preview_scale: int = 1080,
    bg_confidence: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> None:
    """Generate an ROI preview video with overlays.

    Args:
        source: Frame source.
        rois: Per-frame ROI bounding boxes.
        output_path: Path for output MP4.
        preview_scale: Target height for preview.
        bg_confidence: Optional confidence heatmap overlay.
        fps: Video framerate.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = source.resolution
    scale = min(preview_scale / max(h, w), 1.0)
    preview_w = int(w * scale)
    preview_h = int(h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (preview_w, preview_h))

    for t in range(min(source.num_frames, len(rois))):
        frame = source[t]
        frame_u8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)

        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (preview_w, preview_h))

        # Draw ROI rectangle
        roi = rois[t]
        x0 = int(roi.x0 * scale)
        y0 = int(roi.y0 * scale)
        x1 = int(roi.x1 * scale)
        y1 = int(roi.y1 * scale)
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Frame number
        cv2.putText(
            frame_bgr, f"F{t:06d}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        writer.write(frame_bgr)

    writer.release()
    logger.info(f"ROI preview written to {output_path}")
