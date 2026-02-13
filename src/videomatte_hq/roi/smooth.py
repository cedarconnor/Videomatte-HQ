"""ROI temporal smoothing and padding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from videomatte_hq.roi.detect import BBox

if TYPE_CHECKING:
    from videomatte_hq.config import ROIConfig

logger = logging.getLogger(__name__)


def smooth_rois(
    rois: list[Optional[BBox]],
    cfg: "ROIConfig",
    frame_size: tuple[int, int],
) -> list[BBox]:
    """Apply EMA smoothing and padding to raw ROI detections.

    Args:
        rois: Per-frame ROI bounding boxes (may contain None).
        cfg: ROI configuration.
        frame_size: (height, width) of frames.

    Returns:
        Smoothed and padded ROI for every frame (never None).
    """
    h, w = frame_size
    alpha = cfg.smooth_alpha
    pad_ratio = cfg.pad_ratio
    context_px = cfg.context_px

    smoothed: list[BBox] = []

    # Initialize EMA state from first valid ROI
    prev_cx, prev_cy = w / 2, h / 2
    prev_w, prev_h = w, h
    initialized = False

    for roi in rois:
        if roi is None:
            # Use previous smoothed value
            cx, cy = prev_cx, prev_cy
            bw, bh = prev_w, prev_h
        else:
            cx, cy = roi.center
            bw = float(roi.width)
            bh = float(roi.height)

            if initialized:
                # EMA smoothing on center and size
                cx = alpha * cx + (1 - alpha) * prev_cx
                cy = alpha * cy + (1 - alpha) * prev_cy
                bw = alpha * bw + (1 - alpha) * prev_w
                bh = alpha * bh + (1 - alpha) * prev_h
            initialized = True

        prev_cx, prev_cy = cx, cy
        prev_w, prev_h = bw, bh

        # Apply padding
        pad_x = bw * pad_ratio + context_px
        pad_y = bh * pad_ratio + context_px

        x0 = int(max(0, cx - bw / 2 - pad_x))
        y0 = int(max(0, cy - bh / 2 - pad_y))
        x1 = int(min(w, cx + bw / 2 + pad_x))
        y1 = int(min(h, cy + bh / 2 + pad_y))

        smoothed.append(BBox(x0=x0, y0=y0, x1=x1, y1=y1))

    logger.info(
        f"ROI smoothing: alpha={alpha}, pad_ratio={pad_ratio}, context={context_px}px, "
        f"avg_roi_area={sum(r.area for r in smoothed) / len(smoothed):.0f}px²"
    )

    return smoothed
