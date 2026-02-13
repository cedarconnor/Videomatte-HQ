"""ROI format read/write — JSON and CSV."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from videomatte_hq.roi.detect import BBox

logger = logging.getLogger(__name__)


def write_roi_json(path: Path, rois: list[BBox], fps: int = 30) -> None:
    """Write ROI track as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fps": fps,
        "frames": [
            {"frame": i, "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
            for i, r in enumerate(rois)
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"ROI track written to {path}")


def read_roi_json(path: Path) -> list[BBox]:
    """Read ROI track from JSON."""
    data = json.loads(path.read_text())
    rois = [None] * (max(f["frame"] for f in data["frames"]) + 1)
    for f in data["frames"]:
        rois[f["frame"]] = BBox(x0=f["x0"], y0=f["y0"], x1=f["x1"], y1=f["y1"])
    return rois


def write_roi_csv(path: Path, rois: list[BBox]) -> None:
    """Write ROI track as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x0", "y0", "x1", "y1"])
        for i, r in enumerate(rois):
            writer.writerow([i, r.x0, r.y0, r.x1, r.y1])
    logger.info(f"ROI CSV written to {path}")


def read_roi_csv(path: Path) -> list[BBox]:
    """Read ROI track from CSV."""
    rois = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rois.append(BBox(
                x0=int(row["x0"]), y0=int(row["y0"]),
                x1=int(row["x1"]), y1=int(row["y1"]),
            ))
    return rois
