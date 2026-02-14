"""Person detection for ROI tracking.

Uses torchvision's Faster R-CNN or a lightweight detector to locate
people in downscaled frames at a configurable cadence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from videomatte_hq.config import ROIConfig
    from videomatte_hq.io.reader import FrameSource

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Axis-aligned bounding box (pixel coordinates)."""
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def clamp(self, max_w: int, max_h: int) -> "BBox":
        return BBox(
            x0=max(0, self.x0),
            y0=max(0, self.y0),
            x1=min(max_w, self.x1),
            y1=min(max_h, self.y1),
            confidence=self.confidence,
        )


class PersonDetector:
    """Person detector using torchvision Faster R-CNN."""

    PERSON_CLASS_ID = 1  # COCO class for 'person'

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        import torchvision
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        # Load pretrained Faster R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Person detector loaded on {self.device}")

    @torch.no_grad()
    def detect(self, frame: np.ndarray, max_long_side: int = 1080) -> list[BBox]:
        """Detect people in a frame.

        Args:
            frame: (H, W, C) float32 RGB in [0, 1].
            max_long_side: Downscale to this resolution for detection speed.

        Returns:
            List of BBox for detected persons, in original frame coordinates.
        """
        h, w = frame.shape[:2]
        scale = min(max_long_side / max(h, w), 1.0)

        if scale < 1.0:
            det_h, det_w = int(h * scale), int(w * scale)
            det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_frame = frame
            scale = 1.0

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(det_frame.transpose(2, 0, 1)).float().to(self.device)
        results = self.model([tensor])[0]

        boxes = []
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            if label.item() == self.PERSON_CLASS_ID and score.item() >= self.confidence_threshold:
                x0, y0, x1, y1 = box.cpu().numpy()
                # Scale back to original resolution
                boxes.append(BBox(
                    x0=int(x0 / scale),
                    y0=int(y0 / scale),
                    x1=int(x1 / scale),
                    y1=int(y1 / scale),
                    confidence=score.item(),
                ))

        return boxes


def detect_rois(
    source: "FrameSource",
    cfg: "ROIConfig",
    bg_confidence: Optional[np.ndarray] = None,
    bg_plate: Optional[np.ndarray] = None,
    photometric_normalize: bool = True,
) -> list[Optional[BBox]]:
    """Run person detection at configured cadence and interpolate.

    Args:
        source: Frame source.
        cfg: ROI configuration.
        bg_confidence: Optional BG confidence map for motion mask gating.
        bg_plate: Optional BG plate for motion mask computation.
        photometric_normalize: Whether to apply photometric normalization for motion mask.

    Returns:
        List of BBox (one per frame, None if no detection).
    """
    num_frames = source.num_frames
    h, w = source.resolution

    detector = PersonDetector(confidence_threshold=0.5)

    # Detect at cadence
    detections: dict[int, list[BBox]] = {}
    detect_every = cfg.detect_every

    for t in range(0, num_frames, detect_every):
        frame = source[t]
        persons = detector.detect(frame)

        if cfg.multi_person == "single" and persons:
            # Keep largest
            persons = [max(persons, key=lambda b: b.area)]
        elif cfg.multi_person == "union_k" and len(persons) > cfg.k:
            persons = sorted(persons, key=lambda b: b.area, reverse=True)[:cfg.k]

        if persons:
            # Union of all person boxes
            x0 = min(b.x0 for b in persons)
            y0 = min(b.y0 for b in persons)
            x1 = max(b.x1 for b in persons)
            y1 = max(b.y1 for b in persons)
            det_box = BBox(x0, y0, x1, y1, confidence=max(b.confidence for b in persons))
        else:
            det_box = None

        # Motion mask union (Stage 1 design: union with motion mask from BG subtraction)
        if cfg.use_motion_mask and bg_plate is not None and bg_confidence is not None:
            from videomatte_hq.roi.motion_mask import compute_motion_mask

            motion = compute_motion_mask(
                frame, bg_plate, bg_confidence,
                photometric_normalize=photometric_normalize,
            )
            if motion.any():
                ys, xs = np.where(motion)
                motion_box = BBox(
                    x0=int(xs.min()), y0=int(ys.min()),
                    x1=int(xs.max()), y1=int(ys.max()),
                    confidence=0.5,
                )
                if det_box is not None:
                    # Union of detector box and motion box
                    det_box = BBox(
                        x0=min(det_box.x0, motion_box.x0),
                        y0=min(det_box.y0, motion_box.y0),
                        x1=max(det_box.x1, motion_box.x1),
                        y1=max(det_box.y1, motion_box.y1),
                        confidence=det_box.confidence,
                    )
                else:
                    det_box = motion_box

        detections[t] = [det_box] if det_box is not None else []

        if t % (detect_every * 5) == 0:
            logger.debug(f"Detection frame {t}/{num_frames}: {len(detections[t])} detections")

    # Interpolate between detections
    rois: list[Optional[BBox]] = [None] * num_frames
    det_frames = sorted(detections.keys())

    for t in range(num_frames):
        # Find nearest detection frames
        prev_det = max((d for d in det_frames if d <= t), default=None)
        next_det = min((d for d in det_frames if d >= t), default=None)

        if prev_det is not None and detections[prev_det]:
            prev_box = detections[prev_det][0]
        else:
            prev_box = None

        if next_det is not None and detections[next_det]:
            next_box = detections[next_det][0]
        else:
            next_box = None

        if prev_box and next_box and prev_det != next_det:
            # Linear interpolation
            alpha = (t - prev_det) / (next_det - prev_det)
            rois[t] = BBox(
                x0=int(prev_box.x0 + alpha * (next_box.x0 - prev_box.x0)),
                y0=int(prev_box.y0 + alpha * (next_box.y0 - prev_box.y0)),
                x1=int(prev_box.x1 + alpha * (next_box.x1 - prev_box.x1)),
                y1=int(prev_box.y1 + alpha * (next_box.y1 - prev_box.y1)),
            ).clamp(w, h)
        elif prev_box:
            rois[t] = prev_box.clamp(w, h)
        elif next_box:
            rois[t] = next_box.clamp(w, h)
        else:
            # No detection — use full frame
            rois[t] = BBox(0, 0, w, h)

    logger.info(f"ROI detection complete: {sum(1 for r in rois if r is not None)} frames with ROI")
    return rois
