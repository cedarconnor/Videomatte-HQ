"""Prompt-to-box suggestion helpers for initial mask building."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BoxSuggestion:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    source: str
    label: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "x0": int(self.x0),
            "y0": int(self.y0),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "score": float(self.score),
            "source": self.source,
            "label": self.label,
        }


PERSON_KEYWORDS = {
    "person",
    "human",
    "man",
    "woman",
    "boy",
    "girl",
    "actor",
    "subject",
    "talent",
    "foreground person",
}

LEFT_KEYWORDS = {"left", "screen left", "camera left"}
RIGHT_KEYWORDS = {"right", "screen right", "camera right"}
CENTER_KEYWORDS = {"center", "middle", "centre"}


def _normalize_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().lower().split())


def _prompt_prefers(prompt: str) -> Literal["left", "right", "center", "none"]:
    p = _normalize_prompt(prompt)
    if any(k in p for k in LEFT_KEYWORDS):
        return "left"
    if any(k in p for k in RIGHT_KEYWORDS):
        return "right"
    if any(k in p for k in CENTER_KEYWORDS):
        return "center"
    return "none"


def _looks_like_person_prompt(prompt: str) -> bool:
    p = _normalize_prompt(prompt)
    return any(k in p for k in PERSON_KEYWORDS)


def _iou(a: BoxSuggestion, b: BoxSuggestion) -> float:
    xa0, ya0, xa1, ya1 = a.x0, a.y0, a.x1, a.y1
    xb0, yb0, xb1, yb1 = b.x0, b.y0, b.x1, b.y1
    ix0 = max(xa0, xb0)
    iy0 = max(ya0, yb0)
    ix1 = min(xa1, xb1)
    iy1 = min(ya1, yb1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, xa1 - xa0) * max(1, ya1 - ya0))
    area_b = float(max(1, xb1 - xb0) * max(1, yb1 - yb0))
    return inter / max(area_a + area_b - inter, 1e-6)


def _dedupe_boxes(cands: list[BoxSuggestion], iou_threshold: float = 0.9) -> list[BoxSuggestion]:
    out: list[BoxSuggestion] = []
    for c in cands:
        if any(_iou(c, k) >= iou_threshold for k in out):
            continue
        out.append(c)
    return out


def _rank_by_prompt(cands: list[BoxSuggestion], prompt: str, width: int) -> list[BoxSuggestion]:
    pref = _prompt_prefers(prompt)
    if pref == "none":
        return sorted(cands, key=lambda c: c.score, reverse=True)

    if pref == "left":
        target_x = 0.2 * width
    elif pref == "right":
        target_x = 0.8 * width
    else:
        target_x = 0.5 * width

    ranked: list[tuple[float, BoxSuggestion]] = []
    for c in cands:
        cx = 0.5 * (c.x0 + c.x1)
        dist = abs(cx - target_x) / max(width, 1)
        score = float(c.score) - 0.25 * dist
        ranked.append((score, c))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [c for _, c in ranked]


@lru_cache(maxsize=1)
def _person_detector():
    """Lazily instantiate person detector once."""
    from videomatte_hq.roi.detect import PersonDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PersonDetector(device=device, confidence_threshold=0.35)


def _detect_person_boxes(frame_rgb_u8: np.ndarray, max_candidates: int) -> list[BoxSuggestion]:
    rgb_f32 = frame_rgb_u8.astype(np.float32) / 255.0
    detector = _person_detector()
    boxes = detector.detect(rgb_f32, max_long_side=1280)
    out: list[BoxSuggestion] = []
    for b in boxes[: max(1, max_candidates)]:
        out.append(
            BoxSuggestion(
                x0=int(b.x0),
                y0=int(b.y0),
                x1=int(b.x1),
                y1=int(b.y1),
                score=float(b.confidence),
                source="person_detector",
                label="person",
            )
        )
    return out


@lru_cache(maxsize=1)
def _person_detector_weights_available() -> bool:
    """Check whether Faster R-CNN weights already exist locally."""
    try:
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        patterns = [
            "fasterrcnn_resnet50_fpn*.pth",
            "fasterrcnn_resnet50_fpn_v2*.pth",
        ]
        for pat in patterns:
            if any(cache_dir.glob(pat)):
                return True
        return False
    except Exception:
        return False


def _heuristic_boxes(frame_rgb_u8: np.ndarray) -> list[BoxSuggestion]:
    h, w = frame_rgb_u8.shape[:2]
    gray = cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    if float(mag.max()) > 1e-6:
        mag_u8 = np.clip((mag / mag.max()) * 255.0, 0, 255).astype(np.uint8)
    else:
        mag_u8 = np.zeros_like(gray)

    _, bw = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    bw = cv2.dilate(bw, kernel, iterations=1)

    cands: list[BoxSuggestion] = []
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    frame_area = float(h * w)
    for idx in range(1, num_labels):
        x, y, bw_i, bh_i, area = stats[idx]
        if area < 0.005 * frame_area:
            continue
        x0 = int(max(0, x - 8))
        y0 = int(max(0, y - 8))
        x1 = int(min(w - 1, x + bw_i + 8))
        y1 = int(min(h - 1, y + bh_i + 8))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        score = min(0.75, max(0.15, float(area) / frame_area))
        cands.append(
            BoxSuggestion(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                score=score,
                source="heuristic_edges",
                label="salient_region",
            )
        )

    center_w = int(round(w * 0.6))
    center_h = int(round(h * 0.8))
    cx0 = int(round((w - center_w) * 0.5))
    cy0 = int(round((h - center_h) * 0.5))
    cands.append(
        BoxSuggestion(
            x0=max(0, cx0),
            y0=max(0, cy0),
            x1=min(w - 1, cx0 + center_w),
            y1=min(h - 1, cy0 + center_h),
            score=0.2,
            source="fallback_center",
            label="center_subject",
        )
    )
    return cands


def suggest_prompt_boxes(
    frame_rgb_u8: np.ndarray,
    prompt: str,
    max_candidates: int = 5,
) -> list[BoxSuggestion]:
    """Suggest candidate subject boxes from a text prompt and current frame."""
    if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] < 3:
        raise ValueError("Expected frame RGB image with shape (H, W, 3+).")
    if frame_rgb_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 frame for prompt box suggestions.")

    h, w = frame_rgb_u8.shape[:2]
    max_cands = max(1, int(max_candidates))
    all_cands: list[BoxSuggestion] = []

    if _looks_like_person_prompt(prompt):
        if _person_detector_weights_available():
            try:
                all_cands.extend(_detect_person_boxes(frame_rgb_u8=frame_rgb_u8, max_candidates=max_cands))
            except Exception as exc:
                logger.warning("Prompt box detector unavailable; using heuristic suggestions only: %s", exc)
        else:
            logger.info(
                "Prompt box detector weights not found locally; using heuristic suggestions only."
            )

    all_cands.extend(_heuristic_boxes(frame_rgb_u8))
    all_cands = [
        BoxSuggestion(
            x0=max(0, min(w - 1, int(c.x0))),
            y0=max(0, min(h - 1, int(c.y0))),
            x1=max(0, min(w - 1, int(c.x1))),
            y1=max(0, min(h - 1, int(c.y1))),
            score=float(c.score),
            source=c.source,
            label=c.label,
        )
        for c in all_cands
        if int(c.x1) - int(c.x0) >= 3 and int(c.y1) - int(c.y0) >= 3
    ]
    ranked = _rank_by_prompt(_dedupe_boxes(all_cands), prompt=prompt, width=w)
    return ranked[:max_cands]
