"""Point-prompt adapter for interactive foreground/background selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from videomatte_hq.prompts.mask_adapter import (
    _bbox_from_binary,
    _expand_bbox,
    _sample_interior_points,
    _sample_negative_points,
)
from videomatte_hq.protocols import PromptAdapter, SegmentPrompt


@dataclass(slots=True)
class PointPromptAdapter(PromptAdapter):
    """Convert user-placed positive/negative points into SAM3 prompts.

    For initial prompts (empty mask), passes user points directly with no bbox.
    For per-frame re-prompting (propagated mask available), derives bbox and
    interior/negative points FROM the mask — just like MaskPromptAdapter.
    The original user points are only used for the initial frame-0 prompt.
    """

    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    bbox_expand_ratio: float = 0.10
    min_bbox_expand_px: int = 20
    interior_points: int = 5
    suppression_ratio: float = 0.3
    min_suppression_radius: int = 10
    negative_margin_ratio: float = 0.05
    min_negative_margin_px: int = 8

    def adapt(self, mask: np.ndarray, frame_shape: tuple[int, int]) -> SegmentPrompt:
        has_mask = mask is not None and np.any(mask > 0.5)

        if not has_mask:
            # No propagated mask — return original user points (initial prompt).
            return SegmentPrompt(
                bbox=None,
                positive_points=list(self.positive_points),
                negative_points=list(self.negative_points),
                mask=None,
            )

        # Propagated mask available — derive prompt from mask (like MaskPromptAdapter).
        # This ensures coordinates match the current frame resolution and avoids
        # re-injecting static user points on every frame.
        binary = mask >= 0.5 if mask.ndim == 2 else mask[..., 0] >= 0.5
        bbox = _expand_bbox(
            _bbox_from_binary(binary),
            frame_shape,
            expand_ratio=self.bbox_expand_ratio,
            min_expand_px=self.min_bbox_expand_px,
        )
        positive = _sample_interior_points(
            binary,
            k=self.interior_points,
            suppression_ratio=self.suppression_ratio,
            min_suppression_radius=self.min_suppression_radius,
        )
        negative = _sample_negative_points(
            binary,
            bbox,
            margin_ratio=self.negative_margin_ratio,
            min_margin_px=self.min_negative_margin_px,
        )
        return SegmentPrompt(
            bbox=bbox,
            positive_points=positive,
            negative_points=negative,
            mask=None,
        )


def parse_point_prompts(
    json_str: str,
    frame_shape: tuple[int, int],
) -> dict[int, dict[str, list[tuple[float, float]]]]:
    """Parse normalized [0,1] point prompts JSON and convert to pixel coordinates.

    Wire format (normalized 0-1 coords, keyed by frame index):
    {"0": {"positive": [[0.5, 0.3]], "negative": [[0.12, 0.08]]}}

    Returns dict keyed by frame index with pixel-coordinate point lists.
    """
    if not json_str or not json_str.strip():
        return {}

    raw = json.loads(json_str)
    if not isinstance(raw, dict):
        raise ValueError(f"point_prompts_json root must be an object, got {type(raw).__name__}")

    h, w = int(frame_shape[0]), int(frame_shape[1])
    result: dict[int, dict[str, list[tuple[float, float]]]] = {}

    for frame_key, frame_data in raw.items():
        frame_idx = int(frame_key)
        if not isinstance(frame_data, dict):
            raise ValueError(f"Frame {frame_key} entry must be an object")

        pos_raw = frame_data.get("positive", [])
        neg_raw = frame_data.get("negative", [])

        pos_px = [(float(x) * w, float(y) * h) for x, y in pos_raw]
        neg_px = [(float(x) * w, float(y) * h) for x, y in neg_raw]

        result[frame_idx] = {"positive": pos_px, "negative": neg_px}

    return result
