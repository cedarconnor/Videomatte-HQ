"""Box-only prompt adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from videomatte_hq.protocols import PromptAdapter, SegmentPrompt
from videomatte_hq.prompts.mask_adapter import _bbox_from_binary, _normalize_mask


@dataclass(slots=True)
class BoxPromptAdapter(PromptAdapter):
    """Convert mask input to bbox-only prompt payloads."""

    def adapt(self, mask: np.ndarray, frame_shape: tuple[int, int]) -> SegmentPrompt:
        mask_f = _normalize_mask(mask, frame_shape)
        bbox = _bbox_from_binary(mask_f >= 0.5)
        return SegmentPrompt(bbox=bbox, mask=mask_f)
