from __future__ import annotations

import numpy as np

from videomatte_hq.prompts.box_adapter import BoxPromptAdapter
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter


def _sample_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 20:44] = 1.0
    return mask


def test_mask_prompt_adapter_generates_bbox_and_points() -> None:
    mask = _sample_mask()
    adapter = MaskPromptAdapter(interior_points=5)
    prompt = adapter.adapt(mask, frame_shape=mask.shape)

    assert prompt.bbox is not None
    x0, y0, x1, y1 = prompt.bbox
    assert x0 <= 8.0 and y0 <= 4.0
    assert x1 >= 56.0 and y1 >= 60.0

    assert len(prompt.positive_points) > 0
    for x, y in prompt.positive_points:
        xi = int(round(x))
        yi = int(round(y))
        xi = int(np.clip(xi, 0, mask.shape[1] - 1))
        yi = int(np.clip(yi, 0, mask.shape[0] - 1))
        assert mask[yi, xi] >= 0.5

    for x, y in prompt.negative_points:
        xi = int(round(x))
        yi = int(round(y))
        xi = int(np.clip(xi, 0, mask.shape[1] - 1))
        yi = int(np.clip(yi, 0, mask.shape[0] - 1))
        assert mask[yi, xi] < 0.5


def test_mask_prompt_adapter_bbox_expansion_clamps_to_frame() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[0:8, 2:10] = 1.0
    adapter = MaskPromptAdapter(bbox_expand_ratio=0.5, min_bbox_expand_px=20)
    prompt = adapter.adapt(mask, frame_shape=mask.shape)
    assert prompt.bbox is not None
    x0, y0, x1, y1 = prompt.bbox
    assert (x0, y0) == (0.0, 0.0)
    assert x1 <= 32.0 and y1 <= 32.0
    assert x1 > 10.0 and y1 > 8.0


def test_box_prompt_adapter_only_returns_bbox() -> None:
    mask = _sample_mask()
    adapter = BoxPromptAdapter()
    prompt = adapter.adapt(mask, frame_shape=mask.shape)
    assert prompt.bbox is not None
    assert prompt.positive_points == []
    assert prompt.negative_points == []
