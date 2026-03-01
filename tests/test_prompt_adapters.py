from __future__ import annotations

import json

import numpy as np

from videomatte_hq.prompts.box_adapter import BoxPromptAdapter
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter, _nearest_background_point
from videomatte_hq.prompts.point_adapter import PointPromptAdapter, parse_point_prompts
from videomatte_hq.protocols import SegmentPrompt


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


def test_nearest_background_point_checks_immediate_neighbors() -> None:
    binary = np.ones((5, 5), dtype=np.uint8)
    binary[2, 3] = 0  # Immediate neighbor at radius 1.
    assert _nearest_background_point(binary, 2, 2, max_radius=4) == (3, 2)


# ── PointPromptAdapter tests ──


def test_point_adapter_initial_prompt_no_bbox() -> None:
    adapter = PointPromptAdapter(
        positive_points=[(100.0, 200.0), (150.0, 250.0)],
        negative_points=[(10.0, 10.0)],
    )
    empty_mask = np.zeros((480, 640), dtype=np.float32)
    prompt = adapter.adapt(empty_mask, (480, 640))

    assert isinstance(prompt, SegmentPrompt)
    assert prompt.bbox is None
    assert len(prompt.positive_points) == 2
    assert len(prompt.negative_points) == 1
    assert prompt.mask is None


def test_point_adapter_with_propagated_mask_derives_from_mask() -> None:
    """When a propagated mask is available, adapter derives all prompt data
    from the mask (like MaskPromptAdapter) instead of returning static user points."""
    adapter = PointPromptAdapter(
        positive_points=[(320.0, 240.0)],
        negative_points=[],
    )
    mask = np.zeros((480, 640), dtype=np.float32)
    mask[100:400, 200:500] = 1.0
    prompt = adapter.adapt(mask, (480, 640))

    # bbox derived from mask
    assert prompt.bbox is not None
    x0, y0, x1, y1 = prompt.bbox
    assert x0 < 200.0
    assert y0 < 100.0
    assert x1 > 500.0
    assert y1 > 400.0

    # Positive points are interior points from mask, NOT the original user point
    assert len(prompt.positive_points) > 0
    for x, y in prompt.positive_points:
        xi = int(np.clip(round(x), 0, 639))
        yi = int(np.clip(round(y), 0, 479))
        assert mask[yi, xi] >= 0.5, "Positive point should be inside the mask"

    # Negative points derived from mask boundaries
    assert len(prompt.negative_points) > 0
    for x, y in prompt.negative_points:
        xi = int(np.clip(round(x), 0, 639))
        yi = int(np.clip(round(y), 0, 479))
        assert mask[yi, xi] < 0.5, "Negative point should be outside the mask"

    assert prompt.mask is None


def test_parse_point_prompts_normalized_to_pixel() -> None:
    raw = {"0": {"positive": [[0.5, 0.25]], "negative": [[0.1, 0.9]]}}
    result = parse_point_prompts(json.dumps(raw), (480, 640))

    assert 0 in result
    pos = result[0]["positive"]
    neg = result[0]["negative"]

    assert len(pos) == 1
    assert abs(pos[0][0] - 320.0) < 0.01
    assert abs(pos[0][1] - 120.0) < 0.01

    assert len(neg) == 1
    assert abs(neg[0][0] - 64.0) < 0.01
    assert abs(neg[0][1] - 432.0) < 0.01


def test_parse_point_prompts_empty_string() -> None:
    assert parse_point_prompts("", (480, 640)) == {}
    assert parse_point_prompts("  ", (480, 640)) == {}


def test_parse_point_prompts_multiple_frames() -> None:
    raw = {
        "0": {"positive": [[0.5, 0.5]], "negative": []},
        "10": {"positive": [[0.3, 0.7]], "negative": [[0.8, 0.2]]},
    }
    result = parse_point_prompts(json.dumps(raw), (100, 200))

    assert 0 in result
    assert 10 in result
    assert len(result[10]["positive"]) == 1
    assert len(result[10]["negative"]) == 1


def test_point_prompt_generates_correct_prompt_variants() -> None:
    from videomatte_hq.pipeline.stage_segment import UltralyticsSAM3SegmentBackend, SegmentStageConfig

    prompt = SegmentPrompt(
        bbox=None,
        positive_points=[(100.0, 200.0), (150.0, 250.0)],
        negative_points=[(10.0, 10.0)],
        mask=None,
    )

    cfg = SegmentStageConfig()
    backend = UltralyticsSAM3SegmentBackend(cfg)
    variants = backend._prompt_variants(prompt)

    variant_names = [name for name, _ in variants]
    assert "points" in variant_names
    assert "bbox" not in variant_names
    assert "bbox_points" not in variant_names

    for name, data in variants:
        if name == "points":
            assert "points" in data
            assert "labels" in data
            points = data["points"]
            labels = data["labels"]
            assert len(points[0]) == 3
            assert labels[0] == [1, 1, 0]
