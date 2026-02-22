from __future__ import annotations

import numpy as np

from videomatte_hq.pipeline.stage_trimap import (
    build_trimap_from_logits,
    probability_to_logits,
    resize_binary_mask,
    resize_logits,
    sigmoid_logits,
)


def test_probability_logits_roundtrip() -> None:
    p = np.array([[0.01, 0.5, 0.99]], dtype=np.float32)
    logits = probability_to_logits(p)
    out = sigmoid_logits(logits)
    assert np.allclose(out, p, atol=1e-5)


def test_build_trimap_from_logits_thresholds() -> None:
    logits = np.array([[-8.0, 0.0, 8.0]], dtype=np.float32)
    trimap = build_trimap_from_logits(logits, fg_threshold=0.9, bg_threshold=0.1)
    assert np.allclose(trimap, np.array([[0.0, 0.5, 1.0]], dtype=np.float32))


def test_resize_helpers_preserve_shape_contract() -> None:
    logits = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    logits_up = resize_logits(logits, (16, 20))
    mask_up = resize_binary_mask(mask, (16, 20), threshold=0.5)
    assert logits_up.shape == (16, 20)
    assert mask_up.shape == (16, 20)
    assert float(mask_up.mean()) > 0.0
