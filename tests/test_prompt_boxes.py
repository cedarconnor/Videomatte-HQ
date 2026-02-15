from __future__ import annotations

import numpy as np

from videomatte_hq.prompt_boxes import suggest_prompt_boxes


def test_suggest_prompt_boxes_returns_candidates_for_synthetic_subject() -> None:
    h, w = 100, 140
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[24:88, 42:96, :] = 235

    cands = suggest_prompt_boxes(frame_rgb_u8=frame, prompt="object center", max_candidates=5)

    assert len(cands) >= 1
    top = cands[0]
    assert top.x1 > top.x0
    assert top.y1 > top.y0
    assert 0 <= top.x0 < w
    assert 0 <= top.x1 < w
    assert 0 <= top.y0 < h
    assert 0 <= top.y1 < h
