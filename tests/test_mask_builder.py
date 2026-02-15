from __future__ import annotations

import numpy as np

from videomatte_hq.mask_builder import build_prompt_mask_grabcut


def test_build_prompt_mask_grabcut_extracts_subject() -> None:
    h, w = 96, 96
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[20:76, 28:68] = np.array([230, 230, 230], dtype=np.uint8)

    alpha = build_prompt_mask_grabcut(
        frame_rgb_u8=frame,
        box_xyxy=(22, 14, 74, 82),
        fg_points=[(46, 48)],
        bg_points=[(8, 8), (90, 90)],
        point_radius=6,
        iter_count=5,
    )

    assert alpha.shape == (h, w)
    assert alpha.dtype == np.float32
    assert float(alpha[48, 46]) > 0.8  # center of subject
    assert float(alpha[5, 5]) < 0.2  # obvious background
    assert float(alpha.mean()) > 0.05

