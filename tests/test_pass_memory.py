from __future__ import annotations

import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.pass_memory import run_pass_memory


class DummySource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.num_frames = len(frames)
        self.resolution = frames[0].shape[:2]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.frames[idx]

    def __len__(self) -> int:
        return len(self.frames)


def test_pass_memory_rejects_placeholder_backend() -> None:
    cfg = VideoMatteConfig(memory={"window": 10, "backend": "placeholder_nearest_keyframe"})
    kf0 = np.zeros((8, 8), dtype=np.float32)
    kf10 = np.ones((8, 8), dtype=np.float32)
    source = DummySource([np.zeros((8, 8, 3), dtype=np.float32) for _ in range(11)])

    with pytest.raises(ValueError, match="Unsupported memory backend"):
        run_pass_memory(
            source=source,
            keyframe_masks={0: kf0, 10: kf10},
            cfg=cfg,
        )


def test_pass_memory_memory_bank_tracks_moving_subject() -> None:
    h, w = 64, 64
    size = 14
    frames: list[np.ndarray] = []

    for t in range(6):
        frame = np.zeros((h, w, 3), dtype=np.float32)
        x0 = 4 + t * 8
        y0 = 24
        frame[y0:y0 + size, x0:x0 + size, :] = 1.0
        frames.append(frame)

    source = DummySource(frames)

    keyframe_mask = np.zeros((h, w), dtype=np.float32)
    keyframe_mask[24:24 + size, 4:4 + size] = 1.0

    cfg = VideoMatteConfig(
        memory={
            "backend": "appearance_memory_bank",
            "memory_frames": 6,
            "max_anchors": 6,
            "window": 24,
            "query_long_side": 64,
            "spatial_weight": 0.0,
            "temperature": 1.0,
            "confidence_reanchor_threshold": 0.25,
        }
    )

    alphas, _confs = run_pass_memory(
        source=source,
        keyframe_masks={0: keyframe_mask},
        cfg=cfg,
    )

    first_old_x = 4 + size // 2
    last_old_x = first_old_x
    last_new_x = 4 + 5 * 8 + size // 2
    y = 24 + size // 2

    # Subject should be detected at its new location by the last frame.
    assert float(alphas[5][y, last_new_x]) > 0.6
    # Old location should no longer be foreground.
    assert float(alphas[5][y, last_old_x]) < 0.4
