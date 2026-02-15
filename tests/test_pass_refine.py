from __future__ import annotations

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.pass_refine import run_pass_refine


class DummySource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.frames[idx]

    def __len__(self) -> int:
        return len(self.frames)


def test_pass_refine_disabled_passthrough() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.float32)
    alpha = np.random.RandomState(0).rand(32, 32).astype(np.float32)
    conf = np.full((32, 32), 0.7, dtype=np.float32)

    cfg = VideoMatteConfig(refine={"enabled": False})
    out = run_pass_refine(DummySource([frame]), [alpha], [conf], cfg)
    assert np.allclose(out[0], alpha)


def test_pass_refine_updates_boundary_and_locks_confident_regions() -> None:
    h, w = 64, 64
    frame = np.zeros((h, w, 3), dtype=np.float32)
    frame[:, : w // 2, :] = 0.15
    frame[:, w // 2 :, :] = 0.85

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, w // 2 :] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0).astype(np.float32)

    conf = np.full((h, w), 0.9, dtype=np.float32)
    conf[:, 28:36] = 0.1  # uncertain boundary strip

    cfg = VideoMatteConfig(
        refine={
            "enabled": True,
            "backend": "guided_band",
            "unknown_band_px": 6,
            "tile_size": 32,
            "overlap": 8,
            "alpha_bg_threshold": 0.05,
            "alpha_fg_threshold": 0.95,
            "min_confidence": 0.5,
            "guided_radius": 4,
            "guided_eps": 0.01,
            "edge_boost": 0.2,
            "confidence_gain": 1.0,
            "tile_min_coverage": 0.001,
        }
    )

    out = run_pass_refine(DummySource([frame]), [alpha], [conf], cfg)[0]

    # Lock behavior in confident definite regions.
    assert np.allclose(out[:, :20], 0.0)
    assert np.allclose(out[:, 44:], 1.0)

    # Boundary should be actively refined.
    band_delta = np.abs(out[:, 28:36] - alpha[:, 28:36]).mean()
    assert float(band_delta) > 1e-3
