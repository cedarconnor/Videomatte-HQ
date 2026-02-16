from __future__ import annotations

import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.memory_region_constraint import build_memory_region_priors
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


def test_build_memory_region_priors_propagated_bbox() -> None:
    h, w = 64, 64
    frames: list[np.ndarray] = []
    for t in range(5):
        frame = np.zeros((h, w, 3), dtype=np.float32)
        x0 = 8 + t * 6
        frame[16:48, x0 : x0 + 16, :] = 1.0
        frames.append(frame)
    source = DummySource(frames)

    keyframe = np.zeros((h, w), dtype=np.float32)
    keyframe[16:48, 8:24] = 1.0

    cfg = VideoMatteConfig(
        memory={
            "region_constraint_enabled": True,
            "region_constraint_source": "propagated_bbox",
            "region_constraint_backend": "flow",
            "region_constraint_flow_downscale": 1.0,
            "region_constraint_flow_min_coverage": 0.001,
            "region_constraint_flow_max_coverage": 0.8,
            "region_constraint_threshold": 0.2,
            "region_constraint_bbox_margin_px": 4,
            "region_constraint_bbox_expand_ratio": 0.0,
            "region_constraint_dilate_px": 0,
        }
    )

    result = build_memory_region_priors(
        source=source,
        keyframe_masks={0: keyframe},
        cfg=cfg,
    )

    assert result is not None
    assert len(result.priors) == len(frames)
    assert result.guidance_masks is not None
    assert len(result.guidance_masks) == len(frames)
    assert result.backend_used in {"flow", "flow_fallback"}
    assert 0.02 <= float(result.mean_coverage) <= 0.8
    assert all(mask.shape == (h, w) for mask in result.guidance_masks)
    assert all(0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0 for mask in result.guidance_masks)


def test_pass_memory_region_prior_clamps_outside_roi() -> None:
    h, w = 16, 16
    source = DummySource([np.zeros((h, w, 3), dtype=np.float32) for _ in range(2)])
    keyframe = np.ones((h, w), dtype=np.float32)

    roi = np.zeros((h, w), dtype=np.float32)
    roi[:, : w // 2] = 1.0
    region_priors = [roi.copy(), roi.copy()]

    cfg = VideoMatteConfig(memory={"backend": "placeholder_nearest_keyframe", "window": 8})
    alphas, confs = run_pass_memory(
        source=source,
        keyframe_masks={0: keyframe},
        cfg=cfg,
        region_priors=region_priors,
    )

    assert np.allclose(alphas[0][:, : w // 2], 1.0)
    assert np.allclose(alphas[0][:, w // 2 :], 0.0)
    assert float(confs[0][:, w // 2 :].max()) <= float(cfg.memory.region_constraint_outside_confidence_cap) + 1e-6
