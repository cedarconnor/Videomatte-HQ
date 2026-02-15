from __future__ import annotations

import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.pass_temporal_cleanup import run_pass_temporal_cleanup


class DummySource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.frames[idx]

    def __len__(self) -> int:
        return len(self.frames)


def _edge_band(alpha: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    return (alpha > lo) & (alpha < hi)


def test_temporal_cleanup_preserves_edge_band() -> None:
    h, w = 64, 64
    frame = np.zeros((h, w, 3), dtype=np.float32)
    frame[:, : w // 2] = 0.2
    frame[:, w // 2 :] = 0.8

    alpha0 = np.zeros((h, w), dtype=np.float32)
    alpha0[:, w // 2 :] = 1.0
    alpha0[:, 31:33] = 0.5

    alpha1 = np.zeros((h, w), dtype=np.float32)
    alpha1[:, w // 2 + 1 :] = 1.0
    alpha1[:, 32:34] = 0.5

    conf0 = np.ones((h, w), dtype=np.float32)
    conf1 = np.ones((h, w), dtype=np.float32)

    cfg = VideoMatteConfig(
        temporal_cleanup={
            "enabled": True,
            "outside_band_ema": 1.0,
            "min_confidence": 0.0,
            "edge_bg_threshold": 0.05,
            "edge_fg_threshold": 0.95,
            "edge_band_radius_px": 0,
            "edge_snap_enabled": False,
            "clamp_delta": 1.0,
        }
    )

    out = run_pass_temporal_cleanup(
        source=DummySource([frame, frame]),
        alphas=[alpha0, alpha1],
        confidences=[conf0, conf1],
        cfg=cfg,
        anchor_frames={0},
    )

    band = _edge_band(alpha1)
    assert np.allclose(out[1][band], alpha1[band])


def test_temporal_cleanup_smooths_outside_band_when_confident() -> None:
    h, w = 32, 32
    frame = np.zeros((h, w, 3), dtype=np.float32)

    alpha0 = np.zeros((h, w), dtype=np.float32)
    alpha1 = np.ones((h, w), dtype=np.float32) * 0.4
    alpha1[:, 15:17] = 0.5  # edge band strip

    conf0 = np.ones((h, w), dtype=np.float32)
    conf1 = np.ones((h, w), dtype=np.float32)

    cfg = VideoMatteConfig(
        temporal_cleanup={
            "enabled": True,
            "outside_band_ema": 0.5,
            "min_confidence": 0.0,
            "edge_bg_threshold": 0.05,
            "edge_fg_threshold": 0.95,
            "edge_band_radius_px": 0,
            "edge_snap_enabled": False,
            "clamp_delta": 1.0,
        }
    )

    out = run_pass_temporal_cleanup(
        source=DummySource([frame, frame]),
        alphas=[alpha0, alpha1],
        confidences=[conf0, conf1],
        cfg=cfg,
        anchor_frames={0},
    )

    expected = alpha1 * 0.5 + alpha0 * 0.5
    non_edge = ~_edge_band(alpha1)
    assert np.allclose(out[1][non_edge], expected[non_edge], atol=1e-6)


def test_temporal_cleanup_low_confidence_prefers_current() -> None:
    h, w = 32, 32
    frame = np.zeros((h, w, 3), dtype=np.float32)

    alpha0 = np.ones((h, w), dtype=np.float32)
    alpha1 = np.zeros((h, w), dtype=np.float32)
    alpha1[:, 15:17] = 0.5

    conf0 = np.zeros((h, w), dtype=np.float32) + 0.2
    conf1 = np.zeros((h, w), dtype=np.float32) + 0.2

    cfg = VideoMatteConfig(
        temporal_cleanup={
            "enabled": True,
            "outside_band_ema": 0.9,
            "min_confidence": 0.5,
            "edge_bg_threshold": 0.05,
            "edge_fg_threshold": 0.95,
            "edge_band_radius_px": 0,
            "edge_snap_enabled": False,
            "clamp_delta": 1.0,
        }
    )

    out = run_pass_temporal_cleanup(
        source=DummySource([frame, frame]),
        alphas=[alpha0, alpha1],
        confidences=[conf0, conf1],
        cfg=cfg,
        anchor_frames={0},
    )

    non_edge = ~_edge_band(alpha1)
    assert np.allclose(out[1][non_edge], alpha1[non_edge], atol=1e-6)
