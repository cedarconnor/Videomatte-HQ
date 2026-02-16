from __future__ import annotations

import cv2
import numpy as np
import torch

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


def test_pass_refine_guided_trimap_clamps_outside_and_locks_sure_fg() -> None:
    h, w = 80, 80
    frame = np.zeros((h, w, 3), dtype=np.float32)
    frame[:, :, 0] = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    alpha = np.full((h, w), 0.5, dtype=np.float32)
    conf = np.full((h, w), 0.2, dtype=np.float32)

    guidance = np.zeros((h, w), dtype=np.float32)
    guidance[20:60, 26:54] = 1.0

    cfg = VideoMatteConfig(
        refine={
            "enabled": True,
            "backend": "guided_band",
            "unknown_band_px": 6,
            "tile_size": 40,
            "overlap": 8,
            "alpha_bg_threshold": 0.05,
            "alpha_fg_threshold": 0.95,
            "min_confidence": 0.5,
            "guided_radius": 4,
            "guided_eps": 0.01,
            "edge_boost": 0.2,
            "confidence_gain": 1.0,
            "tile_min_coverage": 0.001,
            "region_trimap_enabled": True,
            "region_trimap_threshold": 0.5,
            "region_trimap_fg_erode_px": 3,
            "region_trimap_bg_dilate_px": 8,
            "region_trimap_cleanup_px": 0,
            "region_trimap_keep_largest": True,
            "region_trimap_min_coverage": 0.002,
            "region_trimap_max_coverage": 0.98,
        }
    )

    out = run_pass_refine(
        DummySource([frame]),
        [alpha],
        [conf],
        cfg,
        region_guidance_masks=[guidance],
    )[0]

    fg_binary = guidance >= 0.5
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    sure_fg = cv2.erode(fg_binary.astype(np.uint8), k_erode).astype(bool)
    loose_fg = cv2.dilate(fg_binary.astype(np.uint8), k_dilate).astype(bool)

    assert np.allclose(out[~loose_fg], 0.0)
    assert np.allclose(out[sure_fg], 1.0)


def test_pass_refine_mematte_backend_runs_with_mock_refiner(monkeypatch) -> None:
    h, w = 48, 48
    frame = np.zeros((h, w, 3), dtype=np.float32)
    frame[..., 0] = 0.2
    frame[..., 1] = 0.4
    frame[..., 2] = 0.6

    alpha = np.full((h, w), 0.5, dtype=np.float32)
    conf = np.full((h, w), 0.05, dtype=np.float32)

    class DummyRefiner:
        def infer_tile(self, rgb_tile, trimap_tile, alpha_prior, bg_tile=None):  # noqa: D401
            _, th, tw = rgb_tile.shape
            out = torch.full((1, th, tw), 0.8, dtype=torch.float32)
            out[trimap_tile <= 0.0] = 0.0
            out[trimap_tile >= 1.0] = 1.0
            return out

    monkeypatch.setattr(
        "videomatte_hq.pipeline.pass_refine._load_mematte_refiner",
        lambda _cfg: DummyRefiner(),
    )

    cfg = VideoMatteConfig(
        refine={
            "enabled": True,
            "backend": "mematte",
            "unknown_band_px": 8,
            "tile_size": 24,
            "overlap": 8,
            "alpha_bg_threshold": 0.05,
            "alpha_fg_threshold": 0.95,
            "min_confidence": 0.5,
            "confidence_gain": 1.0,
            "tile_min_coverage": 0.001,
        }
    )

    out = run_pass_refine(DummySource([frame]), [alpha], [conf], cfg)[0]
    assert out.shape == alpha.shape
    # With low confidence and unknown band everywhere, output should move toward dummy 0.8.
    assert float(np.mean(out)) > 0.65
