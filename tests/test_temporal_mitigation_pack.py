from __future__ import annotations

import cv2
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


def _build_clean_step_sequence(num_frames: int, h: int = 64, w: int = 96) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frames: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for t in range(num_frames):
        x = 34 + int(round(2.0 * np.sin(0.45 * t)))
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :x, :] = np.array([0.15, 0.18, 0.22], dtype=np.float32)
        rgb[:, x:, :] = np.array([0.82, 0.80, 0.76], dtype=np.float32)

        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[:, x:] = 1.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2).astype(np.float32)

        frames.append(rgb)
        alphas.append(alpha)
    return frames, alphas


def _inject_edge_jitter(clean: list[np.ndarray], seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for idx, alpha in enumerate(clean):
        a = alpha.copy()
        band = (a > 0.05) & (a < 0.95)
        noise = rng.normal(loc=0.0, scale=0.18, size=a.shape).astype(np.float32)
        a[band] = np.clip(a[band] + noise[band], 0.0, 1.0)
        if idx % 3 == 0:
            a = np.roll(a, shift=1, axis=1)
        if idx % 5 == 0:
            a[:, ::9] = np.clip(a[:, ::9] + 0.10, 0.0, 1.0)
        out.append(np.clip(a, 0.0, 1.0).astype(np.float32))
    return out


def _p95_edge_flicker(alphas: list[np.ndarray], lo: float = 0.05, hi: float = 0.95) -> float:
    values: list[float] = []
    for t in range(1, len(alphas)):
        curr = alphas[t]
        prev = alphas[t - 1]
        band = ((curr > lo) & (curr < hi)) | ((prev > lo) & (prev < hi))
        diff = np.abs(curr - prev)
        values.append(float(diff[band].mean()) if band.any() else float(diff.mean()))
    return float(np.percentile(np.asarray(values, dtype=np.float32), 95.0)) if values else 0.0


def _mean_mae(pred: list[np.ndarray], ref: list[np.ndarray]) -> float:
    vals = [float(np.mean(np.abs(np.asarray(p, dtype=np.float32) - np.asarray(r, dtype=np.float32)))) for p, r in zip(pred, ref)]
    return float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else 0.0


def test_mitigation_edge_band_ema_passes_threshold() -> None:
    frames, clean = _build_clean_step_sequence(num_frames=16)
    noisy = _inject_edge_jitter(clean, seed=11)
    conf = [np.full_like(a, 0.9, dtype=np.float32) for a in noisy]

    cfg_base = VideoMatteConfig(
        temporal_cleanup={
            "enabled": True,
            "outside_band_ema_enabled": True,
            "outside_band_ema": 0.20,
            "min_confidence": 0.4,
            "confidence_clamp_enabled": True,
            "clamp_delta": 0.20,
            "edge_band_ema_enabled": False,
            "edge_snap_enabled": False,
            "edge_bg_threshold": 0.05,
            "edge_fg_threshold": 0.95,
            "edge_band_radius_px": 1,
        }
    )

    cfg_edge_ema = VideoMatteConfig(
        temporal_cleanup={
            **cfg_base.temporal_cleanup.model_dump(),
            "edge_band_ema_enabled": True,
            "edge_band_ema": 0.10,
            "edge_band_min_confidence": 0.4,
        }
    )

    out_base = run_pass_temporal_cleanup(DummySource(frames), noisy, conf, cfg_base, anchor_frames={0})
    out_edge = run_pass_temporal_cleanup(DummySource(frames), noisy, conf, cfg_edge_ema, anchor_frames={0})

    base_flicker = _p95_edge_flicker(out_base)
    edge_flicker = _p95_edge_flicker(out_edge)
    base_mae = _mean_mae(out_base, clean)
    edge_mae = _mean_mae(out_edge, clean)

    assert edge_flicker <= base_flicker * 0.97
    assert edge_mae <= base_mae * 1.05


def test_mitigation_confidence_clamp_passes_threshold() -> None:
    h, w = 48, 64
    frame = np.zeros((h, w, 3), dtype=np.float32)
    frame[:, : w // 2, :] = 0.2
    frame[:, w // 2 :, :] = 0.8

    clean = np.zeros((h, w), dtype=np.float32)
    clean[:, w // 2 :] = 1.0
    clean = cv2.GaussianBlur(clean, (0, 0), sigmaX=1.0).astype(np.float32)

    spike = clean.copy()
    spike[6:20, 4:18] = 1.0
    spike[28:42, 6:20] = 1.0

    alphas = [clean.copy(), spike, clean.copy(), clean.copy()]
    conf = [np.full((h, w), 0.92, dtype=np.float32) for _ in alphas]

    cfg_no_clamp = VideoMatteConfig(
        temporal_cleanup={
            "enabled": True,
            "outside_band_ema_enabled": True,
            "outside_band_ema": 0.65,
            "min_confidence": 0.4,
            "confidence_clamp_enabled": False,
            "clamp_delta": 0.12,
            "edge_band_ema_enabled": False,
            "edge_snap_enabled": False,
            "edge_band_radius_px": 0,
        }
    )

    cfg_clamp = VideoMatteConfig(
        temporal_cleanup={
            **cfg_no_clamp.temporal_cleanup.model_dump(),
            "confidence_clamp_enabled": True,
        }
    )

    out_no_clamp = run_pass_temporal_cleanup(DummySource([frame] * len(alphas)), alphas, conf, cfg_no_clamp, anchor_frames={0})
    out_clamp = run_pass_temporal_cleanup(DummySource([frame] * len(alphas)), alphas, conf, cfg_clamp, anchor_frames={0})

    bg_mask = clean < 0.02
    leak_no_clamp = float(out_no_clamp[2][bg_mask].mean())
    leak_clamp = float(out_clamp[2][bg_mask].mean())

    assert leak_clamp <= leak_no_clamp * 0.70


def test_mitigation_edge_snap_passes_threshold() -> None:
    frames, clean = _build_clean_step_sequence(num_frames=14)
    noisy = _inject_edge_jitter(clean, seed=23)
    conf = [np.full_like(a, 0.9, dtype=np.float32) for a in noisy]

    cfg_base = VideoMatteConfig(
        temporal_cleanup={
            "enabled": True,
            "outside_band_ema_enabled": True,
            "outside_band_ema": 0.18,
            "min_confidence": 0.4,
            "confidence_clamp_enabled": True,
            "clamp_delta": 0.2,
            "edge_band_ema_enabled": True,
            "edge_band_ema": 0.08,
            "edge_band_min_confidence": 0.4,
            "edge_snap_enabled": False,
            "edge_band_radius_px": 1,
        }
    )

    cfg_snap = VideoMatteConfig(
        temporal_cleanup={
            **cfg_base.temporal_cleanup.model_dump(),
            "edge_snap_enabled": True,
            "edge_snap_radius": 2,
            "edge_snap_eps": 0.01,
            "edge_snap_min_confidence": 0.4,
        }
    )

    out_base = run_pass_temporal_cleanup(DummySource(frames), noisy, conf, cfg_base, anchor_frames={0})
    out_snap = run_pass_temporal_cleanup(DummySource(frames), noisy, conf, cfg_snap, anchor_frames={0})

    base_flicker = _p95_edge_flicker(out_base)
    snap_flicker = _p95_edge_flicker(out_snap)
    base_mae = _mean_mae(out_base, clean)
    snap_mae = _mean_mae(out_snap, clean)

    assert snap_flicker <= base_flicker * 1.02
    assert snap_mae <= base_mae * 1.03
