"""Tests for MatAnyone2 stage module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videomatte_hq.pipeline.stage_matanyone2 import (
    MatAnyone2StageConfig,
    MatAnyone2StageResult,
    run_matanyone2_stage,
)


def _make_fake_frames(count: int, h: int = 480, w: int = 640) -> list[np.ndarray]:
    """Create dummy RGB uint8 frames."""
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(count)]


def _make_fake_mask(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a simple centered circle mask."""
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = 1.0
    return mask


class FakeMatAnyone2Model:
    """Mock MatAnyone2 model that returns deterministic alphas."""

    def __init__(self, **kwargs):
        self.loaded = False
        self.kwargs = kwargs

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def process_video(self, frames, first_frame_mask, progress_callback=None):
        # Return alpha as a simple threshold of the mask, resized
        h, w = frames[0].shape[:2]
        max_size = self.kwargs.get("max_size", 1080)
        short_edge = min(h, w)
        if short_edge > max_size:
            scale = max_size / short_edge
            proc_h = int(round(h * scale))
            proc_w = int(round(w * scale))
        else:
            proc_h, proc_w = h, w

        alphas = []
        for i, frame in enumerate(frames):
            alpha = np.ones((proc_h, proc_w), dtype=np.float32) * 0.5
            alpha[proc_h // 4 : 3 * proc_h // 4, proc_w // 4 : 3 * proc_w // 4] = 1.0
            alphas.append(alpha)
            if progress_callback:
                progress_callback(i + 1, len(frames))
        return alphas


def _patched_run(frames, mask, cfg, progress_callback=None):
    """Run matanyone2 stage with the fake model injected."""
    import videomatte_hq.models.matanyone2_wrapper as wrapper_mod
    original_cls = wrapper_mod.MatAnyone2Model
    wrapper_mod.MatAnyone2Model = FakeMatAnyone2Model
    try:
        return run_matanyone2_stage(frames, mask, cfg, progress_callback=progress_callback)
    finally:
        wrapper_mod.MatAnyone2Model = original_cls


def test_run_matanyone2_stage_basic() -> None:
    frames = _make_fake_frames(5, 480, 640)
    mask = _make_fake_mask(480, 640)
    cfg = MatAnyone2StageConfig(max_size=1080)

    result = _patched_run(frames, mask, cfg)

    assert isinstance(result, MatAnyone2StageResult)
    assert len(result.alphas) == 5
    assert result.native_resolution == (480, 640)
    assert result.processing_resolution[0] > 0
    assert result.processing_resolution[1] > 0
    for alpha in result.alphas:
        assert alpha.dtype == np.float32
        assert 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 1.0


def test_run_matanyone2_stage_progress_callback() -> None:
    frames = _make_fake_frames(3, 480, 640)
    mask = _make_fake_mask(480, 640)
    cfg = MatAnyone2StageConfig(max_size=1080)

    progress_calls: list[tuple[int, int]] = []

    def on_progress(current, total):
        progress_calls.append((current, total))

    result = _patched_run(frames, mask, cfg, progress_callback=on_progress)

    assert len(result.alphas) == 3
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3)


def test_run_matanyone2_stage_4k_downscales() -> None:
    """4K input should be downscaled to max_size for processing."""
    frames = _make_fake_frames(2, 2160, 3840)
    mask = _make_fake_mask(2160, 3840)
    cfg = MatAnyone2StageConfig(max_size=1080)

    result = _patched_run(frames, mask, cfg)

    assert result.native_resolution == (2160, 3840)
    # Processing resolution should be <= max_size on short edge
    proc_h, proc_w = result.processing_resolution
    assert min(proc_h, proc_w) <= 1080


def test_run_matanyone2_stage_empty_frames_raises() -> None:
    cfg = MatAnyone2StageConfig()
    with pytest.raises(ValueError, match="No frames"):
        run_matanyone2_stage([], np.zeros((10, 10), dtype=np.float32), cfg)


def test_matanyone2_stage_config_defaults() -> None:
    cfg = MatAnyone2StageConfig()
    assert cfg.max_size == 1080
    assert cfg.warmup == 10
    assert cfg.device == "cuda"
