from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from videomatte_hq.pipeline.stage_refine import RefineStageConfig, refine_sequence


class DummySource:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._frames[index]

    @property
    def resolution(self) -> tuple[int, int]:
        return self._frames[0].shape[:2]


@dataclass
class CountingRefiner:
    calls: int = 0

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        self.calls += 1
        alpha = np.full(trimap.shape, 0.25, dtype=np.float32)
        alpha[trimap >= 1.0] = 1.0
        alpha[trimap <= 0.0] = 0.0
        return alpha


def test_refine_sequence_skip_reuse_behavior() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.float32) for _ in range(4)]
    source = DummySource(frames)

    coarse_masks = []
    coarse_logits = []
    for _ in range(4):
        m = np.zeros((32, 32), dtype=np.float32)
        m[8:24, 8:24] = 1.0
        coarse_masks.append(m)
        coarse_logits.append(np.zeros((32, 32), dtype=np.float32))  # 0.5 prob -> unknown

    cfg = RefineStageConfig(
        refine_enabled=True,
        tile_size=64,
        tile_overlap=0,
        skip_iou_threshold=0.9,
    )
    refiner = CountingRefiner()

    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=cfg,
        refiner=refiner,
    )

    assert len(result.alphas) == 4
    assert result.reused_frames == [1, 2, 3]
    assert refiner.calls == 1


def test_refine_sequence_preview_mode_returns_cleaned_alphas() -> None:
    """When refine_enabled=False, preview alphas are derived from coarse masks."""
    frames = [np.zeros((64, 64, 3), dtype=np.float32)]
    source = DummySource(frames)

    coarse_mask = np.zeros((64, 64), dtype=np.float32)
    coarse_mask[12:56, 12:56] = 1.0
    coarse_masks = [coarse_mask]

    coarse_logits = [np.full((64, 64), -3.0, dtype=np.float32)]
    coarse_logits[0][coarse_mask >= 0.5] = 3.0

    cfg = RefineStageConfig(refine_enabled=False)

    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=cfg,
        refiner=None,
    )

    assert len(result.alphas) == 1
    assert result.reused_frames == []
    alpha = result.alphas[0]
    # Preview alpha should be binary-ish (cleaned up from mask).
    assert float(alpha[32, 32]) > 0.5  # inside mask
    assert float(alpha[0, 0]) < 0.5  # outside mask


def test_refine_sequence_morphological_mode_uses_mask_trimap() -> None:
    """Morphological mode should generate a wider trimap and call the refiner."""
    frames = [np.zeros((64, 64, 3), dtype=np.float32)]
    source = DummySource(frames)

    coarse_mask = np.zeros((64, 64), dtype=np.float32)
    coarse_mask[15:50, 15:50] = 1.0
    coarse_masks = [coarse_mask]

    # Strong logits that would fail with logit-mode (no unknown band)
    coarse_logits = [np.full((64, 64), 8.0, dtype=np.float32)]

    cfg = RefineStageConfig(
        refine_enabled=True,
        trimap_mode="morphological",
        trimap_erosion_px=5,
        trimap_dilation_px=5,
        tile_size=128,
        tile_overlap=0,
    )
    refiner = CountingRefiner()

    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=cfg,
        refiner=refiner,
    )

    assert len(result.alphas) == 1
    assert refiner.calls >= 1


def test_refine_sequence_logit_mode_uses_logits() -> None:
    """Logit mode should use the legacy trimap path."""
    frames = [np.zeros((32, 32, 3), dtype=np.float32)]
    source = DummySource(frames)

    coarse_mask = np.zeros((32, 32), dtype=np.float32)
    coarse_mask[8:24, 8:24] = 1.0
    coarse_masks = [coarse_mask]

    # Logits at 0 => 0.5 prob => all unknown under default thresholds
    coarse_logits = [np.zeros((32, 32), dtype=np.float32)]

    cfg = RefineStageConfig(
        refine_enabled=True,
        trimap_mode="logit",
        tile_size=64,
        tile_overlap=0,
    )
    refiner = CountingRefiner()

    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=cfg,
        refiner=refiner,
    )

    assert len(result.alphas) == 1
    assert refiner.calls >= 1


def test_refine_sequence_fails_if_mematte_never_executes_tiles() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.float32)]
    source = DummySource(frames)

    coarse_mask = np.zeros((32, 32), dtype=np.float32)
    coarse_mask[8:24, 8:24] = 1.0
    coarse_masks = [coarse_mask]

    # Strong foreground logits produce no unknown-band pixels under logit thresholds.
    coarse_logits = [np.full((32, 32), 8.0, dtype=np.float32)]

    cfg = RefineStageConfig(refine_enabled=True, trimap_mode="logit", tile_size=64, tile_overlap=0)
    refiner = CountingRefiner()

    with pytest.raises(RuntimeError, match="MEMatte did not execute"):
        refine_sequence(
            source=source,
            coarse_masks=coarse_masks,
            coarse_logits=coarse_logits,
            cfg=cfg,
            refiner=refiner,
        )
