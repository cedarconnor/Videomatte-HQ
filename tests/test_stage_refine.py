from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest

from videomatte_hq.pipeline.stage_refine import (
    RefineStageConfig,
    _apply_unknown_edge_blend,
    _build_frame_trimap_and_prior,
    refine_sequence,
)


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


class TrackingSource(DummySource):
    def __init__(self, frames: list[np.ndarray]):
        super().__init__(frames)
        self.requested_indices: list[int] = []

    def __getitem__(self, index: int) -> np.ndarray:
        self.requested_indices.append(index)
        return super().__getitem__(index)


@dataclass
class CountingRefiner:
    calls: int = 0

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        self.calls += 1
        alpha = np.full(trimap.shape, 0.25, dtype=np.float32)
        alpha[trimap >= 1.0] = 1.0
        alpha[trimap <= 0.0] = 0.0
        return alpha


@dataclass
class ZeroRefiner:
    calls: int = 0

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.zeros(trimap.shape, dtype=np.float32)


@dataclass
class BatchCountingRefiner:
    calls: int = 0
    batch_calls: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    batch_shapes: list[tuple[int, int]] = field(default_factory=list)

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        self.calls += 1
        alpha = np.full(trimap.shape, 0.25, dtype=np.float32)
        alpha[trimap >= 1.0] = 1.0
        alpha[trimap <= 0.0] = 0.0
        return alpha

    def refine_batch(self, rgb_tiles: list[np.ndarray], trimap_tiles: list[np.ndarray]) -> list[np.ndarray]:
        self.batch_calls += 1
        self.batch_sizes.append(len(trimap_tiles))
        self.batch_shapes.extend(tuple(trimap.shape) for trimap in trimap_tiles)
        return [self.refine(rgb, trimap) for rgb, trimap in zip(rgb_tiles, trimap_tiles)]


def test_refine_sequence_skip_reuse_behavior() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.float32) for _ in range(4)]
    source = TrackingSource(frames)

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
    assert source.requested_indices == [0]


def test_refine_sequence_preview_mode_returns_cleaned_alphas() -> None:
    """When refine_enabled=False, preview alphas are derived from coarse masks."""
    frames = [np.zeros((64, 64, 3), dtype=np.float32)]
    source = TrackingSource(frames)

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
    assert source.requested_indices == []


def test_refine_sequence_trimap_callback_runs_for_reused_frames() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.float32) for _ in range(4)]
    source = TrackingSource(frames)
    coarse_masks = []
    coarse_logits = []
    for _ in range(4):
        m = np.zeros((32, 32), dtype=np.float32)
        m[8:24, 8:24] = 1.0
        coarse_masks.append(m)
        coarse_logits.append(np.zeros((32, 32), dtype=np.float32))

    cfg = RefineStageConfig(refine_enabled=True, tile_size=64, tile_overlap=0, skip_iou_threshold=0.9)
    refiner = CountingRefiner()
    callback_frames: list[int] = []

    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=cfg,
        refiner=refiner,
        trimap_callback=lambda frame_idx, trimap: callback_frames.append(frame_idx),
    )

    assert result.reused_frames == [1, 2, 3]
    assert callback_frames == [0, 1, 2, 3]
    assert refiner.calls == 1
    assert source.requested_indices == [0]


def test_refine_sequence_preview_mode_can_emit_trimap_without_frame_reads() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.float32)]
    source = TrackingSource(frames)

    coarse_mask = np.zeros((32, 32), dtype=np.float32)
    coarse_mask[8:24, 8:24] = 1.0
    coarse_masks = [coarse_mask]
    coarse_logits = [np.zeros((32, 32), dtype=np.float32)]
    callback_shapes: list[tuple[int, int]] = []

    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=RefineStageConfig(refine_enabled=False),
        refiner=None,
        trimap_callback=lambda frame_idx, trimap: callback_shapes.append(tuple(trimap.shape)),
    )

    assert len(result.alphas) == 1
    assert callback_shapes == [(32, 32)]
    assert source.requested_indices == []


def test_refine_sequence_batches_same_sized_tiles_in_chunks() -> None:
    frames = [np.zeros((128, 128, 3), dtype=np.float32)]
    source = DummySource(frames)
    coarse_masks = [np.zeros((128, 128), dtype=np.float32)]
    coarse_logits = [np.zeros((128, 128), dtype=np.float32)]

    refiner = BatchCountingRefiner()
    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=RefineStageConfig(
            refine_enabled=True,
            trimap_mode="logit",
            tile_size=64,
            tile_overlap=0,
            tile_batch_size=2,
        ),
        refiner=refiner,
    )

    assert len(result.alphas) == 1
    assert refiner.calls == 4
    assert refiner.batch_calls == 2
    assert refiner.batch_sizes == [2, 2]
    assert np.allclose(result.alphas[0], 0.25, atol=1e-6)


def test_refine_sequence_groups_batches_by_tile_shape() -> None:
    frames = [np.zeros((96, 128, 3), dtype=np.float32)]
    source = DummySource(frames)
    coarse_masks = [np.zeros((96, 128), dtype=np.float32)]
    coarse_logits = [np.zeros((96, 128), dtype=np.float32)]

    refiner = BatchCountingRefiner()
    refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=RefineStageConfig(
            refine_enabled=True,
            trimap_mode="logit",
            tile_size=64,
            tile_overlap=0,
            tile_batch_size=4,
        ),
        refiner=refiner,
    )

    assert refiner.batch_calls == 2
    assert refiner.batch_sizes == [2, 2]
    assert refiner.batch_shapes == [(64, 64), (64, 64), (32, 64), (32, 64)]


def test_refine_sequence_multitile_generic_refiner_uses_single_tile_fallback() -> None:
    frames = [np.zeros((128, 128, 3), dtype=np.float32)]
    source = DummySource(frames)
    coarse_masks = [np.zeros((128, 128), dtype=np.float32)]
    coarse_logits = [np.zeros((128, 128), dtype=np.float32)]

    refiner = CountingRefiner()
    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=RefineStageConfig(
            refine_enabled=True,
            trimap_mode="logit",
            tile_size=64,
            tile_overlap=0,
            tile_batch_size=2,
        ),
        refiner=refiner,
    )

    assert len(result.alphas) == 1
    assert refiner.calls == 4
    assert np.allclose(result.alphas[0], 0.25, atol=1e-6)


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


def test_morphological_mode_uses_logits_for_soft_prior() -> None:
    mask_up = np.zeros((32, 32), dtype=np.float32)
    mask_up[8:24, 8:24] = 1.0

    coarse_logit = np.full((32, 32), -2.0, dtype=np.float32)
    coarse_logit[8:24, 8:24] = 2.0

    trimap, coarse_prob = _build_frame_trimap_and_prior(
        mask_up,
        coarse_logit,
        mask_up.shape,
        RefineStageConfig(trimap_mode="morphological", trimap_erosion_px=3, trimap_dilation_px=3),
    )

    assert trimap.shape == mask_up.shape
    assert coarse_prob.shape == mask_up.shape
    assert float(coarse_prob[16, 16]) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)), abs=1e-4)
    assert float(coarse_prob[0, 0]) == pytest.approx(1.0 / (1.0 + np.exp(2.0)), abs=1e-4)
    assert 0.0 < float(coarse_prob[16, 16]) < 1.0
    assert 0.0 < float(coarse_prob[0, 0]) < 1.0


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


def test_refine_sequence_hybrid_mode_uses_hybrid_trimap_path() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.float32)]
    source = DummySource(frames)

    coarse_mask = np.zeros((32, 32), dtype=np.float32)
    coarse_mask[8:24, 8:24] = 1.0
    coarse_masks = [coarse_mask]

    coarse_prob = np.zeros((32, 32), dtype=np.float32)
    coarse_prob[8:24, 8:24] = 0.99
    coarse_prob[6:26, 6:26] = np.maximum(coarse_prob[6:26, 6:26], 0.5)
    coarse_logits = [np.log(np.clip(coarse_prob, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - coarse_prob, 1e-6, 1.0 - 1e-6)).astype(np.float32)]

    callback_trimaps: list[np.ndarray] = []
    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=RefineStageConfig(
            refine_enabled=True,
            trimap_mode="hybrid",
            trimap_erosion_px=1,
            trimap_dilation_px=1,
            tile_size=64,
            tile_overlap=0,
        ),
        refiner=CountingRefiner(),
        trimap_callback=lambda frame_idx, trimap: callback_trimaps.append(trimap.copy()),
    )

    assert len(result.alphas) == 1
    assert len(callback_trimaps) == 1
    assert float(callback_trimaps[0][6, 16]) == 0.5


def test_unknown_edge_blend_biases_boundary_back_to_prior() -> None:
    trimap = np.zeros((13, 13), dtype=np.float32)
    trimap[2:11, 2:11] = 0.5
    trimap[5:8, 5:8] = 1.0

    refined = np.zeros_like(trimap, dtype=np.float32)
    coarse_prob = np.zeros_like(trimap, dtype=np.float32)
    coarse_prob[trimap == 0.5] = 1.0

    blended = _apply_unknown_edge_blend(
        refined,
        coarse_prob,
        trimap,
        RefineStageConfig(unknown_edge_blend_px=2),
    )

    assert float(blended[2, 6]) == pytest.approx(0.5, abs=0.08)
    assert float(blended[3, 6]) < 0.05
    assert float(blended[2, 6]) > float(blended[3, 6])
    assert float(blended[6, 6]) == 1.0
    assert float(blended[0, 0]) == 0.0


def test_refine_sequence_unknown_edge_blend_damps_boundary_more_than_band_center() -> None:
    frames = [np.zeros((64, 64, 3), dtype=np.float32)]
    source = DummySource(frames)

    coarse_mask = np.zeros((64, 64), dtype=np.float32)
    coarse_mask[18:46, 18:46] = 1.0
    coarse_masks = [coarse_mask]

    coarse_logits = [np.full((64, 64), -2.0, dtype=np.float32)]
    coarse_logits[0][18:46, 18:46] = 2.0

    cfg = RefineStageConfig(
        refine_enabled=True,
        trimap_mode="morphological",
        trimap_erosion_px=4,
        trimap_dilation_px=4,
        tile_size=128,
        tile_overlap=0,
        unknown_edge_blend_px=2,
    )
    result = refine_sequence(
        source=source,
        coarse_masks=coarse_masks,
        coarse_logits=coarse_logits,
        cfg=cfg,
        refiner=ZeroRefiner(),
    )

    trimap, _ = _build_frame_trimap_and_prior(
        coarse_mask,
        coarse_logits[0],
        coarse_mask.shape,
        cfg,
    )
    unknown = trimap == 0.5
    dist = cv2.distanceTransform(unknown.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    boundary_y, boundary_x = np.argwhere(np.logical_and(unknown, dist <= 1.05))[0]
    center_y, center_x = np.unravel_index(int(np.argmax(dist)), dist.shape)

    alpha = result.alphas[0]
    assert float(alpha[boundary_y, boundary_x]) > 0.0
    assert float(alpha[boundary_y, boundary_x]) > float(alpha[center_y, center_x])


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
