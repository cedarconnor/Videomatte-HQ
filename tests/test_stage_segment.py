from __future__ import annotations

import cv2
import numpy as np

from videomatte_hq.pipeline.stage_qc import compute_iou
from videomatte_hq.pipeline.stage_segment import (
    ChunkedSegmenter,
    StaticMaskSegmentBackend,
    UltralyticsSAM3SegmentBackend,
    _apply_anchor_reference_area_guard,
    _apply_temporal_area_guard,
    _apply_strict_background_suppression,
    _filter_probability_by_prev_component,
    _bbox_from_mask,
    _normalize_precision,
    _ordered_sam_model_candidates,
    _select_mask_candidate,
)
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter
from videomatte_hq.protocols import SegmentPrompt


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


class DummyVideoSource(DummySource):
    def __init__(self, frames: list[np.ndarray], start: int = 0):
        super().__init__(frames)
        self._start = int(start)

    @property
    def is_video(self) -> bool:
        return True

    @property
    def video_path(self) -> str:
        return "dummy.mp4"

    @property
    def video_frame_start(self) -> int:
        return self._start


def test_chunked_segmenter_static_backend_chunk_overlap() -> None:
    frames = [np.zeros((64, 64, 3), dtype=np.float32) for _ in range(10)]
    source = DummySource(frames)

    anchor_mask = np.zeros((64, 64), dtype=np.float32)
    anchor_mask[16:48, 20:44] = 1.0
    prompt = MaskPromptAdapter().adapt(anchor_mask, frame_shape=anchor_mask.shape)

    segmenter = ChunkedSegmenter(
        backend=StaticMaskSegmentBackend(),
        processing_long_side=64,
        max_reanchors_per_chunk=1,
    )
    result = segmenter.segment_sequence(
        source=source,
        prompt=prompt,
        anchor_frame=0,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert len(result.masks) == 10
    assert len(result.logits) == 10
    assert result.anchored_frames == [0, 3, 6, 9]
    assert compute_iou(result.masks[0], result.masks[-1]) > 0.99


def test_bbox_from_mask_supports_expansion() -> None:
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 20:44] = 1.0
    bbox = _bbox_from_mask(mask, expand_ratio=0.1, min_expand_px=6)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 <= 14.0 and y0 <= 10.0
    assert x1 >= 50.0 and y1 >= 54.0


def test_filter_probability_by_prev_component_removes_disconnected_blob() -> None:
    prev = np.zeros((64, 64), dtype=np.float32)
    prev[16:48, 20:44] = 1.0

    prob = np.zeros((64, 64), dtype=np.float32)
    # main person region
    prob[16:48, 22:46] = 0.9
    # disconnected drift blob far on the left
    prob[24:40, 0:8] = 0.9

    filtered = _filter_probability_by_prev_component(prob, prev, threshold=0.5)
    assert float(filtered[:, 0:8].max()) == 0.0
    assert float(filtered[24:40, 22:46].mean()) > 0.5


def test_strict_background_suppression_clamps_far_motion() -> None:
    prev = np.zeros((128, 128), dtype=np.float32)
    prev[40:100, 48:96] = 1.0

    prob = np.zeros((128, 128), dtype=np.float32)
    # valid near-subject region
    prob[42:102, 50:98] = 0.9
    # far moving blob that should be rejected in strict mode
    prob[52:92, 0:24] = 0.9

    strict = _apply_strict_background_suppression(
        prob,
        prev,
        threshold=0.5,
        bbox_expand_ratio=0.05,
        min_bbox_expand_px=8,
        overlap_dilate_ratio=0.02,
        min_overlap_dilate_px=6,
    )
    assert float(strict[:, 0:24].max()) == 0.0
    assert float(strict[42:102, 50:98].mean()) > 0.5


def test_select_mask_candidate_prefers_prompt_match_over_largest_area() -> None:
    arr = np.zeros((2, 32, 32), dtype=np.float32)
    # candidate 0: small centered person-like region
    arr[0, 10:22, 12:20] = 1.0
    # candidate 1: large background-like region
    arr[1, :, :] = 1.0
    arr[1, 9:23, 11:21] = 0.0

    prompt_mask = np.zeros((32, 32), dtype=np.float32)
    prompt_mask[10:22, 12:20] = 1.0
    prompt = SegmentPrompt(
        bbox=(12.0, 10.0, 20.0, 22.0),
        positive_points=[(16.0, 16.0)],
        negative_points=[(2.0, 2.0)],
        mask=prompt_mask,
    )

    selected = _select_mask_candidate(arr, prompt=prompt)
    assert float(selected[16, 16]) >= 0.5
    assert float(selected[2, 2]) < 0.5


def test_select_mask_candidate_handles_empty_stack() -> None:
    arr = np.zeros((0, 24, 16), dtype=np.float32)
    selected = _select_mask_candidate(arr, prompt=None)
    assert selected.shape == (24, 16)
    assert float(selected.sum()) == 0.0


def test_temporal_area_guard_rejects_large_low_iou_jump() -> None:
    prev = np.zeros((96, 96), dtype=np.float32)
    prev[32:64, 36:60] = 1.0

    prob = np.zeros((96, 96), dtype=np.float32)
    # Catastrophic drift candidate far from previous mask.
    prob[0:90, 0:90] = 0.9

    guarded = _apply_temporal_area_guard(
        prob,
        prev,
        threshold=0.5,
        max_area_ratio=3.0,
        min_iou=0.2,
    )
    guarded_bin = (guarded >= 0.5).astype(np.float32)
    assert np.array_equal(guarded_bin, prev)


def test_temporal_area_guard_allows_large_change_when_iou_is_high() -> None:
    prev = np.zeros((96, 96), dtype=np.float32)
    prev[32:64, 36:60] = 1.0

    prob = np.zeros((96, 96), dtype=np.float32)
    # Expanded but still overlapping strongly with previous.
    prob[28:70, 32:68] = 0.9

    guarded = _apply_temporal_area_guard(
        prob,
        prev,
        threshold=0.5,
        max_area_ratio=2.0,
        min_iou=0.2,
    )
    assert float((guarded >= 0.5).sum()) > float(prev.sum())


def test_anchor_reference_area_guard_rejects_gradual_takeover() -> None:
    prev = np.zeros((64, 64), dtype=np.float32)
    prev[20:34, 24:38] = 1.0
    reference_area = float(prev.sum())

    prob = np.zeros((64, 64), dtype=np.float32)
    # Candidate is still centered but much larger than the anchor envelope.
    prob[6:56, 8:58] = 0.9

    guarded = _apply_anchor_reference_area_guard(
        prob,
        prev,
        reference_area=reference_area,
        threshold=0.5,
        max_area_ratio=4.0,
    )
    guarded_bin = (guarded >= 0.5).astype(np.float32)
    assert np.array_equal(guarded_bin, prev)


def test_anchor_reference_area_guard_allows_reasonable_growth() -> None:
    prev = np.zeros((64, 64), dtype=np.float32)
    prev[20:34, 24:38] = 1.0
    reference_area = float(prev.sum())

    prob = np.zeros((64, 64), dtype=np.float32)
    # 3x area growth remains under strict 4x reference ratio.
    prob[16:40, 20:44] = 0.9

    guarded = _apply_anchor_reference_area_guard(
        prob,
        prev,
        reference_area=reference_area,
        threshold=0.5,
        max_area_ratio=4.0,
    )
    assert float((guarded >= 0.5).sum()) > float(prev.sum())


def test_ordered_sam_model_candidates_prefers_local_quality_tiers(monkeypatch, tmp_path) -> None:
    (tmp_path / "sam2_s.pt").write_bytes(b"")
    (tmp_path / "sam_b.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    candidates = _ordered_sam_model_candidates("sam2_l.pt")
    assert candidates[0] == "sam2_l.pt"
    assert candidates[1:3] == ["sam2_s.pt", "sam_b.pt"]


def test_ordered_sam_model_candidates_respects_explicit_path() -> None:
    explicit = r"D:\models\custom_sam.pt"
    candidates = _ordered_sam_model_candidates(explicit)
    assert candidates == [explicit]


def test_normalize_precision_aliases() -> None:
    assert _normalize_precision("fp16") == "fp16"
    assert _normalize_precision("half") == "fp16"
    assert _normalize_precision("bfloat16") == "bf16"
    assert _normalize_precision("fp32") == "fp32"
    assert _normalize_precision("unknown") == "fp32"


def test_ultralytics_backend_build_infer_kwargs_uses_half_on_cuda_fp16() -> None:
    backend = UltralyticsSAM3SegmentBackend(device="cuda:0", precision="fp16")
    kwargs = backend._build_infer_kwargs({"bboxes": [[0, 0, 10, 10]]})
    assert kwargs.get("device") == "cuda:0"
    assert kwargs.get("half") is True


def test_ultralytics_backend_build_infer_kwargs_skips_half_on_fp32() -> None:
    backend = UltralyticsSAM3SegmentBackend(device="cuda:0", precision="fp32")
    kwargs = backend._build_infer_kwargs({})
    assert "half" not in kwargs


def test_ultralytics_backend_reuses_successful_prompt_variant() -> None:
    class _Masks:
        def __init__(self, data: np.ndarray):
            self.data = data

    class _Result:
        def __init__(self, mask: np.ndarray):
            self.masks = _Masks(mask[None, ...].astype(np.float32))

    class _FakeModel:
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []

        def predict(self, source, **kwargs):
            keys = tuple(sorted(k for k in kwargs.keys() if k in {"points", "labels", "bboxes"}))
            self.calls.append(keys)
            # Simulate backend that rejects combined bbox+points prompts.
            if keys == ("bboxes", "labels", "points"):
                raise RuntimeError("unsupported prompt combo")
            h, w = source.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            mask[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] = 1.0
            return [_Result(mask)]

    backend = UltralyticsSAM3SegmentBackend(device="cpu")
    model = _FakeModel()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    prompt = SegmentPrompt(
        bbox=(6.0, 6.0, 26.0, 26.0),
        positive_points=[(16.0, 16.0)],
        negative_points=[],
        mask=None,
    )

    _ = backend._infer_single(model, frame, prompt)
    _ = backend._infer_single(model, frame, prompt)

    # First inference: combined fails then bbox succeeds.
    # Second inference should use cached bbox variant first (no combined retry).
    assert model.calls == [
        ("bboxes", "labels", "points"),
        ("bboxes",),
        ("bboxes",),
    ]


def test_chunked_segmenter_uses_video_fast_path_when_available() -> None:
    class _Backend:
        def __init__(self):
            self.video_calls = 0
            self.chunk_calls = 0

        def segment_video_sequence(self, video_path, prompt, *, start_frame=0, num_frames, frame_shape):
            self.video_calls += 1
            assert video_path == "dummy.mp4"
            assert start_frame == 0
            assert num_frames == 6
            return [np.full(frame_shape, 2.0, dtype=np.float32) for _ in range(num_frames)]

        def segment_chunk(self, frames, prompt, anchor_frame_index=0):
            self.chunk_calls += 1
            return [np.zeros(frames[0].shape[:2], dtype=np.float32) for _ in frames]

    frames = [np.zeros((32, 32, 3), dtype=np.float32) for _ in range(6)]
    source = DummyVideoSource(frames, start=0)
    anchor_mask = np.zeros((32, 32), dtype=np.float32)
    anchor_mask[8:24, 10:22] = 1.0
    prompt = MaskPromptAdapter().adapt(anchor_mask, frame_shape=anchor_mask.shape)
    backend = _Backend()
    segmenter = ChunkedSegmenter(backend=backend, processing_long_side=32)
    result = segmenter.segment_sequence(source=source, prompt=prompt, anchor_frame=0, chunk_size=4, chunk_overlap=1)
    assert backend.video_calls == 1
    assert backend.chunk_calls == 0
    assert len(result.logits) == 6
    assert result.anchored_frames == [0]


def test_chunked_segmenter_uses_video_fast_path_for_offset_start() -> None:
    class _Backend:
        def __init__(self):
            self.video_calls = 0
            self.chunk_calls = 0

        def segment_video_sequence(self, video_path, prompt, *, start_frame=0, num_frames, frame_shape):
            self.video_calls += 1
            assert video_path == "dummy.mp4"
            assert start_frame == 3
            return [np.full(frame_shape, 2.0, dtype=np.float32) for _ in range(num_frames)]

        def segment_chunk(self, frames, prompt, anchor_frame_index=0):
            self.chunk_calls += 1
            return [np.zeros(frames[0].shape[:2], dtype=np.float32) for _ in frames]

    frames = [np.zeros((32, 32, 3), dtype=np.float32) for _ in range(6)]
    source = DummyVideoSource(frames, start=3)
    anchor_mask = np.zeros((32, 32), dtype=np.float32)
    anchor_mask[8:24, 10:22] = 1.0
    prompt = MaskPromptAdapter().adapt(anchor_mask, frame_shape=anchor_mask.shape)
    backend = _Backend()
    segmenter = ChunkedSegmenter(backend=backend, processing_long_side=32)
    result = segmenter.segment_sequence(source=source, prompt=prompt, anchor_frame=0, chunk_size=4, chunk_overlap=1)
    assert backend.video_calls == 1
    assert backend.chunk_calls == 0
    assert len(result.logits) == 6


def test_ultralytics_backend_prepare_video_predictor_for_stream_seeks_and_resets() -> None:
    class _Cap:
        def __init__(self):
            self.pos = 0.0
            self.set_calls: list[tuple[int, float]] = []

        def set(self, prop: int, value: float) -> bool:
            self.set_calls.append((int(prop), float(value)))
            if int(prop) == int(cv2.CAP_PROP_POS_FRAMES):
                self.pos = float(value)
            return True

        def get(self, prop: int) -> float:
            if int(prop) == int(cv2.CAP_PROP_POS_FRAMES):
                return self.pos
            return 0.0

    class _Dataset:
        def __init__(self):
            self.mode = "video"
            self.frames = 20
            self.frame = 0
            self.cap = _Cap()

    class _Predictor:
        def __init__(self):
            self.dataset = _Dataset()
            self.inference_state = {"stale": True}
            self.reset_prompts_calls = 0

        def reset_prompts(self):
            self.reset_prompts_calls += 1

    backend = UltralyticsSAM3SegmentBackend()
    predictor = _Predictor()

    backend._prepare_video_predictor_for_stream(predictor, start_frame=7)

    assert predictor.reset_prompts_calls == 1
    assert predictor.inference_state == {}
    assert predictor.dataset.frame == 7
    assert predictor.dataset.cap.set_calls == [(int(cv2.CAP_PROP_POS_FRAMES), 7.0)]
