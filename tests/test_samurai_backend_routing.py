from __future__ import annotations

import numpy as np
import pytest

from videomatte_hq.propagation_assist import propagate_masks_assist
from videomatte_hq.prompt_mask_range import build_prompt_masks_range
from videomatte_hq.samurai_backend import _normalize_model_cfg_for_builder


def _synthetic_rgb(local_idx: int) -> np.ndarray:
    h, w = 64, 96
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = 12 + int(local_idx)
    frame[16:52, x0 : x0 + 24, :] = 220
    return frame


def test_prompt_range_routes_to_samurai_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_samurai(**kwargs):
        start = int(kwargs["frame_start"])
        end = int(kwargs["frame_end"])
        h, w = 64, 96
        masks = {idx: np.zeros((h, w), dtype=np.float32) for idx in range(start, end + 1)}
        for idx in range(start, end + 1):
            x0 = 12 + idx
            masks[idx][16:52, x0 : x0 + 24] = 1.0
        return masks, "fake samurai"

    import videomatte_hq.prompt_mask_range as prompt_mod

    monkeypatch.setattr(prompt_mod, "propagate_with_samurai_from_prompts", _fake_samurai)

    result = build_prompt_masks_range(
        frame_loader=_synthetic_rgb,
        frame_start=0,
        frame_end=5,
        anchor_frame=0,
        box_xyxy=(8, 10, 44, 56),
        fg_points=[(22, 28)],
        bg_points=[(2, 2)],
        backend="samurai_video_predictor",
        samurai_model_cfg="sam2.1_hiera_l.yaml",
        samurai_checkpoint="checkpoints/sam2.1_hiera_large.pt",
    )

    assert result.backend_used == "samurai_video_predictor"
    assert result.note == "fake samurai"
    assert len(result.masks) == 6
    assert float(result.masks[0].mean()) > 0.05


def test_phase4_samurai_backend_falls_back_to_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("samurai runtime unavailable in test")

    import videomatte_hq.propagation_assist as prop_mod

    monkeypatch.setattr(prop_mod, "propagate_with_samurai_from_mask", _raise)

    anchor = np.zeros((64, 96), dtype=np.float32)
    anchor[16:52, 12:36] = 1.0

    result = propagate_masks_assist(
        frame_loader=_synthetic_rgb,
        frame_start=0,
        frame_end=6,
        anchor_frame=0,
        anchor_mask=anchor,
        backend="samurai_video_predictor",
        fallback_to_flow=True,
    )

    assert result.backend_used == "flow_fallback"
    assert result.note is not None
    assert "Samurai unavailable" in result.note
    assert len(result.masks) >= 1


def test_samurai_model_cfg_pointer_file_is_normalized(tmp_path) -> None:
    ptr = tmp_path / "sam2_hiera_l.yaml"
    ptr.write_text("configs/sam2.1/sam2.1_hiera_l.yaml\n", encoding="utf-8")
    assert _normalize_model_cfg_for_builder(str(ptr)) == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_samurai_model_cfg_configs_path_is_normalized(tmp_path) -> None:
    cfg = tmp_path / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("model:\n  _target_: test\n", encoding="utf-8")
    assert _normalize_model_cfg_for_builder(str(cfg)) == "configs/sam2.1/sam2.1_hiera_l.yaml"
