from __future__ import annotations

from videomatte_hq.config import VideoMatteConfig


def test_config_defaults_to_high_quality_models() -> None:
    cfg = VideoMatteConfig()
    assert cfg.sam3_model == "sam2_l.pt"
    assert cfg.trimap_fallback_band_px == 1
    assert cfg.trimap_mode == "morphological"
    assert cfg.trimap_erosion_px == 20
    assert cfg.trimap_dilation_px == 10


def test_config_from_dict_accepts_nested_sections() -> None:
    cfg = VideoMatteConfig.from_dict(
        {
            "io": {"input": "frames/%06d.png", "output_dir": "out"},
            "segmentation": {"chunk_size": 12, "chunk_overlap": 3},
            "refinement": {"tile_size": 1024, "refine_enabled": False},
            "runtime": {"device": "cpu"},
        }
    )
    assert cfg.input == "frames/%06d.png"
    assert cfg.output_dir == "out"
    assert cfg.chunk_size == 12
    assert cfg.chunk_overlap == 3
    assert cfg.tile_size == 1024
    assert cfg.refine_enabled is False
    assert cfg.device == "cpu"


def test_segment_stage_config_inherits_precision() -> None:
    cfg = VideoMatteConfig(precision="fp32")
    seg = cfg.segment_stage_config()
    assert seg.precision == "fp32"


def test_refine_stage_config_inherits_trimap_mode() -> None:
    cfg = VideoMatteConfig(trimap_mode="logit", trimap_erosion_px=15, trimap_dilation_px=8)
    rcfg = cfg.refine_stage_config()
    assert rcfg.trimap_mode == "logit"
    assert rcfg.trimap_erosion_px == 15
    assert rcfg.trimap_dilation_px == 8


def test_config_round_trip_dict_preserves_trimap_fields() -> None:
    cfg = VideoMatteConfig(trimap_mode="morphological", trimap_erosion_px=25, trimap_dilation_px=12)
    d = cfg.to_dict()
    cfg2 = VideoMatteConfig.from_dict(d)
    assert cfg2.trimap_mode == "morphological"
    assert cfg2.trimap_erosion_px == 25
    assert cfg2.trimap_dilation_px == 12


def test_config_prompt_mode_defaults_to_mask() -> None:
    cfg = VideoMatteConfig()
    assert cfg.prompt_mode == "mask"
    assert cfg.point_prompts_json == ""


def test_config_round_trip_dict_preserves_prompt_mode() -> None:
    json_str = '{"0": {"positive": [[0.5, 0.3]], "negative": []}}'
    cfg = VideoMatteConfig(prompt_mode="points", point_prompts_json=json_str)
    d = cfg.to_dict()
    cfg2 = VideoMatteConfig.from_dict(d)
    assert cfg2.prompt_mode == "points"
    assert cfg2.point_prompts_json == json_str
