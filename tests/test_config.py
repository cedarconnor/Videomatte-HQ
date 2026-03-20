from __future__ import annotations

from videomatte_hq.config import VideoMatteConfig


def test_config_defaults_to_high_quality_models() -> None:
    cfg = VideoMatteConfig()
    assert cfg.sam3_model == "sam2_l.pt"
    assert cfg.trimap_fallback_band_px == 1
    assert cfg.trimap_mode == "morphological"
    assert cfg.trimap_erosion_px == 20
    assert cfg.trimap_dilation_px == 10
    assert cfg.skip_iou_threshold == 0.97
    assert cfg.unknown_edge_blend_px == 4
    assert cfg.tile_batch_size == 1
    assert cfg.mask_hysteresis_enabled is False
    assert cfg.mask_hysteresis_low == 0.45
    assert cfg.mask_hysteresis_high == 0.55
    assert cfg.mask_temporal_smooth_radius == 2
    assert cfg.temporal_smooth_enabled is True
    assert cfg.temporal_smooth_strength == 0.55
    assert cfg.temporal_smooth_motion_threshold == 0.03


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


def test_segment_stage_config_inherits_mask_hysteresis_fields() -> None:
    cfg = VideoMatteConfig(mask_hysteresis_enabled=True, mask_hysteresis_low=0.42, mask_hysteresis_high=0.58)
    seg = cfg.segment_stage_config()
    assert seg.mask_hysteresis_enabled is True
    assert seg.mask_hysteresis_low == 0.42
    assert seg.mask_hysteresis_high == 0.58


def test_temporal_smooth_config_inherits_updated_defaults() -> None:
    cfg = VideoMatteConfig()
    smooth = cfg.temporal_smooth_config()
    assert smooth.enabled is True
    assert smooth.strength == 0.55
    assert smooth.motion_threshold == 0.03


def test_refine_stage_config_inherits_trimap_mode() -> None:
    cfg = VideoMatteConfig(trimap_mode="hybrid", trimap_erosion_px=15, trimap_dilation_px=8)
    rcfg = cfg.refine_stage_config()
    assert rcfg.trimap_mode == "hybrid"
    assert rcfg.trimap_erosion_px == 15
    assert rcfg.trimap_dilation_px == 8


def test_refine_stage_config_inherits_updated_skip_iou_default() -> None:
    cfg = VideoMatteConfig()
    rcfg = cfg.refine_stage_config()
    assert rcfg.skip_iou_threshold == 0.97
    assert rcfg.unknown_edge_blend_px == 4
    assert rcfg.tile_batch_size == 1


def test_config_round_trip_dict_preserves_trimap_fields() -> None:
    cfg = VideoMatteConfig(
        trimap_mode="morphological",
        trimap_erosion_px=25,
        trimap_dilation_px=12,
        unknown_edge_blend_px=6,
    )
    d = cfg.to_dict()
    cfg2 = VideoMatteConfig.from_dict(d)
    assert cfg2.trimap_mode == "morphological"
    assert cfg2.trimap_erosion_px == 25
    assert cfg2.trimap_dilation_px == 12
    assert cfg2.unknown_edge_blend_px == 6


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


def test_config_pipeline_mode_defaults_to_v1() -> None:
    cfg = VideoMatteConfig()
    assert cfg.pipeline_mode == "v1"


def test_config_v2_fields_have_defaults() -> None:
    cfg = VideoMatteConfig()
    assert cfg.matanyone2_repo_dir == "third_party/MatAnyone2"
    assert cfg.matanyone2_max_size == 1080
    assert cfg.matanyone2_warmup == 10
    assert cfg.matanyone2_erode_kernel == 0
    assert cfg.matanyone2_dilate_kernel == 0
    assert cfg.matanyone2_hires_threshold == 1080
    assert cfg.gradient_trimap_base_kernel == 7
    assert cfg.gradient_trimap_max_extra == 20
    assert cfg.gradient_trimap_fg_thresh == 0.95
    assert cfg.gradient_trimap_bg_thresh == 0.05
    assert cfg.gradient_trimap_scale == 0.5


def test_config_round_trip_dict_preserves_v2_fields() -> None:
    cfg = VideoMatteConfig(
        pipeline_mode="v2",
        matanyone2_max_size=720,
        matanyone2_hires_threshold=720,
        gradient_trimap_base_kernel=9,
    )
    d = cfg.to_dict()
    cfg2 = VideoMatteConfig.from_dict(d)
    assert cfg2.pipeline_mode == "v2"
    assert cfg2.matanyone2_max_size == 720
    assert cfg2.matanyone2_hires_threshold == 720
    assert cfg2.gradient_trimap_base_kernel == 9


def test_matanyone2_stage_config_returns_correct_fields() -> None:
    cfg = VideoMatteConfig(
        matanyone2_repo_dir="/custom/path",
        matanyone2_max_size=720,
        device="cpu",
    )
    sc = cfg.matanyone2_stage_config()
    assert sc["repo_dir"] == "/custom/path"
    assert sc["max_size"] == 720
    assert sc["device"] == "cpu"
