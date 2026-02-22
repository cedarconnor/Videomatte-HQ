from __future__ import annotations

from videomatte_hq.config import VideoMatteConfig


def test_config_defaults_to_high_quality_models() -> None:
    cfg = VideoMatteConfig()
    assert cfg.sam3_model == "sam2_l.pt"


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
