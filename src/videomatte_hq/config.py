"""Simplified v2 configuration model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import json

from videomatte_hq.pipeline.stage_refine import RefineStageConfig
from videomatte_hq.pipeline.stage_segment import SegmentStageConfig
from videomatte_hq.postprocess.matte_tuning import MatteTuningConfig


@dataclass(slots=True)
class VideoMatteConfig:
    # ---- IO ----
    input: str = "input_frames/%06d.png"
    output_dir: str = "output"
    output_alpha: str = "alpha/%06d.png"
    frame_start: int = 0
    frame_end: int = -1
    alpha_format: str = "png16"

    # ---- Segmentation (Stage 1) ----
    segment_backend: str = "ultralytics_sam3"
    sam3_model: str = "sam2_l.pt"
    sam3_processing_long_side: int = 960
    anchor_frame: int = 0
    anchor_mask: str = ""
    chunk_size: int = 100
    chunk_overlap: int = 5
    mask_threshold: float = 0.5
    bbox_expand_ratio: float = 0.10
    min_bbox_expand_px: int = 20
    temporal_component_filter: bool = False
    strict_background_suppression: bool = False
    strict_bbox_expand_ratio: float = 0.10
    strict_min_bbox_expand_px: int = 24
    strict_overlap_dilate_ratio: float = 0.03
    strict_min_overlap_dilate_px: int = 12
    strict_temporal_guard: bool = True
    strict_max_area_ratio: float = 3.0
    strict_min_iou: float = 0.20
    strict_anchor_max_area_ratio: float = 4.0
    drift_iou_threshold: float = 0.70
    drift_area_threshold: float = 0.40
    max_reanchors_per_chunk: int = 2

    # ---- Refinement (Stage 2) ----
    refine_enabled: bool = True
    mematte_repo_dir: str = "third_party/MEMatte"
    mematte_checkpoint: str = "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth"
    mematte_max_tokens: int = 18500
    mematte_patch_decoder: bool = True
    tile_size: int = 1536
    tile_overlap: int = 96
    tile_min_unknown_coverage: float = 0.001
    trimap_mode: str = "morphological"
    trimap_erosion_px: int = 20
    trimap_dilation_px: int = 10
    trimap_fg_threshold: float = 0.9
    trimap_bg_threshold: float = 0.1
    trimap_fallback_band_px: int = 1
    skip_iou_threshold: float = 0.98

    # ---- Matte Tuning (Optional) ----
    shrink_grow_px: int = 0
    feather_px: int = 0
    offset_x_px: int = 0
    offset_y_px: int = 0
    matte_tuning_enabled: bool = True

    # ---- Prompt Mode ----
    prompt_mode: str = "mask"          # "mask" | "points"
    point_prompts_json: str = ""       # JSON string with normalized coords keyed by frame index

    # ---- Runtime ----
    device: str = "cuda"
    precision: str = "fp16"
    workers_io: int = 4

    def segment_stage_config(self) -> SegmentStageConfig:
        return SegmentStageConfig(
            backend=self.segment_backend,
            sam3_model=self.sam3_model,
            processing_long_side=self.sam3_processing_long_side,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            device=self.device,
            precision=self.precision,
            mask_threshold=self.mask_threshold,
            bbox_expand_ratio=self.bbox_expand_ratio,
            min_bbox_expand_px=self.min_bbox_expand_px,
            temporal_component_filter=self.temporal_component_filter,
            strict_background_suppression=self.strict_background_suppression,
            strict_bbox_expand_ratio=self.strict_bbox_expand_ratio,
            strict_min_bbox_expand_px=self.strict_min_bbox_expand_px,
            strict_overlap_dilate_ratio=self.strict_overlap_dilate_ratio,
            strict_min_overlap_dilate_px=self.strict_min_overlap_dilate_px,
            strict_temporal_guard=self.strict_temporal_guard,
            strict_max_area_ratio=self.strict_max_area_ratio,
            strict_min_iou=self.strict_min_iou,
            strict_anchor_max_area_ratio=self.strict_anchor_max_area_ratio,
            drift_iou_threshold=self.drift_iou_threshold,
            drift_area_threshold=self.drift_area_threshold,
            max_reanchors_per_chunk=self.max_reanchors_per_chunk,
        )

    def refine_stage_config(self) -> RefineStageConfig:
        return RefineStageConfig(
            refine_enabled=self.refine_enabled,
            mematte_repo_dir=self.mematte_repo_dir,
            mematte_checkpoint=self.mematte_checkpoint,
            mematte_max_tokens=self.mematte_max_tokens,
            mematte_patch_decoder=self.mematte_patch_decoder,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            tile_min_unknown_coverage=self.tile_min_unknown_coverage,
            trimap_mode=self.trimap_mode,
            trimap_erosion_px=self.trimap_erosion_px,
            trimap_dilation_px=self.trimap_dilation_px,
            trimap_fg_threshold=self.trimap_fg_threshold,
            trimap_bg_threshold=self.trimap_bg_threshold,
            trimap_fallback_band_px=self.trimap_fallback_band_px,
            skip_iou_threshold=self.skip_iou_threshold,
            device=self.device,
            precision=self.precision,
        )

    def matte_tuning_config(self) -> MatteTuningConfig:
        return MatteTuningConfig(
            enabled=self.matte_tuning_enabled,
            shrink_grow_px=self.shrink_grow_px,
            feather_px=self.feather_px,
            offset_x_px=self.offset_x_px,
            offset_y_px=self.offset_y_px,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_file(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML output requires PyYAML (`pip install pyyaml`).") from exc
        output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "VideoMatteConfig":
        if not raw:
            return cls()

        merged = dict(raw)
        for section_name in ("io", "segmentation", "refinement", "matte_tuning", "runtime"):
            section = raw.get(section_name)
            if isinstance(section, dict):
                merged.update(section)

        valid_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in merged.items() if k in valid_names}
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "VideoMatteConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        text = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() == ".json":
            payload = json.loads(text) if text.strip() else {}
            return cls.from_dict(payload)

        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML config parsing requires PyYAML (`pip install pyyaml`).") from exc
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config root must be a mapping, got {type(payload).__name__}")
        return cls.from_dict(payload)
