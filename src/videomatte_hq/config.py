"""Configuration schema for the Option B (target-assigned memory propagation) pipeline."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VMBaseModel(BaseModel):
    """Base model with permissive parsing for migration safety."""

    model_config = ConfigDict(extra="ignore")


class ShotType(str, Enum):
    LOCKED_OFF = "locked_off"
    MOVING = "moving"
    UNKNOWN = "unknown"


class AlphaFormat(str, Enum):
    PNG_16 = "png16"
    PNG_8 = "png8"
    DWAA = "dwaa"  # legacy alias accepted by existing UI
    EXR_DWAA = "exr_dwaa"
    EXR_DWAA_HQ = "exr_dwaa_hq"
    EXR_LOSSLESS = "exr_lossless"
    EXR_RAW = "exr_raw"


class IOConfig(VMBaseModel):
    """Input/Output settings."""

    input: str = "input_frames/*.png"
    output_dir: str = "output"
    output_alpha: str = "alpha/%06d.png"

    frame_start: int = 0
    frame_end: int = -1

    shot_type: ShotType = ShotType.UNKNOWN

    alpha_format: AlphaFormat = AlphaFormat.PNG_16
    alpha_dwaa_quality: float = 45.0

    force_overwrite: bool = False


class ProjectConfig(VMBaseModel):
    """Project state and asset paths."""

    path: str = ""
    masks_dir: str = "masks"
    cache_dir: str = "cache"
    autosave: bool = True


class AssignmentConfig(VMBaseModel):
    """Mask-first assignment options."""

    mode: Literal["mask_first"] = "mask_first"
    default_keyframe: int = 0
    require_assignment: bool = True
    unknown_radius_px: int = 64
    fg_erosion_px: int = 6
    bg_dilation_px: int = 12


class MemoryConfig(VMBaseModel):
    """Memory propagation core options."""

    backend: str = "appearance_memory_bank"
    memory_frames: int = 12
    window: int = 120
    max_anchors: int = 20
    confidence_reanchor_threshold: float = 0.35
    query_long_side: int = 960
    spatial_weight: float = 0.1
    temperature: float = 1.0
    auto_anchor_min_gap: int = 0


class RefineConfig(VMBaseModel):
    """High-resolution edge refinement options."""

    enabled: bool = True
    backend: str = "guided_band"
    unknown_band_px: int = 64
    tile_size: int = 1536
    overlap: int = 96
    alpha_bg_threshold: float = 0.05
    alpha_fg_threshold: float = 0.95
    min_confidence: float = 0.5
    guided_radius: int = 8
    guided_eps: float = 0.01
    edge_boost: float = 0.15
    confidence_gain: float = 1.0
    tile_min_coverage: float = 0.002


class TemporalCleanupConfig(VMBaseModel):
    """Post-refinement temporal cleanup options."""

    enabled: bool = True
    outside_band_ema: float = 0.15
    min_confidence: float = 0.5
    reset_on_new_anchor: bool = True
    anchor_reset_frames: int = 6
    edge_bg_threshold: float = 0.05
    edge_fg_threshold: float = 0.95
    edge_band_radius_px: int = 2
    edge_snap_enabled: bool = False
    edge_snap_radius: int = 2
    edge_snap_eps: float = 0.01
    clamp_delta: float = 0.25


class MatteTuningConfig(VMBaseModel):
    """Final matte-shape tuning controls (artist-facing)."""

    enabled: bool = True
    shrink_grow_px: int = 0
    feather_px: int = 0
    offset_x_px: int = 0
    offset_y_px: int = 0


class PreviewConfig(VMBaseModel):
    enabled: bool = False
    scale: int = 1080
    every: int = 10
    modes: list[str] = Field(default_factory=lambda: ["alpha", "checker", "white", "flicker"])


class QCConfig(VMBaseModel):
    enabled: bool = True
    fail_on_regression: bool = False
    output_subdir: str = "qc"
    metrics_filename: str = "optionb_metrics.json"
    report_filename: str = "optionb_report.md"
    sample_output_frames: int = 3
    max_output_roundtrip_mae: float = 0.01
    alpha_range_eps: float = 1e-3
    max_p95_flicker: float = 0.08
    max_p95_edge_flicker: float = 0.12
    min_mean_edge_confidence: float = 0.15
    band_spike_ratio: float = 2.5
    max_band_spike_frames: int = 8


class RuntimeConfig(VMBaseModel):
    device: str = "cuda"
    precision: str = "fp16"
    workers_io: int = 4
    cache_dir: str = ".cache"
    resume: bool = True
    verbose: bool = False


class VideoMatteConfig(VMBaseModel):
    """Root configuration for Option B runtime."""

    io: IOConfig = IOConfig()
    project: ProjectConfig = ProjectConfig()
    assignment: AssignmentConfig = AssignmentConfig()
    memory: MemoryConfig = MemoryConfig()
    refine: RefineConfig = RefineConfig()
    temporal_cleanup: TemporalCleanupConfig = TemporalCleanupConfig()
    matte_tuning: MatteTuningConfig = MatteTuningConfig()
    preview: PreviewConfig = PreviewConfig()
    qc: QCConfig = QCConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def stage_hash(self, stage: str) -> str:
        """Compute a deterministic hash for a stage-relevant config subset."""

        sections: dict[str, list[Any]] = {
            "project": [self.project.model_dump()],
            "assignment": [self.project.model_dump(), self.assignment.model_dump()],
            "memory": [self.assignment.model_dump(), self.memory.model_dump()],
            "refine": [self.memory.model_dump(), self.refine.model_dump()],
            "temporal_cleanup": [self.refine.model_dump(), self.temporal_cleanup.model_dump()],
            "matte_tuning": [self.temporal_cleanup.model_dump(), self.matte_tuning.model_dump()],
            "io": [
                self.matte_tuning.model_dump(),
                self.io.model_dump(),
                self.qc.model_dump(),
            ],
        }

        if stage not in sections:
            raise ValueError(f"Unknown stage for hashing: {stage}")

        payload: dict[str, Any] = {
            "sections": sections[stage],
            "io_context": {
                "input": self.io.input,
                "frame_start": self.io.frame_start,
                "frame_end": self.io.frame_end,
                "shot_type": self.io.shot_type.value,
            },
        }

        json_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_yaml(self, path: Path | str) -> None:
        """Save the config to YAML."""

        import yaml

        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "VideoMatteConfig":
        """Load config from YAML."""

        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
