"""Typed configuration schema for the VideoMatte-HQ pipeline.

Mirrors the full YAML config from design doc §16.2, with default-to-PNG16 output.
Supports loading from YAML file and per-stage config hashing for resume invalidation.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ShotType(str, Enum):
    LOCKED_OFF = "locked_off"
    HANDHELD = "handheld"
    UNKNOWN = "unknown"


class AlphaFormat(str, Enum):
    PNG16 = "png16"
    EXR_DWAA = "exr_dwaa"
    EXR_DWAA_HQ = "exr_dwaa_hq"
    EXR_LOSSLESS = "exr_lossless"
    EXR_RAW = "exr_raw"


class BandMode(str, Enum):
    ADAPTIVE = "adaptive"
    THRESHOLD = "threshold"


class TrimapMethod(str, Enum):
    DISTANCE_TRANSFORM = "distance_transform"
    EROSION = "erosion"


class TemporalMethod(str, Enum):
    FREQUENCY_SEPARATION = "frequency_separation"
    BILATERAL = "bilateral"
    NONE = "none"


class TemporalSmooth(str, Enum):
    FLOW = "flow"
    EMA = "ema"
    NONE = "none"


class DespillMethod(str, Enum):
    BG_PLATE = "bg_plate"
    LOCAL_BG_ESTIMATE = "local_bg_estimate"


class ROIMode(str, Enum):
    AUTO_PERSON_TRACK = "auto_person_track"
    SEGMENTATION = "segmentation"
    BG_SUB = "bg_sub"
    MANUAL = "manual"


class MultiPerson(str, Enum):
    SINGLE = "single"
    UNION_K = "union_k"
    ALL = "all"


class OcclusionFallback(str, Enum):
    AUTO = "auto"
    TEMPORAL_EXTREMES = "temporal_extremes"
    PATCH_INPAINT = "patch_inpaint"
    AI_INPAINT = "ai_inpaint"


class RefFrameMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------

class IOConfig(BaseModel):
    """Input/output configuration."""
    input: str = "frames/%06d.png"
    output_alpha: str = "out/alpha/%06d.png"
    output_fg: Optional[str] = None
    output_preview: str = "out/preview/live_preview.mp4"
    output_dir: str = "out"
    fps: int = 30
    colorspace: str = "auto"
    shot_type: ShotType = ShotType.LOCKED_OFF
    # Alpha output — default to 16-bit PNG per user preference
    alpha_format: AlphaFormat = AlphaFormat.PNG16
    alpha_dwaa_quality: float = 45.0
    alpha_compression_qc: bool = True
    alpha_compression_qc_threshold: float = 0.01
    # Frame range
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


class BackgroundConfig(BaseModel):
    """Background plate estimation (Stage 0.5)."""
    enabled: bool = True
    sample_count: int = 60
    method: str = "temporal_median"
    # Confidence
    variance_threshold: float = 0.05
    occlusion_threshold: float = 0.3
    occlusion_fallback: OcclusionFallback = OcclusionFallback.AUTO
    # Photometric normalization
    photometric_normalize: bool = True
    fg_diff_space: str = "luma"


class ROIConfig(BaseModel):
    """ROI tracking (Stage 1)."""
    mode: ROIMode = ROIMode.AUTO_PERSON_TRACK
    detect_every: int = 15
    pad_ratio: float = 0.25
    context_px: int = 256
    smooth_alpha: float = 0.3
    multi_person: MultiPerson = MultiPerson.UNION_K
    k: int = 2
    use_motion_mask: bool = True
    qc: bool = True


class GlobalConfig(BaseModel):
    """Pass A — Global matte backbone (Stage 2)."""
    model: str = "rvm"
    long_side: int = 2048
    chunk_len: int = 24
    chunk_overlap: int = 6
    use_roi_crop: bool = True


class IntermediateConfig(BaseModel):
    """Pass A′ — Intermediate refinement (Stage 2.5)."""
    enabled: bool = True
    long_side: int = 4096
    model: str = "vitmatte"
    # Guided-filter delta clamping
    guide_filter_radius: int = 8
    guide_filter_eps: float = 0.01
    # Temporal smoothing
    temporal_smooth: TemporalSmooth = TemporalSmooth.FLOW
    smooth_strength: float = 0.3


class BandConfig(BaseModel):
    """Adaptive band definition (Stage 3)."""
    mode: BandMode = BandMode.ADAPTIVE
    # Signal 1: alpha gradient
    alpha_grad_threshold: float = 0.01
    dilate_alpha_px: int = 96
    # Signal 2: RGB edges with alignment filter
    dilate_rgb_px: int = 64
    rgb_proximity_px: int = 192
    edge_alignment_threshold: float = 0.3
    rgb_alpha_range: list[float] = Field(default_factory=lambda: [0.05, 0.95])
    # Signal 3: BG subtraction edges
    dilate_bg_px: int = 64
    bg_enabled: bool = True
    bg_confidence_gate: float = 0.5
    bg_edge_persist_frames: int = 3
    bg_edge_roi_margin_px: int = 128
    # Area cap
    band_max_coverage: float = 0.35
    auto_tighten: bool = True
    # Feather
    feather_px: int = 64
    # Hair-aware
    hair_aware: bool = True
    hair_dilation_multiplier: float = 2.0


class TrimapConfig(BaseModel):
    """Trimap generation (Stage 3)."""
    method: TrimapMethod = TrimapMethod.DISTANCE_TRANSFORM
    unknown_width: int = 32
    unknown_width_hair: int = 48
    unknown_width_body: int = 24
    adaptive_width: bool = True
    adaptive_thresholds: bool = True
    t_fg: float = 0.95
    t_bg: float = 0.05


class TileConfig(BaseModel):
    """Tiling parameters (Stage 3)."""
    tile_size: int = 2048
    tile_size_backoff: list[int] = Field(default_factory=lambda: [2048, 1536, 1024])
    vram_headroom: float = 0.85
    overlap: int = 384
    min_band_coverage: float = 0.005
    blend_space: str = "logit"
    priority: str = "hair_first"


class RefineConfig(BaseModel):
    """Pass B — Edge refinement (Stage 4)."""
    model: str = "vitmatte"
    use_bg_plate: bool = True
    bg_confidence_gate: float = 0.7


class TemporalConfig(BaseModel):
    """Pass C — Temporal stabilization (Stage 5)."""
    method: TemporalMethod = TemporalMethod.FREQUENCY_SEPARATION
    # Structural/detail split
    structural_sigma: float = 4.0
    structural_threshold: float = 0.1
    structural_blend_strength: float = 0.3
    detail_blend_strength: float = 0.7
    # Flow
    flow_model: str = "raft"
    flow_consistency_sigma: float = 2.0
    fallback_threshold: float = 0.1
    # Locked-off shortcut
    locked_off_mode: bool = False
    bilateral_sigma_space: float = 5.0
    bilateral_sigma_intensity: float = 0.1
    band_only: bool = True


class DespillConfig(BaseModel):
    """Edge color despill."""
    enabled: bool = True
    method: DespillMethod = DespillMethod.BG_PLATE
    strength: float = 1.0
    bg_confidence_gate: bool = True
    min_bg_confidence: float = 0.4


class FGOutputConfig(BaseModel):
    """Foreground extraction output."""
    enabled: bool = False
    premultiplied: bool = True
    despilled: bool = True


class PostprocessConfig(BaseModel):
    """Post-processing (Stage 6)."""
    despill: DespillConfig = Field(default_factory=DespillConfig)
    fg_output: FGOutputConfig = Field(default_factory=FGOutputConfig)


class QualityBoostConfig(BaseModel):
    tile_size: int = 1536
    refiner: str = "diffmatte"


class ReferenceFrameConfig(BaseModel):
    """Reference frame mechanism (optional)."""
    enabled: bool = False
    mode: RefFrameMode = RefFrameMode.AUTO
    count: int = 5
    quality_boost: QualityBoostConfig = Field(default_factory=QualityBoostConfig)
    propagation_range_max: int = 30
    propagation_error_limit: float = 15.0
    propagation_motion_limit: float = 50.0
    use_as: str = "prior"


class PreviewConfig(BaseModel):
    """Preview / QC output."""
    enabled: bool = True
    scale: int = 1080
    every: int = 10
    modes: list[str] = Field(default_factory=lambda: ["checker", "alpha", "white", "flicker"])
    checker_size_px: int = 64
    pass_comparison: bool = True


class RuntimeConfig(BaseModel):
    """Runtime / hardware."""
    device: str = "cuda"
    precision: str = "fp16"
    workers_io: int = 4
    cache_dir: str = ".cache/videomatte-hq"
    resume: bool = True
    async_io: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class VideoMatteConfig(BaseModel):
    """Top-level pipeline configuration."""
    io: IOConfig = Field(default_factory=IOConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    globals: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    intermediate: IntermediateConfig = Field(default_factory=IntermediateConfig)
    band: BandConfig = Field(default_factory=BandConfig)
    trimap: TrimapConfig = Field(default_factory=TrimapConfig)
    tiles: TileConfig = Field(default_factory=TileConfig)
    refine: RefineConfig = Field(default_factory=RefineConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    reference_frames: ReferenceFrameConfig = Field(default_factory=ReferenceFrameConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VideoMatteConfig":
        """Load configuration from a YAML file, merging with defaults."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> "VideoMatteConfig":
        """Return default configuration."""
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(by_alias=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def stage_hash(self, stage: str) -> str:
        """Compute a content hash for a specific pipeline stage's config.

        Used for resume invalidation: if the hash changes, cached results
        for this stage and all downstream stages are invalidated.

        Args:
            stage: One of 'background', 'roi', 'global', 'intermediate',
                   'band', 'trimap', 'tiles', 'refine', 'temporal',
                   'postprocess', 'io'.
        """
        section_map: dict[str, Any] = {
            "background": self.background,
            "roi": self.roi,
            "global": self.globals,
            "intermediate": self.intermediate,
            "band": self.band,
            "trimap": self.trimap,
            "tiles": self.tiles,
            "refine": self.refine,
            "temporal": self.temporal,
            "postprocess": self.postprocess,
            "io": self.io,
        }
        section = section_map.get(stage)
        if section is None:
            raise ValueError(f"Unknown stage: {stage}")

        # Deterministic JSON for hashing
        data = section.model_dump()
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
