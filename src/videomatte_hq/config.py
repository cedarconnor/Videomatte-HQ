"""Configuration schema for VideoMatte-HQ pipeline.

Defines Pydantic models for all pipeline stages.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ShotType(Enum):
    LOCKED_OFF = "locked_off"
    MOVING = "moving"
    UNKNOWN = "unknown"


class AlphaFormat(Enum):
    PNG_16 = "png16"
    PNG_8 = "png8"
    DWAA = "dwaa"  # EXR DWAA compression


class BandMode(Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"


class TrimapMethod(Enum):
    DISTANCE_TRANSFORM = "distance_transform"
    EROSION = "erosion"


class TemporalSmooth(Enum):
    NONE = "none"
    EMA = "ema"
    FLOW = "flow"


class MultiPerson(Enum):
    SINGLE = "single"    # largest only
    UNION_K = "union_k"  # union of top K by area


class OcclusionFallback(Enum):
    AUTO = "auto"
    TEMPORAL_EXTREMES = "temporal_extremes"
    PATCH_INPAINT = "patch_inpaint"
    AI_INPAINT = "ai_inpaint"


class IOConfig(BaseModel):
    """Input/Output settings (Stage 0, 7)."""
    input: str = "input_frames/*.png"
    output_dir: str = "output"
    # Output pattern relative to output_dir
    output_alpha: str = "alpha/frame_%05d.png"
    
    frame_start: int = 0
    frame_end: int = -1  # -1 = all
    
    shot_type: ShotType = ShotType.UNKNOWN
    
    # Format
    alpha_format: AlphaFormat = AlphaFormat.PNG_16
    alpha_dwaa_quality: float = 45.0
    
    # Metadata
    force_overwrite: bool = False


class BackgroundConfig(BaseModel):
    """Background plate estimation (Stage 0.5)."""
    enabled: bool = True
    sample_count: int = 15
    variance_threshold: float = 0.05
    photometric_normalize: bool = True
    
    # Occlusion handling
    occlusion_threshold: float = 0.3
    occlusion_fallback: OcclusionFallback = OcclusionFallback.AUTO
    
    # Manual plate override (path to image)
    manual_plate_path: str = ""


class ROIConfig(BaseModel):
    """ROI tracking parameters (Stage 1)."""
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

    @model_validator(mode="after")
    def check_overlap(self) -> "GlobalConfig":
        if self.chunk_overlap >= self.chunk_len:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap}) must be less than chunk length ({self.chunk_len})"
            )
        return self


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
    # Selective processing (speed optimization)
    selective_enabled: bool = True
    selective_rgb_threshold: float = 0.010
    selective_a0_threshold: float = 0.005
    selective_recheck_every: int = 8
    selective_max_skip: int = 6
    selective_delta_decay: float = 0.98


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
    # Performance: compute band/trimap/feather at reduced resolution
    compute_downscale: float = 0.25


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
    # Performance: tiles per GPU batch
    tile_batch_size: int = 4


class RefineConfig(BaseModel):
    """Pass B — Edge refinement (Stage 4)."""
    model: str = "vitmatte"
    use_bg_plate: bool = True
    bg_confidence_gate: float = 0.8


class TemporalConfig(BaseModel):
    """Pass C — Temporal stabilization (Stage 5)."""
    method: str = "frequency_separation"  # 'frequency_separation' or 'none'
    structural_sigma: float = 1.0
    structural_threshold: float = 0.015
    structural_blend_strength: float = 0.7  # conservative
    detail_blend_strength: float = 0.95     # aggressive
    flow_consistency_sigma: float = 1.0
    fallback_threshold: float = 0.2


class PostprocessConfig(BaseModel):
    """Post-processing and Despill (Stage 6)."""
    class DespillConfig(BaseModel):
        enabled: bool = True
        method: str = "advanced"
        spill_color: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0]) # Green
        luma_bias: float = 0.1

    class FGOutputConfig(BaseModel):
        enabled: bool = True
        format: str = "png"
        premultiplied: bool = False

    despill: DespillConfig = DespillConfig()
    fg_output: FGOutputConfig = FGOutputConfig()


class ReferenceFrameConfig(BaseModel):
    """Reference Frame Mechanism (Section 15, optional)."""
    enabled: bool = False
    count: int = 5
    selection_method: str = "auto_quality"
    propagation_range_max: int = 30
    propagation_error_limit: float = 15.0
    propagation_motion_limit: float = 50.0


class PreviewConfig(BaseModel):
    """Live preview generation."""
    enabled: bool = False
    scale: float = 0.5
    every: int = 10
    modes: list[str] = Field(default_factory=lambda: ["alpha", "checker", "white", "flicker"])


class RuntimeConfig(BaseModel):
    """Runtime execution settings."""
    device: str = "cuda"
    precision: str = "fp16"
    workers_io: int = 4
    cache_dir: str = ".cache"
    resume: bool = True
    verbose: bool = False


class VideoMatteConfig(BaseModel):
    """Root configuration for the pipeline."""
    io: IOConfig = IOConfig()
    background: BackgroundConfig = BackgroundConfig()
    roi: ROIConfig = ROIConfig()
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    intermediate: IntermediateConfig = IntermediateConfig()
    band: BandConfig = BandConfig()
    trimap: TrimapConfig = TrimapConfig()
    tiles: TileConfig = TileConfig()
    refine: RefineConfig = RefineConfig()
    temporal: TemporalConfig = TemporalConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    reference_frames: ReferenceFrameConfig = ReferenceFrameConfig()
    preview: PreviewConfig = PreviewConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def stage_hash(self, stage: str) -> str:
        """Compute SHA256 hash of configuration for a given stage."""
        # For simplicity, hash the relevant subsection + IO context (framerate/resolution implied)
        sections = []
        if stage == "background":
            sections.append(self.background.model_dump())
        elif stage == "roi":
            sections.append(self.roi.model_dump())
        elif stage == "global":
            sections.append(self.global_.model_dump())
        elif stage == "intermediate":
             sections.append(self.global_.model_dump())
             sections.append(self.intermediate.model_dump())
        elif stage == "band":
             sections.append(self.intermediate.model_dump())
             sections.append(self.band.model_dump())
             sections.append(self.trimap.model_dump())
             sections.append(self.tiles.model_dump())
        elif stage == "refine":
             sections.append(self.band.model_dump())
             sections.append(self.trimap.model_dump())
             sections.append(self.tiles.model_dump())
             sections.append(self.refine.model_dump())
        elif stage == "temporal":
             sections.append(self.refine.model_dump())
             sections.append(self.temporal.model_dump())
        elif stage == "postprocess":
             sections.append(self.temporal.model_dump())
             sections.append(self.postprocess.model_dump())
        elif stage == "io":
             sections.append(self.postprocess.model_dump())
             sections.append(self.io.model_dump())

        # Always include IO context (frame range etc) for non-IO stages
        # to ensure resume invalidates if frame range changes.
        data = sections
        if stage != "io":
            # Frame span/input affect all stage outputs and must invalidate resume caches.
            data = {
                "section": data,
                "io_context": {
                    "input": self.io.input,
                    "frame_start": self.io.frame_start,
                    "frame_end": self.io.frame_end,
                    "shot_type": self.io.shot_type.value,
                },
            }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
