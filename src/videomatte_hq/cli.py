"""CLI entry point for the Option B VideoMatte-HQ runtime."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

from videomatte_hq.config import AlphaFormat, ShotType, VideoMatteConfig
from videomatte_hq.io.reader import FrameSource
from videomatte_hq.propagation_assist import propagate_masks_assist, select_propagation_frames
from videomatte_hq.project import (
    ensure_project,
    import_keyframe_mask,
    load_keyframe_masks,
    suggest_reprocess_range,
    upsert_keyframe_alpha,
)

logger = logging.getLogger("videomatte_hq")



def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.command("videomatte-hq")
@click.option("--input", "--in", "input_path", default=None, help="Input frames pattern or video file")
@click.option("--out", "output_path", default=None, help="Output alpha pattern or output directory")
@click.option("--config", "config_path", default=None, help="YAML config file")
@click.option("--project", "project_path", default=None, help="Path to .vmhqproj project file")
@click.option("--start", "frame_start", default=None, type=int, help="Start frame")
@click.option("--end", "frame_end", default=None, type=int, help="End frame")
@click.option("--shot-type", default=None, help="Shot type (locked_off/moving/unknown)")
@click.option("--alpha-format", default=None, help="Alpha format (png16/png8/exr_dwaa/exr_lossless)")
@click.option("--alpha-dwaa-quality", default=None, type=float, help="DWAA compression quality")
@click.option("--memory-backend", default=None, help="Memory backend identifier")
@click.option("--memory-frames", default=None, type=int, help="Memory frame budget")
@click.option("--window", default=None, type=int, help="Temporal window length")
@click.option(
    "--memory-region-constraint/--no-memory-region-constraint",
    default=None,
    help="Enable Stage-2 foreground region constraint from propagated/anchor masks",
)
@click.option(
    "--memory-region-source",
    default=None,
    type=click.Choice(
        ["none", "propagated_bbox", "propagated_mask", "nearest_keyframe_bbox"],
        case_sensitive=False,
    ),
    help="Region prior source for Stage 2 constraint",
)
@click.option("--memory-region-anchor-frame", default=None, type=int, help="Region prior anchor frame index")
@click.option(
    "--memory-region-backend",
    default=None,
    help="Region prior propagation backend (flow/samurai_video_predictor/sam2_video_predictor/cutie)",
)
@click.option(
    "--memory-region-fallback-to-flow/--no-memory-region-fallback-to-flow",
    default=None,
    help="Fallback to flow when Samurai/SAM2/Cutie backend is unavailable",
)
@click.option(
    "--memory-region-flow-downscale",
    default=None,
    type=float,
    help="Flow downscale for region prior propagation",
)
@click.option(
    "--memory-region-flow-min-coverage",
    default=None,
    type=float,
    help="Minimum acceptable region prior coverage before fallback",
)
@click.option(
    "--memory-region-flow-max-coverage",
    default=None,
    type=float,
    help="Maximum acceptable region prior coverage before fallback",
)
@click.option(
    "--memory-region-flow-feather-px",
    default=None,
    type=int,
    help="Feather radius during region prior flow propagation",
)
@click.option(
    "--memory-region-samurai-model-cfg",
    default=None,
    help="Samurai/SAM2 model cfg path for region prior propagation",
)
@click.option(
    "--memory-region-samurai-checkpoint",
    default=None,
    help="Samurai/SAM2 checkpoint path for region prior propagation",
)
@click.option(
    "--memory-region-samurai-offload-video-to-cpu/--no-memory-region-samurai-offload-video-to-cpu",
    default=None,
    help="Offload Samurai video buffers to CPU during region prior propagation",
)
@click.option(
    "--memory-region-samurai-offload-state-to-cpu/--no-memory-region-samurai-offload-state-to-cpu",
    default=None,
    help="Offload Samurai inference state to CPU during region prior propagation",
)
@click.option(
    "--memory-region-threshold",
    default=None,
    type=float,
    help="Threshold used to derive region prior from propagated mask",
)
@click.option(
    "--memory-region-bbox-margin-px",
    default=None,
    type=int,
    help="Extra bbox margin for bbox-based region priors",
)
@click.option(
    "--memory-region-bbox-expand-ratio",
    default=None,
    type=float,
    help="Relative bbox expansion ratio for bbox-based region priors",
)
@click.option(
    "--memory-region-dilate-px",
    default=None,
    type=int,
    help="Morphological dilation radius for region prior mask",
)
@click.option(
    "--memory-region-soften-px",
    default=None,
    type=int,
    help="Gaussian soften radius for region prior mask",
)
@click.option(
    "--memory-region-outside-conf-cap",
    default=None,
    type=float,
    help="Confidence cap applied outside constrained region",
)
@click.option(
    "--refine-backend",
    default=None,
    help="Refine backend (guided_band/mematte)",
)
@click.option(
    "--unknown-band-px",
    "--mt-trimap-width-px",
    "unknown_band_px",
    default=None,
    type=int,
    help="Refine trimap unknown-band width in pixels",
)
@click.option(
    "--refine-mematte-repo-dir",
    default=None,
    help="Path to MEMatte repository directory",
)
@click.option(
    "--refine-mematte-checkpoint",
    default=None,
    help="Path to MEMatte checkpoint (.pth)",
)
@click.option(
    "--refine-mematte-max-number-token",
    default=None,
    type=int,
    help="MEMatte max global-attention token count",
)
@click.option(
    "--refine-mematte-patch-decoder/--no-refine-mematte-patch-decoder",
    default=None,
    help="Use MEMatte patch decoder during inference",
)
@click.option(
    "--refine-region-trimap/--no-refine-region-trimap",
    default=None,
    help="Constrain refinement with Samurai/propagated region trimap guidance",
)
@click.option(
    "--refine-region-trimap-threshold",
    default=None,
    type=float,
    help="Threshold applied to propagated guidance mask before trimap generation",
)
@click.option(
    "--refine-region-trimap-fg-erode-px",
    default=None,
    type=int,
    help="Erode radius for sure-foreground lock in guided trimap",
)
@click.option(
    "--refine-region-trimap-bg-dilate-px",
    default=None,
    type=int,
    help="Dilate radius for loose-foreground support in guided trimap",
)
@click.option(
    "--refine-region-trimap-cleanup-px",
    default=None,
    type=int,
    help="Morphological cleanup radius for propagated trimap masks",
)
@click.option(
    "--refine-region-trimap-keep-largest/--no-refine-region-trimap-keep-largest",
    default=None,
    help="Keep only the largest connected component in guided trimap masks",
)
@click.option(
    "--refine-region-trimap-min-coverage",
    default=None,
    type=float,
    help="Minimum guided-trimap coverage accepted before frame fallback",
)
@click.option(
    "--refine-region-trimap-max-coverage",
    default=None,
    type=float,
    help="Maximum guided-trimap coverage accepted before frame fallback",
)
@click.option("--matte-tuning/--no-matte-tuning", default=None, help="Enable final matte tuning controls")
@click.option("--mt-shrink-grow-px", default=None, type=int, help="Final matte choke/expand in pixels")
@click.option("--mt-feather-px", default=None, type=int, help="Final matte feather blur radius in pixels")
@click.option("--mt-offset-x-px", default=None, type=int, help="Final matte X offset in pixels")
@click.option("--mt-offset-y-px", default=None, type=int, help="Final matte Y offset in pixels")
@click.option("--require-assignment/--allow-empty-assignment", default=None, help="Require keyframe assignment")
@click.option("--assign-mask", default=None, help="Import this mask as a keyframe assignment")
@click.option("--assign-frame", default=0, type=int, help="Frame index for --assign-mask")
@click.option(
    "--assign-kind",
    default="initial",
    type=click.Choice(["initial", "correction"], case_sensitive=False),
    help="Assignment kind metadata",
)
@click.option(
    "--apply-suggested-range/--no-apply-suggested-range",
    default=True,
    help="When importing a correction assignment, apply suggested reprocess range",
)
@click.option("--assign-only", is_flag=True, help="Only import assignment and exit")
@click.option(
    "--propagate-from-frame",
    default=None,
    type=int,
    help="Phase 4: propagate additional keyframes from this anchor frame",
)
@click.option("--propagate-range-start", default=None, type=int, help="Propagation range start frame")
@click.option("--propagate-range-end", default=None, type=int, help="Propagation range end frame")
@click.option(
    "--propagate-backend",
    default="flow",
    help="Propagation backend (flow/samurai_video_predictor/sam2_video_predictor/cutie)",
)
@click.option(
    "--propagate-fallback-to-flow/--no-propagate-fallback-to-flow",
    default=True,
    help="Fallback to flow backend if Samurai/SAM2/Cutie backend is unavailable",
)
@click.option("--propagate-stride", default=8, type=int, help="Insert propagated keyframes every N frames")
@click.option(
    "--propagate-max-new-keyframes",
    default=24,
    type=int,
    help="Maximum propagated keyframes to insert",
)
@click.option(
    "--propagate-overwrite-existing/--no-propagate-overwrite-existing",
    default=False,
    help="Overwrite existing keyframes when frame indices collide",
)
@click.option(
    "--propagate-flow-downscale",
    default=0.5,
    type=float,
    help="Flow propagation analysis downscale (0.15..1.0)",
)
@click.option(
    "--propagate-flow-min-coverage",
    default=0.002,
    type=float,
    help="Minimum propagated mask coverage required to save an anchor",
)
@click.option(
    "--propagate-flow-max-coverage",
    default=0.98,
    type=float,
    help="Maximum propagated mask coverage accepted before fallback",
)
@click.option(
    "--propagate-flow-feather-px",
    default=1,
    type=int,
    help="Feather radius applied during flow propagation",
)
@click.option(
    "--propagate-samurai-model-cfg",
    default=None,
    help="Samurai/SAM2 model cfg path for Phase 4 propagation",
)
@click.option(
    "--propagate-samurai-checkpoint",
    default=None,
    help="Samurai/SAM2 checkpoint path for Phase 4 propagation",
)
@click.option(
    "--propagate-samurai-offload-video-to-cpu/--no-propagate-samurai-offload-video-to-cpu",
    default=None,
    help="Offload Samurai video buffers to CPU during Phase 4 propagation",
)
@click.option(
    "--propagate-samurai-offload-state-to-cpu/--no-propagate-samurai-offload-state-to-cpu",
    default=None,
    help="Offload Samurai inference state to CPU during Phase 4 propagation",
)
@click.option("--propagate-kind", default="correction", help="Inserted assignment kind (initial/correction)")
@click.option("--propagate-source", default="cli_propagate", help="Inserted assignment source label")
@click.option("--propagate-only", is_flag=True, help="Only run propagation assist and exit")
@click.option(
    "--debug-stage-samples/--no-debug-stage-samples",
    default=None,
    help="Export sample alpha/rgb/overlay images for each stage (debug_stages).",
)
@click.option(
    "--debug-sample-count",
    default=None,
    type=int,
    help="How many frames to sample for per-stage debug exports.",
)
@click.option(
    "--debug-sample-frames",
    default=None,
    help="Comma-separated absolute frame indices for stage debug sampling.",
)
@click.option(
    "--debug-stage-dir",
    default=None,
    help="Debug stage artifact subdirectory under output_dir.",
)
@click.option("--resume/--no-resume", default=None, help="Use stage cache resume")
@click.option("--qc/--no-qc", default=None, help="Enable/disable Option B QC evaluation")
@click.option(
    "--qc-fail-on-regression/--no-qc-fail-on-regression",
    default=None,
    help="Fail the run if QC regression gates fail",
)
@click.option("--qc-sample-output-frames", default=None, type=int, help="QC output round-trip sample count")
@click.option("--qc-max-output-roundtrip-mae", default=None, type=float, help="QC max output round-trip MAE")
@click.option("--qc-alpha-range-eps", default=None, type=float, help="QC alpha range tolerance epsilon")
@click.option("--qc-max-p95-flicker", default=None, type=float, help="QC max allowed p95 frame flicker")
@click.option("--qc-max-p95-edge-flicker", default=None, type=float, help="QC max allowed p95 edge flicker")
@click.option(
    "--qc-min-mean-edge-confidence",
    default=None,
    type=float,
    help="QC minimum allowed mean edge confidence",
)
@click.option("--qc-band-spike-ratio", default=None, type=float, help="QC band spike ratio threshold")
@click.option(
    "--qc-max-band-spike-frames",
    default=None,
    type=int,
    help="QC max number of frames allowed to exceed band spike threshold",
)
@click.option("--device", default=None, help="Device (cuda/cpu)")
@click.option("--precision", default=None, help="Precision (fp16/fp32)")
@click.option("--workers", default=None, type=int, help="IO workers")
@click.option("--dump-config", is_flag=True, help="Print resolved config YAML and exit")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(
    input_path,
    output_path,
    config_path,
    project_path,
    frame_start,
    frame_end,
    shot_type,
    alpha_format,
    alpha_dwaa_quality,
    memory_backend,
    memory_frames,
    window,
    memory_region_constraint,
    memory_region_source,
    memory_region_anchor_frame,
    memory_region_backend,
    memory_region_fallback_to_flow,
    memory_region_flow_downscale,
    memory_region_flow_min_coverage,
    memory_region_flow_max_coverage,
    memory_region_flow_feather_px,
    memory_region_samurai_model_cfg,
    memory_region_samurai_checkpoint,
    memory_region_samurai_offload_video_to_cpu,
    memory_region_samurai_offload_state_to_cpu,
    memory_region_threshold,
    memory_region_bbox_margin_px,
    memory_region_bbox_expand_ratio,
    memory_region_dilate_px,
    memory_region_soften_px,
    memory_region_outside_conf_cap,
    refine_backend,
    unknown_band_px,
    refine_mematte_repo_dir,
    refine_mematte_checkpoint,
    refine_mematte_max_number_token,
    refine_mematte_patch_decoder,
    refine_region_trimap,
    refine_region_trimap_threshold,
    refine_region_trimap_fg_erode_px,
    refine_region_trimap_bg_dilate_px,
    refine_region_trimap_cleanup_px,
    refine_region_trimap_keep_largest,
    refine_region_trimap_min_coverage,
    refine_region_trimap_max_coverage,
    matte_tuning,
    mt_shrink_grow_px,
    mt_feather_px,
    mt_offset_x_px,
    mt_offset_y_px,
    require_assignment,
    assign_mask,
    assign_frame,
    assign_kind,
    apply_suggested_range,
    assign_only,
    propagate_from_frame,
    propagate_range_start,
    propagate_range_end,
    propagate_backend,
    propagate_fallback_to_flow,
    propagate_stride,
    propagate_max_new_keyframes,
    propagate_overwrite_existing,
    propagate_flow_downscale,
    propagate_flow_min_coverage,
    propagate_flow_max_coverage,
    propagate_flow_feather_px,
    propagate_samurai_model_cfg,
    propagate_samurai_checkpoint,
    propagate_samurai_offload_video_to_cpu,
    propagate_samurai_offload_state_to_cpu,
    propagate_kind,
    propagate_source,
    propagate_only,
    debug_stage_samples,
    debug_sample_count,
    debug_sample_frames,
    debug_stage_dir,
    resume,
    qc,
    qc_fail_on_regression,
    qc_sample_output_frames,
    qc_max_output_roundtrip_mae,
    qc_alpha_range_eps,
    qc_max_p95_flicker,
    qc_max_p95_edge_flicker,
    qc_min_mean_edge_confidence,
    qc_band_spike_ratio,
    qc_max_band_spike_frames,
    device,
    precision,
    workers,
    dump_config,
    verbose,
):
    """VideoMatte-HQ Option B runtime."""

    setup_logging(verbose)

    # Defaults then YAML override.
    cfg = VideoMatteConfig()
    if config_path:
        logger.info(f"Loading config from {config_path}")
        cfg = VideoMatteConfig.from_yaml(config_path)

    # CLI overrides.
    if input_path:
        cfg.io.input = input_path

    if output_path:
        out_p = Path(output_path)
        if out_p.suffix == "":
            cfg.io.output_dir = str(out_p)
        else:
            parent = out_p.parent if str(out_p.parent) not in ("", ".") else Path(cfg.io.output_dir)
            cfg.io.output_dir = str(parent)
            cfg.io.output_alpha = out_p.name

    if project_path:
        cfg.project.path = project_path

    if frame_start is not None:
        cfg.io.frame_start = frame_start
    if frame_end is not None:
        cfg.io.frame_end = frame_end

    if shot_type:
        cfg.io.shot_type = ShotType(shot_type)

    if alpha_format:
        cfg.io.alpha_format = AlphaFormat(alpha_format)
    if alpha_dwaa_quality is not None:
        cfg.io.alpha_dwaa_quality = alpha_dwaa_quality

    if memory_backend:
        cfg.memory.backend = memory_backend
    if memory_frames is not None:
        cfg.memory.memory_frames = memory_frames
    if window is not None:
        cfg.memory.window = window
    if memory_region_constraint is not None:
        cfg.memory.region_constraint_enabled = bool(memory_region_constraint)
    if memory_region_source:
        cfg.memory.region_constraint_source = str(memory_region_source).lower()
    if memory_region_anchor_frame is not None:
        cfg.memory.region_constraint_anchor_frame = int(memory_region_anchor_frame)
    if memory_region_backend:
        cfg.memory.region_constraint_backend = str(memory_region_backend)
    if memory_region_fallback_to_flow is not None:
        cfg.memory.region_constraint_fallback_to_flow = bool(memory_region_fallback_to_flow)
    if memory_region_flow_downscale is not None:
        cfg.memory.region_constraint_flow_downscale = float(memory_region_flow_downscale)
    if memory_region_flow_min_coverage is not None:
        cfg.memory.region_constraint_flow_min_coverage = float(memory_region_flow_min_coverage)
    if memory_region_flow_max_coverage is not None:
        cfg.memory.region_constraint_flow_max_coverage = float(memory_region_flow_max_coverage)
    if memory_region_flow_feather_px is not None:
        cfg.memory.region_constraint_flow_feather_px = int(memory_region_flow_feather_px)
    if memory_region_samurai_model_cfg:
        cfg.memory.region_constraint_samurai_model_cfg = str(memory_region_samurai_model_cfg)
    if memory_region_samurai_checkpoint:
        cfg.memory.region_constraint_samurai_checkpoint = str(memory_region_samurai_checkpoint)
    if memory_region_samurai_offload_video_to_cpu is not None:
        cfg.memory.region_constraint_samurai_offload_video_to_cpu = bool(memory_region_samurai_offload_video_to_cpu)
    if memory_region_samurai_offload_state_to_cpu is not None:
        cfg.memory.region_constraint_samurai_offload_state_to_cpu = bool(memory_region_samurai_offload_state_to_cpu)
    if memory_region_threshold is not None:
        cfg.memory.region_constraint_threshold = float(memory_region_threshold)
    if memory_region_bbox_margin_px is not None:
        cfg.memory.region_constraint_bbox_margin_px = int(memory_region_bbox_margin_px)
    if memory_region_bbox_expand_ratio is not None:
        cfg.memory.region_constraint_bbox_expand_ratio = float(memory_region_bbox_expand_ratio)
    if memory_region_dilate_px is not None:
        cfg.memory.region_constraint_dilate_px = int(memory_region_dilate_px)
    if memory_region_soften_px is not None:
        cfg.memory.region_constraint_soften_px = int(memory_region_soften_px)
    if memory_region_outside_conf_cap is not None:
        cfg.memory.region_constraint_outside_confidence_cap = float(memory_region_outside_conf_cap)
    if refine_backend:
        cfg.refine.backend = str(refine_backend)
    if unknown_band_px is not None:
        cfg.refine.unknown_band_px = unknown_band_px
    if refine_mematte_repo_dir:
        cfg.refine.mematte_repo_dir = str(refine_mematte_repo_dir)
    if refine_mematte_checkpoint:
        cfg.refine.mematte_checkpoint = str(refine_mematte_checkpoint)
    if refine_mematte_max_number_token is not None:
        cfg.refine.mematte_max_number_token = int(refine_mematte_max_number_token)
    if refine_mematte_patch_decoder is not None:
        cfg.refine.mematte_patch_decoder = bool(refine_mematte_patch_decoder)
    if refine_region_trimap is not None:
        cfg.refine.region_trimap_enabled = bool(refine_region_trimap)
    if refine_region_trimap_threshold is not None:
        cfg.refine.region_trimap_threshold = float(refine_region_trimap_threshold)
    if refine_region_trimap_fg_erode_px is not None:
        cfg.refine.region_trimap_fg_erode_px = int(refine_region_trimap_fg_erode_px)
    if refine_region_trimap_bg_dilate_px is not None:
        cfg.refine.region_trimap_bg_dilate_px = int(refine_region_trimap_bg_dilate_px)
    if refine_region_trimap_cleanup_px is not None:
        cfg.refine.region_trimap_cleanup_px = int(refine_region_trimap_cleanup_px)
    if refine_region_trimap_keep_largest is not None:
        cfg.refine.region_trimap_keep_largest = bool(refine_region_trimap_keep_largest)
    if refine_region_trimap_min_coverage is not None:
        cfg.refine.region_trimap_min_coverage = float(refine_region_trimap_min_coverage)
    if refine_region_trimap_max_coverage is not None:
        cfg.refine.region_trimap_max_coverage = float(refine_region_trimap_max_coverage)
    if matte_tuning is not None:
        cfg.matte_tuning.enabled = matte_tuning
    if mt_shrink_grow_px is not None:
        cfg.matte_tuning.shrink_grow_px = mt_shrink_grow_px
    if mt_feather_px is not None:
        cfg.matte_tuning.feather_px = mt_feather_px
    if mt_offset_x_px is not None:
        cfg.matte_tuning.offset_x_px = mt_offset_x_px
    if mt_offset_y_px is not None:
        cfg.matte_tuning.offset_y_px = mt_offset_y_px

    if require_assignment is not None:
        cfg.assignment.require_assignment = require_assignment

    if debug_stage_samples is not None:
        cfg.debug.export_stage_samples = bool(debug_stage_samples)
    if debug_sample_count is not None:
        cfg.debug.sample_count = max(1, int(debug_sample_count))
    if debug_sample_frames:
        parsed_frames: list[int] = []
        for tok in str(debug_sample_frames).split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                parsed_frames.append(int(t))
            except ValueError as exc:
                raise click.BadParameter(
                    f"Invalid --debug-sample-frames token '{t}'. Use comma-separated integers."
                ) from exc
        cfg.debug.sample_frames = parsed_frames
    if debug_stage_dir:
        cfg.debug.stage_dir = str(debug_stage_dir)

    if resume is not None:
        cfg.runtime.resume = resume
    if qc is not None:
        cfg.qc.enabled = qc
    if qc_fail_on_regression is not None:
        cfg.qc.fail_on_regression = qc_fail_on_regression
    if qc_sample_output_frames is not None:
        cfg.qc.sample_output_frames = qc_sample_output_frames
    if qc_max_output_roundtrip_mae is not None:
        cfg.qc.max_output_roundtrip_mae = qc_max_output_roundtrip_mae
    if qc_alpha_range_eps is not None:
        cfg.qc.alpha_range_eps = qc_alpha_range_eps
    if qc_max_p95_flicker is not None:
        cfg.qc.max_p95_flicker = qc_max_p95_flicker
    if qc_max_p95_edge_flicker is not None:
        cfg.qc.max_p95_edge_flicker = qc_max_p95_edge_flicker
    if qc_min_mean_edge_confidence is not None:
        cfg.qc.min_mean_edge_confidence = qc_min_mean_edge_confidence
    if qc_band_spike_ratio is not None:
        cfg.qc.band_spike_ratio = qc_band_spike_ratio
    if qc_max_band_spike_frames is not None:
        cfg.qc.max_band_spike_frames = qc_max_band_spike_frames
    if device:
        cfg.runtime.device = device
    if precision:
        cfg.runtime.precision = precision
    if workers is not None:
        cfg.runtime.workers_io = workers

    if verbose:
        cfg.runtime.verbose = True

    if dump_config:
        import yaml

        click.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        return

    def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
        rgb = np.asarray(frame, dtype=np.float32)
        if rgb.ndim != 3:
            raise ValueError("Input frame must be RGB-like for propagation.")
        if rgb.shape[2] > 3:
            rgb = rgb[..., :3]
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).round().astype(np.uint8)

    # Optional assignment import path.
    project_file, project_state = ensure_project(cfg)
    if assign_mask:
        normalized_kind = str(assign_kind).lower()
        assignment = import_keyframe_mask(
            cfg=cfg,
            project_path=project_file,
            project=project_state,
            frame=assign_frame,
            mask_path=Path(assign_mask),
            source="cli",
            kind=normalized_kind,  # type: ignore[arg-type]
        )
        logger.info(
            f"Imported keyframe mask frame={assignment.frame} asset={assignment.mask_asset} "
            f"project={project_file}"
        )
        if normalized_kind == "correction" and apply_suggested_range:
            start, end = suggest_reprocess_range(
                project=project_state,
                anchor_frame=assign_frame,
                memory_window=cfg.memory.window,
                clip_start=cfg.io.frame_start,
                clip_end=cfg.io.frame_end,
            )
            cfg.io.frame_start = start
            cfg.io.frame_end = end
            logger.info(
                "Applied suggested reprocess range for correction anchor: start=%d end=%d",
                start,
                end,
            )

    if assign_only and propagate_from_frame is None:
        return

    if propagate_from_frame is not None:
        source = FrameSource(
            pattern=cfg.io.input,
            frame_start=cfg.io.frame_start,
            frame_end=cfg.io.frame_end,
            prefetch_workers=0,
        )
        try:
            num_frames = int(source.num_frames)
            if num_frames <= 0:
                raise ValueError("Input has no frames for propagation.")

            clip_abs_start = int(cfg.io.frame_start)
            clip_abs_end = int(cfg.io.frame_end) if int(cfg.io.frame_end) >= 0 else (clip_abs_start + num_frames - 1)

            anchor_candidates = [int(propagate_from_frame), int(propagate_from_frame) - clip_abs_start]
            local_anchor: int | None = None
            for candidate in anchor_candidates:
                if 0 <= candidate < num_frames:
                    local_anchor = int(candidate)
                    break
            if local_anchor is None:
                raise ValueError(
                    f"Propagation anchor frame {propagate_from_frame} is outside loaded range "
                    f"[{clip_abs_start}:{clip_abs_end}]"
                )

            anchor_abs = int(propagate_from_frame)
            if project_state.get_assignment(anchor_abs) is None:
                alt_abs = int(clip_abs_start + local_anchor)
                if project_state.get_assignment(alt_abs) is not None:
                    anchor_abs = alt_abs
                else:
                    raise ValueError(
                        f"Anchor frame {propagate_from_frame} has no assignment. "
                        "Import/build a keyframe first."
                    )

            start_abs = clip_abs_start if propagate_range_start is None else int(propagate_range_start)
            end_abs = clip_abs_end if propagate_range_end is None else int(propagate_range_end)
            start_abs = max(clip_abs_start, min(clip_abs_end, start_abs))
            end_abs = max(clip_abs_start, min(clip_abs_end, end_abs))
            if end_abs < start_abs:
                raise ValueError(f"Invalid propagation range: {start_abs}..{end_abs}")

            local_start = int(start_abs - clip_abs_start)
            local_end = int(end_abs - clip_abs_start)
            if local_anchor < local_start or local_anchor > local_end:
                raise ValueError(
                    f"Anchor frame {anchor_abs} is outside propagation range {start_abs}..{end_abs}"
                )

            anchor_rgb = _to_rgb_u8(source[local_anchor])
            h, w = anchor_rgb.shape[:2]
            keyframe_masks = load_keyframe_masks(project_file, project_state, target_shape=(h, w))
            anchor_alpha = keyframe_masks.get(anchor_abs)
            if anchor_alpha is None:
                raise ValueError(f"Failed to load anchor mask for frame {anchor_abs}")

            def _load_local_rgb(idx: int) -> np.ndarray:
                if idx < local_start or idx > local_end:
                    raise ValueError(f"Local frame {idx} outside range {local_start}..{local_end}")
                return _to_rgb_u8(source[idx])

            prop_result = propagate_masks_assist(
                frame_loader=_load_local_rgb,
                frame_start=local_start,
                frame_end=local_end,
                anchor_frame=local_anchor,
                anchor_mask=anchor_alpha,
                backend=propagate_backend,
                fallback_to_flow=bool(propagate_fallback_to_flow),
                flow_downscale=float(propagate_flow_downscale),
                flow_min_coverage=float(propagate_flow_min_coverage),
                flow_max_coverage=float(propagate_flow_max_coverage),
                flow_feather_px=max(0, int(propagate_flow_feather_px)),
                samurai_model_cfg=str(propagate_samurai_model_cfg or ""),
                samurai_checkpoint=str(propagate_samurai_checkpoint or ""),
                samurai_offload_video_to_cpu=bool(propagate_samurai_offload_video_to_cpu),
                samurai_offload_state_to_cpu=bool(propagate_samurai_offload_state_to_cpu),
                device_hint=str(cfg.runtime.device or "cuda"),
            )

            selected_local = select_propagation_frames(
                frame_start=local_start,
                frame_end=local_end,
                anchor_frame=local_anchor,
                stride=max(1, int(propagate_stride)),
                max_new_keyframes=max(1, int(propagate_max_new_keyframes)),
            )

            existing_frames = {int(item.frame) for item in project_state.keyframes}
            inserted = 0
            for local_idx in selected_local:
                abs_frame = int(clip_abs_start + local_idx)
                if abs_frame == anchor_abs:
                    continue
                if abs_frame in existing_frames and not bool(propagate_overwrite_existing):
                    continue
                alpha = prop_result.masks.get(int(local_idx))
                if alpha is None:
                    continue
                alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
                if float(alpha.mean()) < float(propagate_flow_min_coverage):
                    continue
                upsert_keyframe_alpha(
                    cfg=cfg,
                    project_path=project_file,
                    project=project_state,
                    frame=abs_frame,
                    alpha=alpha,
                    source=f"{str(propagate_source).strip() or 'cli_propagate'}:{prop_result.backend_used}",
                    kind=(str(propagate_kind).lower() if str(propagate_kind).lower() in ("initial", "correction") else "correction"),  # type: ignore[arg-type]
                )
                existing_frames.add(abs_frame)
                inserted += 1

            logger.info(
                "Phase 4 propagation complete: backend=%s inserted=%d range=%d..%d anchor=%d",
                prop_result.backend_used,
                inserted,
                start_abs,
                end_abs,
                anchor_abs,
            )
            if prop_result.note:
                logger.warning("Propagation note: %s", prop_result.note)
        finally:
            source.close()

    if assign_only or propagate_only:
        return

    from videomatte_hq.pipeline.orchestrator import run_pipeline

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
