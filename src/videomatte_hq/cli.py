"""CLI entry point for the Option B VideoMatte-HQ runtime."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from videomatte_hq.config import AlphaFormat, ShotType, VideoMatteConfig
from videomatte_hq.project import ensure_project, import_keyframe_mask, suggest_reprocess_range

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
    "--unknown-band-px",
    "--mt-trimap-width-px",
    "unknown_band_px",
    default=None,
    type=int,
    help="Refine trimap unknown-band width in pixels",
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
    unknown_band_px,
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
    if unknown_band_px is not None:
        cfg.refine.unknown_band_px = unknown_band_px
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

    # Optional assignment import path.
    if assign_mask:
        project_file, project_state = ensure_project(cfg)
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
        if assign_only:
            return

    from videomatte_hq.pipeline.orchestrator import run_pipeline

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
