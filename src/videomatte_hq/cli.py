"""CLI entry point for videomatte-hq.

Usage:
    videomatte-hq --in frames/%06d.exr --out out/alpha/%06d.png
    videomatte-hq --config config.yaml
    videomatte-hq --in video.mp4 --out out/alpha/%06d.png --alpha-format png16
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from videomatte_hq.config import (
    AlphaFormat,
    BandMode,
    ShotType,
    TemporalMethod,
    TrimapMethod,
    VideoMatteConfig,
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
@click.option("--input", "--in", "input_path", required=True, help="Input frames pattern or video file")
@click.option("--out", "output_path", default=None, help="Output alpha pattern (default: out/alpha/%%06d.png)")
@click.option("--config", "config_path", default=None, help="YAML config file (overrides defaults)")
@click.option("--fps", default=None, type=int, help="Framerate (default 30)")
@click.option("--shot-type", default=None, type=click.Choice(["locked_off", "handheld", "unknown"]))
@click.option("--bg-plate", default=None, help="Background plate: 'auto' or path")
@click.option("--roi", default=None, help="ROI mode: auto_person_track / segmentation / bg_sub / manual")
@click.option("--roi-detect-every", default=None, type=int)
@click.option("--roi-pad", default=None, type=float)
@click.option("--roi-context", default=None, type=int)
@click.option("--global-long-side", default=None, type=int)
@click.option("--intermediate-long-side", default=None, type=int)
@click.option("--band-mode", default=None, type=click.Choice(["adaptive", "threshold"]))
@click.option("--tile-size", default=None, type=int)
@click.option("--overlap", default=None, type=int)
@click.option("--trimap-method", default=None, type=click.Choice(["distance_transform", "erosion"]))
@click.option("--trimap-unknown-width", default=None, type=int)
@click.option("--temporal", default=None, type=click.Choice(["frequency_separation", "bilateral", "none"]))
@click.option("--temporal-detail-strength", default=None, type=float)
@click.option("--temporal-structural-strength", default=None, type=float)
@click.option("--alpha-format", default=None, type=click.Choice(["png16", "exr_dwaa", "exr_dwaa_hq", "exr_lossless", "exr_raw"]))
@click.option("--alpha-dwaa-quality", default=None, type=float)
@click.option("--despill/--no-despill", default=None)
@click.option("--preview/--no-preview", default=None)
@click.option("--preview-scale", default=None, type=int)
@click.option("--preview-every", default=None, type=int)
@click.option("--preview-modes", default=None, help="Comma-separated: checker,alpha,white,flicker")
@click.option("--resume/--no-resume", default=None)
@click.option("--device", default=None, help="Device: cuda / cpu")
@click.option("--precision", default=None, type=click.Choice(["fp16", "fp32"]))
@click.option("--frame-start", default=None, type=int)
@click.option("--frame-end", default=None, type=int)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--dump-config", is_flag=True, default=False, help="Dump resolved config as YAML and exit")
def main(
    input_path: str,
    output_path: str | None,
    config_path: str | None,
    fps: int | None,
    shot_type: str | None,
    bg_plate: str | None,
    roi: str | None,
    roi_detect_every: int | None,
    roi_pad: float | None,
    roi_context: int | None,
    global_long_side: int | None,
    intermediate_long_side: int | None,
    band_mode: str | None,
    tile_size: int | None,
    overlap: int | None,
    trimap_method: str | None,
    trimap_unknown_width: int | None,
    temporal: str | None,
    temporal_detail_strength: float | None,
    temporal_structural_strength: float | None,
    alpha_format: str | None,
    alpha_dwaa_quality: float | None,
    despill: bool | None,
    preview: bool | None,
    preview_scale: int | None,
    preview_every: int | None,
    preview_modes: str | None,
    resume: bool | None,
    device: str | None,
    precision: str | None,
    frame_start: int | None,
    frame_end: int | None,
    verbose: bool,
    dump_config: bool,
) -> None:
    """VideoMatte-HQ — High-quality 8K people video matting."""
    setup_logging(verbose)

    # Load base config
    if config_path:
        cfg = VideoMatteConfig.from_yaml(config_path)
    else:
        cfg = VideoMatteConfig.default()

    # Apply CLI overrides
    cfg.io.input = input_path
    if output_path:
        cfg.io.output_alpha = output_path
    if fps is not None:
        cfg.io.fps = fps
    if shot_type:
        cfg.io.shot_type = ShotType(shot_type)
    if alpha_format:
        cfg.io.alpha_format = AlphaFormat(alpha_format)
    if alpha_dwaa_quality is not None:
        cfg.io.alpha_dwaa_quality = alpha_dwaa_quality
    if frame_start is not None:
        cfg.io.frame_start = frame_start
    if frame_end is not None:
        cfg.io.frame_end = frame_end

    # ROI overrides
    if roi:
        from videomatte_hq.config import ROIMode
        cfg.roi.mode = ROIMode(roi)
    if roi_detect_every is not None:
        cfg.roi.detect_every = roi_detect_every
    if roi_pad is not None:
        cfg.roi.pad_ratio = roi_pad
    if roi_context is not None:
        cfg.roi.context_px = roi_context

    # Global pass overrides
    if global_long_side is not None:
        cfg.globals.long_side = global_long_side
    if intermediate_long_side is not None:
        cfg.intermediate.long_side = intermediate_long_side

    # Band/trimap overrides
    if band_mode:
        cfg.band.mode = BandMode(band_mode)
    if tile_size is not None:
        cfg.tiles.tile_size = tile_size
    if overlap is not None:
        cfg.tiles.overlap = overlap
    if trimap_method:
        cfg.trimap.method = TrimapMethod(trimap_method)
    if trimap_unknown_width is not None:
        cfg.trimap.unknown_width = trimap_unknown_width

    # Temporal overrides
    if temporal:
        cfg.temporal.method = TemporalMethod(temporal)
    if temporal_detail_strength is not None:
        cfg.temporal.detail_blend_strength = temporal_detail_strength
    if temporal_structural_strength is not None:
        cfg.temporal.structural_blend_strength = temporal_structural_strength

    # Postprocess overrides
    if despill is not None:
        cfg.postprocess.despill.enabled = despill

    # Preview overrides
    if preview is not None:
        cfg.preview.enabled = preview
    if preview_scale is not None:
        cfg.preview.scale = preview_scale
    if preview_every is not None:
        cfg.preview.every = preview_every
    if preview_modes:
        cfg.preview.modes = preview_modes.split(",")

    # Runtime overrides
    if resume is not None:
        cfg.runtime.resume = resume
    if device:
        cfg.runtime.device = device
    if precision:
        cfg.runtime.precision = precision

    # Dump config and exit if requested
    if dump_config:
        import yaml
        data = cfg.model_dump(by_alias=True)
        click.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
        return

    logger.info("VideoMatte-HQ starting")
    logger.info(f"Input: {cfg.io.input}")
    logger.info(f"Output: {cfg.io.output_alpha}")
    logger.info(f"Alpha format: {cfg.io.alpha_format.value}")
    logger.info(f"Shot type: {cfg.io.shot_type.value}")

    # Run the pipeline
    from videomatte_hq.pipeline.orchestrator import run_pipeline
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
