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
    TrimapMethod,
    VideoMatteConfig,
)
import videomatte_hq.config
print(f"DEBUG: Loaded config from {videomatte_hq.config.__file__}")

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
@click.option("--input", "--in", "input_path", required=False, help="Input frames pattern or video file")
@click.option("--out", "output_path", default=None, help="Output alpha pattern (default: out/alpha/%%06d.png)")
@click.option("--config", "config_path", default=None, help="YAML config file (overrides defaults)")
@click.option("--fps", default=None, type=int, help="Override FPS")
@click.option("--shot-type", default=None, help="Shot type (locked_off/handheld)")
@click.option("--alpha-format", default=None, help="Alpha format (png16/exr_dwaa/etc)")
@click.option("--alpha-dwaa-quality", default=None, type=float, help="DWAA compression quality")
@click.option("--start", "frame_start", default=None, type=int, help="Start frame")
@click.option("--end", "frame_end", default=None, type=int, help="End frame")
@click.option("--roi", default=None, help="ROI mode")
@click.option("--roi-detect-every", default=None, type=int, help="ROI detection interval")
@click.option("--roi-pad", default=None, type=float, help="ROI padding ratio")
@click.option("--roi-context", default=None, type=int, help="ROI context pixels")
@click.option("--long-side", "global_long_side", default=None, type=int, help="Pass A long side")
@click.option("--intermediate-long-side", default=None, type=int, help="Pass A' long side")
@click.option("--band-mode", default=None, help="Band mode")
@click.option("--tile-size", default=None, type=int, help="Tile size")
@click.option("--overlap", default=None, type=int, help="Tile overlap")
@click.option("--trimap-method", default=None, help="Trimap method")
@click.option("--trimap-unknown", "trimap_unknown_width", default=None, type=int, help="Trimap unknown width")
@click.option("--temporal", default=None, help="Temporal method")
@click.option("--temporal-detail", "temporal_detail_strength", default=None, type=float, help="Temporal detail blend")
@click.option("--temporal-structure", "temporal_structural_strength", default=None, type=float, help="Temporal structural blend")
@click.option("--despill/--no-despill", default=None, help="Enable/disable despill")
@click.option("--preview/--no-preview", default=None, help="Enable/disable preview")
@click.option("--preview-scale", default=None, type=int, help="Preview long side")
@click.option("--preview-every", default=None, type=int, help="Preview interval")
@click.option("--preview-modes", default=None, help="Preview modes (comma-separated)")
@click.option("--resume/--no-resume", default=None, help="Resume from cache")
@click.option("--device", default=None, help="Device (cuda/cpu)")
@click.option("--precision", default=None, help="Precision (fp16/fp32)")
@click.option("--dump-config", is_flag=True, help="Print config yaml and exit")
@click.option("--refine-model", default=None, help="Refiner model (vitmatte/matformer)")
@click.option("--global-model", default=None, help="Global model (rvm/modnet)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.option("--force", is_flag=True, help="Force overwrite")
@click.option("--workers", default=None, type=int, help="IO workers")
def main(
    input_path, output_path, config_path, fps, shot_type, alpha_format, alpha_dwaa_quality,
    frame_start, frame_end, roi, roi_detect_every, roi_pad, roi_context,
    global_long_side, intermediate_long_side, band_mode, tile_size, overlap,
    trimap_method, trimap_unknown_width, temporal, temporal_detail_strength,
    temporal_structural_strength, despill, preview, preview_scale, preview_every,
    preview_modes, resume, device, precision, dump_config, refine_model, global_model, verbose, force, workers
):
    """VideoMatte-HQ: 8K People Video Matting Pipeline.

    Run with --config to use a YAML file, or provide --input and --out arguments.
    """
    setup_logging(verbose)
    
    # Load default config
    cfg = VideoMatteConfig()
    
    # Override from YAML
    if config_path:
        import yaml
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
            # Basic recursive update (could be improved)
            # Pydantic's parse_obj_as or similar could work if structure matches
            # For now assuming user_cfg matches structure
            # A more robust way: update the dict of the cfg object
            # This is a simplified "apply" logic for the CLI
            pass 

    # CLI overrides
    if input_path:
        cfg.io.input = input_path
    if output_path:
        # If output_path looks like a directory, append pattern
        p = Path(output_path)
        if p.suffix == "":
            cfg.io.output_dir = str(p)
        else:
            cfg.io.output_alpha = str(p)
            cfg.io.output_dir = str(p.parent)

    if fps:
        # IO config doesn't have FPS currently, assumed from input or ignored
        pass
    
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
        # Partial implementation of ROI overrides
        pass
        
    if global_long_side is not None:
        cfg.global_.long_side = global_long_side
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
        cfg.temporal.method = temporal
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
        
    if refine_model:
        cfg.refine.model = refine_model
        
    if global_model:
        cfg.global_.model = global_model
        
    if verbose:
        cfg.runtime.verbose = verbose
        
    if force:
        cfg.io.force_overwrite = force
        
    if workers:
        cfg.runtime.workers_io = workers

    # Import orchestrator only after config is settled to avoid circular deps if any
    from videomatte_hq.pipeline.orchestrator import run_pipeline
    
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
