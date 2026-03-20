"""CLI entry point for the v2 pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.orchestrator import run_pipeline
from videomatte_hq.prompts.auto_anchor import AutoAnchorResult, build_auto_anchor_mask_for_video

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Videomatte-HQ v2 (SAM3 + MEMatte)")
    p.add_argument("--config", type=str, default="", help="Path to JSON/YAML config file.")

    p.add_argument("--input", type=str, default=None, help="Input sequence pattern or video path.")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--output-alpha", type=str, default=None, help="Output alpha filename pattern.")
    p.add_argument("--frame-start", type=int, default=None, help="Input start frame index.")
    p.add_argument("--frame-end", type=int, default=None, help="Input end frame index.")
    p.add_argument("--alpha-format", type=str, default=None, help="png16 | exr_dwaa | exr_lossless | exr_raw")

    p.add_argument("--anchor-mask", type=str, default=None, help="Anchor mask image for frame 0.")
    p.add_argument("--anchor-frame", type=int, default=None, help="Anchor frame index (currently 0 only).")
    p.add_argument(
        "--auto-anchor",
        dest="auto_anchor",
        action="store_true",
        help="Auto-generate anchor mask for video input when --anchor-mask is not provided.",
    )
    p.add_argument(
        "--no-auto-anchor",
        dest="auto_anchor",
        action="store_false",
        help="Disable auto-anchor generation and require --anchor-mask.",
    )
    p.set_defaults(auto_anchor=None)
    p.add_argument(
        "--auto-anchor-output",
        type=str,
        default=None,
        help="Output path for generated auto-anchor mask (default: <output-dir>/anchor_mask.auto.png).",
    )
    p.add_argument(
        "--segment-backend",
        type=str,
        default=None,
        help="ultralytics_sam3 | static",
    )
    p.add_argument("--sam3-model", type=str, default=None, help="Ultralytics SAM checkpoint name/path.")
    p.add_argument(
        "--sam3-processing-long-side",
        type=int,
        default=None,
        help="Resize long side for SAM stage-1 inference (higher values retain more hair detail but cost more runtime).",
    )
    p.add_argument("--chunk-size", type=int, default=None, help="Segmentation chunk size.")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap.")
    p.add_argument("--mask-hysteresis", action="store_true", dest="mask_hysteresis_enabled", default=None,
                   help="Enable Stage-1 Schmitt-trigger mask hysteresis so borderline pixels keep their previous label until confidence clears a low/high band.")
    p.add_argument("--no-mask-hysteresis", action="store_false", dest="mask_hysteresis_enabled",
                   help="Disable Stage-1 mask hysteresis.")
    p.add_argument("--mask-hysteresis-low", type=float, default=None,
                   help="Stage-1 probability threshold below which pixels flip to background when hysteresis is enabled.")
    p.add_argument("--mask-hysteresis-high", type=float, default=None,
                   help="Stage-1 probability threshold above which pixels flip to foreground when hysteresis is enabled.")

    # ---- Pipeline Mode (v1/v2) ----
    p.add_argument(
        "--pipeline-mode",
        type=str,
        default=None,
        choices=["v1", "v2"],
        help="Pipeline mode: 'v1' (SAM3+MEMatte every frame) or 'v2' (MatAnyone2 temporal matting).",
    )
    p.add_argument("--matanyone2-repo-dir", type=str, default=None,
                   help="Path to cloned MatAnyone2 repository.")
    p.add_argument("--matanyone2-max-size", type=int, default=None,
                   help="Max resolution (short edge) for MatAnyone2 processing (default 1080).")
    p.add_argument("--matanyone2-warmup", type=int, default=None,
                   help="Number of MatAnyone2 warmup iterations on first frame (default 10).")
    p.add_argument("--matanyone2-hires-threshold", type=int, default=None,
                   help="Short edge threshold above which MEMatte refinement runs in v2 mode (default 1080).")

    p.add_argument("--refine-enabled", action="store_true", help="Enable MEMatte refinement.")
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable MEMatte refinement (unsupported in this build; preflight will fail).",
    )
    p.add_argument("--mematte-repo-dir", type=str, default=None, help="Path to local MEMatte repository.")
    p.add_argument("--mematte-checkpoint", type=str, default=None, help="Path to MEMatte checkpoint (.pth).")
    p.add_argument(
        "--allow-external-paths",
        action="store_true",
        help="Allow MEMatte repo/checkpoint paths outside this repository (power-user override).",
    )
    p.add_argument("--tile-size", type=int, default=None, help="Refinement tile size.")
    p.add_argument("--tile-overlap", type=int, default=None, help="Refinement tile overlap.")
    p.add_argument(
        "--tile-batch-size",
        type=int,
        default=None,
        help="Max number of same-sized MEMatte tiles to batch together per Stage-2 call.",
    )
    p.add_argument("--trimap-fg-threshold", type=float, default=None, help="Definite FG probability threshold.")
    p.add_argument("--trimap-bg-threshold", type=float, default=None, help="Definite BG probability threshold.")
    p.add_argument(
        "--trimap-mode",
        type=str,
        default=None,
        choices=["morphological", "hybrid", "logit"],
        help="Trimap generation mode: 'morphological' (stable erosion/dilation band), 'hybrid' (morphology + logit uncertainty), or 'logit' (legacy threshold-based).",
    )
    p.add_argument("--trimap-erosion-px", type=int, default=None, help="Morphological trimap erosion radius in pixels (default 20).")
    p.add_argument("--trimap-dilation-px", type=int, default=None, help="Morphological trimap dilation radius in pixels (default 10).")
    p.add_argument(
        "--trimap-fallback-band-px",
        type=int,
        default=None,
        help="Fallback trimap edge band width in pixels when threshold trimap is empty (hard-mask SAM outputs).",
    )
    p.add_argument(
        "--unknown-edge-blend-px",
        type=int,
        default=None,
        help="Blend refined alpha back toward the coarse prior within this many pixels of unknown-band edges (0 disables).",
    )

    p.add_argument("--mask-temporal-smooth-radius", type=int, default=None,
                   help="Temporal median radius for SAM masks before trimap (0=off, 1=3-frame, 2=5-frame).")
    p.add_argument("--temporal-smooth", action="store_true", dest="temporal_smooth_enabled", default=None)
    p.add_argument("--no-temporal-smooth", action="store_false", dest="temporal_smooth_enabled")
    p.add_argument("--temporal-smooth-strength", type=float, default=None)
    p.add_argument("--temporal-smooth-motion-threshold", type=float, default=None)

    p.add_argument("--shrink-grow-px", type=int, default=None, help="Matte shrink/grow amount.")
    p.add_argument("--feather-px", type=int, default=None, help="Matte feather radius.")
    p.add_argument("--offset-x-px", type=int, default=None, help="Matte x-offset.")
    p.add_argument("--offset-y-px", type=int, default=None, help="Matte y-offset.")

    p.add_argument(
        "--prompt-mode",
        type=str,
        default=None,
        choices=["mask", "points"],
        help="Prompt mode: 'mask' (default, uses anchor mask) or 'points' (interactive foreground/background points).",
    )
    p.add_argument(
        "--point-prompts-json",
        type=str,
        default=None,
        help="JSON string or file path with normalized [0,1] point prompts keyed by frame index.",
    )
    p.add_argument("--device", type=str, default=None, help="Runtime device, e.g. cuda/cpu.")
    p.add_argument("--precision", type=str, default=None, help="Runtime precision hint.")
    p.add_argument("--workers-io", type=int, default=None, help="IO worker count.")
    p.add_argument("--generate-preview-mp4", action="store_true", dest="generate_preview_mp4", default=None,
                   help="Generate an H.264 MP4 preview of the alpha output (default: on).")
    p.add_argument("--no-preview-mp4", action="store_false", dest="generate_preview_mp4",
                   help="Skip MP4 preview generation.")
    p.add_argument("--preview-fps", type=float, default=None,
                   help="Override FPS for the preview MP4 (default: auto-detect from source).")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def _apply_optional(cfg: VideoMatteConfig, name: str, value) -> None:
    if value is not None:
        setattr(cfg, name, value)


def _apply_cli_overrides(cfg: VideoMatteConfig, args: argparse.Namespace) -> VideoMatteConfig:
    _apply_optional(cfg, "input", args.input)
    _apply_optional(cfg, "output_dir", args.output_dir)
    _apply_optional(cfg, "output_alpha", args.output_alpha)
    _apply_optional(cfg, "frame_start", args.frame_start)
    _apply_optional(cfg, "frame_end", args.frame_end)
    _apply_optional(cfg, "alpha_format", args.alpha_format)

    _apply_optional(cfg, "pipeline_mode", args.pipeline_mode)
    _apply_optional(cfg, "matanyone2_repo_dir", args.matanyone2_repo_dir)
    _apply_optional(cfg, "matanyone2_max_size", args.matanyone2_max_size)
    _apply_optional(cfg, "matanyone2_warmup", args.matanyone2_warmup)
    _apply_optional(cfg, "matanyone2_hires_threshold", args.matanyone2_hires_threshold)

    _apply_optional(cfg, "anchor_mask", args.anchor_mask)
    _apply_optional(cfg, "anchor_frame", args.anchor_frame)
    _apply_optional(cfg, "segment_backend", args.segment_backend)
    _apply_optional(cfg, "sam3_model", args.sam3_model)
    _apply_optional(cfg, "sam3_processing_long_side", args.sam3_processing_long_side)
    _apply_optional(cfg, "chunk_size", args.chunk_size)
    _apply_optional(cfg, "chunk_overlap", args.chunk_overlap)
    if args.mask_hysteresis_enabled is not None:
        cfg.mask_hysteresis_enabled = args.mask_hysteresis_enabled
    _apply_optional(cfg, "mask_hysteresis_low", args.mask_hysteresis_low)
    _apply_optional(cfg, "mask_hysteresis_high", args.mask_hysteresis_high)

    if args.refine_enabled:
        cfg.refine_enabled = True
    if args.no_refine:
        cfg.refine_enabled = False
    _apply_optional(cfg, "mematte_repo_dir", args.mematte_repo_dir)
    _apply_optional(cfg, "mematte_checkpoint", args.mematte_checkpoint)
    _apply_optional(cfg, "tile_size", args.tile_size)
    _apply_optional(cfg, "tile_overlap", args.tile_overlap)
    _apply_optional(cfg, "tile_batch_size", args.tile_batch_size)
    _apply_optional(cfg, "trimap_mode", args.trimap_mode)
    _apply_optional(cfg, "trimap_erosion_px", args.trimap_erosion_px)
    _apply_optional(cfg, "trimap_dilation_px", args.trimap_dilation_px)
    _apply_optional(cfg, "trimap_fg_threshold", args.trimap_fg_threshold)
    _apply_optional(cfg, "trimap_bg_threshold", args.trimap_bg_threshold)
    _apply_optional(cfg, "trimap_fallback_band_px", args.trimap_fallback_band_px)
    _apply_optional(cfg, "unknown_edge_blend_px", args.unknown_edge_blend_px)

    if args.mask_temporal_smooth_radius is not None:
        cfg.mask_temporal_smooth_radius = max(0, min(2, args.mask_temporal_smooth_radius))
    if args.temporal_smooth_enabled is not None:
        cfg.temporal_smooth_enabled = args.temporal_smooth_enabled
    _apply_optional(cfg, "temporal_smooth_strength", args.temporal_smooth_strength)
    _apply_optional(cfg, "temporal_smooth_motion_threshold", args.temporal_smooth_motion_threshold)

    _apply_optional(cfg, "shrink_grow_px", args.shrink_grow_px)
    _apply_optional(cfg, "feather_px", args.feather_px)
    _apply_optional(cfg, "offset_x_px", args.offset_x_px)
    _apply_optional(cfg, "offset_y_px", args.offset_y_px)

    _apply_optional(cfg, "prompt_mode", args.prompt_mode)
    if args.point_prompts_json is not None:
        json_val = args.point_prompts_json
        # If it looks like a file path, read from file
        if json_val and not json_val.strip().startswith("{") and Path(json_val).is_file():
            json_val = Path(json_val).read_text(encoding="utf-8")
        cfg.point_prompts_json = json_val
    if args.generate_preview_mp4 is not None:
        cfg.generate_preview_mp4 = args.generate_preview_mp4
    _apply_optional(cfg, "preview_fps", args.preview_fps)
    _apply_optional(cfg, "device", args.device)
    _apply_optional(cfg, "precision", args.precision)
    _apply_optional(cfg, "workers_io", args.workers_io)
    return cfg


def _looks_like_video_input(value: str) -> bool:
    path = Path(str(value))
    if "%" in str(value) or "*" in str(value):
        return False
    suffix = path.suffix.lower()
    return suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _check_runtime_dependency(name: str) -> None:
    __import__(name)


def _is_pathlike_model_reference(model_ref: str) -> bool:
    text = str(model_ref).strip()
    if not text:
        return False
    return any(ch in text for ch in ("/", "\\")) or text.lower().endswith(".pt") and Path(text).exists()


def _resolve_path_under_repo(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _ensure_within_repo(path: Path, *, label: str) -> Path:
    repo = _repo_root().resolve()
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(repo)
    except Exception as exc:
        raise ValueError(
            f"{label} must be inside this repository ({repo}), got: {candidate}. "
            "External MEMatte paths are not allowed for this tool."
        ) from exc
    return candidate


def _run_preflight_checks(cfg: VideoMatteConfig, *, allow_external_paths: bool = False) -> None:
    input_path = Path(str(cfg.input))
    if _looks_like_video_input(cfg.input):
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
    elif not str(cfg.input).strip():
        raise ValueError("input is required.")

    is_v2 = str(cfg.pipeline_mode).strip().lower() == "v2"

    # SAM/ultralytics check — needed for v1 always, and v2 for first-frame mask
    if str(cfg.segment_backend).strip().lower() == "ultralytics_sam3":
        try:
            _check_runtime_dependency("ultralytics")
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is required for segment_backend='ultralytics_sam3'. "
                "Install with `pip install ultralytics`."
            ) from exc
        if _is_pathlike_model_reference(cfg.sam3_model):
            sam_path = Path(str(cfg.sam3_model))
            if not sam_path.exists():
                raise FileNotFoundError(f"SAM model checkpoint not found: {sam_path}")

    if cfg.prompt_mode == "points":
        if not cfg.point_prompts_json.strip():
            raise ValueError("point_prompts_json is required when prompt_mode is 'points'.")
    elif str(cfg.anchor_mask).strip():
        anchor_path = Path(str(cfg.anchor_mask))
        if not anchor_path.exists():
            raise FileNotFoundError(f"Anchor mask not found: {anchor_path}")

    if int(cfg.frame_end) >= 0 and int(cfg.frame_end) < int(cfg.frame_start):
        raise ValueError(
            f"Invalid frame range after preflight: frame_end ({cfg.frame_end}) < frame_start ({cfg.frame_start})."
        )
    if not is_v2:
        if int(cfg.chunk_overlap) >= int(cfg.chunk_size):
            raise ValueError(
                f"chunk_overlap ({cfg.chunk_overlap}) must be less than chunk_size ({cfg.chunk_size})."
            )
    if not (0.0 <= float(cfg.mask_hysteresis_low) <= 1.0 and 0.0 <= float(cfg.mask_hysteresis_high) <= 1.0):
        raise ValueError(
            "mask_hysteresis_low and mask_hysteresis_high must both be within [0, 1]."
        )
    if float(cfg.mask_hysteresis_low) >= float(cfg.mask_hysteresis_high):
        raise ValueError(
            f"mask_hysteresis_low ({cfg.mask_hysteresis_low}) must be less than mask_hysteresis_high ({cfg.mask_hysteresis_high})."
        )
    if int(cfg.anchor_frame) != 0:
        raise ValueError(
            f"Unsupported anchor_frame ({cfg.anchor_frame}) for v2 pipeline. Only anchor_frame=0 is currently supported."
        )

    # ---- v2 MatAnyone2 preflight ----
    if is_v2:
        matanyone2_repo = _resolve_path_under_repo(str(cfg.matanyone2_repo_dir))
        if not matanyone2_repo.exists():
            raise FileNotFoundError(
                f"MatAnyone2 repo dir not found: {matanyone2_repo}. "
                "Clone MatAnyone2 into third_party/MatAnyone2."
            )
        cfg.matanyone2_repo_dir = str(matanyone2_repo)

        # In v2, MEMatte is only required if source exceeds hires_threshold.
        # We can't know the source resolution at preflight time, so we check
        # MEMatte assets only if refine_enabled is True.
        if not bool(cfg.refine_enabled):
            logger.info(
                "v2 pipeline: MEMatte refinement disabled — MatAnyone2 output will be bicubic-upscaled for hires content."
            )
            return
        # Fall through to MEMatte asset checks below
    else:
        if not bool(cfg.refine_enabled):
            logger.info(
                "MEMatte refinement disabled — pipeline will produce SAM-only preview alphas."
            )
            return

    # ---- MEMatte asset checks (v1 always, v2 when refine_enabled) ----
    mematte_repo_candidate = _resolve_path_under_repo(str(cfg.mematte_repo_dir))
    mematte_ckpt_candidate = _resolve_path_under_repo(str(cfg.mematte_checkpoint))
    if allow_external_paths:
        mematte_repo = mematte_repo_candidate
        mematte_ckpt = mematte_ckpt_candidate
    else:
        mematte_repo = _ensure_within_repo(
            mematte_repo_candidate,
            label="MEMatte repo dir",
        )
        mematte_ckpt = _ensure_within_repo(
            mematte_ckpt_candidate,
            label="MEMatte checkpoint",
        )
    # Persist normalized absolute local paths for runtime and metadata.
    cfg.mematte_repo_dir = str(mematte_repo)
    cfg.mematte_checkpoint = str(mematte_ckpt)
    if not mematte_repo.exists():
        raise FileNotFoundError(
            f"MEMatte repo dir not found: {mematte_repo}. "
            "Populate third_party/MEMatte inside this repo."
        )
    if not (mematte_repo / "inference.py").exists():
        raise FileNotFoundError(f"MEMatte repo dir looks invalid (missing inference.py): {mematte_repo}")
    if not mematte_ckpt.exists():
        raise FileNotFoundError(
            f"MEMatte checkpoint not found: {mematte_ckpt}. "
            "Place the checkpoint under third_party/MEMatte/checkpoints inside this repo."
        )


def _resolve_auto_anchor(
    cfg: VideoMatteConfig,
    args: argparse.Namespace,
) -> AutoAnchorResult | None:
    if str(cfg.anchor_mask).strip():
        return None

    auto_anchor_enabled = bool(args.auto_anchor) if args.auto_anchor is not None else _looks_like_video_input(cfg.input)
    if not auto_anchor_enabled:
        raise ValueError(
            "anchor_mask is required for v2 pipeline runs. "
            "Provide --anchor-mask or enable --auto-anchor for video inputs."
        )
    if not _looks_like_video_input(cfg.input):
        raise ValueError(
            "Auto-anchor currently supports video file inputs only. "
            "Provide --anchor-mask for image sequence patterns."
        )

    output_dir = Path(str(cfg.output_dir))
    auto_anchor_out = Path(args.auto_anchor_output) if args.auto_anchor_output else (output_dir / "anchor_mask.auto.png")
    requested_start = int(cfg.frame_start)
    # v2 pipeline needs a tight mask (MatAnyone2 treats it as ground truth);
    # v1 pipeline benefits from a generous mask (SAM re-segments every frame).
    use_tight = str(cfg.pipeline_mode).strip().lower() == "v2"
    result = build_auto_anchor_mask_for_video(
        cfg.input,
        auto_anchor_out,
        device=str(cfg.device),
        frame_start=requested_start,
        tight=use_tight,
    )
    cfg.anchor_mask = str(result.mask_path)
    effective_start = max(requested_start, int(result.probe_frame))
    if effective_start != requested_start:
        logger.info(
            "Auto-anchor probed frame %d (requested frame_start=%d). Using effective frame_start=%d.",
            int(result.probe_frame),
            requested_start,
            effective_start,
        )
        cfg.frame_start = int(effective_start)
    logger.info("Auto-anchor generated via %s: %s", result.method, result.mask_path)
    return result


def _write_cli_run_metadata(
    cfg: VideoMatteConfig,
    *,
    requested_frame_start: int,
    auto_anchor: AutoAnchorResult | None,
    allow_external_paths: bool,
) -> None:
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "input": str(cfg.input),
        "output_dir": str(out_dir.resolve()),
        "pipeline_mode": str(cfg.pipeline_mode),
        "requested_frame_start": int(requested_frame_start),
        "frame_start": int(cfg.frame_start),
        "frame_end": int(cfg.frame_end),
        "segment_backend": str(cfg.segment_backend),
        "sam3_model": str(cfg.sam3_model),
        "refine_enabled": bool(cfg.refine_enabled),
        "trimap_mode": str(cfg.trimap_mode),
        "trimap_erosion_px": int(cfg.trimap_erosion_px),
        "trimap_dilation_px": int(cfg.trimap_dilation_px),
        "trimap_fg_threshold": float(cfg.trimap_fg_threshold),
        "trimap_bg_threshold": float(cfg.trimap_bg_threshold),
        "trimap_fallback_band_px": int(cfg.trimap_fallback_band_px),
        "mematte_repo_dir": str(cfg.mematte_repo_dir),
        "mematte_checkpoint": str(cfg.mematte_checkpoint),
        "device": str(cfg.device),
        "precision": str(cfg.precision),
        "anchor_mask": str(cfg.anchor_mask),
        "allow_external_paths": bool(allow_external_paths),
    }
    if str(cfg.pipeline_mode) == "v2":
        summary["matanyone2_repo_dir"] = str(cfg.matanyone2_repo_dir)
        summary["matanyone2_max_size"] = int(cfg.matanyone2_max_size)
        summary["matanyone2_hires_threshold"] = int(cfg.matanyone2_hires_threshold)
    if auto_anchor is not None:
        summary["auto_anchor"] = {
            "mask_path": str(auto_anchor.mask_path),
            "method": str(auto_anchor.method),
            "probe_frame": int(auto_anchor.probe_frame),
        }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if args.config:
        cfg = VideoMatteConfig.from_file(Path(args.config))
    else:
        cfg = VideoMatteConfig()
    cfg = _apply_cli_overrides(cfg, args)

    requested_frame_start = int(cfg.frame_start)
    auto_anchor_result = None
    if cfg.prompt_mode != "points":
        auto_anchor_result = _resolve_auto_anchor(cfg, args)
    _run_preflight_checks(cfg, allow_external_paths=bool(args.allow_external_paths))
    _write_cli_run_metadata(
        cfg,
        requested_frame_start=requested_frame_start,
        auto_anchor=auto_anchor_result,
        allow_external_paths=bool(args.allow_external_paths),
    )

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
