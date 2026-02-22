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
    p.add_argument("--chunk-size", type=int, default=None, help="Segmentation chunk size.")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap.")

    p.add_argument("--refine-enabled", action="store_true", help="Enable MEMatte refinement.")
    p.add_argument("--no-refine", action="store_true", help="Disable MEMatte refinement.")
    p.add_argument("--mematte-repo-dir", type=str, default=None, help="Path to local MEMatte repository.")
    p.add_argument("--mematte-checkpoint", type=str, default=None, help="Path to MEMatte checkpoint (.pth).")
    p.add_argument("--tile-size", type=int, default=None, help="Refinement tile size.")
    p.add_argument("--tile-overlap", type=int, default=None, help="Refinement tile overlap.")
    p.add_argument("--trimap-fg-threshold", type=float, default=None, help="Definite FG probability threshold.")
    p.add_argument("--trimap-bg-threshold", type=float, default=None, help="Definite BG probability threshold.")

    p.add_argument("--shrink-grow-px", type=int, default=None, help="Matte shrink/grow amount.")
    p.add_argument("--feather-px", type=int, default=None, help="Matte feather radius.")
    p.add_argument("--offset-x-px", type=int, default=None, help="Matte x-offset.")
    p.add_argument("--offset-y-px", type=int, default=None, help="Matte y-offset.")

    p.add_argument("--device", type=str, default=None, help="Runtime device, e.g. cuda/cpu.")
    p.add_argument("--precision", type=str, default=None, help="Runtime precision hint.")
    p.add_argument("--workers-io", type=int, default=None, help="IO worker count.")
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

    _apply_optional(cfg, "anchor_mask", args.anchor_mask)
    _apply_optional(cfg, "anchor_frame", args.anchor_frame)
    _apply_optional(cfg, "segment_backend", args.segment_backend)
    _apply_optional(cfg, "sam3_model", args.sam3_model)
    _apply_optional(cfg, "chunk_size", args.chunk_size)
    _apply_optional(cfg, "chunk_overlap", args.chunk_overlap)

    if args.refine_enabled:
        cfg.refine_enabled = True
    if args.no_refine:
        cfg.refine_enabled = False
    _apply_optional(cfg, "mematte_repo_dir", args.mematte_repo_dir)
    _apply_optional(cfg, "mematte_checkpoint", args.mematte_checkpoint)
    _apply_optional(cfg, "tile_size", args.tile_size)
    _apply_optional(cfg, "tile_overlap", args.tile_overlap)
    _apply_optional(cfg, "trimap_fg_threshold", args.trimap_fg_threshold)
    _apply_optional(cfg, "trimap_bg_threshold", args.trimap_bg_threshold)

    _apply_optional(cfg, "shrink_grow_px", args.shrink_grow_px)
    _apply_optional(cfg, "feather_px", args.feather_px)
    _apply_optional(cfg, "offset_x_px", args.offset_x_px)
    _apply_optional(cfg, "offset_y_px", args.offset_y_px)

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


def _run_preflight_checks(cfg: VideoMatteConfig) -> None:
    input_path = Path(str(cfg.input))
    if _looks_like_video_input(cfg.input):
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
    elif not str(cfg.input).strip():
        raise ValueError("input is required.")

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

    if str(cfg.anchor_mask).strip():
        anchor_path = Path(str(cfg.anchor_mask))
        if not anchor_path.exists():
            raise FileNotFoundError(f"Anchor mask not found: {anchor_path}")

    if int(cfg.frame_end) >= 0 and int(cfg.frame_end) < int(cfg.frame_start):
        raise ValueError(
            f"Invalid frame range after preflight: frame_end ({cfg.frame_end}) < frame_start ({cfg.frame_start})."
        )

    if bool(cfg.refine_enabled):
        mematte_repo = _ensure_within_repo(
            _resolve_path_under_repo(str(cfg.mematte_repo_dir)),
            label="MEMatte repo dir",
        )
        mematte_ckpt = _ensure_within_repo(
            _resolve_path_under_repo(str(cfg.mematte_checkpoint)),
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
    result = build_auto_anchor_mask_for_video(
        cfg.input,
        auto_anchor_out,
        device=str(cfg.device),
        frame_start=requested_start,
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
) -> None:
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "input": str(cfg.input),
        "output_dir": str(out_dir.resolve()),
        "requested_frame_start": int(requested_frame_start),
        "frame_start": int(cfg.frame_start),
        "frame_end": int(cfg.frame_end),
        "segment_backend": str(cfg.segment_backend),
        "sam3_model": str(cfg.sam3_model),
        "refine_enabled": bool(cfg.refine_enabled),
        "mematte_repo_dir": str(cfg.mematte_repo_dir),
        "mematte_checkpoint": str(cfg.mematte_checkpoint),
        "device": str(cfg.device),
        "precision": str(cfg.precision),
        "anchor_mask": str(cfg.anchor_mask),
    }
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
    auto_anchor_result = _resolve_auto_anchor(cfg, args)
    _run_preflight_checks(cfg)
    _write_cli_run_metadata(
        cfg,
        requested_frame_start=requested_frame_start,
        auto_anchor=auto_anchor_result,
    )

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
