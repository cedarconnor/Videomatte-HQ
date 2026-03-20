"""v2 pipeline orchestrator.

Supports two pipeline modes:
  - v1: SAM3 segmentation (every frame) + MEMatte refinement (every frame)
  - v2: SAM3 (frame 0 only) → MatAnyone2 (temporal matting) → MEMatte (>1080p only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.reader import FrameSource
from videomatte_hq.io.writer import AlphaWriter
from videomatte_hq.pipeline.stage_refine import (
    MEMatteEdgeRefiner,
    RefineSequenceResult,
    _refine_frame_tiled,
    _to_rgb_float,
    build_edge_refiner,
    refine_sequence,
)
from videomatte_hq.pipeline.stage_segment import build_segmenter
from videomatte_hq.postprocess.mask_temporal import smooth_logits_temporal, smooth_masks_temporal
from videomatte_hq.postprocess.matte_tuning import apply_matte_tuning
from videomatte_hq.postprocess.temporal_smooth import apply_temporal_smooth
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter
from videomatte_hq.prompts.point_adapter import PointPromptAdapter, parse_point_prompts
from videomatte_hq.protocols import SegmentPrompt, SegmentResult

logger = logging.getLogger(__name__)

QC_TRIMAP_DIR = "qc"
QC_TRIMAP_PATTERN = "trimap.%06d.png"


@dataclass(slots=True)
class PipelineRunResult:
    segment_result: Optional[SegmentResult]
    refine_result: RefineSequenceResult
    output_dir: Path


def _read_anchor_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Anchor mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read anchor mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype == np.uint8:
        mask_f = mask.astype(np.float32) / 255.0
    elif mask.dtype == np.uint16:
        mask_f = mask.astype(np.float32) / 65535.0
    elif np.issubdtype(mask.dtype, np.integer):
        mask_f = mask.astype(np.float32) / float(np.iinfo(mask.dtype).max)
    else:
        mask_f = mask.astype(np.float32)
        max_val = float(mask_f.max()) if mask_f.size else 1.0
        if max_val > 1.0:
            mask_f = mask_f / max(max_val, 1.0)

    h, w = int(shape[0]), int(shape[1])
    if mask_f.shape != (h, w):
        mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask_f, 0.0, 1.0).astype(np.float32)


def _qc_trimap_frame_path(output_dir: Path, frame_idx: int) -> Path:
    return output_dir / QC_TRIMAP_DIR / (QC_TRIMAP_PATTERN % int(frame_idx))


def _write_qc_trimap_preview_png(output_dir: Path, frame_idx: int, trimap: np.ndarray) -> None:
    path = _qc_trimap_frame_path(output_dir, frame_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = np.asarray(trimap, dtype=np.float32)
    out = np.full(tri.shape, 128, dtype=np.uint8)
    out[tri >= 1.0] = 255
    out[tri <= 0.0] = 0
    cv2.imwrite(str(path), out)


# ---------------------------------------------------------------------------
# First-frame mask acquisition (shared by v1 and v2)
# ---------------------------------------------------------------------------


def _get_first_frame_mask(
    cfg: VideoMatteConfig,
    source: FrameSource,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """Get a binary mask for frame 0 via anchor mask, points, or auto-anchor."""
    if cfg.prompt_mode == "points":
        parsed = parse_point_prompts(cfg.point_prompts_json, frame_shape)
        frame0_pts = parsed.get(0, {"positive": [], "negative": []})
        if not frame0_pts["positive"]:
            raise ValueError("Point prompt mode requires at least one positive point on frame 0.")

        prompt_adapter = PointPromptAdapter(
            positive_points=frame0_pts["positive"],
            negative_points=frame0_pts["negative"],
        )
        prompt = prompt_adapter.adapt(
            np.zeros(frame_shape, dtype=np.float32), frame_shape
        )
        logger.info(
            "Point prompt mode: %d positive, %d negative points on frame 0.",
            len(frame0_pts["positive"]),
            len(frame0_pts["negative"]),
        )

        # Run SAM on just frame 0 to get a mask
        seg_cfg = cfg.segment_stage_config()
        seg_cfg.chunk_size = 1
        seg_cfg.chunk_overlap = 0
        segmenter = build_segmenter(seg_cfg, prompt_adapter=prompt_adapter)
        seg_result = segmenter.segment_sequence(
            source=source,
            prompt=prompt,
            anchor_frame=0,
            chunk_size=1,
            chunk_overlap=0,
        )
        if seg_result.masks:
            return (np.asarray(seg_result.masks[0], dtype=np.float32) >= 0.5).astype(np.float32)
        raise RuntimeError("SAM failed to produce a mask for frame 0.")
    else:
        return _read_anchor_mask(cfg.anchor_mask, frame_shape)


# ---------------------------------------------------------------------------
# v1 pipeline (unchanged behavior)
# ---------------------------------------------------------------------------


def _run_pipeline_v1(cfg: VideoMatteConfig) -> PipelineRunResult:
    """Execute v1 segmentation + refinement pipeline."""
    if cfg.prompt_mode == "mask" and not str(cfg.anchor_mask).strip():
        raise ValueError("anchor_mask is required for v2 pipeline runs in mask mode.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = FrameSource(
        pattern=cfg.input,
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        prefetch_workers=max(0, int(cfg.workers_io)),
    )

    try:
        frame_shape = source.resolution
        logger.info("Loaded source: frames=%d, shape=%dx%d", len(source), frame_shape[1], frame_shape[0])

        if cfg.prompt_mode == "points":
            parsed = parse_point_prompts(cfg.point_prompts_json, frame_shape)
            frame0_pts = parsed.get(0, {"positive": [], "negative": []})
            if not frame0_pts["positive"]:
                raise ValueError("Point prompt mode requires at least one positive point on frame 0.")
            prompt_adapter = PointPromptAdapter(
                positive_points=frame0_pts["positive"],
                negative_points=frame0_pts["negative"],
            )
            prompt = prompt_adapter.adapt(
                np.zeros(frame_shape, dtype=np.float32), frame_shape
            )
            logger.info(
                "Point prompt mode: %d positive, %d negative points on frame 0.",
                len(frame0_pts["positive"]),
                len(frame0_pts["negative"]),
            )
        else:
            anchor_mask = _read_anchor_mask(cfg.anchor_mask, frame_shape)
            prompt_adapter = MaskPromptAdapter()
            prompt = prompt_adapter.adapt(anchor_mask, frame_shape)

        segmenter = build_segmenter(cfg.segment_stage_config(), prompt_adapter=prompt_adapter)
        segment_result = segmenter.segment_sequence(
            source=source,
            prompt=prompt,
            anchor_frame=cfg.anchor_frame,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        logger.info(
            "Stage 1 complete: %d frames segmented (%d anchors).",
            len(segment_result.masks),
            len(segment_result.anchored_frames),
        )

        if cfg.mask_temporal_smooth_radius > 0:
            smooth_masks_temporal(segment_result.masks, radius=cfg.mask_temporal_smooth_radius)
            smooth_logits_temporal(segment_result.logits, radius=cfg.mask_temporal_smooth_radius)

        write_start = max(int(cfg.frame_start), 0)
        written_qc_trimaps = 0

        def _trimap_callback(frame_idx: int, trimap: np.ndarray) -> None:
            nonlocal written_qc_trimaps
            _write_qc_trimap_preview_png(output_dir, write_start + frame_idx, trimap)
            written_qc_trimaps += 1

        refine_result = refine_sequence(
            source=source,
            coarse_masks=segment_result.masks,
            coarse_logits=segment_result.logits,
            cfg=cfg.refine_stage_config(),
            trimap_callback=_trimap_callback,
        )
        logger.info(
            "Stage 2 complete: %d alpha frames (reused=%d).",
            len(refine_result.alphas),
            len(refine_result.reused_frames),
        )
        logger.info("Wrote %d QC trimap preview frames to %s/%s", written_qc_trimaps, output_dir, QC_TRIMAP_DIR)

        tuned = apply_matte_tuning(refine_result.alphas, cfg.matte_tuning_config())
        tuned = apply_temporal_smooth(tuned, cfg.temporal_smooth_config())
        writer = AlphaWriter(
            output_pattern=cfg.output_alpha,
            alpha_format=cfg.alpha_format,
            workers=max(1, int(cfg.workers_io)),
            base_dir=output_dir,
        )
        for idx, alpha in enumerate(tuned):
            writer.write(write_start + idx, alpha)
        writer.close()
        logger.info("Wrote %d alpha frames to %s", len(tuned), output_dir)

        if cfg.generate_preview_mp4:
            from videomatte_hq.io.preview_mp4 import generate_alpha_preview_mp4

            preview_fps = cfg.preview_fps if cfg.preview_fps > 0 else source.fps
            try:
                mp4_path = generate_alpha_preview_mp4(
                    output_dir=output_dir,
                    alpha_pattern=cfg.output_alpha,
                    frame_start=write_start,
                    frame_count=len(tuned),
                    fps=preview_fps,
                )
                logger.info("Alpha preview MP4: %s", mp4_path)
            except Exception:
                logger.warning("Failed to generate preview MP4.", exc_info=True)

        return PipelineRunResult(
            segment_result=segment_result,
            refine_result=RefineSequenceResult(
                alphas=tuned,
                reused_frames=refine_result.reused_frames,
            ),
            output_dir=output_dir,
        )
    finally:
        source.close()


# ---------------------------------------------------------------------------
# v2 pipeline (MatAnyone2-based)
# ---------------------------------------------------------------------------


def _run_pipeline_v2(cfg: VideoMatteConfig) -> PipelineRunResult:
    """Execute v2 pipeline: SAM3 (frame 0) → MatAnyone2 → optional MEMatte."""
    from videomatte_hq.pipeline.stage_matanyone2 import MatAnyone2StageConfig, run_matanyone2_stage
    from videomatte_hq.pipeline.stage_trimap import build_trimap_gradient_adaptive
    from videomatte_hq.utils.vram import unload_model

    if cfg.prompt_mode == "mask" and not str(cfg.anchor_mask).strip():
        raise ValueError("anchor_mask is required for v2 pipeline runs in mask mode.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = FrameSource(
        pattern=cfg.input,
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        prefetch_workers=max(0, int(cfg.workers_io)),
    )

    try:
        frame_shape = source.resolution
        native_h, native_w = frame_shape
        num_frames = len(source)
        logger.info(
            "v2 pipeline: %d frames, native=%dx%d, hires_threshold=%d",
            num_frames, native_w, native_h, cfg.matanyone2_hires_threshold,
        )

        # STAGE 0: Get first-frame mask
        logger.info("STAGE 0: Acquiring first-frame mask...")
        first_frame_mask = _get_first_frame_mask(cfg, source, frame_shape)
        logger.info("First-frame mask acquired, coverage=%.3f", float(first_frame_mask.mean()))

        # STAGE 1: MatAnyone2 temporal matting
        logger.info("STAGE 1: Running MatAnyone2...")

        ma2_cfg = MatAnyone2StageConfig(
            repo_dir=cfg.matanyone2_repo_dir,
            max_size=cfg.matanyone2_max_size,
            warmup=cfg.matanyone2_warmup,
            erode_kernel=cfg.matanyone2_erode_kernel,
            dilate_kernel=cfg.matanyone2_dilate_kernel,
            hires_threshold=cfg.matanyone2_hires_threshold,
            device=cfg.device,
            precision=cfg.precision,
        )

        ma2_result = run_matanyone2_stage(
            frames=source,
            first_frame_mask=first_frame_mask,
            cfg=ma2_cfg,
        )
        logger.info(
            "STAGE 1 complete: %d alphas at %dx%d",
            len(ma2_result.alphas),
            ma2_result.processing_resolution[1],
            ma2_result.processing_resolution[0],
        )

        # Save diagnostic: raw MA2 alpha for frame 0 (before any upscale/refinement)
        diag_dir = output_dir / "diag"
        diag_dir.mkdir(parents=True, exist_ok=True)
        if ma2_result.alphas:
            ma2_raw = ma2_result.alphas[0]
            ma2_cov = float((ma2_raw > 0.5).mean())
            logger.info("MA2 raw alpha[0]: shape=%s, coverage=%.4f", ma2_raw.shape, ma2_cov)
            cv2.imwrite(
                str(diag_dir / "ma2_raw_alpha_000.png"),
                np.clip(ma2_raw * 255.0, 0, 255).astype(np.uint8),
            )

        # RESOLUTION CHECK: does source exceed hires_threshold?
        short_edge = min(native_h, native_w)
        needs_hires_refinement = (
            short_edge > cfg.matanyone2_hires_threshold
            and cfg.refine_enabled
        )

        write_start = max(int(cfg.frame_start), 0)
        written_qc_trimaps = 0

        if not needs_hires_refinement:
            # Direct path: bicubic upscale MatAnyone2 alphas to native res
            logger.info(
                "Direct output path (short_edge=%d <= threshold=%d or refine disabled).",
                short_edge, cfg.matanyone2_hires_threshold,
            )
            alphas: list[np.ndarray] = []
            for alpha in ma2_result.alphas:
                if alpha.shape[:2] != (native_h, native_w):
                    alpha_up = cv2.resize(
                        alpha, (native_w, native_h),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    alpha_up = alpha
                alphas.append(np.clip(alpha_up, 0.0, 1.0).astype(np.float32))
        else:
            # Hi-res path: upscale + gradient-adaptive trimap + MEMatte
            logger.info(
                "Hi-res refinement path (short_edge=%d > threshold=%d).",
                short_edge, cfg.matanyone2_hires_threshold,
            )

            refiner = build_edge_refiner(cfg.refine_stage_config())
            refine_cfg = cfg.refine_stage_config()
            alphas = []

            for i in range(num_frames):
                # Upscale MatAnyone2 alpha to native resolution
                alpha_up = cv2.resize(
                    ma2_result.alphas[i],
                    (native_w, native_h),
                    interpolation=cv2.INTER_CUBIC,
                )
                alpha_up = np.clip(alpha_up, 0.0, 1.0).astype(np.float32)

                # Get native-res RGB frame
                rgb = _to_rgb_float(source[i])

                # Build gradient-adaptive trimap
                trimap = build_trimap_gradient_adaptive(
                    frame_rgb=rgb,
                    alpha_upscaled=alpha_up,
                    base_kernel=cfg.gradient_trimap_base_kernel,
                    max_extra=cfg.gradient_trimap_max_extra,
                    fg_thresh=cfg.gradient_trimap_fg_thresh,
                    bg_thresh=cfg.gradient_trimap_bg_thresh,
                    gradient_scale=cfg.gradient_trimap_scale,
                )

                # Write QC trimap
                _write_qc_trimap_preview_png(output_dir, write_start + i, trimap)
                written_qc_trimaps += 1

                # Binarize the coarse prior for MEMatte so the alpha
                # initialization in unknown regions starts from a hard edge.
                # Without this, the soft bicubic-upscaled gradients bias
                # MEMatte toward FG in ambiguous background regions.
                coarse_binary = (alpha_up >= 0.5).astype(np.float32)

                # MEMatte tiled refinement
                refined = _refine_frame_tiled(
                    rgb=rgb,
                    trimap=trimap,
                    coarse_prob=coarse_binary,
                    refiner=refiner,
                    cfg=refine_cfg,
                )
                alphas.append(refined)

                if i == 0 or (i + 1) % 50 == 0:
                    logger.info("v2 refine frame %d/%d", i + 1, num_frames)

            if written_qc_trimaps > 0:
                logger.info("Wrote %d QC trimap frames to %s/%s", written_qc_trimaps, output_dir, QC_TRIMAP_DIR)

        # Post-process: matte tuning + temporal smooth
        tuned = apply_matte_tuning(alphas, cfg.matte_tuning_config())
        tuned = apply_temporal_smooth(tuned, cfg.temporal_smooth_config())

        # Write output
        writer = AlphaWriter(
            output_pattern=cfg.output_alpha,
            alpha_format=cfg.alpha_format,
            workers=max(1, int(cfg.workers_io)),
            base_dir=output_dir,
        )
        for idx, alpha in enumerate(tuned):
            writer.write(write_start + idx, alpha)
        writer.close()
        logger.info("Wrote %d alpha frames to %s", len(tuned), output_dir)

        if cfg.generate_preview_mp4:
            from videomatte_hq.io.preview_mp4 import generate_alpha_preview_mp4

            preview_fps = cfg.preview_fps if cfg.preview_fps > 0 else source.fps
            try:
                mp4_path = generate_alpha_preview_mp4(
                    output_dir=output_dir,
                    alpha_pattern=cfg.output_alpha,
                    frame_start=write_start,
                    frame_count=len(tuned),
                    fps=preview_fps,
                )
                logger.info("Alpha preview MP4: %s", mp4_path)
            except Exception:
                logger.warning("Failed to generate preview MP4.", exc_info=True)

        return PipelineRunResult(
            segment_result=None,
            refine_result=RefineSequenceResult(
                alphas=tuned,
                reused_frames=[],
            ),
            output_dir=output_dir,
        )
    finally:
        source.close()


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def run_pipeline(cfg: VideoMatteConfig) -> PipelineRunResult:
    """Execute the pipeline based on cfg.pipeline_mode."""
    mode = str(cfg.pipeline_mode).strip().lower()
    if mode == "v2":
        logger.info("Pipeline mode: v2 (MatAnyone2)")
        return _run_pipeline_v2(cfg)
    else:
        logger.info("Pipeline mode: v1 (SAM3 + MEMatte)")
        return _run_pipeline_v1(cfg)
