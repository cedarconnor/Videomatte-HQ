"""v2 pipeline orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.reader import FrameSource
from videomatte_hq.io.writer import AlphaWriter
from videomatte_hq.pipeline.stage_refine import RefineSequenceResult, refine_sequence
from videomatte_hq.pipeline.stage_segment import build_segmenter
from videomatte_hq.pipeline.stage_trimap import build_trimap_from_logits, resize_logits
from videomatte_hq.postprocess.matte_tuning import apply_matte_tuning
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter
from videomatte_hq.protocols import SegmentResult

logger = logging.getLogger(__name__)

QC_TRIMAP_DIR = "qc"
QC_TRIMAP_PATTERN = "trimap.%06d.png"


@dataclass(slots=True)
class PipelineRunResult:
    segment_result: SegmentResult
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


def run_pipeline(cfg: VideoMatteConfig) -> PipelineRunResult:
    """Execute v2 segmentation + refinement pipeline."""
    if not str(cfg.anchor_mask).strip():
        raise ValueError("anchor_mask is required for v2 pipeline runs.")

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

        refine_result = refine_sequence(
            source=source,
            coarse_masks=segment_result.masks,
            coarse_logits=segment_result.logits,
            cfg=cfg.refine_stage_config(),
        )
        logger.info(
            "Stage 2 complete: %d alpha frames (reused=%d).",
            len(refine_result.alphas),
            len(refine_result.reused_frames),
        )

        write_start = max(int(cfg.frame_start), 0)
        for idx, logits in enumerate(segment_result.logits):
            logits_up = resize_logits(logits, frame_shape)
            trimap = build_trimap_from_logits(
                logits_up,
                fg_threshold=cfg.trimap_fg_threshold,
                bg_threshold=cfg.trimap_bg_threshold,
            )
            _write_qc_trimap_preview_png(output_dir, write_start + idx, trimap)
        logger.info("Wrote %d QC trimap preview frames to %s/%s", len(segment_result.logits), output_dir, QC_TRIMAP_DIR)

        tuned = apply_matte_tuning(refine_result.alphas, cfg.matte_tuning_config())
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
