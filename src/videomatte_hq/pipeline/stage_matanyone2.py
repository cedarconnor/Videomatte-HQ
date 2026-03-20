"""Stage 1 (v2 pipeline): MatAnyone 2 temporal video matting."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MatAnyone2StageConfig:
    repo_dir: str = "third_party/MatAnyone2"
    max_size: int = 1080
    warmup: int = 10
    erode_kernel: int = 10
    dilate_kernel: int = 10
    hires_threshold: int = 1080
    device: str = "cuda"
    precision: str = "fp16"


@dataclass(slots=True)
class MatAnyone2StageResult:
    alphas: list[np.ndarray]
    processing_resolution: tuple[int, int]  # (h, w) at which MatAnyone2 ran
    native_resolution: tuple[int, int]  # (h, w) of original source


def run_matanyone2_stage(
    frames,
    first_frame_mask: np.ndarray,
    cfg: MatAnyone2StageConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MatAnyone2StageResult:
    """Run MatAnyone 2 on a video sequence.

    Args:
        frames: Indexable sequence of source frames (list or FrameSource).
                Frames are read lazily to avoid pre-loading all into RAM.
        first_frame_mask: Binary mask (float32, 0-1) for the first frame.
        cfg: Stage configuration.
        progress_callback: Optional (current, total) callback.

    Returns:
        MatAnyone2StageResult with alpha mattes at processing resolution.
    """
    from videomatte_hq.models.matanyone2_wrapper import MatAnyone2Model

    if len(frames) == 0:
        raise ValueError("No frames provided to MatAnyone2 stage.")

    native_h, native_w = frames[0].shape[:2]
    logger.info(
        "MatAnyone2 stage: %d frames, native=%dx%d, max_size=%d",
        len(frames), native_w, native_h, cfg.max_size,
    )

    model = MatAnyone2Model(
        repo_dir=cfg.repo_dir,
        device=cfg.device,
        precision=cfg.precision,
        max_size=cfg.max_size,
        warmup=cfg.warmup,
        erode_kernel=cfg.erode_kernel,
        dilate_kernel=cfg.dilate_kernel,
    )
    model.load()

    try:
        alphas = model.process_video(
            frames=frames,
            first_frame_mask=first_frame_mask,
            progress_callback=progress_callback,
        )
    finally:
        model.unload()

    if not alphas:
        raise RuntimeError("MatAnyone2 produced no output frames.")

    proc_h, proc_w = alphas[0].shape[:2]
    logger.info(
        "MatAnyone2 stage complete: %d alphas at %dx%d",
        len(alphas), proc_w, proc_h,
    )

    return MatAnyone2StageResult(
        alphas=alphas,
        processing_resolution=(proc_h, proc_w),
        native_resolution=(native_h, native_w),
    )
