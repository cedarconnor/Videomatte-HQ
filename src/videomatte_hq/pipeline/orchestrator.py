"""Option B pipeline orchestrator.

Stages:
0) Load frames
1) Project + assignment state
2) Memory-query coarse alpha
3) Edge refinement (boundary-band, confidence-gated)
4) Temporal cleanup (confidence-gated, outside boundary)
5) Matte tuning (choke/expand, feather, offset)
6) Write outputs
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.pass_memory import run_pass_memory
from videomatte_hq.pipeline.pass_matte_tuning import run_pass_matte_tuning
from videomatte_hq.pipeline.pass_refine import run_pass_refine
from videomatte_hq.pipeline.pass_temporal_cleanup import run_pass_temporal_cleanup
from videomatte_hq.project import ensure_project, load_keyframe_masks, save_project
from videomatte_hq.qc.optionb import (
    add_output_roundtrip_gate,
    evaluate_optionb_qc,
    failed_gate_names,
    write_optionb_qc_artifacts,
)

logger = logging.getLogger(__name__)

PER_FRAME_CACHE_STAGES = {"memory_alpha", "memory_conf", "refined_alpha", "final_alpha", "tuned_alpha"}


class StageCache:
    """Config-aware stage cache tracker for Option B."""

    STAGE_ORDER = [
        "project",
        "assignment",
        "memory",
        "refine",
        "temporal_cleanup",
        "matte_tuning",
        "io",
    ]

    def __init__(self, cache_dir: Path, config: VideoMatteConfig):
        self.cache_dir = cache_dir
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hashes_path = self.cache_dir / "stage_hashes.json"
        self.prev_hashes = self._load_hashes()

    def _load_hashes(self) -> dict[str, str]:
        if self.hashes_path.exists():
            return json.loads(self.hashes_path.read_text(encoding="utf-8"))
        return {}

    def _save_hashes(self) -> None:
        self.hashes_path.write_text(json.dumps(self.prev_hashes, indent=2), encoding="utf-8")

    def is_stage_valid(self, stage: str) -> bool:
        if stage not in self.STAGE_ORDER:
            return False
        stage_idx = self.STAGE_ORDER.index(stage)
        for i in range(stage_idx + 1):
            s = self.STAGE_ORDER[i]
            current = self.config.stage_hash(s)
            if self.prev_hashes.get(s) != current:
                return False
        return True

    def mark_stage_complete(self, stage: str) -> None:
        if stage not in self.STAGE_ORDER:
            return
        self.prev_hashes[stage] = self.config.stage_hash(stage)
        self._save_hashes()



def _save_cache(cache_dir: Path, name: str, data: Any) -> None:
    path = cache_dir / f"{name}.npy"
    if isinstance(data, list):
        if data and isinstance(data[0], np.ndarray):
            np.save(path, np.stack(data))
            return
        np.save(path, np.asarray(data, dtype=object), allow_pickle=True)
        return
    np.save(path, data, allow_pickle=True)



def _load_cache(cache_dir: Path, name: str, num_frames: int) -> Optional[Any]:
    path = cache_dir / f"{name}.npy"
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0 and data.dtype == object:
        data = data.item()

    if name in PER_FRAME_CACHE_STAGES:
        if isinstance(data, np.ndarray):
            if data.shape[0] != num_frames:
                return None
            if data.dtype == object:
                return data.tolist()
            return [data[i] for i in range(data.shape[0])]
        if isinstance(data, list) and len(data) == num_frames:
            return data
        return None

    return data


def run_pipeline(cfg: VideoMatteConfig) -> None:
    """Execute the Option B runtime pipeline."""

    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg.runtime.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage_cache = StageCache(cache_dir, cfg)

    total_start = time.time()

    # ------------------------------------------------------------------
    # Stage 0: Load frames
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 0: Loading frames")
    logger.info("=" * 60)

    from videomatte_hq.io.reader import FrameSource

    source = FrameSource(
        pattern=cfg.io.input,
        frame_start=cfg.io.frame_start,
        frame_end=cfg.io.frame_end,
        prefetch_workers=cfg.runtime.workers_io,
    )

    num_frames = source.num_frames
    height, width = source.resolution
    logger.info(f"Loaded {num_frames} frames at {width}x{height}")

    # ------------------------------------------------------------------
    # Stage 1: Project + assignments
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 1: Project + keyframe assignments")
    logger.info("=" * 60)

    project_path, project = ensure_project(cfg)
    keyframe_masks = load_keyframe_masks(project_path, project, target_shape=(height, width))

    if cfg.assignment.require_assignment and not keyframe_masks:
        raise RuntimeError(
            "No keyframe assignment found. Add at least one keyframe mask before running."
        )

    if not keyframe_masks:
        logger.warning("No keyframe masks found; falling back to empty mask assignment at frame 0")
        keyframe_masks = {0: np.zeros((height, width), dtype=np.float32)}

    logger.info(f"Loaded {len(keyframe_masks)} keyframe assignment(s) from {project_path}")
    stage_cache.mark_stage_complete("project")
    stage_cache.mark_stage_complete("assignment")

    # ------------------------------------------------------------------
    # Stage 2: Memory coarse alpha
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: Memory coarse alpha")
    logger.info("=" * 60)

    coarse_alphas = None
    confidence_maps = None

    if cfg.runtime.resume and stage_cache.is_stage_valid("memory"):
        coarse_alphas = _load_cache(cache_dir, "memory_alpha", num_frames)
        confidence_maps = _load_cache(cache_dir, "memory_conf", num_frames)
        if coarse_alphas is not None and confidence_maps is not None:
            logger.info("Loaded memory coarse alpha/confidence from cache")

    if coarse_alphas is None or confidence_maps is None:
        coarse_alphas, confidence_maps = run_pass_memory(source, keyframe_masks, cfg)
        _save_cache(cache_dir, "memory_alpha", coarse_alphas)
        _save_cache(cache_dir, "memory_conf", confidence_maps)

    stage_cache.mark_stage_complete("memory")

    # ------------------------------------------------------------------
    # Stage 3: Edge refinement
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3: Edge refinement")
    logger.info("=" * 60)

    refined_alphas = None
    if cfg.runtime.resume and stage_cache.is_stage_valid("refine"):
        refined_alphas = _load_cache(cache_dir, "refined_alpha", num_frames)
        if refined_alphas is not None:
            logger.info("Loaded refined alpha from cache")

    if refined_alphas is None:
        refined_alphas = run_pass_refine(source, coarse_alphas, confidence_maps, cfg)
        _save_cache(cache_dir, "refined_alpha", refined_alphas)

    stage_cache.mark_stage_complete("refine")

    # ------------------------------------------------------------------
    # Stage 4: Temporal cleanup
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 4: Temporal cleanup")
    logger.info("=" * 60)

    final_alphas = None
    if cfg.runtime.resume and stage_cache.is_stage_valid("temporal_cleanup"):
        final_alphas = _load_cache(cache_dir, "final_alpha", num_frames)
        if final_alphas is not None:
            logger.info("Loaded final alpha from cache")

    if final_alphas is None:
        final_alphas = run_pass_temporal_cleanup(
            source=source,
            alphas=refined_alphas,
            confidences=confidence_maps,
            cfg=cfg,
            anchor_frames=set(keyframe_masks.keys()),
        )
        _save_cache(cache_dir, "final_alpha", final_alphas)

    stage_cache.mark_stage_complete("temporal_cleanup")

    # ------------------------------------------------------------------
    # Stage 5: Matte tuning
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 5: Matte tuning")
    logger.info("=" * 60)

    tuned_alphas = None
    if cfg.runtime.resume and stage_cache.is_stage_valid("matte_tuning"):
        tuned_alphas = _load_cache(cache_dir, "tuned_alpha", num_frames)
        if tuned_alphas is not None:
            logger.info("Loaded tuned alpha from cache")

    if tuned_alphas is None:
        tuned_alphas = run_pass_matte_tuning(
            alphas=final_alphas,
            cfg=cfg,
        )
        _save_cache(cache_dir, "tuned_alpha", tuned_alphas)

    stage_cache.mark_stage_complete("matte_tuning")

    # ------------------------------------------------------------------
    # Stage 6: Write outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 6: Writing outputs")
    logger.info("=" * 60)

    from videomatte_hq.io.writer import AlphaWriter

    alpha_format = cfg.io.alpha_format.value
    if alpha_format == "dwaa":
        alpha_format = "exr_dwaa"

    alpha_writer = AlphaWriter(
        output_pattern=cfg.io.output_alpha,
        alpha_format=alpha_format,
        dwaa_quality=cfg.io.alpha_dwaa_quality,
        workers=cfg.runtime.workers_io,
        base_dir=output_dir,
    )

    write_index_start = max(int(cfg.io.frame_start), 0)
    for idx, alpha in enumerate(tuned_alphas):
        alpha_writer.write(write_index_start + idx, alpha)

    alpha_writer.flush()
    alpha_writer.close()

    if cfg.qc.enabled:
        metrics = evaluate_optionb_qc(
            alphas=tuned_alphas,
            confidences=confidence_maps,
            cfg=cfg,
        )
        metrics = add_output_roundtrip_gate(
            metrics,
            output_dir=output_dir,
            output_pattern=cfg.io.output_alpha,
            alphas=tuned_alphas,
            frame_start=write_index_start,
            sample_count=cfg.qc.sample_output_frames,
            max_mae=cfg.qc.max_output_roundtrip_mae,
        )
        metrics_path, report_path = write_optionb_qc_artifacts(
            metrics,
            output_dir=output_dir,
            output_subdir=cfg.qc.output_subdir,
            metrics_filename=cfg.qc.metrics_filename,
            report_filename=cfg.qc.report_filename,
        )
        summary = metrics.get("summary", {})
        logger.info(
            "QC summary: p95_flicker=%.5f p95_edge_flicker=%.5f mean_edge_conf=%.5f",
            float(summary.get("p95_flicker", 0.0)),
            float(summary.get("p95_edge_flicker", 0.0)),
            float(summary.get("mean_edge_confidence", 0.0)),
        )
        logger.info("QC artifacts: metrics=%s report=%s", metrics_path, report_path)

        failed_gates = failed_gate_names(metrics)
        if failed_gates:
            logger.warning("QC gate failures: %s", ", ".join(failed_gates))
            if cfg.qc.fail_on_regression:
                raise RuntimeError(
                    "QC regression gates failed: "
                    + ", ".join(failed_gates)
                    + f". See {metrics_path}."
                )

    if cfg.project.autosave:
        save_project(project_path, project)

    stage_cache.mark_stage_complete("io")

    elapsed = time.time() - total_start
    logger.info(f"Option B pipeline complete in {elapsed:.1f}s ({elapsed / max(num_frames, 1):.2f}s/frame)")
