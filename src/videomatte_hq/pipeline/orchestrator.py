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
from videomatte_hq.diagnostics.stage_debug import (
    export_stage_samples,
    resolve_sample_local_frames,
    write_stage_diagnosis_report,
)
from videomatte_hq.pipeline.memory_region_constraint import build_memory_region_priors
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
STOPPABLE_STAGES = {"assignment", "memory", "refine", "temporal_cleanup", "matte_tuning", "io"}


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


def _write_stage_alpha_preview(
    output_dir: Path,
    stage_name: str,
    alphas: list[np.ndarray],
    *,
    frame_start: int,
    workers_io: int,
) -> Path:
    """Write full per-frame stage preview alphas for manual review/approval."""

    from videomatte_hq.io.writer import AlphaWriter

    stage_output_pattern = f"stages/{stage_name}/alpha/frame_%05d.png"
    writer = AlphaWriter(
        output_pattern=stage_output_pattern,
        alpha_format="png16",
        workers=max(1, int(workers_io)),
        base_dir=output_dir,
    )
    for idx, alpha in enumerate(alphas):
        writer.write(frame_start + idx, alpha)
    writer.flush()
    writer.close()
    return output_dir / "stages" / stage_name / "alpha"


def run_pipeline(cfg: VideoMatteConfig) -> None:
    """Execute the Option B runtime pipeline."""

    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg.runtime.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage_cache = StageCache(cache_dir, cfg)

    total_start = time.time()
    stop_after_stage = str(getattr(cfg.runtime, "stop_after_stage", "io") or "io").strip().lower()
    if stop_after_stage not in STOPPABLE_STAGES:
        raise ValueError(
            f"runtime.stop_after_stage must be one of {sorted(STOPPABLE_STAGES)}, got '{stop_after_stage}'."
        )

    def _stop_requested(stage_name: str) -> bool:
        return stop_after_stage == str(stage_name).strip().lower()

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
    write_index_start = max(int(cfg.io.frame_start), 0)

    debug_enabled = bool(getattr(cfg, "debug", None) and cfg.debug.export_stage_samples)
    auto_stage_diagnosis_on_fail = bool(getattr(cfg.qc, "auto_stage_diagnosis_on_fail", True))
    auto_stage_samples_on_qc_fail = bool(getattr(cfg.debug, "auto_stage_samples_on_qc_fail", True))
    sample_local_frames: list[int] = []
    stage_metrics: dict[str, dict[int, dict[str, float]]] = {}
    stage_order: list[str] = ["stage2_memory", "stage3_refine", "stage4_temporal", "stage5_tuned"]
    diagnosis_written = False
    if debug_enabled:
        sample_local_frames = resolve_sample_local_frames(
            num_frames=int(num_frames),
            frame_start=write_index_start,
            sample_frames=list(getattr(cfg.debug, "sample_frames", []) or []),
            sample_count=int(getattr(cfg.debug, "sample_count", 5)),
        )
        logger.info(
            "Debug stage samples enabled: %d frame(s) -> %s",
            len(sample_local_frames),
            [int(write_index_start + i) for i in sample_local_frames],
        )

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
        raise RuntimeError(
            "No keyframe masks found in project. Build/import at least one valid keyframe assignment "
            "before running the pipeline."
        )

    logger.info(f"Loaded {len(keyframe_masks)} keyframe assignment(s) from {project_path}")
    stage_cache.mark_stage_complete("project")
    stage_cache.mark_stage_complete("assignment")
    if _stop_requested("assignment"):
        if cfg.project.autosave:
            save_project(project_path, project)
        elapsed = time.time() - total_start
        logger.info(
            "Stopping after Stage 1 (assignment) by request in %.1fs.",
            elapsed,
        )
        return

    memory_region_priors = None
    refine_region_guidance_masks = None
    memory_region_result = build_memory_region_priors(
        source=source,
        keyframe_masks=keyframe_masks,
        cfg=cfg,
    )
    if memory_region_result is not None:
        memory_region_priors = memory_region_result.priors
        refine_region_guidance_masks = memory_region_result.guidance_masks
        logger.info(
            "Memory region prior: mode=%s anchor=%d backend=%s mean_cov=%.3f range=[%.3f, %.3f]",
            memory_region_result.mode,
            int(memory_region_result.anchor_absolute_frame),
            memory_region_result.backend_used or memory_region_result.backend_requested or "n/a",
            float(memory_region_result.mean_coverage),
            float(memory_region_result.min_coverage),
            float(memory_region_result.max_coverage),
        )
        if memory_region_result.note:
            logger.warning("Memory region prior note: %s", memory_region_result.note)
        if refine_region_guidance_masks:
            logger.info(
                "Refine guidance: using %d propagated guidance mask(s) for trimap constraints.",
                len(refine_region_guidance_masks),
            )
        if debug_enabled and sample_local_frames:
            stage_metrics["stage1_region_prior"] = export_stage_samples(
                output_dir=output_dir,
                stage_dir_name=str(cfg.debug.stage_dir),
                stage_name="stage1_region_prior",
                source=source,
                alphas=memory_region_priors,
                sample_local_frames=sample_local_frames,
                frame_start=write_index_start,
                confidences=None,
                save_rgb=bool(cfg.debug.save_rgb),
                save_overlay=bool(cfg.debug.save_overlay),
            )
            stage_order = ["stage1_region_prior"] + stage_order

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
        coarse_alphas, confidence_maps = run_pass_memory(
            source=source,
            keyframe_masks=keyframe_masks,
            cfg=cfg,
            region_priors=memory_region_priors,
        )
        _save_cache(cache_dir, "memory_alpha", coarse_alphas)
        _save_cache(cache_dir, "memory_conf", confidence_maps)

    if debug_enabled and sample_local_frames:
        stage_metrics["stage2_memory"] = export_stage_samples(
            output_dir=output_dir,
            stage_dir_name=str(cfg.debug.stage_dir),
            stage_name="stage2_memory",
            source=source,
            alphas=coarse_alphas,
            sample_local_frames=sample_local_frames,
            frame_start=write_index_start,
            confidences=confidence_maps,
            save_rgb=bool(cfg.debug.save_rgb),
            save_overlay=bool(cfg.debug.save_overlay),
        )

    stage_cache.mark_stage_complete("memory")
    if _stop_requested("memory"):
        stage_dir = _write_stage_alpha_preview(
            output_dir,
            stage_name="stage2_memory",
            alphas=coarse_alphas,
            frame_start=write_index_start,
            workers_io=cfg.runtime.workers_io,
        )
        if cfg.project.autosave:
            save_project(project_path, project)
        elapsed = time.time() - total_start
        logger.info(
            "Stopping after Stage 2 (memory) by request. Wrote review alphas to %s in %.1fs.",
            stage_dir,
            elapsed,
        )
        return

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
        refined_alphas = run_pass_refine(
            source,
            coarse_alphas,
            confidence_maps,
            cfg,
            region_guidance_masks=refine_region_guidance_masks,
        )
        _save_cache(cache_dir, "refined_alpha", refined_alphas)

    if debug_enabled and sample_local_frames:
        stage_metrics["stage3_refine"] = export_stage_samples(
            output_dir=output_dir,
            stage_dir_name=str(cfg.debug.stage_dir),
            stage_name="stage3_refine",
            source=source,
            alphas=refined_alphas,
            sample_local_frames=sample_local_frames,
            frame_start=write_index_start,
            confidences=confidence_maps,
            save_rgb=bool(cfg.debug.save_rgb),
            save_overlay=bool(cfg.debug.save_overlay),
        )

    stage_cache.mark_stage_complete("refine")
    if _stop_requested("refine"):
        stage_dir = _write_stage_alpha_preview(
            output_dir,
            stage_name="stage3_refine",
            alphas=refined_alphas,
            frame_start=write_index_start,
            workers_io=cfg.runtime.workers_io,
        )
        if cfg.project.autosave:
            save_project(project_path, project)
        elapsed = time.time() - total_start
        logger.info(
            "Stopping after Stage 3 (refine) by request. Wrote review alphas to %s in %.1fs.",
            stage_dir,
            elapsed,
        )
        return

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

    if debug_enabled and sample_local_frames:
        stage_metrics["stage4_temporal"] = export_stage_samples(
            output_dir=output_dir,
            stage_dir_name=str(cfg.debug.stage_dir),
            stage_name="stage4_temporal",
            source=source,
            alphas=final_alphas,
            sample_local_frames=sample_local_frames,
            frame_start=write_index_start,
            confidences=confidence_maps,
            save_rgb=bool(cfg.debug.save_rgb),
            save_overlay=bool(cfg.debug.save_overlay),
        )

    stage_cache.mark_stage_complete("temporal_cleanup")
    if _stop_requested("temporal_cleanup"):
        stage_dir = _write_stage_alpha_preview(
            output_dir,
            stage_name="stage4_temporal",
            alphas=final_alphas,
            frame_start=write_index_start,
            workers_io=cfg.runtime.workers_io,
        )
        if cfg.project.autosave:
            save_project(project_path, project)
        elapsed = time.time() - total_start
        logger.info(
            "Stopping after Stage 4 (temporal cleanup) by request. Wrote review alphas to %s in %.1fs.",
            stage_dir,
            elapsed,
        )
        return

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

    if debug_enabled and sample_local_frames:
        stage_metrics["stage5_tuned"] = export_stage_samples(
            output_dir=output_dir,
            stage_dir_name=str(cfg.debug.stage_dir),
            stage_name="stage5_tuned",
            source=source,
            alphas=tuned_alphas,
            sample_local_frames=sample_local_frames,
            frame_start=write_index_start,
            confidences=confidence_maps,
            save_rgb=bool(cfg.debug.save_rgb),
            save_overlay=bool(cfg.debug.save_overlay),
        )
        diagnosis_json, diagnosis_md = write_stage_diagnosis_report(
            output_dir=output_dir,
            stage_dir_name=str(cfg.debug.stage_dir),
            stage_order=stage_order,
            per_stage_metrics=stage_metrics,
        )
        logger.info("Debug stage diagnosis artifacts: metrics=%s report=%s", diagnosis_json, diagnosis_md)
        diagnosis_written = True

    stage_cache.mark_stage_complete("matte_tuning")
    if _stop_requested("matte_tuning"):
        stage_dir = _write_stage_alpha_preview(
            output_dir,
            stage_name="stage5_tuned",
            alphas=tuned_alphas,
            frame_start=write_index_start,
            workers_io=cfg.runtime.workers_io,
        )
        if cfg.project.autosave:
            save_project(project_path, project)
        elapsed = time.time() - total_start
        logger.info(
            "Stopping after Stage 5 (matte tuning) by request. Wrote review alphas to %s in %.1fs.",
            stage_dir,
            elapsed,
        )
        return

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

            if (
                auto_stage_diagnosis_on_fail
                and auto_stage_samples_on_qc_fail
                and not diagnosis_written
            ):
                auto_sample_frames = resolve_sample_local_frames(
                    num_frames=int(num_frames),
                    frame_start=write_index_start,
                    sample_frames=(
                        list(getattr(cfg.debug, "auto_sample_frames", []) or [])
                        or list(getattr(cfg.debug, "sample_frames", []) or [])
                    ),
                    sample_count=int(getattr(cfg.debug, "sample_count", 5)),
                )

                if auto_sample_frames:
                    auto_stage_metrics: dict[str, dict[int, dict[str, float]]] = {}
                    auto_stage_order: list[str] = [
                        "stage2_memory",
                        "stage3_refine",
                        "stage4_temporal",
                        "stage5_tuned",
                    ]
                    if memory_region_priors is not None:
                        auto_stage_metrics["stage1_region_prior"] = export_stage_samples(
                            output_dir=output_dir,
                            stage_dir_name=str(cfg.debug.stage_dir),
                            stage_name="stage1_region_prior",
                            source=source,
                            alphas=memory_region_priors,
                            sample_local_frames=auto_sample_frames,
                            frame_start=write_index_start,
                            confidences=None,
                            save_rgb=bool(cfg.debug.save_rgb),
                            save_overlay=bool(cfg.debug.save_overlay),
                        )
                        auto_stage_order = ["stage1_region_prior"] + auto_stage_order
                    auto_stage_metrics["stage2_memory"] = export_stage_samples(
                        output_dir=output_dir,
                        stage_dir_name=str(cfg.debug.stage_dir),
                        stage_name="stage2_memory",
                        source=source,
                        alphas=coarse_alphas,
                        sample_local_frames=auto_sample_frames,
                        frame_start=write_index_start,
                        confidences=confidence_maps,
                        save_rgb=bool(cfg.debug.save_rgb),
                        save_overlay=bool(cfg.debug.save_overlay),
                    )
                    auto_stage_metrics["stage3_refine"] = export_stage_samples(
                        output_dir=output_dir,
                        stage_dir_name=str(cfg.debug.stage_dir),
                        stage_name="stage3_refine",
                        source=source,
                        alphas=refined_alphas,
                        sample_local_frames=auto_sample_frames,
                        frame_start=write_index_start,
                        confidences=confidence_maps,
                        save_rgb=bool(cfg.debug.save_rgb),
                        save_overlay=bool(cfg.debug.save_overlay),
                    )
                    auto_stage_metrics["stage4_temporal"] = export_stage_samples(
                        output_dir=output_dir,
                        stage_dir_name=str(cfg.debug.stage_dir),
                        stage_name="stage4_temporal",
                        source=source,
                        alphas=final_alphas,
                        sample_local_frames=auto_sample_frames,
                        frame_start=write_index_start,
                        confidences=confidence_maps,
                        save_rgb=bool(cfg.debug.save_rgb),
                        save_overlay=bool(cfg.debug.save_overlay),
                    )
                    auto_stage_metrics["stage5_tuned"] = export_stage_samples(
                        output_dir=output_dir,
                        stage_dir_name=str(cfg.debug.stage_dir),
                        stage_name="stage5_tuned",
                        source=source,
                        alphas=tuned_alphas,
                        sample_local_frames=auto_sample_frames,
                        frame_start=write_index_start,
                        confidences=confidence_maps,
                        save_rgb=bool(cfg.debug.save_rgb),
                        save_overlay=bool(cfg.debug.save_overlay),
                    )
                    diagnosis_json, diagnosis_md = write_stage_diagnosis_report(
                        output_dir=output_dir,
                        stage_dir_name=str(cfg.debug.stage_dir),
                        stage_order=auto_stage_order,
                        per_stage_metrics=auto_stage_metrics,
                    )
                    diagnosis_written = True
                    logger.warning(
                        "Auto stage diagnosis exported on QC failure: gates=%s metrics=%s report=%s",
                        ", ".join(failed_gates),
                        diagnosis_json,
                        diagnosis_md,
                    )
                else:
                    logger.warning("Auto stage diagnosis skipped: no sample frames resolved.")

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
