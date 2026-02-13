"""Pipeline orchestrator — main entry point that runs all stages in sequence.

Manages stage dependencies, resume/cache invalidation, and GPU/CPU/IO scheduling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from videomatte_hq.config import VideoMatteConfig

logger = logging.getLogger(__name__)


class StageCache:
    """Manages per-stage caching and config-aware invalidation."""

    # Stages in dependency order — changing an earlier stage invalidates all later ones
    STAGE_ORDER = [
        "background",
        "roi",
        "global",
        "intermediate",
        "band",
        "refine",
        "temporal",
        "postprocess",
        "io",
    ]

    def __init__(self, cache_dir: Path, config: VideoMatteConfig):
        self.cache_dir = cache_dir
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hashes_path = self.cache_dir / "stage_hashes.json"
        self._prev_hashes = self._load_hashes()

    def _load_hashes(self) -> dict[str, str]:
        if self._hashes_path.exists():
            return json.loads(self._hashes_path.read_text())
        return {}

    def _save_hashes(self, hashes: dict[str, str]) -> None:
        self._hashes_path.write_text(json.dumps(hashes, indent=2))

    def is_stage_valid(self, stage: str) -> bool:
        """Check if a stage's cached results are still valid.

        A stage is invalid if its config hash changed, or if any upstream
        stage's config hash changed.
        """
        stage_idx = self.STAGE_ORDER.index(stage) if stage in self.STAGE_ORDER else -1

        # Check this stage and all upstream stages
        for i in range(stage_idx + 1):
            s = self.STAGE_ORDER[i]
            try:
                current_hash = self.config.stage_hash(s)
            except ValueError:
                continue
            prev_hash = self._prev_hashes.get(s)
            if prev_hash != current_hash:
                return False
        return True

    def mark_stage_complete(self, stage: str) -> None:
        """Record the current config hash for a completed stage."""
        try:
            self._prev_hashes[stage] = self.config.stage_hash(stage)
        except ValueError:
            pass
        self._save_hashes(self._prev_hashes)

    def get_first_invalid_stage(self) -> Optional[str]:
        """Find the first stage that needs re-running."""
        for stage in self.STAGE_ORDER:
            if not self.is_stage_valid(stage):
                return stage
        return None


def run_pipeline(cfg: VideoMatteConfig) -> None:
    """Execute the full matting pipeline.

    Stages:
        0.  Decode / load frames
        0.5 Background plate estimation (locked-off shots)
        1.  ROI tracking
        2.  Pass A — Global matte backbone
        2.5 Pass A′ — Intermediate refinement
        3.  Adaptive band + trimap + tile plan
        4.  Pass B — Edge refinement
        5.  Pass C — Temporal stabilization
        6.  Post-processing
        7.  Write outputs + QC
    """
    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.runtime.cache_dir)
    stage_cache = StageCache(cache_dir, cfg)

    # Determine resume starting point
    if cfg.runtime.resume:
        first_invalid = stage_cache.get_first_invalid_stage()
        if first_invalid:
            logger.info(f"Resume: first invalid stage is '{first_invalid}'")
        else:
            logger.info("Resume: all stages valid, re-running output stage only")
    else:
        first_invalid = "background"  # run everything

    total_start = time.time()

    # -----------------------------------------------------------------------
    # Stage 0: Decode / Load
    # -----------------------------------------------------------------------
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
    logger.info(f"Loaded: {num_frames} frames, {width}×{height}")

    # -----------------------------------------------------------------------
    # Stage 0.5: Background plate estimation
    # -----------------------------------------------------------------------
    bg_plate = None
    bg_confidence = None
    if cfg.background.enabled and cfg.io.shot_type.value in ("locked_off", "unknown"):
        logger.info("=" * 60)
        logger.info("Stage 0.5: Background plate estimation")
        logger.info("=" * 60)

        from videomatte_hq.background.plate_estimation import estimate_background_plate
        from videomatte_hq.background.bg_confidence import compute_bg_confidence

        bg_plate, sampled_frames = estimate_background_plate(source, cfg.background)
        bg_confidence = compute_bg_confidence(sampled_frames, bg_plate, cfg.background)

        # Cache results
        bg_plate_path = cache_dir / "bg_plate.npy"
        bg_conf_path = cache_dir / "bg_confidence.npy"
        np.save(bg_plate_path, bg_plate)
        np.save(bg_conf_path, bg_confidence)
        stage_cache.mark_stage_complete("background")
        logger.info("Background plate estimation complete")

    # -----------------------------------------------------------------------
    # Stage 1: ROI tracking
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 1: ROI tracking")
    logger.info("=" * 60)

    from videomatte_hq.roi.detect import detect_rois
    from videomatte_hq.roi.smooth import smooth_rois

    raw_rois = detect_rois(source, cfg.roi, bg_confidence=bg_confidence)
    rois = smooth_rois(raw_rois, cfg.roi, frame_size=(height, width))
    stage_cache.mark_stage_complete("roi")
    logger.info(f"ROI tracking complete: {len(rois)} frame ROIs")

    # -----------------------------------------------------------------------
    # Stage 2: Pass A — Global matte backbone
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: Pass A — Global matte backbone")
    logger.info("=" * 60)

    from videomatte_hq.pipeline.pass_a import run_pass_a

    a0_results = run_pass_a(source, rois, cfg, cache_dir)
    stage_cache.mark_stage_complete("global")
    logger.info("Pass A complete")

    # -----------------------------------------------------------------------
    # Stage 2.5: Pass A′ — Intermediate refinement
    # -----------------------------------------------------------------------
    a0prime_results = a0_results  # fallback if disabled
    if cfg.intermediate.enabled:
        logger.info("=" * 60)
        logger.info("Stage 2.5: Pass A′ — Intermediate refinement")
        logger.info("=" * 60)

        from videomatte_hq.intermediate.pass_a_prime import run_pass_a_prime

        a0prime_results = run_pass_a_prime(source, a0_results, rois, cfg, cache_dir)
        stage_cache.mark_stage_complete("intermediate")
        logger.info("Pass A′ complete")

    # -----------------------------------------------------------------------
    # Stage 3: Adaptive band + trimap + tile plan
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3: Adaptive band + trimap + tiling")
    logger.info("=" * 60)

    from videomatte_hq.band.adaptive_band import compute_adaptive_band
    from videomatte_hq.band.trimap import generate_trimap
    from videomatte_hq.band.feather import compute_feather_mask
    from videomatte_hq.tiling.planner import plan_tiles

    per_frame_data = []
    for t in range(num_frames):
        alpha_prior = a0prime_results[t]
        rgb_frame = source[t]

        band = compute_adaptive_band(
            alpha_prior, rgb_frame, cfg.band,
            bg_confidence=bg_confidence if bg_plate is not None else None,
        )
        trimap = generate_trimap(alpha_prior, cfg.trimap)
        feather = compute_feather_mask(band, cfg.band.feather_px)
        tiles = plan_tiles(band, rois[t], cfg.tiles)

        per_frame_data.append({
            "band": band,
            "trimap": trimap,
            "feather": feather,
            "tiles": tiles,
        })

    stage_cache.mark_stage_complete("band")
    logger.info("Band/trimap/tiling complete")

    # -----------------------------------------------------------------------
    # Stage 4: Pass B — Edge refinement
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 4: Pass B — Edge refinement")
    logger.info("=" * 60)

    from videomatte_hq.pipeline.pass_b import run_pass_b

    a1_results = run_pass_b(
        source, a0prime_results, per_frame_data, cfg,
        bg_plate=bg_plate, bg_confidence=bg_confidence,
    )
    stage_cache.mark_stage_complete("refine")
    logger.info("Pass B complete")

    # -----------------------------------------------------------------------
    # Stage 5: Pass C — Temporal stabilization
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 5: Pass C — Temporal stabilization")
    logger.info("=" * 60)

    from videomatte_hq.temporal.stabilize import run_temporal_stabilization

    final_alphas = run_temporal_stabilization(
        a0prime_results, a1_results, per_frame_data, source, cfg,
    )
    stage_cache.mark_stage_complete("temporal")
    logger.info("Pass C complete")

    # -----------------------------------------------------------------------
    # Stage 6: Post-processing
    # -----------------------------------------------------------------------
    if cfg.postprocess.despill.enabled or cfg.postprocess.fg_output.enabled:
        logger.info("=" * 60)
        logger.info("Stage 6: Post-processing")
        logger.info("=" * 60)

        from videomatte_hq.postprocess.despill import run_despill
        from videomatte_hq.postprocess.fg_extract import extract_foreground

        if cfg.postprocess.despill.enabled and bg_plate is not None:
            for t in range(num_frames):
                # Despill is applied in-place during FG extraction
                pass

        if cfg.postprocess.fg_output.enabled:
            extract_foreground(source, final_alphas, cfg, bg_plate, bg_confidence)

        stage_cache.mark_stage_complete("postprocess")

    # -----------------------------------------------------------------------
    # Stage 7: Write outputs + QC
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 7: Writing outputs + QC")
    logger.info("=" * 60)

    from videomatte_hq.io.writer import AlphaWriter

    alpha_writer = AlphaWriter(
        output_pattern=cfg.io.output_alpha,
        alpha_format=cfg.io.alpha_format.value,
        dwaa_quality=cfg.io.alpha_dwaa_quality,
        workers=cfg.runtime.workers_io,
    )

    for t in range(num_frames):
        alpha_writer.write(t, final_alphas[t])

    alpha_writer.flush()
    alpha_writer.close()

    # Preview / QC
    if cfg.preview.enabled:
        from videomatte_hq.preview.compositor import generate_preview
        generate_preview(source, final_alphas, a0_results, a0prime_results, a1_results, per_frame_data, cfg)

    # QC report
    from videomatte_hq.qc.report import generate_report
    generate_report(cfg, num_frames, per_frame_data, output_dir)

    stage_cache.mark_stage_complete("io")

    elapsed = time.time() - total_start
    logger.info(f"Pipeline complete in {elapsed:.1f}s ({elapsed / num_frames:.2f}s/frame)")
