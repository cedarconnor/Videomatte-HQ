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
PER_FRAME_CACHE_STAGES = {"roi", "band", "global", "intermediate", "refine", "temporal"}


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


def _save_cache(cache_dir: Path, stage: str, data: list[np.ndarray] | dict) -> None:
    """Save stage results to cache."""
    path = cache_dir / f"{stage}.npy"
    try:
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], np.ndarray):
                np.save(path, np.stack(data))
            else:
                # Lists of dataclasses/dicts (e.g., ROI or band plan) are stored as object arrays.
                np.save(path, np.asarray(data, dtype=object), allow_pickle=True)
        else:
            np.save(path, data, allow_pickle=True)
    except Exception as e:
        logger.warning(f"Failed to cache {stage}: {e}")


def _load_cache(cache_dir: Path, stage: str, num_frames: int) -> Optional[Any]:
    """Load stage results from cache."""
    path = cache_dir / f"{stage}.npy"
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        # Unwrap pickled dict/scalar payloads.
        if isinstance(data, np.ndarray) and data.ndim == 0 and data.dtype == object:
            data = data.item()

        # Per-frame stages must match current sequence length, or cache is stale.
        if stage in PER_FRAME_CACHE_STAGES:
            if isinstance(data, np.ndarray):
                cached_frames = int(data.shape[0]) if data.ndim >= 1 else 0
                if cached_frames != num_frames:
                    logger.info(
                        f"Ignoring stale {stage} cache: cached {cached_frames} frames, expected {num_frames}"
                    )
                    return None
                if data.dtype == object:
                    return data.tolist()
                return [data[i] for i in range(cached_frames)]
            if isinstance(data, list):
                if len(data) != num_frames:
                    logger.info(
                        f"Ignoring stale {stage} cache list: cached {len(data)} frames, expected {num_frames}"
                    )
                    return None
                return data
            logger.info(f"Ignoring {stage} cache with unsupported type: {type(data).__name__}")
            return None
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {stage}: {e}")
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
    if hasattr(cfg.io, "output_dir"):
        output_dir = Path(cfg.io.output_dir)
    else:
        output_dir = Path(cfg.io.output_alpha).parent
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

        if stage_cache.is_stage_valid("background"):
            bg_cached = _load_cache(cache_dir, "background", num_frames)
            if (
                isinstance(bg_cached, dict)
                and "bg_plate" in bg_cached
                and "bg_confidence" in bg_cached
            ):
                bg_plate = bg_cached["bg_plate"]
                bg_confidence = bg_cached["bg_confidence"]
                logger.info("Loaded background plate/confidence from cache")

        if bg_plate is None or bg_confidence is None:
            from videomatte_hq.background.plate_estimation import estimate_background_plate
            from videomatte_hq.background.bg_confidence import compute_bg_confidence

            bg_plate, sampled_frames = estimate_background_plate(source, cfg.background)
            bg_confidence = compute_bg_confidence(sampled_frames, bg_plate, cfg.background)

            # Occlusion fallback
            from videomatte_hq.background.bg_confidence import detect_occlusion_mask
            from videomatte_hq.background.plate_estimation import apply_occlusion_fallback

            occlusion_mask = detect_occlusion_mask(bg_confidence, cfg.background.occlusion_threshold)
            if occlusion_mask.any():
                bg_plate, bg_confidence = apply_occlusion_fallback(
                    bg_plate, occlusion_mask, sampled_frames, bg_confidence,
                    method=cfg.background.occlusion_fallback.value,
                )

            _save_cache(
                cache_dir,
                "background",
                {"bg_plate": bg_plate, "bg_confidence": bg_confidence},
            )
            stage_cache.mark_stage_complete("background")
            logger.info("Background plate estimation complete")

    # -----------------------------------------------------------------------
    # Stage 1: ROI tracking
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 1: ROI tracking")
    logger.info("=" * 60)

    rois = None
    if stage_cache.is_stage_valid("roi"):
        rois = _load_cache(cache_dir, "roi", num_frames)
        if rois is not None:
            logger.info("Loaded ROI track from cache")

    if rois is None:
        from videomatte_hq.roi.detect import detect_rois
        from videomatte_hq.roi.smooth import smooth_rois

        raw_rois = detect_rois(
            source, cfg.roi,
            bg_confidence=bg_confidence,
            bg_plate=bg_plate,
            photometric_normalize=cfg.background.photometric_normalize if cfg.background.enabled else False,
        )
        rois = smooth_rois(raw_rois, cfg.roi, frame_size=(height, width))
        _save_cache(cache_dir, "roi", rois)
        stage_cache.mark_stage_complete("roi")
    logger.info(f"ROI tracking complete: {len(rois)} frame ROIs")

    # -----------------------------------------------------------------------
    # Stage 2: Pass A — Global matte backbone
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: Pass A — Global matte backbone")
    logger.info("=" * 60)

    a0_results = None
    if stage_cache.is_stage_valid("global"):
        a0_results = _load_cache(cache_dir, "global", num_frames)
        if a0_results is not None:
             logger.info("Loaded Pass A results from cache")

    if a0_results is None:
        from videomatte_hq.pipeline.pass_a import run_pass_a
        a0_results = run_pass_a(source, rois, cfg, cache_dir)
        _save_cache(cache_dir, "global", a0_results)
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

        results = None
        if stage_cache.is_stage_valid("intermediate"):
            results = _load_cache(cache_dir, "intermediate", num_frames)
            if results is not None:
                logger.info("Loaded Pass A′ results from cache")
        
        if results is None:
            from videomatte_hq.intermediate.pass_a_prime import run_pass_a_prime
            results = run_pass_a_prime(source, a0_results, rois, cfg, cache_dir)
            _save_cache(cache_dir, "intermediate", results)
            stage_cache.mark_stage_complete("intermediate")
            logger.info("Pass A′ complete")
        
        a0prime_results = results

    # -----------------------------------------------------------------------
    # Stage 2.75: Reference frame mechanism (optional)
    # -----------------------------------------------------------------------
    if cfg.reference_frames.enabled:
        logger.info("=" * 60)
        logger.info("Stage 2.75: Reference frame selection + propagation")
        logger.info("=" * 60)

        from videomatte_hq.reference.selector import select_reference_frames
        from videomatte_hq.reference.propagator import propagate_reference

        ref_indices = select_reference_frames(
            source, a0prime_results, count=cfg.reference_frames.count,
        )
        logger.info(f"Reference frames selected: {ref_indices}")

        # For each non-reference frame, attempt to propagate the nearest
        # reference alpha as a stronger prior for Pass B.
        ref_propagated = [None] * num_frames
        for ref_idx in ref_indices:
            ref_propagated[ref_idx] = a0prime_results[ref_idx]  # references use their own alpha

        for t in range(num_frames):
            if ref_propagated[t] is not None:
                continue  # already a reference or already propagated
            # Find nearest reference
            nearest_ref = min(ref_indices, key=lambda r: abs(r - t))
            distance = abs(t - nearest_ref)
            # Simple cumulative heuristic (could be refined with actual flow error)
            est_flow_error = distance * 0.5  # rough estimate
            est_motion = distance * 2.0
            result = propagate_reference(
                ref_alpha=a0prime_results[nearest_ref],
                ref_frame_idx=nearest_ref,
                target_frame_idx=t,
                cumulative_flow_error=est_flow_error,
                cumulative_motion=est_motion,
                error_limit=cfg.reference_frames.propagation_error_limit,
                motion_limit=cfg.reference_frames.propagation_motion_limit,
                range_max=cfg.reference_frames.propagation_range_max,
            )
            if result is not None:
                # Blend reference prior with current A0prime (use as prior, not replacement)
                blend = max(0.0, 1.0 - distance / cfg.reference_frames.propagation_range_max)
                a0prime_results[t] = (
                    blend * result + (1.0 - blend) * a0prime_results[t]
                ).astype(np.float32)
                ref_propagated[t] = a0prime_results[t]

        propagated_count = sum(1 for r in ref_propagated if r is not None)
        logger.info(f"Reference propagation: {propagated_count}/{num_frames} frames influenced")

    # -----------------------------------------------------------------------
    # Stage 3: Adaptive band + trimap + tile plan
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3: Adaptive band + trimap + tiling")
    logger.info("=" * 60)

    # VRAM Probe: Determine tile size once for the whole pipeline (Pass B)
    from videomatte_hq.tiling.vram_probe import select_tile_size
    try:
        tile_size = select_tile_size(
            cfg.refine.model, cfg.runtime.device,
            cfg.tiles.vram_headroom, cfg.tiles.tile_size_backoff,
        )
        logger.info(f"VRAM Probe: selected tile size {tile_size}")
        cfg.tiles.tile_size = tile_size  # Apply to config
    except Exception as e:
        logger.warning(f"VRAM Probe failed, using default tile size {cfg.tiles.tile_size}: {e}")

    per_frame_data = None
    if stage_cache.is_stage_valid("band"):
        per_frame_data = _load_cache(cache_dir, "band", num_frames)
        if per_frame_data is not None:
            logger.info("Loaded band/trimap/tiling from cache")

    if per_frame_data is None:
        from videomatte_hq.band.adaptive_band import compute_adaptive_band
        from videomatte_hq.band.trimap import generate_trimap
        from videomatte_hq.band.feather import compute_feather_mask
        from videomatte_hq.tiling.planner import plan_tiles
        import cv2 as _cv2

        # Downscale factor for band/trimap/feather computation (perf optimization)
        band_downscale = getattr(cfg.band, 'compute_downscale', 0.25)
        if band_downscale >= 1.0:
            band_downscale = None  # disabled
        if band_downscale is not None:
            logger.info(f"Stage 3 downscale: computing at {band_downscale:.0%} resolution")

        per_frame_data = []
        for t in range(num_frames):
            alpha_prior = a0prime_results[t]
            rgb_frame = source[t]
            full_h, full_w = alpha_prior.shape[:2]

            if band_downscale is not None:
                # Downscale inputs for faster band/trimap/feather
                ds_h = max(16, int(full_h * band_downscale))
                ds_w = max(16, int(full_w * band_downscale))
                alpha_ds = _cv2.resize(alpha_prior, (ds_w, ds_h), interpolation=_cv2.INTER_LINEAR)
                rgb_ds = _cv2.resize(rgb_frame, (ds_w, ds_h), interpolation=_cv2.INTER_LINEAR)
                bg_conf_ds = None
                if bg_confidence is not None and bg_plate is not None:
                    bg_conf_ds = _cv2.resize(bg_confidence, (ds_w, ds_h), interpolation=_cv2.INTER_LINEAR)

                scale_ratio = ds_h / full_h

                band_ds = compute_adaptive_band(
                    alpha_ds, rgb_ds, cfg.band,
                    bg_confidence=bg_conf_ds,
                )
                trimap_ds = generate_trimap(alpha_ds, cfg.trimap)
                feather_ds = compute_feather_mask(band_ds, max(1, int(cfg.band.feather_px * scale_ratio)))

                # Upscale masks back to full resolution
                band = _cv2.resize(
                    band_ds.astype(np.uint8), (full_w, full_h), interpolation=_cv2.INTER_NEAREST
                ).astype(bool)
                trimap = _cv2.resize(
                    trimap_ds, (full_w, full_h), interpolation=_cv2.INTER_NEAREST
                ).astype(np.float32)
                feather = _cv2.resize(
                    feather_ds, (full_w, full_h), interpolation=_cv2.INTER_LINEAR
                ).astype(np.float32)
            else:
                # Full resolution (no downscale)
                band = compute_adaptive_band(
                    alpha_prior, rgb_frame, cfg.band,
                    bg_confidence=bg_confidence if bg_plate is not None else None,
                )
                trimap = generate_trimap(alpha_prior, cfg.trimap)
                feather = compute_feather_mask(band, cfg.band.feather_px)

            # Tile plan always at full resolution
            tiles = plan_tiles(band, rois[t], cfg.tiles)

            per_frame_data.append({
                "band": band,
                "trimap": trimap,
                "feather": feather,
                "tiles": tiles,
            })
            
            if (t + 1) % 10 == 0 or t == 0:
                logger.info(f"Stage 3: frame {t+1}/{num_frames}")

        _save_cache(cache_dir, "band", per_frame_data)
        stage_cache.mark_stage_complete("band")
    logger.info("Band/trimap/tiling complete")

    # -----------------------------------------------------------------------
    # Stage 4: Pass B — Edge refinement
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 4: Pass B — Edge refinement")
    logger.info("=" * 60)

    a1_results = None
    if stage_cache.is_stage_valid("refine") and stage_cache.is_stage_valid("band"): # Depend on band
        a1_results = _load_cache(cache_dir, "refine", num_frames)
        if a1_results is not None:
            logger.info("Loaded Pass B results from cache")

    if a1_results is None:
        from videomatte_hq.pipeline.pass_b import run_pass_b

        a1_results = run_pass_b(
            source, a0prime_results, per_frame_data, cfg,
            bg_plate=bg_plate, bg_confidence=bg_confidence,
        )
        _save_cache(cache_dir, "refine", a1_results)
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
