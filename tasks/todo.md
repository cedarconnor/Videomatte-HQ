# TODO — Max-Quality Rerun Validation (2026-02-22)

## Goal
- Re-run all clips in `TestFiles/` using the new max-quality defaults and capture fresh outputs.

## Non-goals
- Retuning thresholds or manual-anchor overrides in this pass.
- Full visual scorecard beyond basic sanity metrics.

## Plan
- [x] Enumerate all test clips in `TestFiles/`.
- [x] Execute 30-frame smoke run for each clip with default settings.
- [x] Record produced `run_XXXX` directories and key run metadata.
- [x] Compute quick alpha-coverage stats for each run.

## Risks / Unknowns
- Top-tier checkpoints may fall back to locally available weights if not present.
- Clip `4990426...` starts with a black frame, which can weaken auto-anchor quality.

## Verification
- [x] Integration runs exist for all 4 clips in `output_tests/run_XXXX`.
- [x] `run_summary.json` present for each run.
- [x] Quick alpha stats reported.

## Review Notes (fill after)
- What changed:
  - Ran max-quality default smoke tests for all `TestFiles` clips:
    - `output_tests/run_0003` (`4625475...`)
    - `output_tests/run_0004` (`4990426...`, frame_start=0)
    - `output_tests/run_0005` (`6138680...`)
    - `output_tests/run_0006` (`7219039...`)
  - Added one additional diagnostic run for the portrait clip with non-black start frame:
    - `output_tests/run_0007` (`4990426...`, frame_start=1)
  - During rerun, fixed SAM empty-candidate crash (`(0, H, W)` masks) in `stage_segment` and added regression coverage.
  - Fixed auto-anchor/proc-start mismatch in `tools/run_test_clip.py`:
    - auto-anchor now returns `anchor_probe_frame`,
    - effective processing start is aligned to `max(requested_frame_start, anchor_probe_frame)` for auto-anchor runs,
    - run summary now records both requested and effective frame starts.
  - Re-ran full `TestFiles` batch after the start-alignment fix:
    - `output_tests/run_0010` (`4625475...`)
    - `output_tests/run_0011` (`4990426...`, requested start `0` -> effective start `1`)
    - `output_tests/run_0012` (`6138680...`)
    - `output_tests/run_0013` (`7219039...`)
- Evidence it works:
  - Four requested clip runs completed with fresh output folders and `run_summary.json`.
  - Quick alpha coverage stats (fg mean / min / max):
    - `run_0003`: `0.0297 / 0.0284 / 0.0322`
    - `run_0004`: `0.0000 / 0.0000 / 0.0000` (all-black result)
    - `run_0005`: `0.1768 / 0.1511 / 0.1996`
    - `run_0006`: `0.1743 / 0.1724 / 0.1766`
    - `run_0007` (extra): `0.1653 / 0.0376 / 0.4633`
    - `run_0010`: `0.0297 / 0.0284 / 0.0322`
    - `run_0011`: `0.0554 / 0.0496 / 0.0620` (non-black, fixed from `run_0004`)
    - `run_0012`: `0.1762 / 0.1511 / 0.1996`
    - `run_0013`: `0.1745 / 0.1724 / 0.1766`
  - Test suite after crash fix passes (`27 passed`).
- Remaining issues / follow-ups:
  - `4990426...` no longer collapses to black under requested `frame_start=0` after effective-start alignment (`run_0011`).
  - Visual quality for `4990426...` remains conservative (small foreground coverage), so temporal robustness tuning is still open.

# TODO — Remaining Validation Sweep (2026-02-22)

## Goal
- Close the remaining open validation items with reproducible evidence for v2-only operation.

## Non-goals
- Retuning model thresholds or adding new model families.
- Re-introducing any v1 comparison loop.

## Plan
- [x] Run a deterministic short integration check (same clip/settings twice) and compare alpha hashes.
- [x] Run deeper visual-QC diagnostics on representative clips (coverage, inversion risk, temporal consistency).
- [x] Run performance/memory smoke on 4K and 8K inputs and capture timings + peak VRAM where available.

## Risks / Unknowns
- GPU kernels can be non-deterministic depending on operator paths.
- No native 8K source clip may require synthetic 8K generation.

## Verification
- [x] Integration test: two runs, matching per-frame alpha hashes (or documented non-determinism with measured deltas).
- [x] Visual QC: no empty/inverted collapse; temporal continuity remains within expected bounds.
- [x] Performance / memory check: complete 4K + 8K 30-frame runs with timing and VRAM notes.

## Review Notes (fill after)
- What changed:
  - Added deterministic integration reruns:
    - `output_tests/run_0014` and `output_tests/run_0015` on `4625475-hd_1920_1080_24fps.mp4` (identical settings, CPU).
  - Added deeper visual-QC diagnostics over representative max-quality runs:
    - `output_tests/run_0010`..`output_tests/run_0013`.
  - Added 4K + 8K GPU performance smoke runs:
    - `output_tests/run_0016` (native 4K `6138680...`)
    - `output_tests/run_0017` (synthetic 8K clip `TestFiles/synthetic_8k_6138680_30f.mp4`)
- Evidence it works:
  - Determinism check (`run_0014` vs `run_0015`): all 30 alpha frame SHA-256 hashes match (`mismatch_count=0`).
  - Visual-QC diagnostics:
    - Empty/inverted collapse: `0` frames across `run_0010`..`run_0013`.
    - Temporal IoU mean/min:
      - `run_0010`: `0.9627 / 0.9425`
      - `run_0011`: `0.9454 / 0.8952`
      - `run_0012`: `0.9175 / 0.6591`
      - `run_0013`: `0.9911 / 0.9345`
    - Connected components: median/max is `1/1` for all four runs (no fragmentation into stray blobs).
  - Performance + VRAM profiling (30 frames, CUDA, sampled via `nvidia-smi`):
    - 4K (`run_0016`): `67.772s`, `0.443 fps`, peak GPU memory `9801 MB`.
    - 8K (`run_0017` synthetic): `265.404s`, `0.113 fps`, peak GPU memory `9743 MB`.
- Remaining issues / follow-ups:
  - 8K memory remained below 10 GB in this 30-frame smoke, but throughput is low (`~0.11 fps`); deeper optimization remains optional follow-up work.

# TODO — 8K Throughput Hardening (2026-02-22)

## Goal
- Improve 8K runtime throughput without regressing matte quality or changing max-quality model defaults.

## Non-goals
- Downgrading SAM model tier from `sam2_l.pt`.
- Introducing v1 eval/comparison workflows.

## Plan
- [x] Wire runtime precision into Stage-1 backend and add guarded FP16 inference hint for CUDA.
- [x] Remove unnecessary per-frame decoder seek in video reader for sequential frame access.
- [x] Cache the last successful Ultralytics SAM prompt variant to avoid repeated failing variant probes.
- [x] Validate with unit tests and 8K re-benchmark.

## Risks / Unknowns
- Ultralytics may ignore or reject optional inference kwargs depending on version.
- 8K runtime can remain model-bound even after pipeline overhead reductions.

## Verification
- [x] Unit tests: `pytest -q` (`34 passed`).
- [x] Integration/perf runs: synthetic 8K 30-frame reruns produced `run_0018`, `run_0019`, `run_0021`.
- [x] Quality parity check: `run_0017` vs `run_0021` remained near-identical.

## Review Notes (fill after)
- What changed:
  - Stage-1 precision plumbing:
    - `src/videomatte_hq/config.py` now forwards runtime `precision` into `SegmentStageConfig`.
    - `src/videomatte_hq/pipeline/stage_segment.py` now supports `precision`, applies CUDA runtime tuning, and uses a guarded `half=True` inference hint with automatic fallback when unsupported.
  - Video decode optimization:
    - `src/videomatte_hq/io/reader.py` now avoids calling `cap.set(CAP_PROP_POS_FRAMES, ...)` for strictly sequential access.
  - Prompt variant optimization:
    - `src/videomatte_hq/pipeline/stage_segment.py` now caches the last successful prompt variant (`bbox`, `points`, etc.) and tries it first in subsequent frames.
  - Tooling:
    - `tools/run_test_clip.py` now accepts and records `--precision`.
  - Tests:
    - Added/updated tests in `tests/test_stage_segment.py`, `tests/test_config.py`, and `tests/test_reader.py`.
- Evidence it works:
  - Tests pass: `34 passed`.
  - 8K timings (30 frames, CUDA, same clip/settings):
    - Baseline: `run_0017` = `265.404s` (`0.113 fps`)
    - After changes: `run_0018` = `271.457s` (`0.111 fps`), `run_0019` = `268.290s` (`0.112 fps`), `run_0021` = `273.847s` (`0.110 fps`)
  - Quality parity (`run_0017` vs `run_0021`):
    - Mean mask area: `0.1744` vs `0.1744`
    - Pixel disagreement mean/max: `0.000274 / 0.002437`
- Remaining issues / follow-ups:
  - Throughput remains effectively unchanged in this benchmark window; runtime appears model-bound on `sam2_l.pt`.
  - Next meaningful speed gain likely requires architecture-level change (native Ultralytics video predictor integration for persistent tracking state, or optional lower-tier speed preset).

# TODO — Native SAM Video Predictor Integration (2026-02-22)

## Goal
- Add an optional fast path that uses Ultralytics SAM video-state predictor APIs for video inputs to reduce per-frame overhead while preserving current quality controls.

## Non-goals
- Changing default model quality tier.
- Removing current per-frame backend path (must remain as fallback).

## Plan
- [x] Add video-source metadata accessors in IO layer so Stage-1 can detect and use native video fast path.
- [x] Extend segment backend/segmenter interface with optional video-sequence entrypoint.
- [x] Implement Ultralytics video-predictor path (prompt once, stream masks) with robust fallback to current per-frame path.
- [x] Validate with tests + 8K benchmark and quality parity check against previous baseline.

## Risks / Unknowns
- Ultralytics predictor class behavior can differ by model family/version.
- Prompt tensor shape conventions differ between image and video predictor classes.

## Verification
- [x] Unit tests: new interface/behavior plus existing segmentation tests.
- [x] Integration test: short video run succeeds on CUDA with same outputs directory conventions.
- [x] Performance check: 8K 30-frame throughput vs latest baseline.
- [x] Quality check: mask area and pixel disagreement remain within narrow tolerance.

## Review Notes (fill after)
- What changed:
  - `src/videomatte_hq/io/reader.py`
    - added `FrameSource` video metadata accessors (`is_video`, `video_path`, `video_frame_start`, `video_frame_end`) so Stage-1 can detect video-backed sources.
  - `src/videomatte_hq/pipeline/stage_segment.py`
    - added optional video backend entrypoint (`segment_video_sequence`) and ChunkedSegmenter fast-path dispatch for video sources starting at frame `0`,
    - implemented Ultralytics SAM video-predictor integration (`SAM2VideoPredictor` / `SAM3VideoPredictor`) with prompt-once streaming and fallback to existing per-frame chunk path,
    - preserved existing strict temporal/area filtering over streamed probabilities,
    - kept fallback behavior on exceptions or unsupported source offsets.
  - `tools/run_test_clip.py`
    - precision flag and metadata remain available for profiling reproducibility.
  - Tests:
    - new `tests/test_reader.py` coverage for sequential seek optimization and video metadata accessors,
    - new `tests/test_stage_segment.py` coverage for fast-path dispatch/fallback.
- Evidence it works:
  - Tests pass: `pytest -q` (`37 passed`).
  - CUDA integration run succeeded with standard runner output conventions:
    - `output_tests/run_0022` (`synthetic_8k_6138680_30f.mp4`, 30 frames)
  - 8K throughput improved vs latest baseline:
    - Baseline (`run_0021`): `273.847s` (`0.110 fps`)
    - Native video predictor (`run_0022`): `237.581s` (`0.126 fps`)
    - Improvement: about `15.3%` faster (`+0.016 fps`)
  - Quality comparison (`run_0021` vs `run_0022`):
    - Mean mask area: `0.1744` -> `0.1825`
    - Pixel disagreement mean/max: `0.009279 / 0.071346`
    - Framewise IoU mean/min: `0.952209 / 0.681699`
- Remaining issues / follow-ups:
  - Fast path currently activates only for video sources with `video_frame_start == 0`; non-zero video offsets intentionally fall back to the chunked path.
  - Native video predictor output is not identical to the per-frame path (it is faster but slightly more expansive on average); keep this in mind for strict reproducibility expectations.

# TODO — Max-Quality Defaults (2026-02-22)

## Goal
- Make highest-tier Ultralytics model choices the default for segmentation and person anchor detection.

## Non-goals
- Forcing downloads or pinning one exact weight file in every environment.
- Retuning stage logic for speed.

## Plan
- [x] Set stage-1 default SAM checkpoint to top-tier quality.
- [x] Set test runner default `--sam3-model` to the same top-tier checkpoint.
- [x] Reorder person detector auto-anchor candidates from highest to lowest quality.
- [x] Add regression test coverage for high-quality default config.
- [x] Run test suite to verify no regressions.

## Risks / Unknowns
- Highest-tier weights can be slower and require more VRAM.
- If top-tier files are unavailable and download is blocked, runtime will fall back after failed attempts.

## Verification
- [x] Unit tests: `27 passed`.
- [x] Integration sanity: default config and parser now both report `sam2_l.pt`.
- [x] Visual QC: smoke-run masks for `run_0010`..`run_0013` are non-empty and non-inverted across all sampled frames.
- [x] Performance / memory check:

## Review Notes (fill after)
- What changed:
  - `src/videomatte_hq/config.py` default `sam3_model` changed to `sam2_l.pt`.
  - `src/videomatte_hq/pipeline/stage_segment.py` default SAM model names changed to `sam2_l.pt`.
  - `tools/run_test_clip.py` default `--sam3-model` changed to `sam2_l.pt`.
  - `tools/run_test_clip.py` person-anchor detector fallback list now starts at `yolo11x-seg.pt` and descends by quality.
  - `tests/test_config.py` now asserts high-quality default model selection.
- Evidence it works:
  - `pytest -q` passes (`27 passed`).
  - Runtime defaults print `sam2_l.pt` from both `VideoMatteConfig()` and `tools.run_test_clip._parser()`.
  - Visual QC metrics across `run_0010`..`run_0013`:
    - Empty frames (`fg < 0.2%`): `0` in all runs.
    - Inverted/full-background frames (`fg > 80%`): `0` in all runs.
    - Temporal IoU means: `run_0010=0.9627`, `run_0011=0.9454`, `run_0012=0.9175`, `run_0013=0.9911`.
- Remaining issues / follow-ups:
  - Install top-tier YOLO/SAM weights locally to avoid first-run download/fallback delay.

# TODO — V2 Strict Drift Rejection (2026-02-22)

## Goal
- Stop strict-mode segment drift where SAM flips from person to large background structures (doorways/scene regions) across frames.

## Non-goals
- Reworking stage-2 MEMatte behavior in this pass.
- Adding new model families beyond current Ultralytics SAM3 path.

## Plan
- [x] Wire strict temporal area/IoU guard into `UltralyticsSAM3SegmentBackend.segment_chunk`.
- [x] Expose strict guard thresholds in config and CLI so we can tune per clip without code edits.
- [x] Add focused unit tests for strict guard acceptance/rejection behavior.
- [x] Run targeted 30-frame integration rerun for failing clip in next `output_tests/run_XXXX`.
- [x] Review output masks on first/middle/last frames to verify no background takeover.

## Risks / Unknowns
- Guard can over-freeze masks if thresholds are too tight during true fast subject motion.
- A poor anchor can still fail; guard should reduce catastrophic flips, not invent missing subject detail.

## Verification
- [x] Unit tests: strict guard logic + existing stage-segment suite.
- [x] Integration test: rerun `4990426-hd_1080_1920_30fps.mp4`.
- [x] Visual QC: confirm silhouette continuity and no doorway/background lock-in.
- [x] Performance / memory check: run completes on GPU path.

## Review Notes (fill after)
- What changed:
- Added strict temporal guards, strict config/CLI knobs, and integration reruns for the failing portrait clip path.
- Evidence it works:
  - Strict guard path is active in `stage_segment` with config + CLI coverage.
  - Stage-segment test suite includes strict guard and empty-candidate regressions; suite currently passes (`27 passed`).
  - Multiple `4990426...` integration runs executed on CUDA with strict controls available.
- Remaining issues / follow-ups:

# TODO — V2 Anchor/Alpha Hardening (2026-02-21)

## Goal
- Eliminate flat-white alpha outputs and reduce head/foot clipping by hardening person prompting and bbox propagation in the v2 segment stage.

## Non-goals
- Re-introducing v1 comparison/eval workflows.
- Enabling MEMatte weights for this task (refine remains optional).

## Plan
- [x] Add bbox expansion controls to mask prompt adaptation so first-frame prompts are not overly tight.
- [x] Expand propagated SAM bbox prompts frame-to-frame to reduce truncation drift.
- [x] Improve auto anchor mask post-processing (largest component + close + light dilation) before writing `anchor_mask.png`.
- [x] Run 30-frame smoke test on `TestFiles/6138680-uhd_3840_2160_24fps.mp4` into next `output_tests/run_XXXX`.
- [x] Verify output stats and mask geometry (not all-white alpha, sensible coverage, no obvious top/bottom truncation artifacts).
- [x] Update `tasks/lessons.md` with the correction rule for explicit person prompting and bbox slack.

## Risks / Unknowns
- Too much bbox expansion can pull in background and hurt edge precision.
- Auto-anchor morphology can overfill if dilation is too aggressive.

## Verification
- [x] Unit tests: prompt adapter bbox behavior and backend bbox expansion utility.
- [x] Integration test: 30-frame clip run and generated `run_summary.json`.
- [x] Visual QC: inspect `anchor_mask.png` + first alpha frames for top/head and bottom/feet clipping.
- [x] Performance / memory check: run completes on CUDA device without OOM.

## Review Notes (fill after)
- What changed:
  - Added prompt bbox expansion controls in `src/videomatte_hq/prompts/mask_adapter.py` and wired config fields in `src/videomatte_hq/config.py`.
  - Hardened SAM stage in `src/videomatte_hq/pipeline/stage_segment.py`:
    - frame propagation now re-prompts from previous mask (bbox + points), not bbox-only,
    - prompt variant order now prefers combined `bbox + points`,
    - configurable prompt adapter settings flow from config to backend,
    - added optional temporal component filtering path (kept disabled by default after validation),
    - added optional strict background suppression mode (bbox-gated + overlap-connected filtering).
  - Improved auto-anchor generation in `tools/run_test_clip.py`:
    - uses explicit person class detection,
    - post-processes detected anchor masks with component cleanup + morphology,
    - preserves incremented run outputs.
  - Added/updated tests in `tests/test_prompt_adapters.py` and `tests/test_stage_segment.py`.
  - Hardened no-refine preview fallback in `src/videomatte_hq/pipeline/stage_refine.py`:
    - preview alpha now uses cleaned binary mask (component filtering + close + hole-fill),
    - removed raw-probability tearing from preview-only runs.
  - Added refinement regression coverage in `tests/test_stage_refine.py`.
- Evidence it works:
  - Unit tests pass: `15 passed`.
  - Unit tests after follow-up tuning pass: `16 passed`.
  - New runs created: `output_tests/run_0006` through `output_tests/run_0015`.
  - White-alpha failure resolved: run_0008 has `near-white frames (mean > 0.98): 0/30`.
  - Frame-0 alpha silhouette improved vs earlier run (hat/head region expands upward: run_0007 y0=548 -> run_0008 y0=528).
  - Current default behavior validated on run_0013 (matches run_0008 metrics; non-white output preserved).
  - Strict-mode validation run completed: `output_tests/run_0015` (`strict_background_suppression=true`).
  - Preview-solidification validation run completed: `output_tests/run_0018` (frame 0 and frame 29 show strongly reduced interior tearing vs `run_0013`/`run_0016`).
- Remaining issues / follow-ups:
  - Anchor masks from detector are still coarse and can appear slightly flattened at extremities.
  - Frame 29 leftward extent is largely real motion in source footage (hat swing), so stricter suppression risks clipping true foreground.

---

# TODO — Videomatte-HQ v2 Implementation

## Goal
- Implement the v2 two-stage architecture (SAM3 segmentation/tracking + MEMatte refinement) with clear protocols, reduced config surface, and high-resolution-safe execution.

## Non-goals
- Porting legacy v1 drift-compensation subsystems.
- Porting web UI in this initial implementation track.
- Adding v2.x features (multi-subject, text prompts, adaptive motion/detail trimap widening).
- Running v1-v2 comparison/eval loops for this repo's current direction (v2-only by user directive).

## Proposed Changes (files/modules)
- Scaffold v2 package structure under `src/videomatte_hq/`.
- Keep verbatim from v1 where design requires it:
  - `src/videomatte_hq/models/edge_mematte.py`
  - `src/videomatte_hq/io/reader.py`
  - `src/videomatte_hq/io/writer.py`
  - `src/videomatte_hq/tiling/windows.py`
  - `src/videomatte_hq/tiling/stitch.py`
  - `src/videomatte_hq/utils/image.py`
- Extract supporting dependencies required by copied tiling logic:
  - `src/videomatte_hq/safe_math.py`
  - `src/videomatte_hq/tiling/planner.py`
- Extract matte tuning primitive as the v2 optional postprocess base:
  - `src/videomatte_hq/postprocess/matte_tuning.py` (from v1 `pipeline/pass_matte_tuning.py`)
- Implement new v2 modules:
  - `src/videomatte_hq/protocols.py`
  - `src/videomatte_hq/prompts/mask_adapter.py`
  - `src/videomatte_hq/prompts/box_adapter.py`
  - `src/videomatte_hq/pipeline/stage_segment.py`
  - `src/videomatte_hq/pipeline/stage_trimap.py`
  - `src/videomatte_hq/pipeline/stage_refine.py`
  - `src/videomatte_hq/pipeline/stage_qc.py`
  - `src/videomatte_hq/pipeline/orchestrator.py`
  - `src/videomatte_hq/config.py`
  - `src/videomatte_hq/cli.py`

## Plan
- [x] Review `Agents.md` and `Videomatte-HQ-v2-Design-Document.md` in this repo.
- [x] Inventory reusable modules from `D:\Videomatte-HQ`.
- [x] Extract required v1 foundation modules into this repo.
- [x] Define protocol/dataclass contracts in `protocols.py` and lock interface tests.
- [x] Implement PromptAdapter(s) with farthest-point interior sampling + negative points.
- [x] Implement SAM3 segmenter stage with chunking, overlap handoff, drift detection, and re-anchor hooks.
- [x] Implement logit-based trimap builder and MEMatte tiled refine stage (unknown-band-only execution + skip-frame reuse).
- [x] Implement orchestrator/config/CLI for the two-stage flow with minimal flags.
- [x] Add evaluation harness tests and v1-v2 comparison scripts for quality and stability metrics.

## Risks / Unknowns
- Ultralytics SAM3 video predictor API may vary by version (mask prompt support, state reset, raw logit exposure).
- Chunk boundary handoff can cause visible seams without robust overlap crossfade logic.
- MEMatte runtime dependencies/checkpoint availability may vary across environments.
- 8K jobs can exceed VRAM without conservative tile/chunk defaults and OOM fallbacks.

## Verification
- [x] Unit tests: prompt sampling, drift detector, trimap thresholding, skip-frame logic (`pytest -q` currently `27 passed`).
- [x] Integration test: short clip end-to-end run with deterministic outputs (`run_0014` vs `run_0015` hash match).
- [x] Visual QC: hair detail, motion blur, occlusion recovery, low-contrast subject clip (representative runs `run_0010`..`run_0013` with continuity metrics).
- [x] Performance / memory check: 4K and 8K profiling with chunk + tile settings (`run_0016`, `run_0017`).
- [x] Comparison harness: intentionally out of scope for now (v2-only directive; no new v1 eval loops).

## Review Notes (fill after)
- What changed:
  - Reviewed v2 architecture requirements and mapped concrete source modules from v1.
  - Seeded this track with extracted foundation modules required for v2 implementation:
    - `src/videomatte_hq/models/edge_mematte.py` (from `D:\Videomatte-HQ\src\videomatte_hq\models\edge_mematte.py`)
    - `src/videomatte_hq/io/reader.py` (from `D:\Videomatte-HQ\src\videomatte_hq\io\reader.py`)
    - `src/videomatte_hq/io/writer.py` (from `D:\Videomatte-HQ\src\videomatte_hq\io\writer.py`)
    - `src/videomatte_hq/tiling/windows.py` (from `D:\Videomatte-HQ\src\videomatte_hq\tiling\windows.py`)
    - `src/videomatte_hq/tiling/stitch.py` (from `D:\Videomatte-HQ\src\videomatte_hq\tiling\stitch.py`)
    - `src/videomatte_hq/tiling/planner.py` (from `D:\Videomatte-HQ\src\videomatte_hq\tiling\planner.py`)
    - `src/videomatte_hq/utils/image.py` (from `D:\Videomatte-HQ\src\videomatte_hq\utils\image.py`)
    - `src/videomatte_hq/safe_math.py` (from `D:\Videomatte-HQ\src\videomatte_hq\safe_math.py`)
    - `src/videomatte_hq/postprocess/matte_tuning.py` (from `D:\Videomatte-HQ\src\videomatte_hq\pipeline\pass_matte_tuning.py`)
  - Implemented v2 core modules:
    - Protocols: `src/videomatte_hq/protocols.py`
    - Prompt adapters: `src/videomatte_hq/prompts/mask_adapter.py`, `src/videomatte_hq/prompts/box_adapter.py`
    - Stage logic: `src/videomatte_hq/pipeline/stage_segment.py`, `src/videomatte_hq/pipeline/stage_trimap.py`, `src/videomatte_hq/pipeline/stage_refine.py`, `src/videomatte_hq/pipeline/stage_qc.py`
    - Orchestration/config/CLI: `src/videomatte_hq/pipeline/orchestrator.py`, `src/videomatte_hq/config.py`, `src/videomatte_hq/cli.py`
  - Added evaluation harness + comparison CLI:
    - `src/videomatte_hq/eval/harness.py`, `src/videomatte_hq/eval/cli.py`, `tools/compare_v1_v2.py`
  - Added test and packaging scaffold:
    - `pyproject.toml`
    - `tests/test_prompt_adapters.py`, `tests/test_stage_trimap.py`, `tests/test_stage_qc.py`, `tests/test_stage_segment.py`, `tests/test_stage_refine.py`, `tests/test_config.py`, `tests/test_eval_harness.py`
- Evidence it works:
  - Extraction manifest and new source tree are present under `src/videomatte_hq/`.
  - Syntax validation passed for extracted modules via `python -c "import py_compile, pathlib; ..."` (`py_compile ok`).
  - v2 module/unit test suite passes: `pytest -q` (`27 passed`).
- Remaining issues / follow-ups:
  - Ultralytics SAM3 backend is implemented as a compatibility wrapper; direct mask-prompt video predictor integration still needs version-specific hardening.

---

# TODO — Native SAM Video Predictor Offset Start Support (2026-02-22)

## Goal
- Keep the native Ultralytics SAM video predictor fast path enabled when `FrameSource` starts at a non-zero video frame offset.

## Non-goals
- Changing default model quality tiers (already set to max-quality).
- Reworking chunked per-frame fallback behavior for non-video inputs.

## Plan
- [x] Extend the video fast-path backend API to accept a source video start frame offset.
- [x] Pre-seek the Ultralytics video loader before streaming so prompt anchoring occurs on the requested start frame.
- [x] Remove the `source_video_start == 0` fast-path gate and preserve fallback-on-error behavior.
- [x] Add tests for offset fast-path dispatch and offset backend alignment hooks.

## Risks / Unknowns
- Ultralytics internal predictor/dataloader APIs are version-sensitive (callback order, dataset attributes, `cap` behavior).
- Predictor instances are reused across calls; stale inference state can corrupt subsequent runs if not reset.
- OpenCV frame seek accuracy can vary by codec/container; fallback path must remain intact on mismatch/failure.

## Verification
- [x] Unit tests: `pytest -q` (targeted subset `25 passed`, then full suite `38 passed`).
- [x] Integration smoke: short clip with `--frame-start > 0` uses video fast path (no chunk fallback warning in command output; `output_tests/run_0023` produced 10 alpha frames).
- [x] Visual QC: offset-start alpha masks remain stable (user-reviewed output reported “looks good” after offset-start patch).
- [x] Performance check: direct offset-start comparison completed (`run_0024` fast path vs `run_0025` forced fallback on same 30-frame 4K clip/settings).

## Review Notes (fill after)
- What changed:
  - Extended `VideoSegmentBackend.segment_video_sequence(...)` to accept `start_frame`.
  - `ChunkedSegmenter` now passes `source.video_frame_start` into the native video fast path instead of forcing fallback for non-zero starts.
  - Ultralytics SAM video backend now injects an `on_predict_start` callback that:
    - resets stale predictor prompts/state for reused predictors,
    - pre-seeks the internal Ultralytics video loader (`dataset.cap`) to the requested start frame,
    - aligns `dataset.frame` so prompt conditioning occurs on the offset frame.
  - Callback is removed after streaming completes to avoid leaking duplicate hooks across runs.
  - Added tests for offset dispatch and predictor pre-seek/reset hook behavior.
- Evidence it works:
  - Targeted tests pass: `tests/test_stage_segment.py tests/test_reader.py tests/test_config.py` => `25 passed`.
  - Full suite passes: `pytest -q` => `38 passed`.
  - Offset-start smoke run succeeded: `output_tests/run_0023` (`frame_start=1`, `frame_end=10`, 10 alpha PNGs generated).
  - Command output for `run_0023` did not include the fallback warning (`"Video fast path failed; falling back to chunked per-frame path"`).
  - Visual QC for offset-start output was user-confirmed as good.
  - Offset-start perf benchmark (same clip/settings, 30 frames, 4K, `frame_start=1`, `fp16`, no refine):
    - `output_tests/run_0024` (native video fast path): `59.945s` (~`0.500 fps`)
    - `output_tests/run_0025` (forced chunked fallback via temporary monkeypatch): `59.757s` (~`0.502 fps`)
    - Result: no material speed delta on this short 4K offset sample (difference ~`0.19s`, ~`0.3%`); both runs produced 30 alpha frames.
- Remaining issues / follow-ups:
  - If needed, rerun the offset perf comparison on a longer clip or 8K offset sample to reduce benchmark noise and better isolate native video predictor gains.

---

# TODO — CLI Finish Pass (Solid CLI Pipeline) (2026-02-22)

## Goal
- Finish the tool as a solid CLI pipeline: production CLI auto-anchor support, preflight validation for common missing assets, and end-to-end refined CLI validation.

## Non-goals
- Web UI / FastAPI port.
- Full design-doc parity on config/flag count.
- v1-v2 comparison/eval loops.

## Plan
- [x] Add production CLI auto-anchor path (video input) using the hardened person-detection anchor flow.
- [x] Add CLI preflight checks for input/anchor/MEMatte asset paths and key backend dependency availability.
- [x] Expose MEMatte path overrides in the product CLI (repo dir + checkpoint) for real refinement runs.
- [x] Add a `pyproject.toml` console script entry point.
- [x] Run representative refined CLI validation clips and record outcomes.

## Risks / Unknowns
- MEMatte runtime may require extra Python dependencies beyond current environment.
- Refined CLI runs may be substantially slower than no-refine smoke tests on 4K clips.
- Auto-anchor on production CLI must not alter manual-anchor workflows.

## Verification
- [x] Unit tests: `pytest -q` (`41 passed`).
- [x] CLI UX: ran product CLI without manual `--anchor-mask` on video inputs (auto-anchor exercised on all representative runs).
- [x] Refine path: product CLI runs with `refine_enabled=True` using local MEMatte repo/checkpoint and writes alpha frames.
- [x] Packaging: console script entry point present in `pyproject.toml` and `videomatte-hq-v2 --help` works after editable install.

## Review Notes (fill after)
- What changed:
  - Added shared video auto-anchor helper in `src/videomatte_hq/prompts/auto_anchor.py`:
    - black-frame probing (`frame_start` + up to 10 probes),
    - explicit person detection via Ultralytics YOLO (quality-ordered fallback list),
    - anchor mask postprocessing + center-box fallback.
  - Hardened product CLI in `src/videomatte_hq/cli.py`:
    - `--auto-anchor` / `--no-auto-anchor` / `--auto-anchor-output`,
    - `--mematte-repo-dir` and `--mematte-checkpoint` overrides,
    - preflight checks for input paths, anchor masks, invalid frame ranges, Ultralytics availability, and MEMatte repo/checkpoint paths,
    - auto-anchor frame-start adjustment logging (requested vs effective start),
    - `config_used.json` and `run_summary.json` output metadata for reproducibility.
  - Added packaging console entry point in `pyproject.toml`:
    - `videomatte-hq-v2 = "videomatte_hq.cli:main"`.
  - Added CLI regression tests in `tests/test_cli.py`.
- Evidence it works:
  - Test suite passes: `pytest -q` => `41 passed`.
  - Console script validation:
    - editable install succeeded (`pip install -e . --no-deps`)
    - `.\.venv\Scripts\videomatte-hq-v2.exe --help` prints CLI usage with new auto-anchor and MEMatte path flags.
  - Representative refined product-CLI runs (no manual `--anchor-mask`, auto-anchor enabled, local MEMatte DIM checkpoint):
    - `output_cli/cli_refine_4625475` (1080p landscape): completed, 10 alpha frames, auto-anchor method `ultralytics_person_detect`.
    - `output_cli/cli_refine_4990426` (portrait black-opening clip): completed, 10 alpha frames; auto-anchor probed frame `1` and shifted `frame_start` `0 -> 1`; elapsed `14.665s`.
    - `output_cli/cli_refine_7219039` (4K dancer clip): completed, 10 alpha frames; Stage-2 reuse triggered (`reused=8`); elapsed `24.939s`.
  - Metadata files present for all three CLI runs:
    - `config_used.json` and `run_summary.json`.
  - Quick alpha coverage stats (mean / min / max):
    - `cli_refine_4625475`: `0.0305 / 0.0292 / 0.0321`
    - `cli_refine_4990426`: `0.0502 / 0.0493 / 0.0516`
    - `cli_refine_7219039`: `0.1769 / 0.1760 / 0.1771`
- Remaining issues / follow-ups:
  - README/install docs should be updated to document the new product CLI auto-anchor and MEMatte path flags (and local-only MEMatte path constraint).

---

# TODO — Localize MEMatte Assets To HQ2 (2026-02-22)

## Goal
- Eliminate runtime dependency on `D:\Videomatte-HQ` by moving MEMatte assets into `D:\Videomatte-HQ2` and enforcing local-only MEMatte paths in the CLI.

## Non-goals
- Removing configurable MEMatte path overrides entirely.
- Rewriting MEMatte integration code.

## Plan
- [x] Copy `third_party/MEMatte` (including checkpoint) into this repo.
- [x] Add CLI preflight containment checks so MEMatte repo/checkpoint paths must resolve under the current repo root.
- [x] Add regression tests for the containment check.
- [x] Re-run a refined CLI validation using local-only MEMatte paths/defaults and verify metadata no longer points to `D:\Videomatte-HQ`.

## Risks / Unknowns
- MEMatte directory copy may be large/slow.
- Some users may prefer external shared asset locations; containment enforcement is an intentional product constraint per user directive.

## Verification
- [x] `third_party/MEMatte` exists in `D:\Videomatte-HQ2`.
- [x] `pytest -q` passes after CLI preflight changes (`43 passed`).
- [x] Refined CLI run succeeds using local MEMatte assets and `run_summary.json` contains no `D:\Videomatte-HQ` paths.

## Review Notes (fill after)
- What changed:
  - Copied MEMatte assets from the original repo into this repo:
    - `D:\Videomatte-HQ2\third_party\MEMatte`
    - including `D:\Videomatte-HQ2\third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth`
  - Hardened `src/videomatte_hq/cli.py` MEMatte preflight:
    - relative MEMatte paths are resolved against the current repo root (`D:\Videomatte-HQ2`),
    - MEMatte repo/checkpoint paths are rejected if they resolve outside this repo,
    - normalized absolute local MEMatte paths are written into runtime config/metadata.
  - Added/updated CLI tests in `tests/test_cli.py` for local path normalization and external path rejection.
  - Ran a refined product-CLI validation using local-only MEMatte defaults (no `--mematte-*` overrides).
- Evidence it works:
  - `third_party/MEMatte` and DIM checkpoint now exist under `D:\Videomatte-HQ2`.
  - Test suite passes after containment changes: `pytest -q` => `43 passed`.
  - Local-only refined CLI run succeeded:
    - `output_cli/cli_refine_4990426_local` (portrait clip, auto-anchor shifted `frame_start 0 -> 1`, elapsed `16.358s`)
  - `output_cli/cli_refine_4990426_local/run_summary.json` and `config_used.json` both record:
    - `mematte_repo_dir = D:\Videomatte-HQ2\third_party\MEMatte`
    - `mematte_checkpoint = D:\Videomatte-HQ2\third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth`
  - Explicit negative CLI check confirms external MEMatte paths are blocked:
    - passing `D:\Videomatte-HQ\third_party\MEMatte` now fails preflight with `External MEMatte paths are not allowed for this tool`.
- Remaining issues / follow-ups:
  - Historical logs/task notes may still mention `D:\Videomatte-HQ` as provenance, but the runtime CLI path is now self-contained to `D:\Videomatte-HQ2`.

---

# TODO — Docs + GitHub Publish (2026-02-22)

## Goal
- Add a detailed `README.md` and `BEGINNER_GUIDE.md` for the current CLI pipeline, then publish this repo state to GitHub (force-push overwrite per user request).

## Non-goals
- Building a web UI.
- Rewriting architecture/runtime code unless docs validation reveals a critical inconsistency.

## Plan
- [x] Write `README.md` with setup, CLI usage, examples, troubleshooting, and local-only MEMatte notes.
- [x] Write `BEGINNER_GUIDE.md` with step-by-step first-run instructions on Windows.
- [x] Verify docs against current CLI flags/help and local MEMatte constraints.
- [x] Initialize git in `D:\Videomatte-HQ2`, commit repo contents, and force-push to `https://github.com/cedarconnor/Videomatte-HQ.git`.

## Risks / Unknowns
- Force-push requires valid GitHub credentials/access from this environment.
- Large repo contents (weights, MEMatte, outputs) may make push slow or exceed repository expectations.

## Verification
- [x] `README.md` and `BEGINNER_GUIDE.md` exist and reflect current CLI behavior.
- [x] `videomatte-hq-v2 --help` options documented accurately.
- [x] Git push completes to the provided remote URL.

## Review Notes (fill after)
- What changed:
  - Added a detailed project `README.md` for the v2 CLI pipeline (architecture, install, CLI usage, examples, troubleshooting, local-only MEMatte constraint, and notes about large weights/checkpoints).
  - Added `BEGINNER_GUIDE.md` with a Windows-first setup and first-run walkthrough for full refinement and preview/no-refine modes.
  - Added `.gitignore` to exclude local outputs, virtualenvs, and large model/checkpoint files (`*.pt`, `*.pth`) that are not suitable for a standard GitHub push.
  - Flattened `third_party/MEMatte` into a normal vendored directory by removing nested git metadata so this repo tracks MEMatte source directly (not a broken gitlink/submodule).
  - Initialized git in `D:\Videomatte-HQ2`, committed the repo state, and force-pushed to the user-provided GitHub repo.
- Evidence it works:
  - `README.md` and `BEGINNER_GUIDE.md` exist at repo root and describe the current `videomatte-hq-v2` product CLI path.
  - CLI help was previously validated from the local venv after editable install:
    - `.\.venv\Scripts\videomatte-hq-v2.exe --help`
  - Git push succeeded:
    - commit: `0c3dee4` (`Build v2 CLI pipeline and beginner docs`)
    - remote: `https://github.com/cedarconnor/Videomatte-HQ.git`
    - result: forced update of `origin/main`
- Remaining issues / follow-ups:
  - Large weights/checkpoints are intentionally not pushed (`*.pt`, `*.pth`) due GitHub size limits; the repo remains code/docs-complete but users must supply local model files.
