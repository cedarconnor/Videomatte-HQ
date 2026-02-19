# UI Redesign Execution Plan

## Scope
Implement the `UI_REDESIGN_SPEC.md` redesign with progressive disclosure:
- Wizard Mode (default)
- Pro Mode (dashboard)

## Plan
- [x] Review current requirements (`UI_REDESIGN_SPEC.md`, `Agents.md`) and confirm constraints.
- [x] Phase 1: Introduce mode architecture and layout shells.
  - Add `WizardLayout` stepper container.
  - Add `DashboardLayout` shell (left stage list, sticky header action area, right context panel).
  - Add mode switch + persistence (`Wizard` default) in `RunTab`.
- [x] Phase 2: Build Wizard happy path (functional, v1).
  - Step 1 Setup: input/output basics + validation.
  - Step 2 Subject: mask builder focus (box/points/autodetect + keyframe workflow).
  - Step 3 Refine: simplified controls (`Edge Tightness`, `Edge Softness`, `De-Spill`).
  - Step 4 Render: run summary + start pipeline + progress handoff.
- [x] Phase 3: Pro Mode cleanup.
  - Keep full power controls but improve structure and naming.
  - Ensure sticky global Start Pipeline is always visible.
  - Ensure stage navigation is reliable and less cluttered.
- [x] Phase 4: Basic/Advanced visibility split.
  - Apply clear `Basic` defaults.
  - Hide backend internals specified as `HIDDEN` in spec.
- [x] Phase 5: UX delighters.
  - Smart defaults for input.
  - Immediate validation warnings.
  - Keyboard shortcuts in mask builder (`F`, `B`, `Enter`).
  - Toast action for completed jobs.
- [ ] Phase 6: QA + docs.
  - Verify all UI controls map correctly to API/config.
  - Run frontend build + pytest.
  - Update README + Beginner Guide for new UI flow.

## In Progress
- Continuing **Phase 6**:
  - `RunTab.tsx` is still monolithic and should be split into wizard/pro section components.
  - Additional Pro help-context UX can be added in right panel.

## Review Notes
- Current `RunTab.tsx` is monolithic; first pass will add shells and mode routing without breaking existing run behavior.
- Completed in this slice:
  - Run stage navigation is now in the main app left sidebar (under `Run Job`) when Pro mode is active.
  - Inner Pro stage sidebar is hidden to remove nested navigation confusion.
  - Pro stage labels and major section titles were renamed to plain-English wording.
  - Advanced-only stages (background/framing/global) are hidden by default with a clear notice.
  - Added UI delighters:
    - Auto-suggest input from `TestFiles` when exactly one clip is present.
    - Live output-folder warning if destination exists and overwrite is off.
    - Mask builder shortcuts: `F`, `B`, `Enter`.
    - Completion toast action: `View Result (A/B)` opens Job Queue.
  - Validation: `cd web && npm run build` passed; `python -m pytest -q` passed (`43 passed, 1 skipped`).

---

# CLI + UI Debug Track (6138680-uhd_3840_2160_24fps.mp4)

## Scope
Stabilize subject selection quality in Stage 1 for CLI-driven runs and eliminate UI-to-CLI execution failures so the full phase pipeline can run reliably on:
- `TestFiles/6138680-uhd_3840_2160_24fps.mp4`

## Plan
- [x] Reproduce and baseline current behavior on the target clip.
  - Capture current Stage 1 priors/masks and QC artifacts.
  - Record exact failure signatures for CLI and UI-triggered runs.
- [x] Fix CLI Stage 1 subject-guidance robustness.
  - Improve Stage 1 mask/region-prior stability to reduce background bleed and subject loss.
  - Add guardrails so degenerate guidance is detected and handled earlier.
- [x] Fix UI-triggered command-line run reliability.
  - Ensure backend job runner uses a stable Python runtime selection.
  - Improve failure reporting when subprocess exits abnormally.
  - Add safe defaults/guardrails for first full run on 4K clips.
- [x] Verify phase-by-phase quality progression.
  - Run targeted passes and inspect stage sample outputs (`stage1_region_prior` -> `stage5_tuned`).
  - Re-run with adjusted settings until Stage 1 guidance is clean and downstream stages are stable.
- [x] Regression verification.
  - Run `python -m pytest -q`.
  - Run `cd web && npm run build`.
- [x] Document outcomes.
  - Update this file’s review notes with root cause, fixes, and residual risks.

## In Progress
- Completed validation + residual tuning needed:
  - Stage 1 guidance stability is now bounded across full-range replay (no near-empty collapse).
  - UI and CLI both run reliably from `.venv` path with clearer runtime failure messages.
  - Added Stage 4 motion-warp mitigation path (config + CLI controls) and validated A/B behavior.
  - Remaining quality risk is Stage 4 edge flicker, not Stage 1 subject acquisition.

## Review Notes
- Root cause (CLI quality):
  - Stage 1 propagated guidance could collapse/expand on sparse-anchor segments (full-range baseline had min coverage near `0.010`).
- Root cause (UI runtime failures):
  - UI-submitted jobs could execute under a non-venv interpreter in some launch contexts, producing opaque command-line failures.
  - Full-range 4K defaults increased risk of native aborts before users could iterate.
- Implemented fixes:
  - Added dynamic anchor-relative coverage guardrails + fallback stabilization in `memory_region_constraint.py`.
  - Added CLI startup/runtime dependency error formatting with explicit `.venv` remediation in `cli.py`.
  - Updated UI job runner to prefer `.venv` Python and emit actionable failure reasons in `jobs.py`.
  - Set UI default frame end to `30` in `RunTab.tsx` for safer first-pass runs.
  - Added motion-aware Stage 4 temporal blending support in `pass_temporal_cleanup.py`.
  - Added CLI toggles for motion-warp cleanup (`--tc-motion-warp`, `--tc-motion-warp-max-side`) and config fields.
  - Added mitigation regression coverage in `tests/test_temporal_mitigation_pack.py`.
- Verification evidence:
  - Full-range Stage 1 replay (`163` frames): stabilized note emitted with fallback application; prior coverage floor no longer near zero.
  - End-to-end validation runs (`0..30`) completed for both motion-warp and no-motion-warp modes, with all stages exported (`stage1_region_prior`..`stage5_tuned`).
  - Motion-warp A/B (same clip/settings): Stage 3->4 diagnosis score improved (`7.870` -> `4.866`), while QC `p95_edge_flicker` remained high (`0.481` vs `0.482`) and still failed gate.
  - Full-clip outline render completed (`163` frames) to `output_quality_full_outline/alpha/` with tuned Stage 4 settings; runtime `1037.7s`.
  - Full-clip QC summary: `p95_flicker=0.07505`, `p95_edge_flicker=0.49342`, `mean_edge_conf=0.45521`; only `edge_flicker` gate failed.
  - Regression checks passed: `44 passed, 1 skipped`; frontend production build passed.
- Residual risk:
  - QC `edge_flicker` gate still fails on this clip (best observed around `~0.47` vs gate `0.12`); this appears dominated by high-motion edge behavior rather than Stage 1 subject acquisition/runtime failures.
