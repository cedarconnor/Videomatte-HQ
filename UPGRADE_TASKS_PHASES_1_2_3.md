# VideoMatte-HQ Edge Stability Upgrade Plan (Phases 1-3)

## Goal
Stabilize Stage 2/3 edge behavior and make regressions diagnosable in one run.

## Phase 1 - Auto Stage Diagnosis On QC Failure
- [x] Add config flags for automatic stage diagnosis on QC gate failure.
- [x] Auto-export per-stage samples when QC fails (even if manual debug export is off).
- [x] Write stage diagnosis JSON/Markdown automatically for failed QC runs.
- [x] Expose new controls in UI and CLI.
- [x] Add focused orchestrator test coverage.

Pass criteria:
- Any run with failed QC gates produces `debug_stages/diagnosis.json` and `debug_stages/diagnosis.md`.

## Phase 2 - Temporal Stabilization Option Pack
- [x] Add toggleable edge-flicker mitigations (edge EMA, confidence-gated clamp, optional edge snap).
- [x] Add objective pass/fail metrics per mitigation on test subset.
- [x] Keep defaults conservative and deterministic.

Pass criteria:
- At least one mitigation lowers `p95_edge_flicker` without increasing leakage metrics.

## Phase 3 - Locked Workflow Presets + Safe Defaults
- [ ] Add simple quality presets for SAM2/Samurai + MatAnyone + MEMatte workflow.
- [ ] Bind QC thresholds and runtime knobs to preset levels.
- [ ] Document recommended preset per hardware tier.

Pass criteria:
- New users can run default preset without manual low-level tuning.
