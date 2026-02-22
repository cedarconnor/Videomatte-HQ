# Lessons Learned

## 2026-02-21 — Person Prompt Specificity And Prompt Slack
**Mistake pattern:**  
- Relied on generic/over-tight mask prompting, which can produce clipped silhouettes and unstable propagation.

**New rule:**  
- For human matting anchors, explicitly prompt `person` and propagate prompts as `bbox + points` with bbox slack; avoid point-only-first fallback ordering.

**Prevention checklist:**  
- [ ] Verify prompt construction explicitly targets person class for auto-anchors.
- [ ] Ensure mask->prompt includes bbox expansion and combined prompt priority.
- [ ] Validate on a short clip that alpha means are not near-white and silhouette extremities are preserved.
- [ ] Cross-check suspected drift/overreach against source video frames before hardening constraints.
- [ ] When using pixel thresholds in stage-1 filtering, scale expectations to processing resolution, not source resolution.

## 2026-02-21 — Preview Mode Should Not Use Raw Coarse Probabilities
**Mistake pattern:**  
- In no-refine runs, using raw coarse probabilities produced torn matte interiors and unstable holes.

**New rule:**  
- For no-refine preview output, emit a cleaned binary alpha mask (component-filtered + hole-filled + morphology), not raw logits-derived probabilities.

**Prevention checklist:**  
- [ ] Validate frame 0 and final frame for interior holes before accepting preview output.
- [ ] Keep preview cleanup scoped to no-refine mode so MEMatte path is unaffected.

## 2026-02-22 — Honor Explicit Quality Tier Requests In Defaults
**Mistake pattern:**  
- Continued using lower-tier or missing default checkpoints after user asked for maximum quality defaults.

**New rule:**  
- When user requests maximum quality, set default model selection to top-tier checkpoints and order auto-detection fallbacks highest-to-lowest quality.

**Prevention checklist:**  
- [ ] Confirm config defaults match highest available quality tier (`sam2_l.pt` or stronger when supported).
- [ ] Confirm CLI defaults mirror config defaults for reproducibility.
- [ ] Verify auto-anchor detector candidate ordering starts from `x/l` tiers before `n`.
- [ ] Add/keep unit tests that fail if defaults regress to lower-tier models.

## 2026-02-22 — Handle Empty SAM Candidate Stacks
**Mistake pattern:**  
- Assumed SAM mask stacks always contain at least one candidate; high-tier model outputs sometimes returned `(0, H, W)` and crashed selection.

**New rule:**  
- Treat empty candidate stacks as valid no-detection frames and return an all-background probability map instead of raising.

**Prevention checklist:**  
- [ ] Add regression tests for zero-length candidate stacks.
- [ ] Keep result conversion tolerant to missing masks for all supported model variants.
- [ ] Re-run a multi-clip integration batch after model-tier changes.

## 2026-02-22 — Align Processing Start With Auto-Anchor Probe Frame
**Mistake pattern:**  
- Auto-anchor could be built from a later non-black probe frame while segmentation still started at the original `frame_start`, causing prompt/content misalignment and black collapse on black-opening clips.

**New rule:**  
- When auto-anchor is derived from a probed frame index, use `effective_frame_start = max(requested_frame_start, anchor_probe_frame)` for processing and record both values in run metadata.

**Prevention checklist:**  
- [ ] Include both requested and effective frame starts in run summaries.
- [ ] Validate black-opening clips with `frame_start=0` after any anchor logic change.
- [ ] Keep manual-anchor behavior unchanged unless explicitly requested.

## 2026-02-22 — Confirm Completion Criteria Before Gap Analysis
**Mistake pattern:**  
- Framed remaining work against full design-document completion when the user's actual acceptance target was a solid CLI pipeline.

**New rule:**  
- Before listing “what is left,” explicitly anchor the answer to the user's finish criteria (CLI pipeline vs full product/UI/doc parity).

**Prevention checklist:**  
- [ ] Restate the acceptance target in one line before enumerating gaps.
- [ ] Separate CLI blockers from optional design-alignment improvements.
- [ ] Avoid presenting future-scope items (UI, v2.x features) as blockers unless the user asks for them.

## 2026-02-22 — Final Tool Must Not Depend On Original Repo Paths
**Mistake pattern:**  
- Validated the CLI using external MEMatte paths from `D:\Videomatte-HQ`, leaving the final tool dependent on the original repository layout.

**New rule:**  
- For final-tool signoff, all runtime assets/dependencies used by the CLI (especially MEMatte repo + checkpoints) must be contained under `D:\Videomatte-HQ2` and preflight should reject external MEMatte paths.

**Prevention checklist:**  
- [ ] Verify `third_party/MEMatte` exists inside the current repo before refined CLI signoff.
- [ ] Ensure CLI preflight rejects MEMatte repo/checkpoint paths outside the current repo root.
- [ ] Re-run final refined CLI validation using local-only defaults/paths and check `run_summary.json`.
