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
