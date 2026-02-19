# Lessons Learned

## UI/UX Corrections
- Keep run-stage navigation in one place. Do not duplicate stage lists in both app sidebar and page body.
- Default labels should describe intent in plain language, not internal CLI parameter names.
- Keep the primary run action (`Start Pipeline`) sticky and visible without scrolling.
- Do not unmount `RunTab` when switching app tabs; preserve in-progress wizard state by keeping the component mounted.
- Browser drag/drop cannot reliably provide absolute local file paths; always provide explicit local browse actions for desktop workflows.
- Do not force wizard mask-building into single-frame mode when Stage 1 quality depends on Samurai range propagation defaults.

## Workflow Guardrails
- Validate frontend and backend after UI refactors (`npm run build` and `pytest -q`) before reporting completion.
- When Stage-1 subject priors clip limbs/hair, increase region-prior expansion defaults (`bbox_margin`, `bbox_expand_ratio`, `dilate_px`) rather than tightening downstream refiners.
