# Lessons Learned

## UI/UX Corrections
- Keep run-stage navigation in one place. Do not duplicate stage lists in both app sidebar and page body.
- Default labels should describe intent in plain language, not internal CLI parameter names.
- Keep the primary run action (`Start Pipeline`) sticky and visible without scrolling.

## Workflow Guardrails
- Validate frontend and backend after UI refactors (`npm run build` and `pytest -q`) before reporting completion.
