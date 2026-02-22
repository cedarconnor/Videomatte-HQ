# agents.md — Videomatte-HQ

This repo builds a **high-quality, high-resolution video matting pipeline** (up to 4K–8K) with a simplified **v2 two-stage architecture**:

1) **Coarse segmentation / tracking** (SAM3 via Ultralytics backend, swappable)
2) **Edge refinement matting** (MeMatte, tiled, swappable)

Primary goals:
- **Clean, debuggable pipeline** with minimal moving parts
- **Temporal stability without blur** (avoid “warp-and-blend” artifacts)
- **Scalable to high resolution** (tiling + unknown-band compute)
- **Fast iteration** (preview mode + re-anchoring strategy)

---

## Repo Conventions

### Suggested Layout (v1 frozen, v2 active)
- `videomatte_hq/v1/`  
  - Legacy pipeline (frozen; bugfixes only)
- `videomatte_hq/v2/`
  - `stage_segment.py` — SAM3 segmentation + tracking backend (pluggable)
  - `stage_trimap.py` — trimap generation (adaptive unknown band)
  - `stage_refine.py` — MeMatte refinement (tiled)
  - `stage_qc.py` — drift/jitter detection + re-anchor triggers
  - `pipeline.py` — orchestration + config
- `videomatte_hq/common/`
  - IO, video decode/encode, caching, tiling utils, device mgmt, logging
- `configs/` — example configs (yaml/json)
- `tests/` — unit tests + small integration tests
- `tasks/`
  - `todo.md` — execution plans + progress tracking
  - `lessons.md` — self-improvement rules + recurring pitfalls

### Definitions
- **Coarse mask**: segmentation output (binary or logits)
- **Trimap**: {0=BG, 0.5=unknown, 1=FG}
- **Alpha**: float matte [0..1]
- **Unknown band**: trimap region where refinement is allowed to modify alpha

---

## Workflow Orchestration (MANDATORY)

### 1) Plan Mode Default
**Enter plan mode** for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, **STOP and re-plan immediately** — don’t keep pushing.
- Use plan mode for **verification steps**, not just building.
- Write **detailed specs upfront** to reduce ambiguity.

**Plan Mode Output** (write into `tasks/todo.md`):
- Goal / non-goals
- Proposed changes (files/modules)
- Risks + mitigations
- Verification checklist (how we prove it works)

### 2) Subagent Strategy
Use subagents liberally to keep the main context window clean.
- Offload **research, exploration, and parallel analysis** to subagents.
- For complex problems, throw more compute at it via subagents.
- **One task per subagent** for focused execution.

Examples of subagent tasks:
- “Investigate Ultralytics SAM3 `track()` API behavior & state persistence”
- “Compare MeMatte vs ViTMatte for 8K edge refinement”
- “Design drift detection heuristics for long sequences”

### 3) Self-Improvement Loop
After ANY correction from the user:
- Update `tasks/lessons.md` with:
  - the mistake pattern
  - the corrected rule
  - a prevention checklist
- Ruthlessly iterate until the mistake rate drops.
- At the **start of each session**, skim lessons relevant to current work.

### 4) Verification Before Done
Never mark a task complete without proving it works.
- Diff behavior between `main` and your changes when relevant.
- Ask yourself: **“Would a staff engineer approve this?”**
- Run tests, check logs, demonstrate correctness (with reproducible steps).

### 5) Demand Elegance (Balanced)
For non-trivial changes:
- Pause and ask: **“Is there a more elegant way?”**
- If a fix feels hacky: **“Knowing everything I know now, implement the elegant solution.”**
- Skip this for simple, obvious fixes — don’t over-engineer.
- Challenge your own work before presenting it.

### 6) Autonomous Bug Fixing
When given a bug report:
- Just fix it. Don’t ask for hand-holding.
- Point at logs/errors/failing tests — then resolve them.
- Zero context switching required from the user.
- Fix failing CI tests without being told how.

---

## Task Management (MANDATORY)

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

### `tasks/todo.md` Template
```md
# TODO — <topic>

## Goal
-

## Non-goals
-

## Plan
- [ ] Step 1
- [ ] Step 2
- [ ] Step 3

## Risks / Unknowns
-

## Verification
- [ ] Unit tests:
- [ ] Integration test:
- [ ] Visual QC:
- [ ] Performance / memory check:

## Review Notes (fill after)
- What changed:
- Evidence it works:
- Remaining issues / follow-ups: