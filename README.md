# VideoMatte-HQ

Offline Option B video matting pipeline for people footage.

The current implementation is a mask-first, assignment-driven workflow:
- Stage 0: load frames
- Stage 1: load/create project and keyframe assignments
- Stage 2: memory-query coarse alpha
- Stage 3: boundary-band refinement
- Stage 4: confidence-gated temporal cleanup
- Stage 5: matte tuning (shrink/grow, feather, XY offset)
- Stage 6: output write + QC metrics/gates

## What Changed
- Old multi-stage flow-stabilized pipeline is removed from runtime.
- Project-backed mask-first assignment (`.vmhqproj`) is required by default.
- Correction anchors support suggested partial reprocess ranges.
- Built-in QC metrics and regression gates can fail runs automatically.

## Requirements
- Python 3.10+
- Windows/Linux/macOS
- Optional CUDA GPU (CPU also supported)

## Install
```bash
python -m venv .venv
.venv\Scripts\pip install -e .
```

## Quick Start (CLI)

1. Import an initial assignment mask:
```bash
videomatte-hq \
  --input input_frames/frame_%05d.png \
  --project output/project.vmhqproj \
  --assign-mask masks/mask_00000.png \
  --assign-frame 0 \
  --assign-kind initial \
  --assign-only
```

2. Run the Option B pipeline with QC gates enabled:
```bash
videomatte-hq \
  --input input_frames/frame_%05d.png \
  --out output/alpha/%05d.png \
  --project output/project.vmhqproj \
  --require-assignment \
  --qc \
  --qc-fail-on-regression
```

3. Add a correction anchor and auto-apply suggested reprocess range:
```bash
videomatte-hq \
  --input input_frames/frame_%05d.png \
  --project output/project.vmhqproj \
  --assign-mask masks/mask_00120.png \
  --assign-frame 120 \
  --assign-kind correction \
  --apply-suggested-range
```

4. Apply matte tuning (trimap width, grow, feather, offset):
```bash
videomatte-hq \
  --input input_frames/frame_%05d.png \
  --out output/alpha/%05d.png \
  --project output/project.vmhqproj \
  --unknown-band-px 64 \
  --mt-shrink-grow-px 1 \
  --mt-feather-px 1 \
  --mt-offset-x-px 0 \
  --mt-offset-y-px 0
```

## Launcher
`run_videomatte.bat` includes tuned QC defaults and hard gating:
- `QC_FAIL_ON_REGRESSION=1`
- `QC_MAX_P95_FLICKER=0.005`
- `QC_MAX_P95_EDGE_FLICKER=0.02`
- `QC_MIN_MEAN_EDGE_CONFIDENCE=0.22`
- `QC_BAND_SPIKE_RATIO=1.8`
- `QC_MAX_BAND_SPIKE_FRAMES=3`
- `QC_MAX_OUTPUT_ROUNDTRIP_MAE=0.002`

## Key CLI Flags

### I/O and run control
- `--input`, `--out`, `--project`
- `--start`, `--end`
- `--device`, `--precision`, `--workers`
- `--resume/--no-resume`

### Assignment workflow
- `--require-assignment/--allow-empty-assignment`
- `--assign-mask`, `--assign-frame`
- `--assign-kind {initial,correction}`
- `--apply-suggested-range/--no-apply-suggested-range`
- `--assign-only`

### Memory core
- `--memory-backend`
- `--memory-frames`
- `--window`

### QC and regression gates
- `--qc/--no-qc`
- `--qc-fail-on-regression/--no-qc-fail-on-regression`
- `--qc-sample-output-frames`
- `--qc-max-output-roundtrip-mae`
- `--qc-alpha-range-eps`
- `--qc-max-p95-flicker`
- `--qc-max-p95-edge-flicker`
- `--qc-min-mean-edge-confidence`
- `--qc-band-spike-ratio`
- `--qc-max-band-spike-frames`

### Matte tuning
- `--unknown-band-px` (alias: `--mt-trimap-width-px`)
- `--matte-tuning/--no-matte-tuning`
- `--mt-shrink-grow-px`
- `--mt-feather-px`
- `--mt-offset-x-px`
- `--mt-offset-y-px`

## QC Outputs
When QC is enabled, artifacts are written under:
- `output_dir/qc/optionb_metrics.json`
- `output_dir/qc/optionb_report.md`

Metrics include:
- alpha validity/range checks
- p95 temporal flicker
- p95 edge-band flicker
- mean edge confidence
- band coverage spike detection
- sampled output roundtrip MAE (written output vs in-memory alpha)

If `fail_on_regression` is enabled, any failed gate causes a non-zero run exit.

## Web UI
Run:
```bash
run_web.bat
```
Then open `http://localhost:5173`.

### Run Job tab
- Input/output + frame range
- Mask-first assignment import
- Initial Mask Builder (Phase 2): load frame, text-prompt box suggestions, draw/adjust box, add FG/BG points, build + import mask
- Anchor type: `initial` / `correction`
- Auto-apply suggested reprocess range
- One-click matte tuning presets: `Subtle`, `Balanced`, `Aggressive`, plus `Reset`
- Matte tuning controls (trimap width, shrink/grow, feather, offset X/Y)
- Runtime settings
- QC & regression-gate settings (all QC thresholds exposed)

### Quality Control tab
- A/B wipe comparison
- Alpha/checker/white/black/overlay composite modes
- Overlay color + opacity controls for matte inspection
- Dynamic path discovery from recent jobs (`/api/qc/info`)

## API Endpoints (current)
- `POST /api/jobs`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/logs`
- `POST /api/jobs/{job_id}/cancel`
- `POST /api/project/state`
- `POST /api/assignments/import`
- `POST /api/assignments/suggest-range`
- `POST /api/assignments/frame-preview`
- `POST /api/assignments/build-mask`
- `POST /api/assignments/suggest-boxes`
- `GET /api/qc/info`

## Config Schema (runtime)
Top-level sections used by Option B runtime:
- `io`
- `project`
- `assignment`
- `memory`
- `refine`
- `temporal_cleanup`
- `matte_tuning`
- `preview`
- `qc`
- `runtime`

Example (`my_config.yaml`):
```yaml
io:
  input: "input_frames/frame_%05d.png"
  output_dir: "output"
  output_alpha: "alpha/%05d.png"
  frame_start: 0
  frame_end: -1

project:
  path: "output/project.vmhqproj"

assignment:
  require_assignment: true

memory:
  backend: "appearance_memory_bank"
  memory_frames: 12
  window: 120

refine:
  enabled: true
  backend: "guided_band"
  unknown_band_px: 64
  tile_size: 1536
  overlap: 96

temporal_cleanup:
  enabled: true
  outside_band_ema: 0.15
  min_confidence: 0.5

matte_tuning:
  enabled: true
  shrink_grow_px: 0
  feather_px: 0
  offset_x_px: 0
  offset_y_px: 0

qc:
  enabled: true
  fail_on_regression: true
  max_p95_flicker: 0.005
  max_p95_edge_flicker: 0.02
  min_mean_edge_confidence: 0.22
  band_spike_ratio: 1.8
  max_band_spike_frames: 3
  max_output_roundtrip_mae: 0.002

runtime:
  device: "cuda"
  precision: "fp16"
  workers_io: 4
  resume: true
```

Run with config:
```bash
videomatte-hq --config my_config.yaml
```

## Development
Run tests:
```bash
.venv\Scripts\python -m pytest -q
```

Frontend build:
```bash
cd web
npm run build
```
