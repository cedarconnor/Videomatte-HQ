# Videomatte-HQ v2 (CLI + Local Web UI)

High-quality video matting pipeline with a simplified two-stage architecture:

1. Stage 1: Ultralytics SAM-based subject segmentation/tracking
2. Stage 2: MEMatte edge refinement (tiled, high-resolution)

This repo is now both:

- a product CLI (`videomatte-hq-v2`)
- a local-first web UI (`videomatte-hq-web` + `run_web.bat`)

## Status (Current Behavior)

- Full `SAM -> trimap -> MEMatte` pipeline is implemented and used in production runs (not a stub).
- MEMatte refinement is mandatory in this build.
  - `--no-refine` / preview fallback runs are intentionally not supported.
  - If MEMatte cannot run, the pipeline fails.
- If Stage 2 receives no unknown trimap band, the run fails with a clear error (`MEMatte did not execute on any tiles ...`) so you can retune trimap thresholds.
- QC trimap previews are written for runs and can be inspected in the web UI.

## Key Features

### Core pipeline

- Video input and image sequence input
- Auto-anchor generation for video inputs (person-targeted)
- Manual anchor mask support
- Black-opening video handling (auto-anchor probe may shift effective `frame_start`)
- High-quality default SAM model (`sam2_l.pt`)
- Tiled MEMatte refinement for high resolutions (4K/8K support)
- Temporal reuse optimization in Stage 2 (`skip_iou_threshold`)

### Local web UI (v2 CLI wrapper)

- Run / Jobs / QC / Settings tabs
- File and folder browsing (`Browse` buttons, no manual path typing required)
- Live job progress (polling + parsed CLI log progress)
- SQLite-backed job history persistence across backend restarts
- QC image preview caching (LRU)
- QC alpha preview + QC trimap preview (actual Stage-2 trimap input)
- Visible `Trimap Refine Band` control on the Run page (preset slider + exact thresholds)
- Jobs page remediation hint for `MEMatte did not execute on any tiles` failures

## MEMatte Path Policy (Default + Override)

Default behavior:

- The tool expects MEMatte assets under this repo (recommended self-contained setup):
  - `third_party/MEMatte`
  - `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth`
- CLI/web preflight enforces repo-local paths by default.

Power-user override (supported):

- CLI: `--allow-external-paths`
- Web UI: `Allow External MEMatte Paths` checkbox on the Run page

This lets you point at an external MEMatte checkout/checkpoint if needed.

## Repo Layout (important parts)

- `src/videomatte_hq/cli.py`: product CLI entry point + preflight
- `src/videomatte_hq/config.py`: config model and defaults
- `src/videomatte_hq/pipeline/orchestrator.py`: Stage 1 -> trimap -> Stage 2 wiring + QC trimap preview writes
- `src/videomatte_hq/pipeline/stage_segment.py`: SAM segmentation/tracking
- `src/videomatte_hq/pipeline/stage_trimap.py`: logit-based trimap generation
- `src/videomatte_hq/pipeline/stage_refine.py`: MEMatte tiled refinement (strict, no preview fallback)
- `src/videomatte_hq/models/edge_mematte.py`: detectron2-free MEMatte wrapper
- `src/videomatte_hq/prompts/auto_anchor.py`: auto-anchor generation (video, person detection)
- `src/videomatte_hq_web/server.py`: web server entry point (thin wrapper)
- `src/videomatte_hq_web/server_runtime.py`: FastAPI app + QC preview API + browse API
- `src/videomatte_hq_web/jobs.py`: local subprocess job runner + progress parsing + SQLite job metadata
- `web/`: React/Vite frontend (Run/Jobs/QC/Settings)
- `third_party/MEMatte`: local MEMatte source (checkpoint path expected under `checkpoints/`)

## Requirements

- Windows (primary workflow tested here) or another OS supported by Python/PyTorch/OpenCV
- Python 3.10+
- GPU recommended for practical speed (`cuda`)
- Node.js (only if using the web UI frontend in dev mode)

Core Python dependencies are declared in `pyproject.toml`.

Notes about MEMatte runtime dependencies:

- The wrapper includes compatibility shims for several packages (detectron2, timm, fairscale, fvcore).
- Some MEMatte code paths may still require `einops`.
- If refinement fails with import/runtime messages mentioning `einops`, install it:

```powershell
pip install einops
```

## Install (CLI)

### 1. Create/activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the project

```powershell
pip install -e .
```

This installs the console command:

```powershell
videomatte-hq-v2 --help
```

If needed, you can run the module directly:

```powershell
python -m videomatte_hq.cli --help
```

## Install (Web UI)

### 1. Install Python web extras

```powershell
pip install -e .[web]
```

### 2. Install frontend dependencies

```powershell
cd web
npm install
cd ..
```

### 3. Start backend + frontend together

```powershell
run_web.bat
```

What `run_web.bat` does:

- starts the FastAPI backend (`127.0.0.1:8000`)
- starts the Vite frontend dev server on a free port (`5173..5199`)
- prints the exact frontend/backend URLs

If `web/dist` exists, the backend can also serve the built frontend as a single-port fallback.

## Quick Start (CLI, Video, Auto-Anchor, Full Refinement)

The CLI auto-generates an anchor mask for video inputs when `--anchor-mask` is not provided.

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\4990426-hd_1080_1920_30fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\my_run" `
  --frame-start 0 `
  --frame-end 30 `
  --device cuda `
  --precision fp16
```

What happens:

- Auto-anchor is generated and saved to `<output-dir>\anchor_mask.auto.png`
- If the requested first frame is too dark/black, auto-anchor may probe forward and shift the effective `frame_start`
- Stage 1 segmentation/tracking runs
- Stage 2 MEMatte refinement runs (required)
- Alpha frames are written to `<output-dir>\alpha\`
- QC trimap preview frames are written to `<output-dir>\qc\trimap.000000.png`, ...

## Quick Start (Web UI)

1. Start the UI with `run_web.bat`
2. Open the printed frontend URL
3. In `Run`:
   - choose input/output with `Browse`
   - keep MEMatte refine enabled (required)
   - use `Auto-Anchor Preview` if you want to inspect the initial anchor
   - start with a short frame range (for example `0..30`)
4. If a job fails with `MEMatte did not execute on any tiles`:
   - increase `Trimap Refine Band` to `Wider` or `Maximum`
   - re-run
   - inspect QC trimap preview (gray region = MEMatte unknown band)
5. Inspect output in `QC` and `Jobs`, then run a longer range

## Image Sequence Workflow (Manual Anchor Required)

Auto-anchor currently supports video file inputs only.

For an image sequence, provide an explicit `--anchor-mask`:

```powershell
videomatte-hq-v2 `
  --input "D:\frames\%06d.png" `
  --output-dir "D:\output\matte_run" `
  --frame-start 0 `
  --frame-end 100 `
  --anchor-mask "D:\output\anchor_mask.png" `
  --device cuda `
  --precision fp16
```

Anchor mask format:

- grayscale image preferred
- `uint8` or `uint16` supported
- white = foreground, black = background
- resized internally to match frame resolution if needed

## Output Files

Each product run writes (minimum):

- `alpha\000000.png`, `alpha\000001.png`, ... (default `png16`)
- `qc\trimap.000000.png`, `qc\trimap.000001.png`, ... (QC trimap preview, `uint8` PNG)
- `config_used.json` (exact resolved config used for the run)
- `run_summary.json` (compact metadata summary)
- `anchor_mask.auto.png` (when auto-anchor is used)

`run_summary.json` includes (among other fields):

- `requested_frame_start`
- effective `frame_start` after auto-anchor probe adjustment
- `anchor_mask` path
- auto-anchor metadata (`method`, `probe_frame`)
- MEMatte paths used for the run
- `allow_external_paths` flag

## Common Commands

### Full refinement, explicit auto-anchor flag (optional for videos)

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\4625475-hd_1920_1080_24fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\cli_refine_4625475" `
  --frame-start 0 `
  --frame-end 9 `
  --auto-anchor `
  --device cuda `
  --precision fp16
```

### Use a config file (JSON/YAML)

```powershell
videomatte-hq-v2 --config ".\my_config.json"
```

Config files can use flat keys or grouped nested sections (`io`, `segmentation`, `refinement`, `runtime`).

### Override trimap thresholds (CLI) when MEMatte unknown band is empty

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\4625475-hd_1920_1080_24fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\retuned_run" `
  --frame-start 0 `
  --frame-end 30 `
  --trimap-fg-threshold 0.97 `
  --trimap-bg-threshold 0.03 `
  --device cuda `
  --precision fp16
```

### Use external MEMatte assets (power-user override)

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\7219039-uhd_3840_2160_25fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\external_mematte_run" `
  --allow-external-paths `
  --mematte-repo-dir "D:\SomewhereElse\MEMatte" `
  --mematte-checkpoint "D:\SomewhereElse\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth"
```

## Performance Notes

- GPU (`cuda`) is strongly recommended
- `fp16` is the default precision hint on CUDA
- Stage 1 uses chunking and may use a native Ultralytics video fast path for video sources
- Stage 2 can skip MEMatte refinement on nearly identical consecutive masks (`skip_iou_threshold`)
- 4K/8K runs are supported but can still be slow at max-quality SAM defaults (`sam2_l.pt`)
- Wider trimap unknown bands generally improve robustness but increase MEMatte compute

## Troubleshooting

### `anchor_mask is required for v2 pipeline runs`

Cause:

- You are running on an image sequence without `--anchor-mask`
- Or you disabled auto-anchor for a video (`--no-auto-anchor`)

Fix:

- Provide `--anchor-mask`
- Or allow/enable auto-anchor for video inputs

### Auto-anchor shifts `frame_start`

Cause:

- The requested starting frame was too dark/black

Behavior:

- CLI probes forward (up to a small number of frames), chooses a usable frame for anchor generation, and may set an effective `frame_start`

Where to confirm:

- logs
- `run_summary.json`

### `MEMatte refinement is mandatory for this tool`

Cause:

- `--no-refine` was passed or `refine_enabled=false` was set in config/UI payload

Fix:

- Remove `--no-refine`
- Keep MEMatte refinement enabled (required in this build)

### `MEMatte did not execute on any tiles (trimap unknown band was empty ...)`

Cause:

- The trimap thresholds produced no gray unknown region for MEMatte to refine
- Common with small/distant subjects or over-confident Stage 1 masks

Fix:

- Widen the trimap unknown band:
  - Web UI: increase `Trimap Refine Band` to `Wider` or `Maximum`
  - CLI: raise `--trimap-fg-threshold` and lower `--trimap-bg-threshold` (for example `0.97 / 0.03`)
- Inspect `QC` trimap preview (gray = unknown region MEMatte can refine)
- Re-check anchor quality / try a shorter frame range first

### `External MEMatte paths are not allowed for this tool`

Cause:

- You passed `--mematte-repo-dir` or `--mematte-checkpoint` outside this repo without enabling override

Fix:

- Use the local copies under `D:\Videomatte-HQ2\third_party\MEMatte`, or
- Add `--allow-external-paths` (CLI) / enable `Allow External MEMatte Paths` (web UI)

### `MEMatte repo dir not found` / `MEMatte checkpoint not found`

Cause:

- MEMatte path is wrong or assets are missing

Fix:

- Verify the repo/checkpoint paths exist
- If using the self-contained setup, ensure:
  - `third_party/MEMatte`
  - `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth`

### `Ultralytics is required for segment_backend='ultralytics_sam3'`

Fix:

```powershell
pip install ultralytics
```

### CUDA not used even though `--device cuda` was passed

Possible causes:

- PyTorch CUDA build not installed
- GPU/driver issue

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Validation Notes (Current Repo State)

The current repo has been validated with:

- refined CLI runs (auto-anchor + MEMatte)
- black-opening portrait clip handling (auto-anchor probe + effective start shift)
- 4K and 8K smoke/perf runs
- strict MEMatte-only refinement enforcement
- QC trimap preview generation + web QC preview support
- local web UI (Run/Jobs/QC) with progress parsing, browse UX, and failure guidance

## Developer / Testing Utilities

This repo also includes `tools/run_test_clip.py` for smoke tests and repeated run benchmarking.

That tool is useful for rapid iteration, but the product-facing paths are:

- `videomatte-hq-v2` / `python -m videomatte_hq.cli`
- `videomatte-hq-web` / `run_web.bat` (local UI)
