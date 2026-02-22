# Videomatte-HQ v2 (CLI)

High-quality video matting pipeline with a simplified two-stage architecture:

1. Stage 1: Ultralytics SAM-based subject segmentation/tracking
2. Stage 2: MEMatte edge refinement (tiled, high-resolution)

This repo is configured as a CLI-first tool.

The CLI is intentionally constrained to MEMatte paths inside this repo (not external repos/paths).
Expected local runtime paths:

- `third_party/MEMatte`
- `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth`

Notes:
- Large weights/checkpoints may be excluded from GitHub pushes due file-size limits.
- If a checkpoint is missing in your clone, place it back under the expected local path.
- The CLI rejects MEMatte paths outside this repo on purpose.

## Status

- Full SAM -> trimap -> MEMatte refinement path is implemented (not a stub)
- Product CLI supports:
  - video input
  - image sequence input
  - auto-anchor generation for video inputs
  - manual anchor masks
  - no-refine preview mode
  - refined output mode (MEMatte)
- Console script is available after install: `videomatte-hq-v2`

## Repo Layout (important parts)

- `src/videomatte_hq/cli.py`: product CLI entry point
- `src/videomatte_hq/config.py`: config model and defaults
- `src/videomatte_hq/pipeline/orchestrator.py`: Stage 1 -> Stage 2 pipeline wiring
- `src/videomatte_hq/pipeline/stage_segment.py`: SAM segmentation/tracking
- `src/videomatte_hq/pipeline/stage_trimap.py`: logit-based trimap generation
- `src/videomatte_hq/pipeline/stage_refine.py`: MEMatte tiled refinement and preview fallback
- `src/videomatte_hq/models/edge_mematte.py`: detectron2-free MEMatte wrapper
- `src/videomatte_hq/prompts/auto_anchor.py`: auto-anchor generation (video, person detection)
- `third_party/MEMatte`: local MEMatte source (checkpoint path expected under `checkpoints/`)

## Requirements

- Windows (tested in this repo workflow) or another OS supported by Python/PyTorch/OpenCV
- Python 3.10+
- GPU recommended for practical speed (`cuda`)

Core Python dependencies are declared in `pyproject.toml`.

Notes about MEMatte runtime dependencies:
- The wrapper includes compatibility shims for several packages (detectron2, timm, fairscale, fvcore).
- Some MEMatte code paths may still require `einops`.
- If refinement fails with import/runtime messages mentioning `einops`, install it:
  - `pip install einops`

## Install

### 1. Create/activate a virtual environment

PowerShell example:

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

If you prefer not to install the console script, you can run:

```powershell
python -m videomatte_hq.cli --help
```

## Quick Start (Video, Auto-Anchor, Full Refinement)

The CLI will auto-generate an anchor mask for video inputs if `--anchor-mask` is not provided.

Example:

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
- If the starting frame is black/dark, the CLI may probe forward and shift to an effective start frame
- Stage 1 segmentation runs
- Stage 2 MEMatte refinement runs (default behavior)
- Alpha frames are written to `<output-dir>\alpha\`

## Quick Start (No-Refine Preview)

Use this for faster debugging/iteration.

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\7219039-uhd_3840_2160_25fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\preview_run" `
  --frame-start 0 `
  --frame-end 20 `
  --no-refine `
  --device cuda `
  --precision fp16
```

Notes:

- This uses the Stage 2 preview fallback path (cleaned binary/preview alpha), not MEMatte
- Useful for validating prompting, tracking stability, and anchor quality before spending refinement time

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

Each run writes:

- `alpha\000000.png`, `alpha\000001.png`, ... (default `png16`)
- `config_used.json` (exact resolved config used for the run)
- `run_summary.json` (compact metadata summary)
- `anchor_mask.auto.png` (when auto-anchor is used)

`run_summary.json` includes:

- `requested_frame_start`
- effective `frame_start` after auto-anchor probe adjustment
- `anchor_mask` path
- auto-anchor metadata (`method`, `probe_frame`)
- normalized local MEMatte paths

## Important Runtime Constraint (Local MEMatte Only)

This tool is intentionally configured to be self-contained under `D:\Videomatte-HQ2`.

The CLI preflight will reject MEMatte repo/checkpoint paths outside this repository.

Examples of accepted paths:

- `third_party/MEMatte`
- `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth`
- absolute paths that still resolve inside `D:\Videomatte-HQ2\...`

Example of rejected path:

- `D:\Videomatte-HQ\third_party\MEMatte`

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

### Override a few fields on top of a config file

```powershell
videomatte-hq-v2 `
  --config ".\my_config.yaml" `
  --output-dir ".\output_cli\override_run" `
  --frame-end 60 `
  --no-refine
```

## Performance Notes

- GPU (`cuda`) is strongly recommended
- `fp16` is the default precision hint on CUDA
- Stage 1 uses chunking and may use a native Ultralytics video fast path for video sources
- Stage 2 can skip MEMatte refinement on nearly identical consecutive masks (`skip_iou_threshold`)
- 4K/8K runs are supported but can still be slow at max-quality SAM defaults (`sam2_l.pt`)

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

### `External MEMatte paths are not allowed for this tool`

Cause:
- You passed `--mematte-repo-dir` or `--mematte-checkpoint` pointing outside this repo

Fix:
- Use the local copies under `D:\Videomatte-HQ2\third_party\MEMatte`
- Or omit the flags and use defaults

### `MEMatte repo dir not found` / `MEMatte checkpoint not found`

Cause:
- `third_party/MEMatte` or the checkpoint file is missing

Fix:
- Ensure these exist inside this repo:
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

Checks:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Validation Notes (Current Repo State)

The current repo has been validated with:

- full refined CLI runs (auto-anchor + MEMatte)
- black-opening portrait clip handling (auto-anchor probe + effective start shift)
- 4K refined run with Stage 2 skip-frame reuse
- local-only MEMatte path enforcement

## Developer / Testing Utilities

This repo also includes `tools/run_test_clip.py` for smoke tests and repeated run benchmarking.

That tool is useful for rapid iteration, but the product-facing path is `videomatte-hq-v2` / `python -m videomatte_hq.cli`.
