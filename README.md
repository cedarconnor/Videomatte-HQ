# Videomatte-HQ v2

High-quality video matting pipeline that extracts clean alpha mattes from video. Two-stage architecture: **SAM2** segments and tracks the subject, then **MEMatte** refines edges to capture hair, fur, and translucent detail.

Includes a **CLI** for scripted/batch workflows and a **local web UI** for interactive use.

## How It Works

```
Input Video
    |
    v
[Stage 1] SAM2 Segmentation + Tracking
    |  Auto-anchor or point-picker identifies the subject
    |  Chunked processing for temporal consistency
    |
    v
[Trimap] Morphological erosion/dilation
    |  Creates "definite FG / unknown / definite BG" zones
    |
    v
[Stage 2] MEMatte Tiled Refinement
    |  Processes unknown band at full resolution
    |  Hann-window tile blending (supports 4K-8K)
    |
    v
Alpha Matte Output (16-bit PNG per frame)
```

## Quick Start

### Install

```powershell
git clone <repo-url> D:\Videomatte-HQ2
cd D:\Videomatte-HQ2
install.bat
```

The installer handles Python venv, PyTorch with CUDA, all dependencies, the web frontend, and model downloads. See [Beginner Guide](BEGINNER_GUIDE.md) for detailed setup instructions.

### Web UI

```powershell
run_web.bat
```

Opens a local browser UI with:
- **Run** tab — configure input, frame range, subject selection (auto-anchor or point picker), trimap/MEMatte settings
- **Jobs** tab — monitor running jobs, view logs, progress
- **QC** tab — inspect alpha output, trimap overlays, wipe comparison
- **Help** tab — workflow guide, MEMatte tuning reference, troubleshooting

### CLI

```powershell
# Activate the environment
.venv\Scripts\activate

# Auto-anchor, short test range
videomatte-hq-v2 ^
  --input "D:\clips\interview.mp4" ^
  --output-dir output ^
  --frame-start 0 --frame-end 29 ^
  --device cuda --precision fp16

# Point prompts (SAM3 propagates from frame 0)
videomatte-hq-v2 ^
  --input "D:\clips\dancer.mp4" ^
  --output-dir output ^
  --prompt-mode points ^
  --point-prompts-json "{\"0\":{\"positive\":[[0.5,0.4]],\"negative\":[]}}"

# Image sequence with manual anchor mask
videomatte-hq-v2 ^
  --input "D:\frames\%06d.png" ^
  --output-dir output ^
  --anchor-mask anchor.png ^
  --frame-start 0 --frame-end 100
```

## Output

Each run produces:

```
output/
  alpha/000000.png ...    16-bit alpha mattes
  qc/trimap.000000.png    Trimap visualization (gray = unknown band)
  config_used.json        Exact settings used
  run_summary.json        Run metadata and timing
  anchor_mask.auto.png    Auto-generated anchor (if applicable)
```

## Subject Selection Methods

| Method | Best for | How |
|--------|----------|-----|
| **Auto-Anchor** | Single person, video input | YOLO detects person on frame 0, SAM2 tracks across all frames |
| **Point Picker** | Any subject, precise control | Left-click foreground, right-click background on frame 0. SAM2 propagates. |
| **Manual Anchor** | Image sequences, multi-subject | Provide a black/white mask PNG marking the subject |

## Key Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `trimap_mode` | `morphological` | How the unknown band is generated |
| `trimap_erosion_px` | `20` | Shrinks FG boundary inward (px) |
| `trimap_dilation_px` | `10` | Expands unknown outward (px) |
| `tile_size` | `1536` | MEMatte tile size (larger = more VRAM) |
| `tile_overlap` | `96` | Tile overlap for seam-free blending |
| `sam3_model` | `sam2_l.pt` | SAM checkpoint (auto-downloaded) |
| `device` | `cuda` | `cuda` or `cpu` |
| `precision` | `fp16` | `fp16` (faster) or `fp32` |

For hair and fine detail:
- **Decrease** erosion to 10-12 (keeps more definite FG)
- **Increase** dilation to 20-25 (catches wispy strands)
- **Increase** tile_size to 2048 if VRAM allows

See the **Help** tab in the web UI for a full tuning reference.

## Requirements

- **Python** 3.10+
- **NVIDIA GPU** recommended (8+ GB VRAM). CPU works but is very slow.
- **Node.js** 18+ (only for web UI frontend dev mode)
- **MEMatte** repo + checkpoint under `third_party/MEMatte/` (see [Beginner Guide](BEGINNER_GUIDE.md))

## Project Structure

```
src/
  videomatte_hq/           CLI pipeline
    cli.py                  Entry point
    config.py               Configuration dataclass
    pipeline/               Orchestrator + stage implementations
    models/                 MEMatte wrapper (detectron2-free)
    prompts/                Auto-anchor, point picker, mask adapter
    tiling/                 Tile planning + Hann-window stitching
    io/                     Video/sequence reader + async alpha writer
  videomatte_hq_web/        FastAPI backend for web UI
web/                        React + Vite + TypeScript frontend
third_party/MEMatte/        MEMatte source + checkpoints
tests/                      Unit + integration tests
tools/                      Dev/debug utilities
```

## Troubleshooting

**"Hair looks hard-edged"** — Increase trimap dilation (20-25px), decrease erosion (10-12px). Switch to `morphological` trimap mode if using `logit`.

**"Frame out of range"** — `frame_end` is an inclusive index, not a count. A 30-frame video has valid indices 0-29. The UI auto-clamps this.

**"MEMatte did not execute on any tiles"** — The unknown band is empty. Increase erosion/dilation or switch to `morphological` trimap mode.

**CUDA not detected** — Verify with `python -c "import torch; print(torch.cuda.is_available())"`. Reinstall PyTorch with the correct CUDA version via `install.bat`.

**MEMatte checkpoint not found** — Ensure `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth` exists. See [Beginner Guide](BEGINNER_GUIDE.md) for download instructions.

## License

Research use. See individual model licenses for SAM2 (Meta), MEMatte, and Ultralytics YOLO.
