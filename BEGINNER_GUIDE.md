# Beginner Guide

Step-by-step setup and first run on Windows. Covers both the CLI and web UI.

## Prerequisites

| Requirement | Check | Install |
|-------------|-------|---------|
| Python 3.10+ | `python --version` | [python.org/downloads](https://www.python.org/downloads/) — check "Add to PATH" |
| NVIDIA GPU + driver | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) |
| Node.js 18+ (web UI only) | `node --version` | [nodejs.org](https://nodejs.org/) |
| Git | `git --version` | [git-scm.com](https://git-scm.com/) |

## 1. Clone and Install

```powershell
cd D:\
git clone <repo-url> Videomatte-HQ2
cd Videomatte-HQ2
```

Run the automated installer:

```powershell
install.bat
```

It will:
1. Ask which CUDA version to use (or CPU-only)
2. Create a Python virtual environment (`.venv`)
3. Install PyTorch with GPU support
4. Install all project dependencies
5. Build the web frontend
6. Download the SAM2 model (~450 MB)
7. Check for MEMatte assets
8. Run the test suite

### Manual install (if you prefer)

```powershell
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA 12.4 (adjust for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project + web extras
pip install -e ".[web,dev]"
pip install einops timm

# Web frontend
cd web && npm install && npm run build && cd ..
```

## 2. Set Up MEMatte

The pipeline needs the MEMatte model for edge refinement. Two files are required:

| Asset | Expected path |
|-------|---------------|
| MEMatte source code | `third_party/MEMatte/` (must contain `inference.py`) |
| MEMatte checkpoint | `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth` |

### Download MEMatte

```powershell
# Clone the MEMatte repo (if not already present)
git clone https://github.com/AcademicFuworker/MEMatte third_party/MEMatte

# Download the checkpoint (~112 MB)
# Get MEMatte_ViTS_DIM.pth from the MEMatte GitHub releases page
# and place it at: third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth
```

Verify:

```powershell
Test-Path third_party\MEMatte\inference.py        # should be True
Test-Path third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth  # should be True
```

### Alternative: external MEMatte

If your MEMatte repo/checkpoint are stored outside this project, add `--allow-external-paths` (CLI) or check "Allow External MEMatte Paths" (web UI).

## 3. First Run (Web UI)

The web UI is the easiest way to get started.

```powershell
run_web.bat
```

This starts two servers and prints their URLs:
- **Backend** — `http://127.0.0.1:8000`
- **Frontend** — `http://127.0.0.1:5173` (or next available port)

Open the frontend URL in your browser.

### Workflow

1. **Run tab** — Browse to your input video, set a short frame range (e.g. 0–29)
2. Choose subject selection:
   - **Auto-Anchor** — click "Auto-Anchor Preview" to detect the person automatically
   - **Point Picker** — left-click on the subject (foreground), right-click on background
3. Click **Preflight** to validate everything
4. Click **Start Job** to run the pipeline
5. **Jobs tab** — watch progress and logs
6. **QC tab** — inspect alpha mattes and trimap overlays
7. **Help tab** — tuning reference, workflow guide, common issues

### Tips

- Start with a short range (0–29) before processing the full video
- The UI auto-clamps `frame_end` to the video's actual last frame
- SAM-only preview mode (uncheck "MEMatte Refine") gives fast results without soft edges
- Point picker: 2–5 foreground points + 1–2 background points usually works well

## 4. First Run (CLI)

```powershell
.venv\Scripts\activate

videomatte-hq-v2 ^
  --input "D:\Videomatte-HQ2\TestFiles\my_clip.mp4" ^
  --output-dir output_cli\first_run ^
  --frame-start 0 --frame-end 29 ^
  --device cuda --precision fp16
```

Check the output:

```powershell
dir output_cli\first_run\alpha\       # alpha matte frames
dir output_cli\first_run\qc\          # trimap visualization
type output_cli\first_run\run_summary.json
```

## 5. Tuning MEMatte for Better Edges

The default settings work well for most content. For difficult cases (fine hair, fur, translucent fabric), adjust the trimap parameters.

### How the trimap works

```
SAM binary mask edge
|<-- erosion (inward) -->|<-- unknown band -->|<-- dilation (outward) -->|
|                        |                    |                          |
|   Definite Foreground  |  MEMatte refines   |   Definite Background    |
|   (alpha = 1.0)        |  alpha values here |   (alpha = 0.0)          |
```

### Recommended adjustments

| Goal | Erosion | Dilation | Tile size |
|------|---------|----------|-----------|
| Default (balanced) | 20 | 10 | 1536 |
| Better hair detail | 10–12 | 20–25 | 2048 |
| Very fine strands | 8 | 25–30 | 2048 |
| Reduce VRAM usage | 20 | 10 | 1024 |

In the web UI, these settings are on the Run page under "Trimap Generation".

### CLI example with tuned settings

```powershell
videomatte-hq-v2 ^
  --input video.mp4 --output-dir output ^
  --trimap-erosion-px 12 --trimap-dilation-px 20 ^
  --tile-size 2048 --tile-overlap 128 ^
  --device cuda --precision fp16
```

## 6. Common Issues

### "Frame out of range" crash

`frame_end` is the last frame **index** (inclusive), not the frame count. A 30-frame video has indices 0–29, so use `--frame-end 29`. The web UI auto-corrects this.

### "MEMatte did not execute on any tiles"

The unknown band is empty — MEMatte has nothing to refine. Fix:
- Use `morphological` trimap mode (the default)
- Increase erosion and/or dilation to widen the unknown band

### Auto-anchor picks the wrong person

Use the Point Picker instead, or provide a manual `--anchor-mask` (white = subject, black = background).

### Black first frame shifts frame_start

Videos that start with black frames trigger the auto-anchor probe, which skips forward to a usable frame. Check `run_summary.json` for the effective `frame_start`.

### Slow processing

- Use `--device cuda` (CPU is 10-50x slower)
- Use `--precision fp16` (default, faster than fp32)
- Reduce `--tile-size` to 1024 (less VRAM, faster tiles)
- Process a short range first, extend once you're happy with quality

## 7. Full CLI Reference

```
videomatte-hq-v2 --help
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Video file or image sequence pattern |
| `--output-dir` | `output` | Output directory |
| `--frame-start` | `0` | First frame index |
| `--frame-end` | `-1` (all) | Last frame index (inclusive) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--precision` | `fp16` | `fp16` or `fp32` |
| `--anchor-mask` | (auto) | Manual anchor mask path |
| `--auto-anchor` | (default for video) | Force auto-anchor on |
| `--no-auto-anchor` | — | Disable auto-anchor |
| `--prompt-mode` | `mask` | `mask` or `points` |
| `--trimap-mode` | `morphological` | `morphological` or `logit` |
| `--trimap-erosion-px` | `20` | Erosion into FG (px) |
| `--trimap-dilation-px` | `10` | Dilation into BG (px) |
| `--tile-size` | `1536` | MEMatte tile size (px) |
| `--tile-overlap` | `96` | Tile overlap (px) |
| `--no-refine` | — | SAM-only output (no MEMatte) |
| `--config` | — | Load settings from JSON/YAML file |
| `--verbose` | — | Debug logging |
| `--allow-external-paths` | — | Allow MEMatte paths outside repo |

## 8. Using a Config File

Save settings to a JSON or YAML file instead of passing flags:

```json
{
  "input": "D:\\clips\\interview.mp4",
  "output_dir": "output",
  "frame_start": 0,
  "frame_end": 149,
  "device": "cuda",
  "precision": "fp16",
  "trimap_erosion_px": 12,
  "trimap_dilation_px": 20,
  "tile_size": 2048
}
```

```powershell
videomatte-hq-v2 --config my_settings.json
```
