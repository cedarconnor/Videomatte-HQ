# Videomatte-HQ v2

High-quality video matting pipeline for extracting production-grade alpha mattes from video at resolutions up to 8K. Two pipeline modes target different workflows:

- **v1** — SAM2 segmentation on every frame + MEMatte refinement on every frame. Best for short clips or when per-frame SAM tracking is needed.
- **v2** — MatAnyone2 temporal matting + MEMatte only for content above 1080p. Faster, more temporally stable, and the recommended default for most work.

Includes a **CLI** for scripted/batch workflows and a **local web UI** for interactive use.

## Pipeline Architecture

### v2 Pipeline (recommended)

```
Input Video (any resolution, any length)
    |
    v
[Stage 0] First-Frame Mask (SAM2, single frame only)
    |  Auto-anchor, point picker, or manual mask
    |  SAM2 unloaded after this stage
    |
    v
[Stage 1] MatAnyone2 Temporal Matting
    |  Memory-based propagation from first-frame mask
    |  Processes at <= 1080p for efficiency
    |  Produces temporally consistent alpha mattes
    |
    v
[Resolution Check] min(height, width) > 1080?
    |
    +-- NO  --> Bicubic upscale to native res --> Post-process --> Output
    |
    +-- YES --> [Stage 2] Hi-Res Refinement
                  |  Upscale MatAnyone2 alpha to native resolution
                  |  Gradient-adaptive trimap (wider band around complex edges)
                  |  MEMatte tiled refinement at native resolution
                  |  Hann-window tile blending (supports 4K-8K)
                  |
                  v
              Post-process --> Output
```

### v1 Pipeline (legacy)

```
Input Video
    |
    v
[Stage 1] SAM2 Segmentation + Tracking (every frame)
    |  Chunked processing with overlap blending
    v
[Trimap] Morphological erosion/dilation
    v
[Stage 2] MEMatte Tiled Refinement (every frame)
    v
Alpha Matte Output
```

## Quick Start

### Install

```powershell
git clone <repo-url> D:\Videomatte-HQ2
cd D:\Videomatte-HQ2
install.bat
```

The installer creates a Python venv, installs PyTorch with CUDA, clones and sets up third-party models (MEMatte + MatAnyone2), downloads all model checkpoints, builds the web frontend, and runs the test suite.

See [INSTALL.md](INSTALL.md) for detailed manual setup instructions and troubleshooting.

### Web UI

```powershell
run_web.bat
```

The web UI provides:
- **Run** tab — pipeline mode toggle (v1/v2), input/output config, subject selection (auto-anchor or point picker), MatAnyone2 and MEMatte settings
- **Jobs** tab — monitor running jobs, view logs, progress
- **QC** tab — inspect alpha output, trimap overlays, wipe comparison, zoom for fine detail
- **Help** tab — workflow guide, tuning reference, troubleshooting

### CLI

```powershell
# Activate the environment
.venv\Scripts\activate

# v2 pipeline (recommended) — auto-anchor, 4K input
videomatte-hq-v2 ^
  --input "D:\clips\interview_4k.mp4" ^
  --output-dir output ^
  --pipeline-mode v2 ^
  --device cuda --precision fp16

# v2 with custom frame range
videomatte-hq-v2 ^
  --input "D:\clips\dancer.mp4" ^
  --output-dir output ^
  --pipeline-mode v2 ^
  --frame-start 0 --frame-end 100

# v1 pipeline (SAM2 every frame + MEMatte every frame)
videomatte-hq-v2 ^
  --input "D:\clips\interview.mp4" ^
  --output-dir output ^
  --pipeline-mode v1 ^
  --frame-start 0 --frame-end 29

# Point prompts (SAM2 propagates from frame 0)
videomatte-hq-v2 ^
  --input "D:\clips\dancer.mp4" ^
  --output-dir output ^
  --prompt-mode points ^
  --point-prompts-json "{\"0\":{\"positive\":[[0.5,0.4]],\"negative\":[]}}"
```

## Output

Each run produces:

```
output/
  alpha/000000.png ...    16-bit alpha mattes at native resolution
  alpha_preview.mp4       H.264 preview video
  qc/trimap.000000.png    Trimap visualization (white=FG, gray=unknown, black=BG)
  config_used.json        Exact settings used
  run_summary.json        Run metadata
  anchor_mask.auto.png    Auto-generated anchor mask (if applicable)
```

## Subject Selection Methods

| Method | Best for | How |
|--------|----------|-----|
| **Auto-Anchor** | Single person, video input | YOLO detects person on frame 0, used as first-frame mask |
| **Point Picker** | Any subject, precise control | Left-click foreground, right-click background on frame 0 |
| **Manual Anchor** | Image sequences, multi-subject | Provide a black/white mask PNG marking the subject |

## Key Configuration

### v2 Pipeline Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `pipeline_mode` | `v1` | `v1` (SAM2+MEMatte every frame) or `v2` (MatAnyone2) |
| `matanyone2_max_size` | `1080` | MatAnyone2 processing resolution (short edge) |
| `matanyone2_warmup` | `10` | Warmup iterations on first frame |
| `matanyone2_hires_threshold` | `1080` | Resolution above which MEMatte refinement activates |
| `gradient_trimap_base_kernel` | `7` | Base unknown band width (px, scales with resolution) |
| `gradient_trimap_max_extra` | `20` | Max additional width in high-gradient regions |

### Common Settings

| Setting | Default | Purpose |
|---------|---------|---------|
| `trimap_erosion_px` | `20` | (v1) Shrinks FG boundary inward |
| `trimap_dilation_px` | `10` | (v1) Expands unknown outward |
| `tile_size` | `1536` | MEMatte tile size (larger = more VRAM) |
| `mask_temporal_smooth_radius` | `2` | Temporal median on masks (0=off, 1=3-frame, 2=5-frame) |
| `temporal_smooth_enabled` | `true` | Post-process alpha EMA smoothing |
| `temporal_smooth_strength` | `0.55` | EMA strength (0-1) |
| `device` | `cuda` | `cuda` or `cpu` |
| `precision` | `fp16` | `fp16` (faster) or `fp32` |

## Performance

Tested on NVIDIA A6000 (48 GB VRAM):

| Resolution | Pipeline | MatAnyone2 | MEMatte Refine | Total (per frame) |
|------------|----------|------------|----------------|-------------------|
| 1080p | v2 | 0.7 s/frame | skipped | ~0.7 s |
| 4K | v2 | 0.35 s/frame | 1.7 s/frame | ~2.1 s |
| 8K | v2 | 0.7 s/frame | 8 s/frame | ~8.7 s |
| 1080p | v1 | n/a | 0.3 s/frame | ~0.7 s (SAM + MEMatte) |

VRAM usage: MatAnyone2 and MEMatte run sequentially (not concurrently). Peak is during MEMatte at 8K (~15 GB).

## Requirements

- **Python** 3.10+
- **NVIDIA GPU** recommended (8+ GB VRAM for 1080p, 24+ GB for 4K, 48 GB for 8K)
- **Node.js** 18+ (for web UI frontend)
- Third-party models installed by `install.bat`:
  - **MEMatte** — edge-aware alpha refinement
  - **MatAnyone2** — temporal video matting (v2 pipeline)
  - **SAM2** — first-frame segmentation (auto-downloaded by Ultralytics)

## Project Structure

```
src/
  videomatte_hq/               CLI pipeline
    cli.py                      Entry point
    config.py                   Configuration dataclass
    pipeline/
      orchestrator.py            v1/v2 dispatcher
      stage_segment.py           SAM2 segmentation (v1)
      stage_matanyone2.py        MatAnyone2 temporal matting (v2)
      stage_refine.py            MEMatte tiled refinement
      stage_trimap.py            Trimap generation (morphological + gradient-adaptive)
    models/
      edge_mematte.py            MEMatte wrapper (detectron2-free)
      matanyone2_wrapper.py      MatAnyone2 wrapper
    utils/
      vram.py                    VRAM management for sequential model stages
    prompts/                     Auto-anchor, point picker, mask adapter
    tiling/                      Tile planning + Hann-window stitching
    io/                          Video/sequence reader + async alpha writer
  videomatte_hq_web/             FastAPI backend for web UI
web/                             React + Vite + TypeScript frontend
third_party/
  MEMatte/                       MEMatte source + checkpoints
  MatAnyone2/                    MatAnyone2 source + pretrained_models
tests/                           Unit + integration tests
```

## Troubleshooting

**Alpha flicker between frames** — Two-layer fix is on by default: (1) `mask_temporal_smooth_radius=2` applies a 5-frame median to masks before trimap. (2) `temporal_smooth_enabled=true` applies motion-adaptive post-process alpha smoothing. If motion looks too sticky, reduce `temporal_smooth_strength`.

**"Hair looks hard-edged"** — In v1: increase trimap dilation (20-25px), decrease erosion (10-12px). In v2: increase `gradient_trimap_max_extra` (30-40) for a wider unknown band around complex edges.

**Background leaking into matte** — Check the auto-anchor mask in the output directory. If it includes background objects, use point picker or manual mask mode instead.

**CUDA out of memory** — Reduce `tile_size` (try 1024), or reduce `matanyone2_max_size` (try 720). At 8K, MEMatte requires ~15 GB VRAM.

**CUDA not detected** — Verify with `python -c "import torch; print(torch.cuda.is_available())"`. Reinstall PyTorch with the correct CUDA version via `install.bat`.

**MatAnyone2 not found** — Run `install.bat` which clones the repo and downloads the checkpoint. Or manually: `git clone https://github.com/pq-yang/MatAnyone2 third_party/MatAnyone2`

**MEMatte checkpoint not found** — Ensure `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth` exists. See [INSTALL.md](INSTALL.md) for download instructions.

## License

Research use. See individual model licenses for SAM2 (Meta), MEMatte, MatAnyone2, and Ultralytics YOLO.
