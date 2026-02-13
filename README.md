# VideoMatte-HQ

**High-quality offline 8K people video matting** — four-pass pipeline producing temporally stable alpha mattes at native resolution.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License MIT](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Four-pass architecture** — stable backbone → 4K refinement → 8K edge tiles → temporal stabilization
- **Temporal consistency** — frequency-separation stabilization keeps edges stable without corrupting the silhouette
- **Smart tiling** — boundary-only tiles with VRAM-aware sizing (auto backoff 2048 → 1536 → 1024)
- **Adaptive band** — three-signal edge detection with directional alignment, area cap, and auto-tighten
- **Background plate estimation** — temporal median with confidence map and photometric normalization
- **Live preview** — 2×2 mosaic QC (checkerboard / alpha / white / flicker heatmap)
- **Pluggable models** — swap backbone (RVM), refiner (ViTMatte), and flow (RAFT) independently
- **Resume support** — config-hash-aware stage caching, picks up where it left off

## Pipeline

```
Stage 0    → Load frames (PNG/EXR/video)
Stage 0.5  → Background plate estimation (locked-off shots)
Stage 1    → ROI tracking (person detection + motion mask)
Stage 2    → Pass A  — Global matte backbone (RVM @ 2K, temporal chunks)
Stage 2.5  → Pass A′ — Intermediate refinement (ViTMatte @ 4K, guided-filter delta)
Stage 3    → Adaptive band + distance-transform trimap + tile planning
Stage 4    → Pass B  — Edge refinement (ViTMatte tiles @ native res, logit stitching)
Stage 5    → Pass C  — Temporal stabilization (RAFT flow, structural/detail split)
Stage 6    → Post-processing (despill, foreground extraction)
Stage 7    → Write outputs + QC report
```

## Requirements

- **Python** 3.10+
- **PyTorch** 2.0+ with CUDA
- **GPU** with 8GB+ VRAM (24–48GB recommended for 8K)
- **FFmpeg** (for preview video output)

## Installation

```bash
# Clone
git clone https://github.com/your-org/videomatte-hq.git
cd videomatte-hq

# Install (editable, with dev tools)
pip install -e ".[dev]"

# Or use the launcher
run_videomatte.bat
```

## Quick Start

### Using the batch launcher

Double-click **`run_videomatte.bat`** or run from terminal:

```bash
run_videomatte.bat
```

This will process the included test video and output alpha mattes to `out/alpha/`.

### Command line

```bash
# Process a video file
videomatte-hq --in video.mp4 --out out/alpha/%06d.png

# Process an image sequence
videomatte-hq --in frames/%06d.exr --out out/alpha/%06d.png --fps 24

# EXR output (lossless)
videomatte-hq --in video.mp4 --out out/alpha/%06d.exr --alpha-format exr_lossless

# With config file
videomatte-hq --in video.mp4 --config my_config.yaml

# Handheld shot (disables BG estimation)
videomatte-hq --in video.mp4 --out out/alpha/%06d.png --shot-type handheld

# Custom settings
videomatte-hq \
  --in video.mp4 \
  --out out/alpha/%06d.png \
  --global-long-side 2048 \
  --tile-size 1536 \
  --temporal frequency_separation \
  --temporal-detail-strength 0.7 \
  --preview \
  --preview-modes checker,alpha,white,flicker
```

### Dump resolved config

```bash
videomatte-hq --in video.mp4 --dump-config
```

## Configuration

All settings can be specified via YAML config file or CLI flags. CLI flags override config file values.

<details>
<summary><strong>Full YAML config reference</strong></summary>

```yaml
io:
  input: "frames/%06d.png"
  output_alpha: "out/alpha/%06d.png"
  output_fg: null
  fps: 30
  shot_type: "locked_off"        # locked_off | handheld | unknown
  alpha_format: "png16"          # png16 | exr_dwaa | exr_lossless | exr_raw
  alpha_dwaa_quality: 45.0

background:
  enabled: true
  sample_count: 60
  variance_threshold: 0.05
  photometric_normalize: true

roi:
  mode: "auto_person_track"
  detect_every: 15
  pad_ratio: 0.25
  context_px: 256

global:
  model: "rvm"
  long_side: 2048
  chunk_len: 24
  chunk_overlap: 6

intermediate:
  enabled: true
  long_side: 4096
  model: "vitmatte"
  guide_filter_radius: 8

band:
  mode: "adaptive"
  band_max_coverage: 0.35
  auto_tighten: true
  hair_aware: true

trimap:
  method: "distance_transform"
  unknown_width: 32

tiles:
  tile_size: 2048               # auto backoff: 2048 → 1536 → 1024
  overlap: 384
  vram_headroom: 0.85

refine:
  model: "vitmatte"

temporal:
  method: "frequency_separation"
  structural_blend_strength: 0.3
  detail_blend_strength: 0.7
  flow_model: "raft"

postprocess:
  despill:
    enabled: true
    strength: 1.0

preview:
  enabled: true
  scale: 1080
  every: 10
  modes: ["checker", "alpha", "white", "flicker"]

runtime:
  device: "cuda"
  precision: "fp16"
  workers_io: 4
  resume: true
```

</details>

## Output

| Output | Path | Description |
|---|---|---|
| Alpha matte | `out/alpha/%06d.png` | 16-bit PNG (default) or EXR |
| Foreground | `out/fg/%06d.png` | Optional premultiplied RGB |
| Preview | `out/preview/live_preview.mp4` | 2×2 QC mosaic |
| QC report | `out/qc/report.html` | Metrics, problem frames, stats |
| ROI track | `out/qc/roi.json` | Per-frame bounding boxes |

## Models

| Pass | Model | Role | License |
|---|---|---|---|
| A (backbone) | **RVM** (RobustVideoMatting) | Temporal matte @ 2K | Apache 2.0 |
| A′ + B (refiner) | **ViTMatte** | Detail matte @ 4K/8K | MIT |
| C (flow) | **RAFT** | Optical flow for stabilization | BSD-3 |
| ROI | **Faster R-CNN** | Person detection | BSD-3 |

Model weights are downloaded automatically on first run.

## Project Structure

```
videomatte-hq/
├── pyproject.toml
├── run_videomatte.bat
├── README.md
├── TestFiles/
│   └── 6138680-uhd_3840_2160_24fps.mp4
└── src/videomatte_hq/
    ├── cli.py                  # CLI entry point
    ├── config.py               # YAML config schema
    ├── safe_math.py            # Logit/sigmoid utilities
    ├── io/                     # Frame read/write, colorspace, EXR compression
    ├── background/             # BG plate, confidence, photometric normalization
    ├── roi/                    # Person detection, tracking, motion mask
    ├── models/                 # RVM, ViTMatte, RAFT wrappers
    ├── intermediate/           # Pass A′ + guided filter
    ├── band/                   # Adaptive band, trimap, feather
    ├── tiling/                 # Tile planning, VRAM probe, stitching
    ├── temporal/               # Flow, frequency separation, stabilization
    ├── postprocess/            # Despill, foreground extraction
    ├── preview/                # Checkerboard, mosaic compositor
    ├── qc/                     # Metrics, problem detector, report
    ├── reference/              # Reference frame selection & propagation
    └── pipeline/               # Orchestrator, Pass A, Pass B execution
```

## License

MIT
