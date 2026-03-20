# Videomatte-HQ v2 — Installation Guide

## Automated Install (Recommended)

The `install.bat` script handles everything:

```powershell
git clone <repo-url> D:\Videomatte-HQ2
cd D:\Videomatte-HQ2
install.bat
```

It will:
1. Verify Python 3.10+ is installed
2. Create a `.venv` virtual environment
3. Install PyTorch with CUDA (interactive prompt for CUDA version)
4. Install the project and all Python dependencies
5. Clone MEMatte and MatAnyone2 into `third_party/`
6. Download model checkpoints (~600 MB total)
7. Build the web frontend (requires Node.js)
8. Run the test suite to verify the installation

### Installer Options

```powershell
install.bat                 # Interactive — prompts for CUDA version
install.bat --cuda 12.4     # CUDA 12.4 (recommended for modern GPUs)
install.bat --cuda 12.1     # CUDA 12.1
install.bat --cuda 11.8     # CUDA 11.8 (older drivers)
install.bat --cpu            # CPU only (no GPU, very slow)
```

---

## Manual Install (Step by Step)

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | [python.org/downloads](https://www.python.org/downloads/) — check "Add to PATH" |
| Git | any | [git-scm.com](https://git-scm.com/) |
| NVIDIA GPU | 8+ GB VRAM | Required for practical use. CPU works but is extremely slow |
| NVIDIA Driver | 525+ | Must match your CUDA version |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) — only needed for web UI |
| ffmpeg | any | [ffmpeg.org](https://ffmpeg.org/) — for preview MP4 generation |

### Step 1: Clone the Repository

```powershell
git clone <repo-url> D:\Videomatte-HQ2
cd D:\Videomatte-HQ2
```

### Step 2: Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Step 3: Install PyTorch with CUDA

Pick the command matching your CUDA version. Check with `nvidia-smi` — the "CUDA Version" in the top-right is your maximum supported version.

```powershell
# CUDA 12.4 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA is working:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX ...
```

### Step 4: Install the Project

```powershell
pip install -e ".[web,dev]"
```

This installs:
- Core: numpy, opencv-python, imageio, pillow, pyyaml, ultralytics
- Web UI: fastapi, uvicorn
- Dev: pytest

### Step 5: Install Additional Dependencies

```powershell
pip install einops timm hydra-core omegaconf kornia safetensors
```

These are needed by the MatAnyone2 and MEMatte model wrappers.

### Step 6: Set Up MEMatte

```powershell
# Clone the repository
git clone https://github.com/AcademicFuworker/MEMatte third_party/MEMatte

# Download the checkpoint (~160 MB)
# Place it at: third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth
#
# Download from the MEMatte GitHub releases page or Google Drive link
# in the MEMatte README.
mkdir third_party\MEMatte\checkpoints
```

The checkpoint file must be at exactly:
```
third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth
```

Verify:
```powershell
dir third_party\MEMatte\inference.py
dir third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth
```

### Step 7: Set Up MatAnyone2 (Required for v2 Pipeline)

```powershell
# Clone the repository
git clone https://github.com/pq-yang/MatAnyone2 third_party/MatAnyone2

# Download the checkpoint (~135 MB) — automatic via their download utility
python -c "import sys; sys.path.insert(0,'third_party/MatAnyone2'); from hugging_face.tools.download_util import load_file_from_url; load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth', 'third_party/MatAnyone2/pretrained_models')"
```

The checkpoint file must be at exactly:
```
third_party/MatAnyone2/pretrained_models/matanyone2.pth
```

Verify:
```powershell
dir third_party\MatAnyone2\matanyone2\inference\inference_core.py
dir third_party\MatAnyone2\pretrained_models\matanyone2.pth
```

### Step 8: Download SAM2 Model

SAM2 is auto-downloaded by Ultralytics on first use, but you can pre-download:

```powershell
python -c "from ultralytics import SAM; SAM('sam2_l.pt')"
```

This downloads `sam2_l.pt` (~450 MB) to the current directory.

### Step 9: Build Web Frontend

```powershell
cd web
npm install
npm run build
cd ..
```

### Step 10: Verify Installation

```powershell
# Run tests
python -m pytest tests/ -x -q

# Verify CLI
videomatte-hq-v2 --help

# Quick smoke test (v2 pipeline, 5 frames)
videomatte-hq-v2 ^
  --input "TestFiles\4625475-hd_1920_1080_24fps.mp4" ^
  --output-dir output_test ^
  --pipeline-mode v2 ^
  --frame-start 0 --frame-end 4 ^
  --auto-anchor
```

---

## Directory Layout After Install

```
D:\Videomatte-HQ2\
  .venv\                          Python virtual environment
  src\videomatte_hq\              Pipeline source code
  src\videomatte_hq_web\          Web UI backend
  web\                            Web UI frontend (React/Vite)
  third_party\
    MEMatte\
      inference.py                MEMatte source
      checkpoints\
        MEMatte_ViTS_DIM.pth      MEMatte checkpoint (160 MB)
    MatAnyone2\
      matanyone2\                 MatAnyone2 source
      pretrained_models\
        matanyone2.pth            MatAnyone2 checkpoint (135 MB)
  sam2_l.pt                       SAM2-Large checkpoint (450 MB)
  tests\                          Test suite
  install.bat                     Automated installer
  run_web.bat                     Web UI launcher
```

---

## Updating

```powershell
cd D:\Videomatte-HQ2
git pull
.venv\Scripts\activate
pip install -e ".[web,dev]"
python -m pytest tests/ -x -q
```

---

## Uninstalling

Delete the entire directory. Everything is self-contained:

```powershell
rmdir /s /q D:\Videomatte-HQ2
```

---

## Troubleshooting Install Issues

### "Python is not recognized"

Install Python 3.10+ from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**.

### PyTorch CUDA mismatch

If `torch.cuda.is_available()` returns `False` after installing with a CUDA index:

1. Check your driver: `nvidia-smi` — note the CUDA version in the top-right
2. Reinstall PyTorch with the matching CUDA version
3. If your driver says CUDA 12.x, use `cu124`. If 11.x, use `cu118`

### "No module named 'hydra'"

Run: `pip install hydra-core omegaconf`

These are MatAnyone2 dependencies not listed in the project's core requirements.

### MEMatte "missing inference.py"

The MEMatte repo must be cloned directly into `third_party/MEMatte/` (not a subdirectory):

```powershell
# Wrong:  third_party/MEMatte/MEMatte/inference.py
# Right:  third_party/MEMatte/inference.py
git clone https://github.com/AcademicFuworker/MEMatte third_party/MEMatte
```

### npm errors during frontend build

Ensure Node.js 18+ is installed. If you see permission errors:

```powershell
cd web
rmdir /s /q node_modules
npm install
npm run build
```

### "CUDA out of memory" during install test

The test suite runs CPU-only unit tests. If you see OOM errors, it's likely from a different process using GPU memory. Close other GPU applications and retry.
