# VideoMatte-HQ Beginner Guide

This guide is for first-time users who want the easiest path.

This build is locked to the production workflow:
`SAM2/Samurai tracking -> MatAnyone coarse alpha -> MEMatte refinement -> optional matte cleanup`.

![Mask Builder — load a frame, auto-detect the subject, and build a mask](docs/images/mask_builder_result.png)

## 1) Before You Start

- You need Windows and Python 3.10 or newer.
- You should have:
  - A video file (or image sequence)
  - Optional: a mask image for a keyframe (usually frame 0)

## 2) One-Time Setup

1. Open **PowerShell** in the folder `D:\Videomatte-HQ`.
2. Run these commands:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
```

This creates the local Python environment and installs the tool.

## 3) Start the Web App (Recommended)

1. In the same PowerShell window, run:

```powershell
run_web.bat
```

2. Open your browser to:

`http://localhost:5173`

## 4) Run Your First Matte

In the **Run Job** tab:

1. Set **Input Path**:
   - Video file path, or frame pattern like `input_frames/frame_%05d.png`
2. Set **Output Directory**:
   - Example: `output`
3. In **Subject Assignment (Mask-First)**:
   - **Keyframe Index**: `0`
   - **Anchor Type**: `Initial`
   - If you already have a mask image:
     - Set **Mask Path**
     - Click **Import Mask**
    - If you do not have a mask image, use **Initial Mask Builder (Phase 3)**:
      - Click **Load Frame**
      - (Optional) Enter a prompt like `person` and click **Suggest Boxes**
      - Draw one box around the subject
      - Add a few FG points on the subject and BG points on the background (if needed)
      - Click **Build Anchor Mask** (locked to SAM2/Samurai in Multiple Mask Frames mode)
      - If the subject moves a lot, click **Build + Import Range** (also SAM2/Samurai)
      - Set **Samurai Model Cfg Path** and **Samurai Checkpoint Path** before building
    - Optional after your first keyframe is imported/built: use **Phase 4: Long-Range Propagation Assist**
      - The current **Keyframe Index** is used as the anchor
      - Backend is **SAM2/Samurai Video Predictor** (set model cfg/checkpoint first)
      - Set your propagation range start/end
      - Click **Propagate Keyframes** to auto-add correction anchors across the shot
4. In **Matte Tuning**:
    - Start with preset **Balanced**
5. In **Memory Propagation (Stage 2)**:
    - Keep **Enable Region Constraint** turned on (recommended)
    - Keep **Region Source** on `Tracked Subject Mask` (this is the locked default)
6. Optional in **Edge Refinement (Stage 3)**:
   - Backend is locked to `mematte` for high-resolution edge detail
   - Set:
      - **MEMatte Repo Dir**: `third_party/MEMatte`
      - **MEMatte Checkpoint**: `third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth`
      - **MEMatte Max Tokens**: start with `12000` to `18500`
7. Click **Start Pipeline**

## 5) Check Progress and Quality

- **Job Queue** tab: shows running/completed jobs and logs
- **Quality Control** tab: compare input vs output with the A/B wipe slider
  - Use the dropdown to switch between **Alpha (Raw)**, **Checkerboard**, **White BG**, **Black BG**, or **Overlay**
  - Try **Overlay** mode to check edge quality
  - Use **J/K** or arrow keys to navigate frames, **Shift** for 10-frame jumps
- Optional: in **Run Job > Debug Stage Exports**, enable stage samples before running
  - This writes per-stage images and a diagnosis report, which is helpful when quality breaks on specific frames
- Optional: leave **Auto-export stage diagnostics when QC fails** enabled
  - If QC fails, the app will automatically write `debug_stages/diagnosis.json` + `debug_stages/diagnosis.md` so you can see which stage introduced the issue

## 6) If a Section Looks Wrong

Use a correction keyframe:

1. Make a corrected mask for the bad frame.
2. In **Subject Assignment**:
   - Set that frame number
   - Set **Anchor Type** to `Correction`
   - Choose the corrected mask path
   - Click **Import Mask**
3. Keep **Auto-Apply Suggested Range** enabled.
4. Run again.

## 7) Where Files Are Saved

- Alpha output frames: `output\alpha\...`
- QC report: `output\qc\optionb_report.md`
- QC metrics JSON: `output\qc\optionb_metrics.json`
- Project file: usually `output\project.vmhqproj`
- Stage debug artifacts (if enabled): `output\debug_stages\...`

## 8) Quick Matte Tuning Tips

- **Shrink/Grow**:
  - Positive = expands matte
  - Negative = tightens matte
- **Feather**:
  - Higher = softer edges
- **Offset X/Y**:
  - Moves matte left/right/up/down by pixels
- **Trimap Width**:
  - Wider values can help difficult hair edges
- **Temporal Cleanup (Stage 4) Flicker Controls**:
  - Turn on **Smooth inside edge band (micro-EMA)** for edge shimmer
  - Keep **Use confidence-gated clamp** enabled for stability in hard motion
  - Use **Edge snap guidance filter** only when edges wobble and you need extra tightening

## 9) Quick Troubleshooting

- If the web UI does not open:
  - Make sure `run_web.bat` is still running
  - Open `http://localhost:5173` manually
- If run fails saying assignment is required:
  - Import at least one keyframe mask first
- If output looks too harsh:
  - Increase feather slightly (for example 1 to 2)
- If output looks too loose:
  - Use a small negative shrink/grow (for example `-1`)
- If background is leaking into the matte:
  - Keep **Memory Propagation > Enable Region Constraint** on
  - Set **Memory Propagation > Propagation Backend** to **SAM2/Samurai Video Predictor** and fill cfg/checkpoint paths
  - Increase **BBox Margin** slowly only if limbs are getting clipped
  - Enable **Debug Stage Exports** and check if `stage2_memory` is where leakage starts
- If the mask builder or pipeline features fail with errors:
  - Make sure you ran `run_web.bat` from the project folder (it uses the local `.venv` Python)
  - Check that both servers are running (backend on port 8000, frontend on port 5173)
  - Verify runtime manually:
    - `.\.venv\Scripts\python -c "import torch, torchvision; import importlib; importlib.import_module('sam2.build_sam')"`
  - If that command fails with `WinError 127` or `c10_cuda.dll`:
    - `.\.venv\Scripts\pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- If `mematte` backend fails to start:
  - Confirm `third_party/MEMatte` exists
  - Confirm checkpoint file path exists
  - Re-check your Python environment and rerun after model paths are fixed
