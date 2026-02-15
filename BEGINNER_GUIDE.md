# VideoMatte-HQ Beginner Guide

This guide is for first-time users who want the easiest path.

## 1) Before You Start

- You need Windows and Python 3.10 or newer.
- You should have:
  - A video file (or image sequence)
  - At least one mask image for a keyframe (usually frame 0)

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
   - **Mask Path**: your mask image path
   - Click **Import Mask**
4. In **Matte Tuning**:
   - Start with preset **Balanced**
5. Click **Start Pipeline**

## 5) Check Progress and Quality

- **Job Queue** tab: shows running/completed jobs and logs
- **Quality Control** tab: compare input vs output
  - Try **Overlay** mode to check edge quality

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

