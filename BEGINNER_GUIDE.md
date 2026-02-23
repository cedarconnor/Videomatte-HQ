# Beginner Guide (Windows, First Successful Run)

This guide walks you through getting your first alpha matte output from `D:\Videomatte-HQ2`.

It covers both:

- the product CLI (`videomatte-hq-v2`)
- the local web UI (`run_web.bat`)

It assumes:

- You are on Windows
- You want the real refined pipeline (SAM + MEMatte)
- You want a short test run first, then a longer run

## 0. What You Are Running

This tool has two stages:

1. SAM-based segmentation/tracking (finds and tracks the subject)
2. MEMatte refinement (improves edges/hair/soft details)

Important behavior in this build:

- MEMatte refinement is required (no preview/no-refine mode)
- If MEMatte gets no unknown trimap band to refine, the run fails with a clear error and you widen the trimap band thresholds
- The web UI and QC previews are designed to help you do that quickly

## 1. Open PowerShell in the Repo

```powershell
cd D:\Videomatte-HQ2
```

## 2. Activate the Virtual Environment

If `.venv` already exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

If it does not exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install the Project (CLI)

Install the repo so the CLI command is available:

```powershell
pip install -e .
```

Test it:

```powershell
videomatte-hq-v2 --help
```

If needed, you can always run the module directly:

```powershell
python -m videomatte_hq.cli --help
```

## 4. Confirm MEMatte Assets Exist (Local or External)

### Recommended (self-contained inside this repo)

Check these paths:

```powershell
Test-Path .\third_party\MEMatte
Test-Path .\third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth
```

Both should return `True`.

### Optional (external MEMatte path)

You can also use a MEMatte repo/checkpoint outside `D:\Videomatte-HQ2`, but you must opt in:

- CLI: add `--allow-external-paths`
- Web UI: enable `Allow External MEMatte Paths`

If you do not opt in, preflight will reject external MEMatte paths.

## 5. (Optional) Install the Web UI Dependencies

Skip this if you only want the CLI.

Install Python web extras:

```powershell
pip install -e .[web]
```

Install frontend packages:

```powershell
cd web
npm install
cd ..
```

## 6. First Successful Run (CLI, Short Range)

Use a short range first so failures are fast and easy to inspect.

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\4625475-hd_1920_1080_24fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\first_run" `
  --frame-start 0 `
  --frame-end 9 `
  --device cuda `
  --precision fp16
```

What this does:

- Uses default segmenter (`ultralytics_sam3`)
- Uses the default high-quality SAM checkpoint (`sam2_l.pt`)
- Auto-generates an anchor mask (video input, no manual anchor provided)
- Runs MEMatte refinement (required)
- Writes alpha frames and QC trimap preview frames

## 7. Check the Output

Open the output folder:

- `D:\Videomatte-HQ2\output_cli\first_run`

You should see:

- `alpha\` (alpha matte images)
- `qc\` (trimap preview images)
- `config_used.json`
- `run_summary.json`
- `anchor_mask.auto.png` (if auto-anchor was used)

Quick checks:

```powershell
Get-ChildItem .\output_cli\first_run\alpha\*.png | Measure-Object
Get-ChildItem .\output_cli\first_run\qc\trimap*.png | Measure-Object
Get-Content .\output_cli\first_run\run_summary.json
```

## 8. If the First Frame Is Black (Auto-Anchor Start Shift)

Some videos begin with a black or near-black frame.

The CLI handles this automatically for auto-anchor runs:

- it probes forward to find a usable frame for anchor generation
- it may shift the effective `frame_start`

You may see a log like:

- `Auto-anchor probed frame 1 (requested frame_start=0). Using effective frame_start=1.`

Confirm in `run_summary.json`:

- `requested_frame_start`
- `frame_start` (effective)

## 9. If You Get `MEMatte did not execute on any tiles`

This means MEMatte was enabled and loaded, but the trimap unknown band was empty (no gray region for Stage 2 to refine).

### Fast fix (CLI)

Widen the trimap band by making FG/BG thresholds more strict:

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\4625475-hd_1920_1080_24fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\first_run_retuned" `
  --frame-start 0 `
  --frame-end 9 `
  --trimap-fg-threshold 0.97 `
  --trimap-bg-threshold 0.03 `
  --device cuda `
  --precision fp16
```

Why this helps:

- fewer pixels become “definitely FG” or “definitely BG”
- more pixels remain gray/unknown
- MEMatte gets a region to refine

## 10. First Successful Run (Web UI)

Start the backend + frontend:

```powershell
run_web.bat
```

Open the printed frontend URL (the script prints the exact port).

### Suggested beginner flow in the UI

1. Go to `Run`
2. Use `Browse` to pick input and output folder
3. Click `Auto-Anchor Preview` (optional but useful)
4. Keep a short range (for example `0..30`)
5. Click `Preflight`
6. Click `Start Job`
7. Open `Jobs` and watch progress/logs
8. Open `QC` and inspect:
   - alpha preview
   - trimap preview (gray = MEMatte unknown band)

### If the web job fails with the MEMatte no-tile error

Use the visible `Trimap Refine Band` control on the `Run` page:

- try `Wider`
- then `Maximum` if needed

The UI also shows exact thresholds (FG/BG) so you can see what changed.

## 11. Using Your Own Video

Replace the input path with your own file and keep the first run short:

```powershell
videomatte-hq-v2 `
  --input "D:\MyClips\person_shot.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\my_clip_run" `
  --frame-start 0 `
  --frame-end 30 `
  --device cuda `
  --precision fp16
```

Tips:

- Start with `0..30` or `0..60`
- Confirm the anchor mask looks correct
- Check the `qc\trimap...png` outputs (or the web QC tab)
- Then extend the frame range

## 12. Manual Anchor Mask (If Auto-Anchor Picks the Wrong Subject)

If auto-anchor picks the wrong person/object:

1. Create a black/white anchor mask for the starting frame
2. Save it as PNG (white subject, black background)
3. Run with `--anchor-mask`

Example:

```powershell
videomatte-hq-v2 `
  --input "D:\MyClips\group_shot.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\manual_anchor_run" `
  --frame-start 0 `
  --frame-end 100 `
  --anchor-mask "D:\Masks\group_shot_anchor.png" `
  --device cuda `
  --precision fp16
```

To force manual mode and prevent auto-anchor:

```powershell
--no-auto-anchor
```

## 13. External MEMatte Assets (Optional)

If your MEMatte repo/checkpoint are stored outside `D:\Videomatte-HQ2`, enable the override.

### CLI example

```powershell
videomatte-hq-v2 `
  --input "D:\MyClips\person_shot.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\external_mematte" `
  --allow-external-paths `
  --mematte-repo-dir "D:\Models\MEMatte" `
  --mematte-checkpoint "D:\Models\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth"
```

### Web UI

- Check `Allow External MEMatte Paths` on the `Run` page
- Enter/browse the external MEMatte repo/checkpoint paths

## 14. Common Errors and Fixes

### Error: `MEMatte refinement is mandatory for this tool`

Cause:

- `--no-refine` was passed
- or a config/web payload attempted to disable refinement

Fix:

- Remove `--no-refine`
- Keep MEMatte enabled (required in this build)

### Error: `MEMatte did not execute on any tiles ...`

Cause:

- The trimap unknown band is empty

Fix:

- Widen the trimap band:
  - Web UI: `Trimap Refine Band` -> `Wider` or `Maximum`
  - CLI: try `--trimap-fg-threshold 0.97 --trimap-bg-threshold 0.03`
- Inspect QC trimap preview (gray pixels should exist around edges/uncertain regions)

### Error: `External MEMatte paths are not allowed for this tool`

Cause:

- You used external MEMatte paths without enabling override

Fix:

- Add `--allow-external-paths` (CLI), or
- Enable `Allow External MEMatte Paths` (web UI), or
- Use repo-local MEMatte assets under `third_party\MEMatte`

### Error: `MEMatte repo dir not found` / `MEMatte checkpoint not found`

Fix:

- Verify the paths exist (local or external)
- For the self-contained repo setup, make sure:
  - `D:\Videomatte-HQ2\third_party\MEMatte`
  - `D:\Videomatte-HQ2\third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth`

### Error: `Ultralytics is required for segment_backend='ultralytics_sam3'`

Fix:

```powershell
pip install ultralytics
```

### Error: `anchor_mask is required for v2 pipeline runs`

Usually means:

- You are using an image sequence and did not provide `--anchor-mask`
- Or you disabled auto-anchor for a video (`--no-auto-anchor`)

Fix:

- Provide `--anchor-mask`, or
- Use a video input and allow auto-anchor

### Error: `Invalid frame range ... frame_end < frame_start`

Fix:

- Make sure `--frame-end >= --frame-start`

## 15. Useful Next Commands

Show CLI help:

```powershell
videomatte-hq-v2 --help
```

Run with verbose logs:

```powershell
videomatte-hq-v2 ... --verbose
```

Use a config file:

```powershell
videomatte-hq-v2 --config .\my_run.yaml
```

## 16. Beginner Workflow Recommendation (Current Best Practice)

Use this sequence for new clips:

1. Run a short refined range (`10-30` frames)
2. Inspect `anchor_mask.auto.png`
3. Inspect alpha outputs and `qc\trimap...png` (or web QC tab)
4. If MEMatte no-tile error occurs, widen the trimap band and rerun the short range
5. When the short range looks good, extend to a larger frame range

This is the fastest reliable workflow now that no-refine preview mode is disabled.

## 17. Where To Confirm What Actually Ran

Use these files written for every product CLI run:

- `config_used.json` (full resolved config)
- `run_summary.json` (compact summary, auto-anchor info, MEMatte paths, `allow_external_paths`)

These are the first place to check if results look different than expected.
