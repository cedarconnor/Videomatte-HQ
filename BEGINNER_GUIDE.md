# Beginner Guide (Windows, First Successful Run)

This guide walks you through getting your first alpha matte output from `D:\Videomatte-HQ2` using the product CLI.

It assumes:

- You are on Windows
- You want the real refined pipeline (SAM + MEMatte)
- You want everything contained in `D:\Videomatte-HQ2` (no paths to `D:\Videomatte-HQ`)

## 0. What You Are Running

This tool has two stages:

1. SAM-based segmentation/tracking (finds and tracks the subject)
2. MEMatte refinement (improves edges/hair/soft details)

The CLI can also auto-generate an anchor mask for video files, so you usually do not need to paint one manually for a quick run.

## 1. Open PowerShell in the Repo

Open PowerShell and go to the repo folder:

```powershell
cd D:\Videomatte-HQ2
```

## 2. Activate the Virtual Environment

If the `.venv` already exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

If it does not exist, create and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install the Project (CLI Command)

Install the repo so the `videomatte-hq-v2` command is available:

```powershell
pip install -e .
```

Test that the CLI is installed:

```powershell
videomatte-hq-v2 --help
```

If that fails, you can still run:

```powershell
python -m videomatte_hq.cli --help
```

## 4. Confirm MEMatte Assets Are Local (Inside This Repo)

This tool is configured to reject MEMatte paths outside `D:\Videomatte-HQ2`.

Check these paths exist:

```powershell
Test-Path .\third_party\MEMatte
Test-Path .\third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth
```

Both should return `True`.

If not:

- restore/copy the `third_party\MEMatte` folder into `D:\Videomatte-HQ2`
- make sure the checkpoint is present under `third_party\MEMatte\checkpoints\`

## 5. Run Your First Video (Full Refinement)

Use one of the test clips included in this repo:

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

- Uses the default segmenter (`ultralytics_sam3`)
- Uses default high-quality SAM checkpoint (`sam2_l.pt`)
- Auto-generates an anchor mask (because input is a video and no `--anchor-mask` is provided)
- Runs MEMatte refinement (because refinement is enabled by default)
- Writes alpha output frames

## 6. Check the Output

Open the output folder:

```powershell
D:\Videomatte-HQ2\output_cli\first_run
```

You should see:

- `alpha\` (alpha matte images)
- `config_used.json`
- `run_summary.json`
- `anchor_mask.auto.png` (auto-generated anchor mask)

Quick checks:

```powershell
Get-ChildItem .\output_cli\first_run\alpha\*.png | Measure-Object
Get-Content .\output_cli\first_run\run_summary.json
```

## 7. What If the First Frame Is Black?

Some videos begin with a black or near-black frame.

The CLI handles this automatically for auto-anchor runs:

- it probes forward to find a usable frame for anchor generation
- it may shift the effective `frame_start`

You will see a log like:

- `Auto-anchor probed frame 1 (requested frame_start=0). Using effective frame_start=1.`

You can confirm this in `run_summary.json`:

- `requested_frame_start`
- `frame_start` (effective)

## 8. Faster Test Mode (Skip MEMatte)

If you just want to test tracking/anchor quality first, disable refinement:

```powershell
videomatte-hq-v2 `
  --input "D:\Videomatte-HQ2\TestFiles\4990426-hd_1080_1920_30fps.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\preview_check" `
  --frame-start 0 `
  --frame-end 15 `
  --no-refine `
  --device cuda `
  --precision fp16
```

This runs a preview/no-refine path. It is faster and useful for debugging, but not the final highest-quality output.

## 9. Using Your Own Video

Replace the `--input` path with your own file:

```powershell
videomatte-hq-v2 `
  --input "D:\MyClips\person_shot.mp4" `
  --output-dir "D:\Videomatte-HQ2\output_cli\my_clip_run" `
  --frame-start 0 `
  --frame-end 120 `
  --device cuda `
  --precision fp16
```

Tips:

- Start with a short frame range first (`0..30` or `0..60`)
- Confirm the auto-anchor mask looks correct
- Then expand the frame range

## 10. Manual Anchor Mask (If Auto-Anchor Fails or Picks the Wrong Subject)

Auto-anchor is convenient, but not perfect.

If it picks the wrong person/object:

1. Create a black/white anchor mask image for the starting frame
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

## 11. Common Errors and Fixes

### Error: `External MEMatte paths are not allowed for this tool`

You passed a MEMatte path outside `D:\Videomatte-HQ2`.

Fix:

- Remove `--mematte-repo-dir` and `--mematte-checkpoint` (defaults already point inside this repo), or
- Use paths under `D:\Videomatte-HQ2\third_party\MEMatte`

Do not use:

- `D:\Videomatte-HQ\third_party\MEMatte`

### Error: `MEMatte repo dir not found`

Fix:

- Ensure `D:\Videomatte-HQ2\third_party\MEMatte` exists

### Error: `MEMatte checkpoint not found`

Fix:

- Ensure this file exists:
  - `D:\Videomatte-HQ2\third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth`

### Error: `Ultralytics is required for segment_backend='ultralytics_sam3'`

Fix:

```powershell
pip install ultralytics
```

### Error: `anchor_mask is required for v2 pipeline runs`

This usually means:

- You are using an image sequence input and did not provide `--anchor-mask`
- Or you disabled auto-anchor on a video (`--no-auto-anchor`)

Fix:

- Provide `--anchor-mask`, or
- Use a video input and allow auto-anchor

### Error: `Invalid frame range ... frame_end < frame_start`

Fix:

- Make sure `--frame-end` is greater than or equal to `--frame-start`

## 12. Useful Next Commands

Show CLI help:

```powershell
videomatte-hq-v2 --help
```

Run with verbose logs:

```powershell
videomatte-hq-v2 ... --verbose
```

Use config file instead of many flags:

```powershell
videomatte-hq-v2 --config .\my_run.yaml
```

## 13. Beginner Workflow Recommendation

Use this sequence for new clips:

1. Run `--no-refine` on 10-30 frames to check anchor/tracking
2. Inspect `anchor_mask.auto.png` and a few alpha outputs
3. Run full refine on the same short range
4. If good, extend to a larger frame range

This saves time and avoids waiting on MEMatte for obviously bad anchors.

## 14. Where to Ask “Did It Actually Use the Right Settings?”

Use these files:

- `config_used.json` (full resolved config)
- `run_summary.json` (compact summary + auto-anchor info)

These are written for every product CLI run.
