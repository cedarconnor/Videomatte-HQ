"""Full pass on 9748555-uhd_2160_4096_25fps.mp4 with all fixes applied."""
import os, sys, json, logging, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_full")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.orchestrator import run_pipeline
from videomatte_hq.prompts.auto_anchor import build_auto_anchor_mask_for_video

VIDEO = os.path.join(ROOT, "TestFiles", "9748555-uhd_2160_4096_25fps.mp4")
OUT = os.path.join(ROOT, "output_runs", "full_v2_fixed_9748555")
os.makedirs(OUT, exist_ok=True)

# Generate tight anchor
anchor_path = os.path.join(OUT, "anchor_mask.auto.png")
logger.info("Generating tight anchor mask...")
r = build_auto_anchor_mask_for_video(VIDEO, anchor_path, device="cuda", frame_start=0, tight=True)
logger.info("Anchor: %s (method=%s, frame=%d)", r.mask_path, r.method, r.probe_frame)

cfg = VideoMatteConfig(
    input=VIDEO,
    output_dir=OUT,
    frame_start=0,
    frame_end=-1,  # all frames
    pipeline_mode="v2",
    anchor_mask=str(r.mask_path),
    matanyone2_erode_kernel=0,
    matanyone2_dilate_kernel=0,
    matanyone2_max_size=1080,
    matanyone2_warmup=10,
    refine_enabled=True,
    device="cuda",
    precision="fp16",
    generate_preview_mp4=True,
    preview_fps=0.0,
)

from dataclasses import asdict
with open(os.path.join(OUT, "config_used.json"), "w") as f:
    json.dump(asdict(cfg), f, indent=2)

t0 = time.time()
logger.info("Starting full v2 pipeline...")
result = run_pipeline(cfg)
elapsed = time.time() - t0

import cv2
import numpy as np

# Count written alpha frames from disk
alpha_dir = os.path.join(OUT, "alpha")
n = len([f for f in os.listdir(alpha_dir) if f.endswith(".png")]) if os.path.isdir(alpha_dir) else 0
logger.info("Done: %d alpha frames in %.1f seconds (%.2f fps)", n, elapsed, n / max(elapsed, 1))

# Quick coverage summary from disk
coverages = []
for i in range(n):
    path = os.path.join(alpha_dir, f"{i:06d}.png")
    if not os.path.exists(path):
        continue
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        continue
    if a.ndim == 3:
        a = a[..., 0]
    af = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
    c = float((af > 0.5).mean())
    coverages.append(c)
    if i == 0 or (i + 1) % 100 == 0 or i == n - 1:
        logger.info("  frame %d/%d: cov>0.5=%.4f", i, n, c)

coverages = np.array(coverages)
logger.info("Coverage stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            coverages.mean(), coverages.std(), coverages.min(), coverages.max())

# Save summary
summary = {
    "input": VIDEO,
    "output_dir": OUT,
    "num_frames": n,
    "elapsed_seconds": round(elapsed, 1),
    "fps": round(n / max(elapsed, 1), 2),
    "coverage_mean": round(float(coverages.mean()), 4),
    "coverage_std": round(float(coverages.std()), 4),
    "coverage_min": round(float(coverages.min()), 4),
    "coverage_max": round(float(coverages.max()), 4),
}
with open(os.path.join(OUT, "run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
logger.info("Summary saved to %s", os.path.join(OUT, "run_summary.json"))
