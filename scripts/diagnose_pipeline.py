"""Pipeline diagnostic: measure boundary accuracy at each stage.

Writes per-stage visualizations and metrics to output_runs/full_v2_9770853/diag_stages/.

Usage:
    python scripts/diagnose_pipeline.py
"""
import json
import os
import sys
import random

import cv2
import numpy as np

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "output_runs", "full_v2_9770853")
DIAG_DIR = os.path.join(RUN_DIR, "diag_stages")
VIDEO_PATH = os.path.join(ROOT, "TestFiles", "9770853-uhd_2160_4096_25fps.mp4")

os.makedirs(DIAG_DIR, exist_ok=True)

# ─── helpers ─────────────────────────────────────────────────────────────────

def read_frame(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    print(f"Video: {w}x{h} @ {fps}fps, read frame {frame_idx} shape={frame.shape}")
    return frame  # BGR uint8


def measure_mask(name, mask, frame_shape=None):
    """Print coverage and boundary stats for a binary mask."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.uint8) if mask.dtype == np.uint8 else (mask > 0.5).astype(np.uint8)
    total = binary.size
    fg = int(binary.sum())
    coverage = fg / total
    print(f"  {name}: coverage={coverage:.4f} ({fg}/{total} px)")

    if fg > 0 and fg < total:
        # Bounding box
        ys, xs = np.where(binary > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        bw = bbox[2] - bbox[0] + 1
        bh = bbox[3] - bbox[1] + 1
        print(f"  {name}: bbox=({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) size={bw}x{bh}")

        # Perimeter / area ratio (compactness)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            area = sum(cv2.contourArea(c) for c in contours)
            if area > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                print(f"  {name}: perimeter={perimeter:.0f}, area={area:.0f}, compactness={compactness:.3f}")
    return binary, coverage


def boundary_distance_stats(mask_a, mask_b, name_a="A", name_b="B"):
    """Compute distance from mask_a boundary to mask_b boundary."""
    ba = (mask_a > 0).astype(np.uint8)
    bb = (mask_b > 0).astype(np.uint8)

    # Get boundary pixels of each mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_a = cv2.dilate(ba, kernel) - cv2.erode(ba, kernel)
    edge_b = cv2.dilate(bb, kernel) - cv2.erode(bb, kernel)

    if edge_a.sum() == 0 or edge_b.sum() == 0:
        print(f"  boundary distance {name_a}→{name_b}: one mask has no boundary")
        return None

    # Distance from edge_a pixels to nearest edge_b pixel
    dist_to_b = cv2.distanceTransform(1 - edge_b, cv2.DIST_L2, 5)
    distances = dist_to_b[edge_a > 0]

    stats = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
        "std": float(np.std(distances)),
    }
    print(f"  boundary distance {name_a}→{name_b}: "
          f"mean={stats['mean']:.1f} median={stats['median']:.1f} "
          f"p95={stats['p95']:.1f} max={stats['max']:.1f}")
    return stats


def overlay_boundary(frame_bgr, mask, color=(0, 255, 0), thickness=2):
    """Draw mask boundary on frame."""
    vis = frame_bgr.copy()
    binary = (mask > 0).astype(np.uint8) if mask.dtype == np.uint8 else (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def make_checker(h, w, block=32):
    """Create a checkerboard pattern."""
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((y // block) + (x // block)) % 2 == 0:
                checker[y:y+block, x:x+block] = (200, 200, 200)
            else:
                checker[y:y+block, x:x+block] = (240, 240, 240)
    return checker


def composite_checker(frame_bgr, alpha_f):
    """Composite frame over checkerboard using alpha."""
    h, w = frame_bgr.shape[:2]
    checker = make_checker(h, w)
    if alpha_f.ndim == 2:
        a = alpha_f[..., None]
    else:
        a = alpha_f
    comp = (frame_bgr.astype(np.float32) * a + checker.astype(np.float32) * (1 - a))
    return np.clip(comp, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0: Analyze the auto-anchor mask
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STAGE 0: AUTO-ANCHOR MASK ANALYSIS")
print("=" * 70)

frame0 = read_frame(VIDEO_PATH, 0)
h, w = frame0.shape[:2]
print(f"Frame shape: {frame0.shape} (H={h}, W={w})")

# Load the saved anchor mask
anchor_path = os.path.join(RUN_DIR, "anchor_mask.auto.png")
anchor_mask = cv2.imread(anchor_path, cv2.IMREAD_UNCHANGED)
if anchor_mask.ndim == 3:
    anchor_mask = anchor_mask[..., 0]
print(f"\nSaved anchor mask: shape={anchor_mask.shape}, dtype={anchor_mask.dtype}")
measure_mask("saved_anchor", anchor_mask)

# Generate a FRESH YOLO seg mask (before postprocessing) for comparison
print("\n--- Generating fresh YOLO seg mask (without postprocessing) ---")
try:
    from ultralytics import YOLO

    # Find best available model
    yolo_models = [
        "yolo11x-seg.pt", "yolo11l-seg.pt", "yolo11m-seg.pt",
        "yolo11s-seg.pt", "yolo11n-seg.pt",
        "yolov8x-seg.pt", "yolov8l-seg.pt", "yolov8m-seg.pt",
    ]
    raw_yolo_mask = None
    for model_name in yolo_models:
        try:
            model = YOLO(model_name)
            results = model.predict(source=frame0, device="cuda", classes=[0], conf=0.2, verbose=False)
            if results and results[0].masks is not None:
                mask_data = results[0].masks.data
                arr = mask_data.detach().cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] > 0:
                    areas = arr.reshape(arr.shape[0], -1).sum(axis=1)
                    idx = int(np.argmax(areas))
                    m = arr[idx].astype(np.float32)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    raw_yolo_mask = (m >= 0.5).astype(np.uint8) * 255
                    print(f"  Used YOLO model: {model_name}")
                    break
        except Exception:
            continue

    if raw_yolo_mask is not None:
        measure_mask("raw_yolo_seg", raw_yolo_mask)

        # Now apply the CURRENT postprocessing to show the bloat
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from videomatte_hq.prompts.auto_anchor import _postprocess_anchor_mask, _largest_component
        postprocessed = _postprocess_anchor_mask(raw_yolo_mask)
        measure_mask("postprocessed_anchor", postprocessed)

        # Measure the boundary expansion
        boundary_distance_stats(raw_yolo_mask, postprocessed, "raw_yolo_edge", "postprocessed_edge")

        # Save comparison visualizations
        vis_raw = overlay_boundary(frame0, raw_yolo_mask, (0, 255, 0), 3)  # green = raw YOLO
        vis_post = overlay_boundary(vis_raw, postprocessed, (0, 0, 255), 3)  # red = postprocessed
        vis_saved = overlay_boundary(frame0, anchor_mask, (0, 0, 255), 3)  # red = saved anchor
        cv2.imwrite(os.path.join(DIAG_DIR, "stage0_yolo_raw_vs_postprocessed.png"), vis_post)
        cv2.imwrite(os.path.join(DIAG_DIR, "stage0_saved_anchor_boundary.png"), vis_saved)
        print(f"  Saved: stage0_yolo_raw_vs_postprocessed.png (green=raw, red=postprocessed)")
        print(f"  Saved: stage0_saved_anchor_boundary.png (red=saved anchor)")

        # Save raw vs postprocessed masks side by side
        raw_3ch = cv2.cvtColor(raw_yolo_mask, cv2.COLOR_GRAY2BGR)
        post_3ch = cv2.cvtColor(postprocessed, cv2.COLOR_GRAY2BGR)
        sidebyside = np.hstack([raw_3ch, post_3ch])
        cv2.imwrite(os.path.join(DIAG_DIR, "stage0_masks_raw_vs_post.png"), sidebyside)

        # Save the raw YOLO mask (what we SHOULD be using for v2)
        cv2.imwrite(os.path.join(DIAG_DIR, "stage0_raw_yolo_mask.png"), raw_yolo_mask)
        print(f"  Saved: stage0_raw_yolo_mask.png")
    else:
        print("  WARNING: Could not generate YOLO mask (no model available)")

except ImportError:
    print("  WARNING: ultralytics not installed, skipping YOLO comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Simulate MatAnyone2 mask preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 1: MATANYONE2 MASK PREPROCESSING")
print("=" * 70)

# Load config
with open(os.path.join(RUN_DIR, "config_used.json")) as f:
    cfg = json.load(f)

# Read anchor mask as float [0,1]
anchor_f = anchor_mask.astype(np.float32) / 255.0

# Simulate the wrapper's mask preprocessing
mask_np = (np.clip(anchor_f, 0.0, 1.0) * 255.0).astype(np.uint8)

# Try to import MA2 utilities
ma2_repo = os.path.join(ROOT, "third_party", "MatAnyone2")
sys.path.insert(0, ma2_repo)
try:
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

    erode_k = cfg.get("matanyone2_erode_kernel", 10)
    dilate_k = cfg.get("matanyone2_dilate_kernel", 10)
    print(f"MA2 erode_kernel={erode_k}, dilate_kernel={dilate_k}")

    # Before dilate/erode
    measure_mask("before_ma2_preprocessing", mask_np)

    # Apply dilate then erode (matching wrapper order)
    random.seed(42)  # gen_dilate/gen_erosion use random
    if dilate_k > 0:
        mask_after_dilate = gen_dilate(mask_np.astype(np.float32), dilate_k, dilate_k)
        measure_mask("after_ma2_dilate", mask_after_dilate.astype(np.uint8))
    else:
        mask_after_dilate = mask_np.astype(np.float32)

    random.seed(42)
    if erode_k > 0:
        mask_after_erode = gen_erosion(mask_after_dilate, erode_k, erode_k)
        measure_mask("after_ma2_dilate+erode", mask_after_erode.astype(np.uint8))
    else:
        mask_after_erode = mask_after_dilate

    # Simulate resize to processing resolution
    min_side = min(h, w)
    max_size = cfg.get("matanyone2_max_size", 1080)
    if max_size > 0 and min_side > max_size:
        new_h = int(h / min_side * max_size)
        new_w = int(w / min_side * max_size)
        mask_resized = cv2.resize(mask_after_erode.astype(np.float32),
                                   (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"  Mask resized to processing res: {new_w}x{new_h}")
        measure_mask("ma2_mask_at_proc_res", mask_resized.astype(np.uint8))
    else:
        new_h, new_w = h, w
        mask_resized = mask_after_erode

    # Save visualizations
    vis_ma2_mask = np.clip(mask_after_erode, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(DIAG_DIR, "stage1_ma2_input_mask.png"), vis_ma2_mask)
    print(f"  Saved: stage1_ma2_input_mask.png")

    # Boundary distance from raw YOLO to MA2 preprocessed mask
    if raw_yolo_mask is not None:
        boundary_distance_stats(raw_yolo_mask, vis_ma2_mask, "raw_yolo_edge", "ma2_input_edge")

except ImportError as e:
    print(f"  WARNING: Could not import MatAnyone2 utils: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Analyze MatAnyone2 OUTPUT (from saved alphas)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 2: MATANYONE2 OUTPUT ANALYSIS")
print("=" * 70)

alpha_path = os.path.join(RUN_DIR, "alpha", "000000.png")
alpha_out = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)
if alpha_out.ndim == 3:
    alpha_out = alpha_out[..., 0]
print(f"Final alpha: shape={alpha_out.shape}, dtype={alpha_out.dtype}")

if alpha_out.dtype == np.uint16:
    alpha_f = alpha_out.astype(np.float32) / 65535.0
else:
    alpha_f = alpha_out.astype(np.float32) / 255.0

measure_mask("final_alpha_binary", (alpha_f > 0.5).astype(np.uint8) * 255)

# Check the trimap that was used
trimap_path = os.path.join(RUN_DIR, "qc", "trimap.000000.png")
trimap = cv2.imread(trimap_path, cv2.IMREAD_UNCHANGED)
if trimap is not None:
    if trimap.ndim == 3:
        trimap = trimap[..., 0]
    t = trimap.astype(np.float32) / 255.0
    fg_pct = float((t >= 0.99).mean())
    unk_pct = float(((t > 0.01) & (t < 0.99)).mean())
    bg_pct = float((t <= 0.01).mean())
    print(f"  Trimap: FG={fg_pct:.4f} Unknown={unk_pct:.4f} BG={bg_pct:.4f}")

# Boundary comparison: raw YOLO (ground truth proxy) vs final alpha
if raw_yolo_mask is not None:
    alpha_binary = (alpha_f > 0.5).astype(np.uint8) * 255
    boundary_distance_stats(raw_yolo_mask, alpha_binary, "raw_yolo_edge", "final_alpha_edge")

    # Create multi-boundary overlay
    vis_multi = frame0.copy()
    vis_multi = overlay_boundary(vis_multi, raw_yolo_mask, (0, 255, 0), 2)  # green = YOLO
    vis_multi = overlay_boundary(vis_multi, anchor_mask, (255, 0, 0), 2)  # blue = anchor
    vis_multi = overlay_boundary(vis_multi, alpha_binary, (0, 0, 255), 2)  # red = final alpha
    cv2.imwrite(os.path.join(DIAG_DIR, "stage2_boundaries_comparison.png"), vis_multi)
    print(f"  Saved: stage2_boundaries_comparison.png (green=YOLO, blue=anchor, red=final)")

    # Composite visualization
    comp = composite_checker(frame0, alpha_f[..., None])
    cv2.imwrite(os.path.join(DIAG_DIR, "stage2_checker_composite.png"), comp)
    print(f"  Saved: stage2_checker_composite.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: MEMatte alpha_prior bias analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 3: MEMATTE ALPHA_PRIOR BIAS ANALYSIS")
print("=" * 70)

if trimap is not None:
    t = trimap.astype(np.float32) / 255.0

    # Current behavior: alpha_prior = (trimap >= 0.5) → unknown=FG
    alpha_prior_current = (t >= 0.5).astype(np.float32)
    # Correct behavior: alpha_prior = (trimap > 0.5) → unknown=0.0
    alpha_prior_fixed = (t > 0.5).astype(np.float32)

    unk_mask = (t > 0.01) & (t < 0.99)
    unk_count = int(unk_mask.sum())

    if unk_count > 0:
        current_fg_in_unk = float(alpha_prior_current[unk_mask].mean())
        fixed_fg_in_unk = float(alpha_prior_fixed[unk_mask].mean())
        print(f"  Unknown band: {unk_count} pixels ({unk_count/t.size*100:.2f}% of image)")
        print(f"  alpha_prior in unknown (current >= 0.5): {current_fg_in_unk:.3f} (ALL FG!)")
        print(f"  alpha_prior in unknown (fixed > 0.5):    {fixed_fg_in_unk:.3f}")
        print(f"  BUG: Current code treats ALL unknown pixels as FG prior")

    # Visualize the bias
    bias_vis = np.zeros((t.shape[0], t.shape[1], 3), dtype=np.uint8)
    bias_vis[t <= 0.01] = (0, 0, 0)       # BG = black
    bias_vis[t >= 0.99] = (255, 255, 255)  # FG = white
    bias_vis[unk_mask] = (0, 0, 255)       # Unknown = red (biased to FG)
    cv2.imwrite(os.path.join(DIAG_DIR, "stage3_trimap_unknown_bias.png"), bias_vis)
    print(f"  Saved: stage3_trimap_unknown_bias.png (red = unknown region biased to FG)")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Expansion measurement at each pipeline step
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 4: CUMULATIVE EXPANSION MEASUREMENT")
print("=" * 70)

results = {}

if raw_yolo_mask is not None:
    masks_to_compare = [
        ("1_raw_yolo", raw_yolo_mask),
        ("2_postprocessed_anchor", postprocessed),
        ("3_saved_anchor", anchor_mask),
    ]
    try:
        masks_to_compare.append(("4_ma2_preprocessed", vis_ma2_mask))
    except NameError:
        pass
    masks_to_compare.append(("5_final_alpha", (alpha_f > 0.5).astype(np.uint8) * 255))

    ref_mask = raw_yolo_mask
    ref_coverage = float((ref_mask > 0).sum()) / ref_mask.size

    for name, mask in masks_to_compare:
        binary = (mask > 0).astype(np.uint8) if mask.dtype == np.uint8 else (mask > 0.5).astype(np.uint8)
        coverage = float(binary.sum()) / binary.size
        expansion_ratio = coverage / max(ref_coverage, 1e-8)
        extra_pixels = int(binary.sum()) - int((ref_mask > 0).sum())

        stats = boundary_distance_stats(ref_mask, mask, "raw_yolo", name)
        results[name] = {
            "coverage": coverage,
            "expansion_vs_yolo": expansion_ratio,
            "extra_pixels": extra_pixels,
            "boundary_dist": stats,
        }
        print(f"  {name}: coverage={coverage:.4f}, expansion={expansion_ratio:.2f}x, "
              f"extra_px={extra_pixels:+d}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: Aspect ratio check
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 5: ASPECT RATIO / DIMENSION CHECK")
print("=" * 70)

print(f"Source video (cv2): W={w}, H={h} → numpy shape (H,W)=({h},{w})")
print(f"Anchor mask shape: {anchor_mask.shape}")
print(f"Final alpha shape: {alpha_out.shape}")

if trimap is not None:
    print(f"Trimap shape: {trimap.shape}")

# Check if MA2 processing resolution preserves aspect ratio
min_side = min(h, w)
max_size_cfg = cfg.get("matanyone2_max_size", 1080)
if max_size_cfg > 0 and min_side > max_size_cfg:
    proc_h = int(h / min_side * max_size_cfg)
    proc_w = int(w / min_side * max_size_cfg)
    native_ar = w / h
    proc_ar = proc_w / proc_h
    print(f"MA2 processing res: {proc_w}x{proc_h}")
    print(f"Native aspect ratio: {native_ar:.6f}")
    print(f"Processing aspect ratio: {proc_ar:.6f}")
    print(f"Aspect ratio error: {abs(native_ar - proc_ar) / native_ar * 100:.3f}%")

    # Check for rounding issues
    # The correct formula should preserve aspect ratio better
    scale = max_size_cfg / min_side
    correct_h = round(h * scale)
    correct_w = round(w * scale)
    print(f"Using round() instead of int(): {correct_w}x{correct_h} "
          f"(AR={correct_w/correct_h:.6f}, error={abs(native_ar - correct_w/correct_h)/native_ar*100:.3f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

print("""
IDENTIFIED ISSUES (in order of impact):

1. AUTO-ANCHOR MASK OVER-DILATION [HIGH IMPACT]
   File: src/videomatte_hq/prompts/auto_anchor.py : _postprocess_anchor_mask()
   - Applies morphological close (1.5% of bbox) + dilation (5% of bbox, max 81px kernel)
   - For a large person in 4K, this creates a ~40px halo around the true boundary
   - The anchor mask coverage is ~56% for a single person (should be ~25-30%)
   FIX: For v2 pipeline (MatAnyone2), remove dilation entirely or reduce to 1%.
        MA2 expects a TIGHT boundary mask, not an expanded one.

2. MATANYONE2 ERODE/DILATE DOUBLES THE BLOAT [MEDIUM IMPACT]
   File: src/videomatte_hq/models/matanyone2_wrapper.py : process_video()
   - gen_dilate(10) + gen_erosion(10) is designed for rough USER-drawn masks
   - When input is already an accurate YOLO seg mask, this adds unnecessary expansion
   - The dilate→erode is a close operation that smooths concavities but keeps expansion
   FIX: Set matanyone2_erode_kernel=0 and matanyone2_dilate_kernel=0 for auto-anchor,
        OR make these only apply when the mask source is user-drawn.

3. MEMATTE ALPHA_PRIOR BIASES UNKNOWN→FG [MEDIUM IMPACT]
   File: src/videomatte_hq/pipeline/stage_refine.py : _prepare_mematte_inputs()
   - alpha_prior = (tri_f >= 0.5) → all unknown pixels (0.5) become FG (1.0)
   - This tells MEMatte that ALL uncertain areas are foreground
   - Combined with the bloated boundary, MEMatte expands rather than contracts
   FIX: Change >= 0.5 to > 0.5 so unknown pixels get alpha_prior=0.0 (neutral)

4. GRADIENT TRIMAP BUILT FROM BLOATED ALPHA [CASCADING]
   File: src/videomatte_hq/pipeline/stage_trimap.py : build_trimap_gradient_adaptive()
   - Trimap unknown band is centered on the bloated MA2 boundary (wrong location)
   - Even if MEMatte works perfectly, it refines at the wrong boundary
   - 71% of image is marked definite FG, only 1.7% is unknown
   FIX: Fixing issues 1-2 above will fix this cascading issue.

5. ASPECT RATIO TRUNCATION [LOW IMPACT on this file, varies]
   File: src/videomatte_hq/models/matanyone2_wrapper.py : process_video()
   - Uses int() truncation instead of round() for processing resolution
   - Can introduce ~0.05% aspect ratio error (small but accumulates with resize)
   FIX: Use round() instead of int() for new_h, new_w calculation.
""")

# Save results
with open(os.path.join(DIAG_DIR, "diagnostic_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Diagnostic results saved to {DIAG_DIR}/diagnostic_results.json")
print(f"Visualization images saved to {DIAG_DIR}/")
