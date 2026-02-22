# VIDEOMATTE-HQ v2 — Architecture Design Document

**Simplified Two-Stage Video Matting Pipeline**

Replaces the 6-stage MatAnyone/SAMURAI/MEMatte pipeline with a clean **SAM 3 Segmentation** + **MEMatte Refinement** architecture.

*February 2026*

---

## 1. Problem Statement

The current Videomatte-HQ repository has grown to approximately 12,000 lines of Python across 78 files. The 6-stage pipeline has accumulated significant complexity, with multiple subsystems existing solely to compensate for upstream instability in the MatAnyone coarse segmentation stage.

### 1.1 Current Pipeline Architecture

```
Stage 0: Frame loading
Stage 1: Project + keyframe assignments (.vmhqproj)
  └─ Memory region constraint (SAM2/SAMURAI video predictor propagation)
Stage 2: Coarse alpha (MatAnyone OR custom appearance memory bank)
Stage 3: Edge refinement (MeMatte tiled OR guided filter, with region trimap)
Stage 4: Temporal cleanup (EMA, edge-band smoothing, edge snap, motion warp)
Stage 5: Matte tuning (shrink/grow, feather, offset)
Stage 6: Output write + QC metrics + auto-diagnosis
```

### 1.2 Root Causes of Complexity

**Three separate heavy models.** SAM2/SAMURAI (region priors), MatAnyone (coarse propagation), and MEMatte (edge refinement) each require their own repository vendoring, shim layers, and build toolchains. The MatAnyone stage alone requires Hydra configuration, a custom InferenceCore, and a warmup phase.

**Stability patches dominate the codebase.** Approximately 1,600 lines (`samurai_backend.py`, `pass_memory.py`, `memory_region_constraint.py`) exist solely to feed and constrain MatAnyone. The confidence-gated temporal cleanup and region constraint systems (~700 combined lines) exist to patch problems caused by MatAnyone drift. These are symptoms, not features.

**Windows build fragility.** MatAnyone requires Hydra. SAMURAI requires custom CUDA kernel compilation. Both are hostile to Windows environments and frequently brick user installations in ComfyUI contexts.

**Configuration explosion.** The pipeline exposes over 130 configuration fields, most of which are tuning knobs for compensating upstream drift rather than meaningful artistic controls.

---

## 2. Proposed Architecture

The v2 architecture reduces the pipeline to two fundamental stages, separated by three protocol interfaces that allow backend-agnostic operation.

### 2.1 High-Level Pipeline

```
Interfaces:
  ├─ Segmenter protocol       → segment_sequence(source, prompt) → masks + logits
  ├─ PromptAdapter protocol    → adapt(mask) → bbox / points / text
  └─ EdgeRefiner protocol      → refine(rgb, trimap) → alpha

Stage 1: Segmentation (low-res, temporal)
  ├─ PromptAdapter converts user mask → bbox + N interior points + negatives
  ├─ Segmenter (default: Ultralytics SAM 3) propagates across chunked frames
  ├─ Drift detector: IoU vs previous frame, re-anchor if < threshold
  └─ Output: per-frame soft logits + binary masks

Stage 2: Refinement (full-res, per-frame)
  ├─ Trimap built from SAM 3 logits (not morphology)
  ├─ EdgeRefiner (default: MEMatte) runs tiled on unknown band only
  ├─ Skip-frame optimization: if IoU with previous > 0.98, reuse alpha
  └─ Output: production alpha mattes

Optional: Matte tuning (shrink/grow/feather)
```

### 2.2 Why SAM 3 Replaces the Entire Middle Stack

SAM 3 (Segment Anything Model 3) is Meta's successor to SAM 2, with improved temporal consistency and video object segmentation capabilities. It directly replaces three current subsystems:

| Current Subsystem | Lines of Code | Replaced By |
|---|---|---|
| `samurai_backend.py` | 686 | SAM 3 via Ultralytics |
| `pipeline/pass_memory.py` | 548 | SAM 3 video tracking |
| `memory_region_constraint.py` | 367 | Eliminated (SAM 3 does not drift) |
| `propagation_assist.py` | ~350 | SAM 3 built-in propagation |
| `pass_temporal_cleanup.py` | ~300 | Eliminated (better coarse masks) |
| **Total removed** | **~2,250 lines** | **~200-line SAM 3 module** |

---

## 3. SAM 3 Backend Selection

### 3.1 Ultralytics vs HuggingFace Transformers

Two ported SAM 3 implementations are available. Ultralytics is recommended as the default backend, with HuggingFace as a documented alternative.

| Criterion | Ultralytics | HuggingFace Transformers |
|---|---|---|
| Installation | `pip install ultralytics` | `pip install transformers` |
| Windows MSVC | Not required | Not required |
| Video tracking | Built-in `model.track()` with state | Manual propagation loop required |
| Mask prompting | Limited via high-level API; use `SAM3VideoPredictor` for direct access | Full mask/box/point prompt control |
| API maturity | Well-tested deployment ecosystem | Research-grade, more boilerplate |
| ComfyUI dist. | Rock-solid, no compiler needed | Good, no compiler needed |

### 3.2 Mask-Prompt Initialization (Critical Caveat)

The Ultralytics SAM 3 high-level video API currently exposes text and bounding box prompts for `model.track()`. It does not accept a raw binary mask array directly. To use true mask prompting on frame 0, bypass the high-level API and use the underlying predictor class:

```python
from ultralytics.models.sam.predict import SAM3VideoPredictor

predictor = SAM3VideoPredictor(
    overrides=dict(task="segment", mode="predict", model="sam3.pt")
)

# Inject mask prompt directly into predictor state for frame 0
# Then propagate via predictor(source="video.mp4", stream=True)
```

If direct mask prompting proves unavailable in the target version, the PromptAdapter system (Section 4.1) transparently converts user masks to bbox + multi-point prompts as a fallback. This fallback discards precise edge data from the user's drawn mask but produces equivalent tracking results for most subjects.

### 3.3 SAM 3 as Interchangeable Backend (Design Requirement)

SAM 3 via Ultralytics must be treated as one backend implementation, not as the architecture itself. The exact video tracking API and state persistence model is still maturing. The Segmenter protocol (Section 4.2) ensures that if limitations are discovered (multi-object ID stability, occlusion handling, long-sequence drift), the backend can be swapped to:

- Ultralytics SAM 2 dynamic interactive predictor
- HuggingFace / Meta reference code for custom anchoring loops
- Future SAM variants or alternative VOS models

without touching the orchestrator or refinement stage.

---

## 4. Protocol Interfaces

Three protocol interfaces form the architectural spine. All pipeline components depend only on these protocols, never on concrete implementations.

### 4.1 PromptAdapter Protocol

Converts user-provided masks into prompts accepted by any segmentation backend. Multiple adapter implementations support different prompting strategies.

```python
class PromptAdapter(Protocol):
    """Convert a user mask into segmenter-compatible prompts."""

    def adapt(
        self,
        mask: np.ndarray,       # [H, W] float32 in [0, 1]
        frame_shape: tuple[int, int],
    ) -> SegmentPrompt:
        ...


@dataclass
class SegmentPrompt:
    bbox: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    mask: np.ndarray | None = None  # raw mask if backend supports it
```

**Adapter implementations:**

- **MaskPromptAdapter** (default): Derives bbox from mask extent, samples K interior positive points via farthest-point sampling on the distance transform, and places 4 negative points outside bbox corners. This is the primary fallback when direct mask prompting is unavailable.
- **BoxPromptAdapter:** Extracts bbox only. Minimal, for backends that only accept box prompts.
- **TextPromptAdapter:** Future-facing; SAM 3 concept segmentation via natural language. Not implemented in v2.0 but the interface slot is reserved.

**Farthest-point sampling for positive points (critical detail):**

A single center-point prompt is fragile for thin, concave, or multi-part shapes. Instead, sample K well-spread interior points using the distance transform of the binary mask:

```python
def _sample_interior_points(mask: np.ndarray, k: int = 5) -> list[tuple[float, float]]:
    binary = (mask >= 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    points = []
    for _ in range(k):
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        points.append((float(x), float(y)))
        cv2.circle(dist, (int(x), int(y)), max(10, int(dist.max() * 0.3)), 0, -1)
    return points
```

This ensures prompt coverage across the full mask extent, preventing tracking loss on elongated or articulated subjects.

### 4.2 Segmenter Protocol

```python
class Segmenter(Protocol):
    """Temporal video segmentation backend."""

    def segment_sequence(
        self,
        source: FrameSource,
        prompt: SegmentPrompt,
        anchor_frame: int = 0,
        chunk_size: int = 100,
        chunk_overlap: int = 5,
    ) -> SegmentResult:
        ...


@dataclass
class SegmentResult:
    masks: list[np.ndarray]   # [H, W] float32, thresholded binary masks
    logits: list[np.ndarray]  # [H, W] float32, pre-sigmoid soft probabilities
    anchored_frames: list[int]  # frame indices where re-anchoring occurred
```

The Segmenter always returns both thresholded masks and raw soft logits. The logits are consumed by the trimap builder in Stage 2 (Section 5.2) to produce confidence-aware unknown bands.

### 4.3 EdgeRefiner Protocol

```python
class EdgeRefiner(Protocol):
    """High-resolution alpha matte refinement."""

    def refine(
        self,
        rgb: np.ndarray,       # [H, W, 3] float32 in [0, 1]
        trimap: np.ndarray,    # [H, W] float32: 0.0=BG, 0.5=unknown, 1.0=FG
    ) -> np.ndarray:           # [H, W] float32 alpha in [0, 1]
        ...
```

The default implementation is MEMatte (existing detectron2-free wrapper). The protocol allows future swaps to ViTMatte or newer matting models without changing the orchestrator.

---

## 5. Stage Specifications

### 5.1 Stage 1: SAM 3 Segmentation

**Purpose:** Propagate a subject mask across all video frames at reduced resolution, producing per-frame binary masks and soft logits.

#### 5.1.1 Resolution Strategy

SAM 3 processes frames at a configurable long-side resolution (default: 960px). For 8K source material (7680×4320), this represents an 8× downscale. The coarse masks are upscaled to source resolution before Stage 2. This separation keeps VRAM usage tractable and leverages SAM 3's strength in semantic tracking rather than sub-pixel edge extraction.

#### 5.1.2 Chunked Processing (Mandatory for 8K)

SAM 3's tracking mechanism maintains an internal memory bank of past frames for temporal consistency. On long sequences, this memory bank will exhaust GPU VRAM. Chunked processing is not optional for 8K workflows; it is a hard architectural requirement.

```
Processing strategy:
  1. Split sequence into chunks of N frames (default: 100)
  2. Overlap M frames between adjacent chunks (default: 5)
  3. For chunk 0: seed with user's initial prompt
  4. For chunk K>0: seed with refined output mask from last frame of chunk K-1
  5. Clear SAM 3 inference state between chunks
  6. In overlap zone: cross-fade masks from adjacent chunks
```

**Re-anchoring at chunk boundaries:** The final output mask of chunk A is fed back through the PromptAdapter to generate a fresh prompt for frame 0 of chunk B. This prevents drift accumulation across the full sequence while maintaining temporal coherence within each chunk.

#### 5.1.3 Drift Detection and Re-Anchoring

Even within a single chunk, SAM 3 tracking can encounter problems during occlusions, rapid motion, or concept drift. A lightweight IoU-based drift detector monitors mask stability:

```python
def _check_drift(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    iou_threshold: float = 0.70,
    area_change_threshold: float = 0.40,
) -> bool:
    """Return True if drift is detected."""
    intersection = np.logical_and(current_mask > 0.5, previous_mask > 0.5).sum()
    union = np.logical_or(current_mask > 0.5, previous_mask > 0.5).sum()
    iou = intersection / max(union, 1)
    area_ratio = current_mask.sum() / max(previous_mask.sum(), 1)
    return iou < iou_threshold or abs(1.0 - area_ratio) > area_change_threshold
```

When drift is detected, the system re-anchors by re-prompting SAM 3 with the last known-good mask (the previous frame's output converted via PromptAdapter). This doubles as a quality gate: if drift persists after re-anchoring, the frame is flagged for manual review.

#### 5.1.4 Capturing Soft Logits

SAM 3 outputs raw logits (pre-sigmoid activation values) before thresholding into binary masks. These logits encode the model's per-pixel confidence and are essential for the trimap strategy in Stage 2. Both masks and logits are stored in the SegmentResult:

- Logit > 0.0 after sigmoid corresponds to > 50% foreground probability
- The magnitude of the logit encodes certainty: large positive values are confident foreground, large negative values are confident background, values near zero are uncertain

Logit capture is implementation-specific. In Ultralytics, access via the results object's raw tensor outputs rather than the thresholded mask properties.

### 5.2 Stage 2: Upscale and MEMatte Refinement

**Purpose:** Upscale coarse SAM 3 masks to source resolution and refine edge detail using MEMatte tiled inference on the unknown band only.

#### 5.2.1 Logit-Based Trimap Generation (Key Improvement)

The v1 pipeline used morphological operations (`cv2.erode`, `cv2.dilate`) on binary masks to create the unknown band for MEMatte. This produces mathematically uniform, jagged bands that ignore actual visual features.

**v2 uses SAM 3's raw soft logits** to generate a confidence-aware trimap. The unknown band naturally widens around motion blur, fuzzy hair, and transparency (where the model is less certain) and stays razor-thin along hard edges like clothing seams.

```python
def build_trimap_from_logits(
    logits: np.ndarray,     # [H, W] raw SAM 3 logits (pre-sigmoid)
    fg_threshold: float = 0.9,
    bg_threshold: float = 0.1,
) -> np.ndarray:
    """Build trimap from soft logits instead of morphology."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))  # sigmoid
    trimap = np.full_like(probs, 0.5, dtype=np.float32)  # default: unknown
    trimap[probs >= fg_threshold] = 1.0   # definite foreground
    trimap[probs <= bg_threshold] = 0.0   # definite background
    return trimap
```

**Why this is better:** Around hair strands where SAM 3 outputs probabilities of 0.3–0.7, the trimap correctly marks a wide unknown band. Along a hard jacket edge where SAM 3 outputs probabilities of 0.01 or 0.99, the unknown band shrinks to nearly nothing. MEMatte receives better guidance and produces cleaner results with fewer artifacts.

#### 5.2.2 Upscale Strategy

SAM 3 logits and masks are computed at processing resolution (e.g., 960px long side). Before trimap generation, both are upscaled to source resolution:

- Logits: bilinear interpolation (preserves continuous probability gradients)
- Binary masks: bilinear interpolation followed by re-threshold at 0.5 (for skip-frame IoU checks)

Trimap generation happens after upscale, at full source resolution, so the unknown band has pixel-accurate alignment with the source frames.

#### 5.2.3 Tiled MEMatte Inference

MEMatte processes tiles at full source resolution. The tiling logic from v1 (tile iteration, Hann window blending, overlap stitching) is preserved with minor simplification:

- Remove the guided filter fallback path (MEMatte is the only refiner)
- Remove region trimap guidance (no longer needed without MatAnyone)
- Keep confidence-gated blending: MEMatte output is composited over the upscaled coarse mask using the trimap's unknown band as a blend mask
- Definite FG (trimap = 1.0) and definite BG (trimap = 0.0) regions are locked; MEMatte only runs on tiles containing unknown pixels

The existing detectron2-free MEMatte wrapper (`edge_mematte.py`, ~497 lines) is copied verbatim to v2. This wrapper's shim layers for detectron2, timm, fairscale, and einops are the most valuable engineering in the repository and represent dozens of hours of compatibility work.

#### 5.2.4 Skip-Frame Optimization

Not every frame requires full tiled MEMatte inference. If the SAM 3 mask is nearly identical between consecutive frames (IoU > 0.98), the previous frame's refined alpha is reused. This can reduce refinement compute by 50–70% on locked-off or slow-motion shots:

```python
prev_mask = None
prev_alpha = None
for t in range(num_frames):
    mask = coarse_masks[t]
    if prev_mask is not None and _compute_iou(mask, prev_mask) > skip_iou_threshold:
        outputs.append(prev_alpha)  # reuse
        continue
    alpha = _tiled_mematte_inference(refiner, source[t], trimap[t], cfg)
    outputs.append(alpha)
    prev_mask, prev_alpha = mask, alpha
```

---

## 6. Component Disposition Map

Every file in the v1 source tree falls into one of four categories: keep verbatim, simplify, replace, or delete.

| Component | Lines | Fate | Rationale |
|---|---|---|---|
| `models/edge_mematte.py` | 497 | ✅ **KEEP** | Detectron2-free MEMatte wrapper. Core asset. |
| `io/reader.py` | ~280 | ✅ **KEEP** | Frame loading / FrameSource. Stable. |
| `io/writer.py` | ~150 | ✅ **KEEP** | Alpha output writer. Stable. |
| `tiling/` (stitch, windows) | ~100 | ✅ **KEEP** | Hann window + tile stitching. |
| `utils/image.py` | ~40 | ✅ **KEEP** | Image conversion utilities. |
| `pipeline/pass_refine.py` | 377 | 🔶 **SIMPLIFY** | Keep MEMatte tiled path only. Cut guided filter fallback, region trimap. |
| `pipeline/pass_matte_tuning.py` | ~80 | 🔶 **SIMPLIFY** | Keep shrink/grow/feather. Useful artist control. |
| `config.py` | 297 | 🔶 **SIMPLIFY** | ~130 fields to ~30 fields. |
| `cli.py` | 1,276 | 🔶 **SIMPLIFY** | ~50 CLI flags to ~10. |
| `pipeline/orchestrator.py` | 701 | 🔶 **REWRITE** | 2-stage orchestrator, ~150 lines. |
| `qc/` | ~500 | 🔶 **SIMPLIFY** | Basic sanity checks only. Cut regression gates. |
| `samurai_backend.py` | 686 | ❌ **DELETE** | SAM 3 replaces SAMURAI entirely. |
| `pipeline/pass_memory.py` | 548 | ❌ **DELETE** | MatAnyone backend removed. SAM 3 replaces. |
| `memory_region_constraint.py` | 367 | ❌ **DELETE** | Drift compensation no longer needed. |
| `pipeline/pass_temporal_cleanup.py` | ~300 | ❌ **DELETE** | Replaced by SAM 3 temporal consistency + drift detector. |
| `propagation_assist.py` | ~350 | ❌ **DELETE** | SAM 3 built-in propagation. |
| `background/`, `band/`, `intermediate/`, `temporal/`, `reference/`, `roi/` | ~1,770 | ❌ **DELETE** | All drift-compensation subsystems. No longer needed. |
| `sam_builder.py`, `prompt_boxes.py`, `prompt_mask_range.py` | ~900 | ❌ **DELETE** | Replaced by PromptAdapter system. |
| `models/global_rvm.py`, `flow_raft.py` | ~220 | ❌ **DELETE** | RVM and optical flow no longer needed. |

**Estimated total: ~12,000 lines reduced to ~2,500–3,000 lines.**

---

## 7. Project File Structure

```
src/videomatte_hq/
├── __init__.py
├── cli.py                       # Simplified CLI (~150 lines)
├── config.py                    # Simplified config (~80 lines, ~30 fields)
├── protocols.py                 # Segmenter, PromptAdapter, EdgeRefiner protocols
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py          # 2-stage orchestrator (~150 lines)
│   ├── stage_segment.py         # SAM 3 Segmenter implementation (~250 lines)
│   └── stage_refine.py          # MEMatte tiled refinement (~250 lines)
├── prompts/
│   ├── __init__.py
│   ├── mask_adapter.py          # Mask → bbox + farthest-point prompts (~80 lines)
│   └── box_adapter.py           # Mask → bbox-only fallback (~30 lines)
├── models/
│   ├── __init__.py
│   └── edge_mematte.py          # VERBATIM from v1 (detectron2-free wrapper)
├── io/
│   ├── __init__.py
│   ├── reader.py                # VERBATIM from v1 (FrameSource)
│   └── writer.py                # VERBATIM from v1 (AlphaWriter)
├── tiling/
│   ├── __init__.py
│   ├── stitch.py                # VERBATIM from v1
│   └── windows.py               # VERBATIM from v1
├── postprocess/
│   ├── __init__.py
│   └── matte_tuning.py          # Optional shrink/grow/feather
└── utils/
    ├── __init__.py
    └── image.py                 # VERBATIM from v1
```

---

## 8. Simplified Configuration

The v2 configuration reduces from ~130 fields to ~30, organized into four groups:

```python
class VideoMatteConfig(BaseModel):
    # ---- IO ----
    input: str = "input_frames/*.png"
    output_dir: str = "output"
    output_alpha: str = "alpha/%06d.png"
    frame_start: int = 0
    frame_end: int = -1
    alpha_format: str = "png16"

    # ---- Segmentation (Stage 1) ----
    sam3_model: str = "sam3_large"
    sam3_processing_long_side: int = 960
    anchor_frame: int = 0
    anchor_mask: str = ""             # path to initial mask image
    chunk_size: int = 100             # frames per SAM 3 processing chunk
    chunk_overlap: int = 5            # overlap frames between chunks
    drift_iou_threshold: float = 0.70 # re-anchor if IoU drops below this
    drift_area_threshold: float = 0.40

    # ---- Refinement (Stage 2) ----
    refine_enabled: bool = True
    mematte_repo_dir: str = "third_party/MEMatte"
    mematte_checkpoint: str = "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth"
    mematte_max_tokens: int = 18500
    tile_size: int = 1536
    tile_overlap: int = 96
    trimap_fg_threshold: float = 0.9  # logit probability threshold for definite FG
    trimap_bg_threshold: float = 0.1  # logit probability threshold for definite BG
    skip_iou_threshold: float = 0.98  # reuse previous alpha if mask IoU exceeds this

    # ---- Matte Tuning (Optional) ----
    shrink_grow_px: int = 0
    feather_px: int = 0

    # ---- Runtime ----
    device: str = "cuda"
    workers_io: int = 4
```

---

## 9. Evaluation Harness

A before/after comparison harness prevents the simplification from introducing quality regressions. This must be run before declaring v2 production-ready.

### 9.1 Test Clips

Select 3–5 representative clips covering the failure modes most likely to differ between v1 and v2:

| Clip Type | What It Tests | Expected v2 Behavior |
|---|---|---|
| Hair detail | Fine strand separation, transparency | Logit trimap should outperform morphological trimap |
| Motion blur | Fast subject movement, edge smearing | SAM 3 logits create wider unknown band in blur zones |
| Occlusion | Subject passes behind object and re-emerges | Drift detector re-anchors; SAM 3 should recover |
| Wide shot | Small subject, large BG area | PromptAdapter multi-point sampling keeps lock |
| Low contrast | Subject color similar to background | SAM 3 semantic understanding should outperform MatAnyone color matching |

### 9.2 Metrics

- **Temporal IoU stability:** frame-to-frame IoU of output alphas. Standard deviation should be < 0.02 for static shots.
- **Area jitter:** frame-to-frame change in foreground pixel count. Flags mask shrink/expand drift.
- **Centroid jitter:** frame-to-frame centroid displacement. Flags spatial drift.
- **Edge F-score:** if ground-truth mattes are available, compute boundary F-score at 1px and 3px thresholds.
- **QC heuristics:** simplified versions of v1's QC gates, run as warnings rather than hard failures.

---

## 10. Migration Strategy

Clean-slate implementation on a `v2/` branch, not in-place refactoring.

1. **Create `v2/` branch** with the new file structure from Section 7.
2. **Copy verbatim:** `edge_mematte.py`, `io/reader.py`, `io/writer.py`, `tiling/`, `utils/image.py`. Do not modify these files.
3. **Implement protocols:** `protocols.py` with Segmenter, PromptAdapter, EdgeRefiner. These are small (< 50 lines total) but define the architecture.
4. **Implement PromptAdapter:** `mask_adapter.py` with farthest-point sampling. Unit test with roundtrip sanity checks (mask → prompt → SAM 3 → mask, verify IoU > 0.9).
5. **Implement Stage 1:** `stage_segment.py` with chunked SAM 3 processing and drift detection. Test on a single short clip first.
6. **Implement Stage 2:** `stage_refine.py` with logit-based trimap and tiled MEMatte. Verify MEMatte still loads and infers correctly.
7. **Implement orchestrator:** `orchestrator.py` wiring Stage 1 → Stage 2. Minimal config, minimal CLI.
8. **Run evaluation harness** (Section 9) against v1 output on all test clips. Compare quality metrics.
9. **Port web UI last:** update FastAPI backend endpoints to match the 2-stage flow. Simplify the wizard to: upload frames → draw mask → click Run.

### 10.1 Build Priority Order

| # | Task | Depends On | Est. Effort |
|---|---|---|---|
| 1 | Protocol interfaces | Nothing | 1 hour |
| 2 | PromptAdapter + unit tests | #1 | 2 hours |
| 3 | Chunked SAM 3 segmenter | #1, #2 | Half day |
| 4 | Logit-based trimap + MEMatte refiner | #1 | Half day |
| 5 | Orchestrator + config + CLI | #3, #4 | Half day |
| 6 | Evaluation harness + comparison | #5 | 1 day |
| 7 | Web UI port | #5 | 1 day |

---

## 11. Future Considerations

### 11.1 Adaptive Trimap Width by Motion/Detail

The logit-based trimap is a major improvement over morphology, but can be further enhanced by widening the unknown band where edge gradients are high (hair, fur, transparency), where mask confidence is low (semi-transparent materials), or where motion blur is detected (per-frame blur metric). This is a v2.1 enhancement, not a v2.0 requirement.

### 11.2 ComfyUI Node Architecture

The strict two-stage separation maps directly to a node-based architecture. A future ComfyUI integration requires only two nodes: a SAM 3 Tracker node (wrapping the Segmenter protocol) and a MEMatte Refiner node (wrapping the EdgeRefiner protocol). The PromptAdapter runs internally within the tracker node. This is architecturally identical to the existing ComfyUI_SimpleTiles_Uprez pattern.

### 11.3 Alternative Refiners

The EdgeRefiner protocol enables future swaps without orchestrator changes. Candidates include ViTMatte (already partially integrated in v1), newer transformer-based matting models, and potential ensemble approaches combining multiple refiners on different unknown-band zones.

### 11.4 Multi-Subject Tracking

SAM 3 supports multi-object tracking natively. The v2 architecture can be extended to track multiple subjects by providing per-subject prompts and maintaining per-subject SegmentResults. Each subject's masks flow independently through Stage 2 refinement. This is left as a v2.x extension.

### 11.5 Text-Prompted Segmentation

SAM 3 introduces concept-level segmentation via natural language prompts. A TextPromptAdapter could enable workflows where the user simply types "the person in the red jacket" instead of drawing a mask. The protocol interface already reserves this slot.
