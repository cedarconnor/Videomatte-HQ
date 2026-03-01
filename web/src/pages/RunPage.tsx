import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getInputSuggestions,
  getVideoInfo,
  pickNative,
  pointPickerFrameUrl,
  preflight,
  previewAutoAnchor,
  previewPointPrompt,
  submitJob,
} from "../api";
import { PointPickerCanvas, type PickerPoint } from "../components/PointPickerCanvas";
import { FrameTimeline } from "../components/FrameTimeline";
import type {
  PointPromptPreviewResponse,
  PreflightResponse,
  PreviewResponse,
  UiPreferences,
  VideoMatteConfigForm,
} from "../types";

interface Props {
  onJobQueued: (jobId: string) => void;
  uiPrefs: UiPreferences;
}

type PickerTarget = "input" | "output_dir" | "anchor_mask" | "mematte_repo_dir" | "mematte_checkpoint";

const TRIMAP_BAND_PRESETS = [
  { label: "Balanced", fg: 0.9, bg: 0.1, note: "Default" },
  { label: "Wide", fg: 0.85, bg: 0.15, note: "More MEMatte work" },
  { label: "Wider", fg: 0.8, bg: 0.2, note: "Wider unknown band" },
  { label: "Maximum", fg: 0.7, bg: 0.3, note: "Largest threshold band" },
] as const;

function trimapPresetIndexFor(fg: number, bg: number): number {
  let best = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  TRIMAP_BAND_PRESETS.forEach((p, idx) => {
    const score = Math.abs(Number(fg) - p.fg) + Math.abs(Number(bg) - p.bg);
    if (score < bestScore) {
      bestScore = score;
      best = idx;
    }
  });
  return best;
}

function buildInitialForm(prefs: UiPreferences): VideoMatteConfigForm {
  return {
    input: "",
    output_dir: prefs.default_output_dir || "output_ui",
    output_alpha: "alpha/%06d.png",
    frame_start: 0,
    frame_end: 30,
    anchor_mask: "",
    segment_backend: "ultralytics_sam3",
    sam3_model: "sam2_l.pt",
    chunk_size: 100,
    chunk_overlap: 5,
    refine_enabled: true,
    mematte_repo_dir: prefs.default_mematte_repo_dir,
    mematte_checkpoint: prefs.default_mematte_checkpoint,
    tile_size: 1536,
    tile_overlap: 96,
    trimap_mode: "morphological",
    trimap_erosion_px: 20,
    trimap_dilation_px: 10,
    trimap_fg_threshold: 0.9,
    trimap_bg_threshold: 0.1,
    trimap_fallback_band_px: 1,
    device: prefs.default_device || "cuda",
    precision: prefs.default_precision || "fp16",
    prompt_mode: "mask",
    point_prompts: {},
  };
}

function validateForm(form: VideoMatteConfigForm): string[] {
  const errors: string[] = [];
  if (!form.input.trim()) errors.push("Input path is required.");
  if (!form.output_dir.trim()) errors.push("Output folder is required.");
  if (!Number.isFinite(form.frame_start) || form.frame_start < 0) errors.push("Frame start must be >= 0.");
  if (!Number.isFinite(form.frame_end) || form.frame_end < form.frame_start) errors.push("Frame end must be >= frame start.");
  if (!Number.isFinite(form.chunk_size) || form.chunk_size < 2) errors.push("Chunk size must be >= 2.");
  if (!Number.isFinite(form.chunk_overlap) || form.chunk_overlap < 0) errors.push("Chunk overlap must be >= 0.");
  if (form.chunk_overlap >= form.chunk_size) errors.push("Chunk overlap must be smaller than chunk size.");
  if (form.refine_enabled) {
    if (!Number.isFinite(form.tile_size) || form.tile_size < 64) errors.push("Tile size must be >= 64.");
    if (!Number.isFinite(form.tile_overlap) || form.tile_overlap < 0) errors.push("Tile overlap must be >= 0.");
    if (form.tile_overlap >= form.tile_size) errors.push("Tile overlap must be smaller than tile size.");
  }
  if (form.trimap_mode === "morphological") {
    if (!Number.isFinite(form.trimap_erosion_px) || form.trimap_erosion_px < 1) errors.push("Trimap erosion must be >= 1 px.");
    if (!Number.isFinite(form.trimap_dilation_px) || form.trimap_dilation_px < 1) errors.push("Trimap dilation must be >= 1 px.");
  }
  if (!Number.isFinite(form.trimap_fg_threshold) || form.trimap_fg_threshold < 0 || form.trimap_fg_threshold > 1) {
    errors.push("Trimap FG threshold must be between 0 and 1.");
  }
  if (!Number.isFinite(form.trimap_bg_threshold) || form.trimap_bg_threshold < 0 || form.trimap_bg_threshold > 1) {
    errors.push("Trimap BG threshold must be between 0 and 1.");
  }
  if (Number.isFinite(form.trimap_bg_threshold) && Number.isFinite(form.trimap_fg_threshold) && form.trimap_bg_threshold >= form.trimap_fg_threshold) {
    errors.push("Trimap BG threshold must be lower than trimap FG threshold.");
  }
  if (!Number.isFinite(form.trimap_fallback_band_px) || form.trimap_fallback_band_px < 0) {
    errors.push("Trimap fallback band must be >= 0.");
  }
  if (form.prompt_mode === "points") {
    const frame0 = form.point_prompts["0"];
    if (!frame0 || frame0.positive.length === 0) {
      errors.push("Point mode requires at least one positive (foreground) point.");
    }
  }
  return errors;
}

export function RunPage({ onJobQueued, uiPrefs }: Props) {
  const [form, setForm] = useState<VideoMatteConfigForm>(() => buildInitialForm(uiPrefs));
  const [autoAnchor, setAutoAnchor] = useState<boolean | null>(null);
  const [allowExternalPaths, setAllowExternalPaths] = useState(false);
  const [verbose, setVerbose] = useState(false);
  const [busy, setBusy] = useState<null | "preflight" | "anchor" | "submit" | "point-preview">(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("Configure a run, then preflight or submit.");
  const [preflightResult, setPreflightResult] = useState<PreflightResponse | null>(null);
  const [anchorPreview, setAnchorPreview] = useState<PreviewResponse | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [picking, setPicking] = useState(false);

  // Point picker state
  const [pickerPoints, setPickerPoints] = useState<PickerPoint[]>([]);
  const [pickerFrame, setPickerFrame] = useState(0);
  const [pickerFrameSrc, setPickerFrameSrc] = useState<string | null>(null);
  const [pointPreview, setPointPreview] = useState<PointPromptPreviewResponse | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.45);
  const [videoFrameCount, setVideoFrameCount] = useState(0);

  useEffect(() => {
    void getInputSuggestions().then(setSuggestions).catch(() => {});
  }, []);

  // Load video info and auto-clamp frame_end when input changes
  useEffect(() => {
    if (!form.input.trim()) {
      setPickerFrameSrc(null);
      setVideoFrameCount(0);
      return;
    }
    void getVideoInfo(form.input)
      .then((info) => {
        setVideoFrameCount(info.frame_count);
        if (info.frame_count > 0) {
          setForm((prev) => ({
            ...prev,
            frame_end: Math.min(prev.frame_end, info.frame_count - 1),
          }));
        }
      })
      .catch(() => setVideoFrameCount(0));
  }, [form.input]);

  // Update frame image when picker frame or input changes
  useEffect(() => {
    if (!form.input.trim()) {
      setPickerFrameSrc(null);
      return;
    }
    const url = pointPickerFrameUrl(form.input, pickerFrame, form.frame_start);
    setPickerFrameSrc(url);
    setPointPreview(null);
  }, [form.input, pickerFrame, form.frame_start]);

  const validationErrors = useMemo(() => validateForm(form), [form]);
  const canSubmit = useMemo(() => form.input.trim().length > 0 && validationErrors.length === 0, [form.input, validationErrors]);
  const trimapPresetIndex = useMemo(
    () => trimapPresetIndexFor(form.trimap_fg_threshold, form.trimap_bg_threshold),
    [form.trimap_fg_threshold, form.trimap_bg_threshold],
  );
  const trimapPreset = TRIMAP_BAND_PRESETS[trimapPresetIndex];

  function update<K extends keyof VideoMatteConfigForm>(key: K, value: VideoMatteConfigForm[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  const handlePointsChange = useCallback((points: PickerPoint[]) => {
    setPickerPoints(points);
    setPointPreview(null);
    // Sync to form
    const positive: [number, number][] = points
      .filter((p) => p.label === "positive")
      .map((p) => [p.x, p.y]);
    const negative: [number, number][] = points
      .filter((p) => p.label === "negative")
      .map((p) => [p.x, p.y]);
    setForm((prev) => ({
      ...prev,
      point_prompts: { ...prev.point_prompts, "0": { positive, negative } },
    }));
  }, []);

  function ensureLocalValidation(actionLabel: string): boolean {
    if (validationErrors.length === 0) return true;
    setError(validationErrors.join(" "));
    setStatus(`${actionLabel} blocked by local validation.`);
    return false;
  }

  async function handlePreflight() {
    if (!ensureLocalValidation("Preflight")) return;
    setBusy("preflight");
    setError(null);
    try {
      const res = await preflight(form, autoAnchor, allowExternalPaths);
      setPreflightResult(res);
      setStatus(
        res.anchor_required
          ? "Preflight passed, but an anchor mask is required (auto-anchor disabled or not applicable)."
          : "Preflight passed.",
      );
    } catch (e) {
      setError(String((e as Error).message || e));
      setStatus("Preflight failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleAutoAnchorPreview() {
    if (!ensureLocalValidation("Auto-anchor preview")) return;
    setBusy("anchor");
    setError(null);
    try {
      const res = await previewAutoAnchor(form);
      setAnchorPreview(res);
      if (res.effective_frame_start !== form.frame_start) {
        update("frame_start", res.effective_frame_start);
      }
      update("anchor_mask", res.mask_path);
      setStatus(`Auto-anchor ready (${res.method}), probe frame ${res.probe_frame}.`);
    } catch (e) {
      setError(String((e as Error).message || e));
      setStatus("Auto-anchor preview failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handlePointPreview() {
    if (!form.input.trim()) return;
    const frame0 = form.point_prompts["0"];
    if (!frame0 || frame0.positive.length === 0) {
      setError("Place at least one positive (foreground) point first.");
      return;
    }
    setBusy("point-preview");
    setError(null);
    try {
      const res = await previewPointPrompt(form, pickerFrame, frame0.positive, frame0.negative);
      setPointPreview(res);
      setStatus(`Point preview ready, coverage: ${(res.mask_coverage * 100).toFixed(1)}%`);
    } catch (e) {
      setError(String((e as Error).message || e));
      setStatus("Point preview failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleSubmit() {
    if (!ensureLocalValidation("Job submit")) return;
    setBusy("submit");
    setError(null);
    try {
      // Build config for submission
      const submitConfig: Partial<VideoMatteConfigForm> = { ...form };
      if (form.prompt_mode === "points") {
        // Serialize point prompts to JSON for the backend
        (submitConfig as Record<string, unknown>).point_prompts_json = JSON.stringify(form.point_prompts);
      }
      const res = await submitJob(submitConfig, {
        auto_anchor: form.prompt_mode === "points" ? false : autoAnchor,
        allow_external_paths: allowExternalPaths,
        verbose,
      });
      setStatus(`Job queued: ${res.id}`);
      onJobQueued(res.id);
    } catch (e) {
      setError(String((e as Error).message || e));
      setStatus("Job submit failed.");
    } finally {
      setBusy(null);
    }
  }

  async function nativePick(target: PickerTarget, mode: "file" | "dir", title: string, fileTypes?: [string, string][]) {
    if (picking) return;
    setPicking(true);
    try {
      const initial = typeof form[target] === "string" ? String(form[target]) : "";
      const result = await pickNative(mode, title, initial, fileTypes);
      if (result) update(target, result);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setPicking(false);
    }
  }

  function applyTrimapPreset(index: number) {
    const preset = TRIMAP_BAND_PRESETS[Math.max(0, Math.min(TRIMAP_BAND_PRESETS.length - 1, Number(index) || 0))];
    update("trimap_fg_threshold", preset.fg);
    update("trimap_bg_threshold", preset.bg);
  }

  const isPointMode = form.prompt_mode === "points";
  const pickerFrameEnd = videoFrameCount > 0 ? Math.min(form.frame_end, videoFrameCount - 1) : form.frame_end;

  // Clamp pickerFrame when frame range changes
  useEffect(() => {
    setPickerFrame((prev) => Math.max(form.frame_start, Math.min(pickerFrameEnd, prev)));
  }, [form.frame_start, pickerFrameEnd]);

  return (
    <div className="panel-grid">
      <section className="panel panel-main">
        <div className="panel-head">
          <h2>Run Pipeline</h2>
          <p>Thin UI for the current v2 CLI. Start with a short range, inspect QC, then run the full clip.</p>
        </div>

        <div className="field-grid">
          <label title="Path to input video file (.mp4, .mov, .mkv) or image sequence pattern (e.g. frames/%06d.png)">
            Input Video / Sequence
            <div className="field-with-button">
              <input
                value={form.input}
                onChange={(e) => update("input", e.target.value)}
                placeholder="D:\\Videomatte-HQ2\\TestFiles\\clip.mp4"
              />
              <button type="button" onClick={() => nativePick("input", "file", "Select Input Video / Sequence", [["Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.mxf"], ["All files", "*.*"]])} disabled={picking}>Browse</button>
            </div>
          </label>
          <label title="Directory where alpha mattes, QC trimaps, and run metadata will be saved">
            Output Folder
            <div className="field-with-button">
              <input value={form.output_dir} onChange={(e) => update("output_dir", e.target.value)} />
              <button type="button" onClick={() => nativePick("output_dir", "dir", "Select Output Folder")} disabled={picking}>Browse</button>
            </div>
          </label>
          <label title="Filename pattern for alpha output frames inside the output folder. Use %06d for zero-padded frame numbers.">
            Alpha Pattern
            <input value={form.output_alpha} onChange={(e) => update("output_alpha", e.target.value)} />
          </label>
          {!isPointMode && (
            <label title="Binary mask image marking the subject in the anchor frame. If blank, auto-anchor will detect the person automatically.">
              Anchor Mask (optional if auto-anchor)
              <div className="field-with-button">
                <input value={form.anchor_mask} onChange={(e) => update("anchor_mask", e.target.value)} />
                <button type="button" onClick={() => nativePick("anchor_mask", "file", "Select Anchor Mask", [["Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.exr"], ["All files", "*.*"]])} disabled={picking}>Browse</button>
              </div>
            </label>
          )}

          <label title="First frame number to process (0-indexed). Skipped frames at the start will not produce output.">
            Frame Start
            <input type="number" value={form.frame_start} onChange={(e) => update("frame_start", Number(e.target.value))} />
          </label>
          <label title="Last frame number to process (inclusive). Use a short range (e.g. 30) for testing before running the full clip.">
            Frame End
            <input type="number" value={form.frame_end} onChange={(e) => update("frame_end", Number(e.target.value))} />
          </label>
          <label title="Compute device for inference. CUDA uses the GPU for much faster processing; CPU is a fallback.">
            Device
            <select value={form.device} onChange={(e) => update("device", e.target.value)}>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label title="Floating-point precision. FP16 (half) is faster and uses less VRAM; FP32 is more precise but slower.">
            Precision
            <select value={form.precision} onChange={(e) => update("precision", e.target.value)}>
              <option value="fp16">FP16</option>
              <option value="fp32">FP32</option>
            </select>
          </label>
        </div>

        <div className="hint-box">
          <strong>Trimap Generation</strong>
          <div className="field-grid">
            <label title="How the trimap unknown band is generated. Morphological uses erosion/dilation on the binary mask for reliable wide bands. Logit uses probability thresholds from SAM output (may produce very narrow bands).">
              Mode
              <select value={form.trimap_mode} onChange={(e) => update("trimap_mode", e.target.value)}>
                <option value="morphological">Morphological (recommended)</option>
                <option value="logit">Logit threshold (legacy)</option>
              </select>
            </label>
          </div>
          {form.trimap_mode === "morphological" ? (
            <>
              <p className="muted">
                Creates unknown band via erosion + dilation of the SAM binary mask. Produces wide bands (default ~30px) that MEMatte can refine effectively.
              </p>
              <div className="field-grid">
                <label title="How many pixels to erode (shrink) the SAM foreground mask inward. Places the 'definite FG' boundary well inside the SAM edge, giving MEMatte room to find the actual boundary.">
                  Erosion (px)
                  <input type="number" min={1} step={1} value={form.trimap_erosion_px} onChange={(e) => update("trimap_erosion_px", Number(e.target.value))} disabled={!form.refine_enabled} />
                </label>
                <label title="How many pixels to dilate (expand) the SAM mask outward. Places the 'definite BG' boundary outside the SAM edge, ensuring background is correctly classified.">
                  Dilation (px)
                  <input type="number" min={1} step={1} value={form.trimap_dilation_px} onChange={(e) => update("trimap_dilation_px", Number(e.target.value))} disabled={!form.refine_enabled} />
                </label>
              </div>
              <div className="muted">
                Total unknown band: ~{form.trimap_erosion_px + form.trimap_dilation_px}px (erosion {form.trimap_erosion_px}px inward + dilation {form.trimap_dilation_px}px outward)
              </div>
            </>
          ) : (
            <>
              <p className="muted">
                Legacy mode: creates unknown band from logit probability thresholds. May produce very narrow bands with confident SAM outputs.
              </p>
              <div className="scrub-row">
                <span className="muted">Balanced</span>
                <input
                  type="range"
                  min={0}
                  max={TRIMAP_BAND_PRESETS.length - 1}
                  step={1}
                  value={trimapPresetIndex}
                  onChange={(e) => applyTrimapPreset(Number(e.target.value))}
                  aria-label="Trimap refine band"
                />
                <span className="muted">Maximum</span>
              </div>
              <div className="hint-list">
                {TRIMAP_BAND_PRESETS.map((preset, idx) => (
                  <button
                    key={preset.label}
                    type="button"
                    className={`tag-button ${idx === trimapPresetIndex ? "active" : ""}`}
                    onClick={() => applyTrimapPreset(idx)}
                    title={`${preset.note} (fg=${preset.fg.toFixed(2)}, bg=${preset.bg.toFixed(2)})`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <div className="muted">
                Preset: <strong>{trimapPreset.label}</strong> ({trimapPreset.note}) · FG &ge; {form.trimap_fg_threshold.toFixed(2)} · BG &le; {form.trimap_bg_threshold.toFixed(2)}
              </div>
              <div className="muted">
                Hard-mask fallback band: <strong>{form.trimap_fallback_band_px}px</strong> (ensures MEMatte runs when threshold trimap is empty)
              </div>
            </>
          )}
        </div>

        <details className="stacked-details">
          <summary>Segmentation & Refinement (advanced)</summary>
          <div className="field-grid">
            <label title="Segmentation model backend. ultralytics_sam3 uses SAM2 via Ultralytics; static replays pre-computed masks.">
              Segment Backend
              <select value={form.segment_backend} onChange={(e) => update("segment_backend", e.target.value)}>
                <option value="ultralytics_sam3">ultralytics_sam3</option>
                <option value="static">static</option>
              </select>
            </label>
            <label title="SAM2 model checkpoint name or path. Larger models (sam2_l) are slower but more accurate. Ultralytics will auto-download if not found locally.">
              SAM3 Model
              <input value={form.sam3_model} onChange={(e) => update("sam3_model", e.target.value)} />
            </label>
            <label title="Number of frames processed per SAM video chunk. Larger chunks maintain better temporal consistency but use more VRAM.">
              Chunk Size
              <input type="number" value={form.chunk_size} onChange={(e) => update("chunk_size", Number(e.target.value))} />
            </label>
            <label title="Number of frames that overlap between consecutive chunks. Overlap frames are blended to prevent chunk boundary artifacts. Must be less than chunk size.">
              Chunk Overlap
              <input type="number" value={form.chunk_overlap} onChange={(e) => update("chunk_overlap", Number(e.target.value))} />
            </label>
            <label className="check-line" title="When enabled, MEMatte refines SAM masks to produce soft alpha edges with hair detail. When disabled, outputs hard binary masks from SAM only (preview mode).">
              <input
                type="checkbox"
                checked={form.refine_enabled}
                onChange={(e) => update("refine_enabled", e.target.checked)}
              />
              MEMatte Refine {form.refine_enabled ? "Enabled" : "Disabled — SAM-only preview"}
            </label>
            <label title="Path to the local MEMatte repository directory containing inference.py and model code.">
              MEMatte Repo Dir
              <div className="field-with-button">
                <input value={form.mematte_repo_dir} onChange={(e) => update("mematte_repo_dir", e.target.value)} disabled={!form.refine_enabled} />
                <button type="button" onClick={() => nativePick("mematte_repo_dir", "dir", "Select MEMatte Repo Dir")} disabled={!form.refine_enabled || picking}>Browse</button>
              </div>
            </label>
            <label title="Path to the MEMatte model weights file (.pth). Default uses ViT-S DIM checkpoint.">
              MEMatte Checkpoint
              <div className="field-with-button">
                <input value={form.mematte_checkpoint} onChange={(e) => update("mematte_checkpoint", e.target.value)} disabled={!form.refine_enabled} />
                <button type="button" onClick={() => nativePick("mematte_checkpoint", "file", "Select MEMatte Checkpoint", [["Checkpoint files", "*.pth *.pt"], ["All files", "*.*"]])} disabled={!form.refine_enabled || picking}>Browse</button>
              </div>
            </label>
            <label title="Size of each square tile (in pixels) for MEMatte refinement. Larger tiles use more VRAM but reduce stitching seams. Typical: 1024-2048.">
              Tile Size
              <input type="number" value={form.tile_size} onChange={(e) => update("tile_size", Number(e.target.value))} disabled={!form.refine_enabled} />
            </label>
            <label title="Overlap between adjacent MEMatte tiles (in pixels). Overlapping regions are blended with Hann windows to eliminate tile boundary artifacts.">
              Tile Overlap
              <input type="number" value={form.tile_overlap} onChange={(e) => update("tile_overlap", Number(e.target.value))} disabled={!form.refine_enabled} />
            </label>
            <label title="[Logit mode] Probability threshold above which pixels are classified as definite foreground (1.0). Higher = narrower unknown band.">
              Trimap FG Threshold
              <input
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={form.trimap_fg_threshold}
                onChange={(e) => update("trimap_fg_threshold", Number(e.target.value))}
              />
            </label>
            <label title="[Logit mode] Probability threshold below which pixels are classified as definite background (0.0). Lower = narrower unknown band.">
              Trimap BG Threshold
              <input
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={form.trimap_bg_threshold}
                onChange={(e) => update("trimap_bg_threshold", Number(e.target.value))}
              />
            </label>
            <label title="[Logit mode] When the threshold-based trimap produces an empty unknown band, this fallback creates a morphological band of the specified width around the mask edge.">
              Trimap Fallback Band (px)
              <input
                type="number"
                step={1}
                min={0}
                value={form.trimap_fallback_band_px}
                onChange={(e) => update("trimap_fallback_band_px", Number(e.target.value))}
              />
            </label>
          </div>
        </details>

        <div className="inline-options">
          {!isPointMode && (
            <div className="choice-group" title="Controls whether a person detection model auto-generates the anchor mask. Auto enables it for video inputs only. Force On/Off overrides.">
              <span>Auto-Anchor</span>
              <button className={autoAnchor === null ? "active" : ""} onClick={() => setAutoAnchor(null)} type="button" title="Automatically use auto-anchor for video inputs, require manual anchor for image sequences">Auto</button>
              <button className={autoAnchor === true ? "active" : ""} onClick={() => setAutoAnchor(true)} type="button" title="Always auto-generate anchor mask via YOLO person detection">Force On</button>
              <button className={autoAnchor === false ? "active" : ""} onClick={() => setAutoAnchor(false)} type="button" title="Disable auto-anchor; you must provide an anchor mask manually">Force Off</button>
            </div>
          )}
          <label className="check-line" title="Enable detailed DEBUG-level logging in the CLI process for troubleshooting.">
            <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
            Verbose CLI logs
          </label>
          <label className="check-line" title="Allow MEMatte repo and checkpoint paths outside the Videomatte-HQ2 repository. Useful if MEMatte is installed elsewhere on this machine.">
            <input
              type="checkbox"
              checked={allowExternalPaths}
              onChange={(e) => setAllowExternalPaths(e.target.checked)}
            />
            Allow External MEMatte Paths
          </label>
        </div>

        {suggestions.length > 0 && !form.input && (
          <div className="hint-box">
            <strong>Detected test inputs:</strong>
            <div className="hint-list">
              {suggestions.slice(0, 6).map((s) => (
                <button key={s} type="button" className="tag-button" onClick={() => update("input", s)}>
                  {s.split(/[\\/]/).pop()}
                </button>
              ))}
            </div>
          </div>
        )}

        {validationErrors.length > 0 && (
          <div className="validation-list">
            {validationErrors.map((msg) => <div key={msg}>{msg}</div>)}
          </div>
        )}

        <div className="action-row">
          <button onClick={handlePreflight} disabled={busy !== null || !canSubmit} title="Validate all paths, dependencies, and settings without starting the pipeline. Quick sanity check.">
            {busy === "preflight" ? "Checking..." : "Preflight"}
          </button>
          {!isPointMode && (
            <button onClick={handleAutoAnchorPreview} disabled={busy !== null || !canSubmit} title="Run YOLO person detection on the first non-black frame and preview the generated anchor mask overlay.">
              {busy === "anchor" ? "Building..." : "Auto-Anchor Preview"}
            </button>
          )}
          <button className="primary" onClick={handleSubmit} disabled={busy !== null || !canSubmit} title="Queue the full pipeline job: segmentation + MEMatte refinement + alpha output.">
            {busy === "submit" ? "Queueing..." : "Start Job"}
          </button>
        </div>

        <div className="status-strip">
          <span>{status}</span>
          {error && <span className="status-error">{error}</span>}
        </div>
      </section>

      <aside className="panel">
        <div className="panel-head compact">
          <h3>Preflight Summary</h3>
        </div>
        {preflightResult ? (
          <dl className="kv-list">
            <div><dt>Video Input</dt><dd>{String(preflightResult.is_video_input)}</dd></div>
            {!isPointMode && <div><dt>Auto-Anchor Effective</dt><dd>{String(preflightResult.auto_anchor_effective)}</dd></div>}
            {!isPointMode && <div><dt>Anchor Required</dt><dd>{String(preflightResult.anchor_required)}</dd></div>}
            <div><dt>Prompt Mode</dt><dd>{form.prompt_mode}</dd></div>
            <div><dt>Frame Range</dt><dd>{preflightResult.frame_start}..{preflightResult.frame_end}</dd></div>
            <div><dt>Refine</dt><dd>{preflightResult.refine_enabled ? "MEMatte" : "SAM-only preview"}</dd></div>
            <div><dt>MEMatte Repo</dt><dd className="small">{preflightResult.mematte_repo_dir}</dd></div>
            <div><dt>MEMatte Ckpt</dt><dd className="small">{preflightResult.mematte_checkpoint}</dd></div>
            <div><dt>External Paths</dt><dd>{String(Boolean(preflightResult.allow_external_paths))}</dd></div>
          </dl>
        ) : (
          <p className="muted">Run preflight to validate paths, MEMatte assets, and frame range.</p>
        )}
      </aside>

      <section className="panel panel-span">
        <div className="panel-head compact">
          <h3>Subject Selection</h3>
        </div>

        <div className="prompt-tabs">
          <button
            type="button"
            className={`prompt-tab ${!isPointMode ? "active" : ""}`}
            onClick={() => update("prompt_mode", "mask")}
          >
            Auto-Anchor
          </button>
          <button
            type="button"
            className={`prompt-tab ${isPointMode ? "active" : ""}`}
            onClick={() => update("prompt_mode", "points")}
          >
            Point Picker
          </button>
        </div>

        {isPointMode ? (
          <>
            <p className="muted" style={{ marginTop: 0, marginBottom: "0.6rem" }}>
              Left-click to place foreground points, right-click for background. SAM3 uses these to identify the subject.
            </p>
            <PointPickerCanvas
              frameSrc={pickerFrameSrc}
              points={pickerPoints}
              onPointsChange={handlePointsChange}
              overlayDataUrl={pointPreview?.overlay_preview_data_url}
              overlayOpacity={overlayOpacity}
              disabled={busy !== null}
            />
            <p className="muted" style={{ margin: "0.4rem 0 0", fontSize: "0.78rem" }}>
              Points apply to frame 0 for initial segmentation. SAM3 automatically propagates the selection across all frames.
            </p>
            <div className="point-picker-controls">
              <button
                type="button"
                onClick={handlePointPreview}
                disabled={busy !== null || pickerPoints.filter((p) => p.label === "positive").length === 0}
              >
                {busy === "point-preview" ? "Running SAM3..." : "Preview SAM3"}
              </button>
              <button
                type="button"
                onClick={() => { setPickerPoints([]); setPointPreview(null); update("point_prompts", {}); }}
                disabled={busy !== null || pickerPoints.length === 0}
              >
                Clear All
              </button>
              <button
                type="button"
                onClick={() => {
                  const next = pickerPoints.slice(0, -1);
                  handlePointsChange(next);
                }}
                disabled={busy !== null || pickerPoints.length === 0}
              >
                Undo
              </button>
              {pointPreview && (
                <div className="opacity-control">
                  <span>Overlay</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                  />
                </div>
              )}
              <span className="muted">
                {pickerPoints.filter((p) => p.label === "positive").length} fg / {pickerPoints.filter((p) => p.label === "negative").length} bg
              </span>
            </div>
            <FrameTimeline
              frame={pickerFrame}
              frameStart={form.frame_start}
              frameEnd={pickerFrameEnd}
              onFrameChange={setPickerFrame}
              disabled={busy !== null}
            />
          </>
        ) : (
          <>
            <p className="muted" style={{ marginTop: 0 }}>
              Generated anchor mask, source frame, and overlay. Use "Auto-Anchor Preview" to generate and inspect the initial subject mask.
            </p>
            {!anchorPreview ? (
              <p className="muted">Click "Auto-Anchor Preview" above to detect the subject automatically.</p>
            ) : (
              <>
                <div className="anchor-meta">
                  <span>Method: <code>{anchorPreview.method}</code></span>
                  <span>Probe frame: <code>{anchorPreview.probe_frame}</code></span>
                  <span>Coverage: <code>{anchorPreview.mask_coverage.toFixed(4)}</code></span>
                  <span>Mask file: <code>{anchorPreview.mask_path}</code></span>
                </div>
                <div className="preview-grid">
                  <figure>
                    <img src={anchorPreview.frame_preview_data_url} alt="anchor frame preview" />
                    <figcaption>Frame</figcaption>
                  </figure>
                  <figure>
                    <img src={anchorPreview.mask_preview_data_url} alt="anchor mask preview" />
                    <figcaption>Mask</figcaption>
                  </figure>
                  <figure>
                    <img src={anchorPreview.overlay_preview_data_url} alt="anchor overlay preview" />
                    <figcaption>Overlay</figcaption>
                  </figure>
                </div>
              </>
            )}
          </>
        )}
      </section>

    </div>
  );
}
