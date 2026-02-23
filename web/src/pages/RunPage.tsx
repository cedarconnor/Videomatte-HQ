import { useEffect, useMemo, useState } from "react";
import { getInputSuggestions, preflight, previewAutoAnchor, submitJob } from "../api";
import { PathBrowserModal } from "../components/PathBrowserModal";
import type { PreflightResponse, PreviewResponse, UiPreferences, VideoMatteConfigForm } from "../types";

interface Props {
  onJobQueued: (jobId: string) => void;
  uiPrefs: UiPreferences;
}

type PickerTarget = "input" | "output_dir" | "anchor_mask" | "mematte_repo_dir" | "mematte_checkpoint";

type PickerState = {
  target: PickerTarget;
  mode: "file" | "dir";
  title: string;
} | null;

const TRIMAP_BAND_PRESETS = [
  { label: "Balanced", fg: 0.9, bg: 0.1, note: "Default" },
  { label: "Wide", fg: 0.94, bg: 0.06, note: "More MEMatte work" },
  { label: "Wider", fg: 0.97, bg: 0.03, note: "Fix hard/empty trimaps" },
  { label: "Maximum", fg: 0.99, bg: 0.01, note: "Slowest / most permissive" }
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
    trimap_fg_threshold: 0.9,
    trimap_bg_threshold: 0.1,
    device: prefs.default_device || "cuda",
    precision: prefs.default_precision || "fp16"
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
  if (!Number.isFinite(form.tile_size) || form.tile_size < 64) errors.push("Tile size must be >= 64.");
  if (!Number.isFinite(form.tile_overlap) || form.tile_overlap < 0) errors.push("Tile overlap must be >= 0.");
  if (form.tile_overlap >= form.tile_size) errors.push("Tile overlap must be smaller than tile size.");
  if (!Number.isFinite(form.trimap_fg_threshold) || form.trimap_fg_threshold < 0 || form.trimap_fg_threshold > 1) {
    errors.push("Trimap FG threshold must be between 0 and 1.");
  }
  if (!Number.isFinite(form.trimap_bg_threshold) || form.trimap_bg_threshold < 0 || form.trimap_bg_threshold > 1) {
    errors.push("Trimap BG threshold must be between 0 and 1.");
  }
  if (Number.isFinite(form.trimap_bg_threshold) && Number.isFinite(form.trimap_fg_threshold) && form.trimap_bg_threshold >= form.trimap_fg_threshold) {
    errors.push("Trimap BG threshold must be lower than trimap FG threshold.");
  }
  return errors;
}

export function RunPage({ onJobQueued, uiPrefs }: Props) {
  const [form, setForm] = useState<VideoMatteConfigForm>(() => buildInitialForm(uiPrefs));
  const [autoAnchor, setAutoAnchor] = useState<boolean | null>(null);
  const [allowExternalPaths, setAllowExternalPaths] = useState(false);
  const [verbose, setVerbose] = useState(false);
  const [busy, setBusy] = useState<null | "preflight" | "anchor" | "submit">(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("Configure a run, then preflight or submit.");
  const [preflightResult, setPreflightResult] = useState<PreflightResponse | null>(null);
  const [anchorPreview, setAnchorPreview] = useState<PreviewResponse | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [picker, setPicker] = useState<PickerState>(null);

  useEffect(() => {
    void getInputSuggestions().then(setSuggestions).catch(() => {});
  }, []);

  const validationErrors = useMemo(() => validateForm(form), [form]);
  const canSubmit = useMemo(() => form.input.trim().length > 0 && validationErrors.length === 0, [form.input, validationErrors]);
  const trimapPresetIndex = useMemo(
    () => trimapPresetIndexFor(form.trimap_fg_threshold, form.trimap_bg_threshold),
    [form.trimap_fg_threshold, form.trimap_bg_threshold]
  );
  const trimapPreset = TRIMAP_BAND_PRESETS[trimapPresetIndex];

  function update<K extends keyof VideoMatteConfigForm>(key: K, value: VideoMatteConfigForm[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

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
      const res = await preflight({ ...form, refine_enabled: true }, autoAnchor, allowExternalPaths);
      setPreflightResult(res);
      setStatus(
        res.anchor_required
          ? "Preflight passed, but an anchor mask is required (auto-anchor disabled or not applicable)."
          : "Preflight passed."
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

  async function handleSubmit() {
    if (!ensureLocalValidation("Job submit")) return;
    setBusy("submit");
    setError(null);
    try {
      const res = await submitJob(
        { ...form, refine_enabled: true },
        { auto_anchor: autoAnchor, allow_external_paths: allowExternalPaths, verbose }
      );
      setStatus(`Job queued: ${res.id}`);
      onJobQueued(res.id);
    } catch (e) {
      setError(String((e as Error).message || e));
      setStatus("Job submit failed.");
    } finally {
      setBusy(null);
    }
  }

  function openPicker(target: PickerTarget, mode: "file" | "dir", title: string) {
    setPicker({ target, mode, title });
  }

  function applyTrimapPreset(index: number) {
    const preset = TRIMAP_BAND_PRESETS[Math.max(0, Math.min(TRIMAP_BAND_PRESETS.length - 1, Number(index) || 0))];
    update("trimap_fg_threshold", preset.fg);
    update("trimap_bg_threshold", preset.bg);
  }

  return (
    <div className="panel-grid">
      <section className="panel panel-main">
        <div className="panel-head">
          <h2>Run Pipeline</h2>
          <p>Thin UI for the current v2 CLI. Start with a short range, inspect QC, then run the full clip.</p>
        </div>

        <div className="field-grid">
          <label>
            Input Video / Sequence
            <div className="field-with-button">
              <input
                value={form.input}
                onChange={(e) => update("input", e.target.value)}
                placeholder="D:\\Videomatte-HQ2\\TestFiles\\clip.mp4"
              />
              <button type="button" onClick={() => openPicker("input", "file", "Select Input Video / Sequence")}>Browse</button>
            </div>
          </label>
          <label>
            Output Folder
            <div className="field-with-button">
              <input value={form.output_dir} onChange={(e) => update("output_dir", e.target.value)} />
              <button type="button" onClick={() => openPicker("output_dir", "dir", "Select Output Folder")}>Browse</button>
            </div>
          </label>
          <label>
            Alpha Pattern
            <input value={form.output_alpha} onChange={(e) => update("output_alpha", e.target.value)} />
          </label>
          <label>
            Anchor Mask (optional if auto-anchor)
            <div className="field-with-button">
              <input value={form.anchor_mask} onChange={(e) => update("anchor_mask", e.target.value)} />
              <button type="button" onClick={() => openPicker("anchor_mask", "file", "Select Anchor Mask")}>Browse</button>
            </div>
          </label>

          <label>
            Frame Start
            <input type="number" value={form.frame_start} onChange={(e) => update("frame_start", Number(e.target.value))} />
          </label>
          <label>
            Frame End
            <input type="number" value={form.frame_end} onChange={(e) => update("frame_end", Number(e.target.value))} />
          </label>
          <label>
            Device
            <select value={form.device} onChange={(e) => update("device", e.target.value)}>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label>
            Precision
            <select value={form.precision} onChange={(e) => update("precision", e.target.value)}>
              <option value="fp16">FP16</option>
              <option value="fp32">FP32</option>
            </select>
          </label>
        </div>

        <div className="hint-box">
          <strong>Trimap Refine Band</strong>
          <p className="muted">
            Widen this when jobs fail with "MEMatte did not execute on any tiles" or when QC trimap previews show almost no gray unknown band.
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
            Preset: <strong>{trimapPreset.label}</strong> ({trimapPreset.note}) · FG ≥ {form.trimap_fg_threshold.toFixed(2)} · BG ≤ {form.trimap_bg_threshold.toFixed(2)}
          </div>
        </div>

        <details className="stacked-details">
          <summary>Segmentation & Refinement (advanced)</summary>
          <div className="field-grid">
            <label>
              Segment Backend
              <select value={form.segment_backend} onChange={(e) => update("segment_backend", e.target.value)}>
                <option value="ultralytics_sam3">ultralytics_sam3</option>
                <option value="static">static</option>
              </select>
            </label>
            <label>
              SAM3 Model
              <input value={form.sam3_model} onChange={(e) => update("sam3_model", e.target.value)} />
            </label>
            <label>
              Chunk Size
              <input type="number" value={form.chunk_size} onChange={(e) => update("chunk_size", Number(e.target.value))} />
            </label>
            <label>
              Chunk Overlap
              <input type="number" value={form.chunk_overlap} onChange={(e) => update("chunk_overlap", Number(e.target.value))} />
            </label>
            <label className="check-line">
              <input
                type="checkbox"
                checked={true}
                disabled
                readOnly
              />
              MEMatte Refine Enabled (required)
            </label>
            <label>
              MEMatte Repo Dir
              <div className="field-with-button">
                <input value={form.mematte_repo_dir} onChange={(e) => update("mematte_repo_dir", e.target.value)} />
                <button type="button" onClick={() => openPicker("mematte_repo_dir", "dir", "Select MEMatte Repo Dir")}>Browse</button>
              </div>
            </label>
            <label>
              MEMatte Checkpoint
              <div className="field-with-button">
                <input value={form.mematte_checkpoint} onChange={(e) => update("mematte_checkpoint", e.target.value)} />
                <button type="button" onClick={() => openPicker("mematte_checkpoint", "file", "Select MEMatte Checkpoint")}>Browse</button>
              </div>
            </label>
            <label>
              Tile Size
              <input type="number" value={form.tile_size} onChange={(e) => update("tile_size", Number(e.target.value))} />
            </label>
            <label>
              Tile Overlap
              <input type="number" value={form.tile_overlap} onChange={(e) => update("tile_overlap", Number(e.target.value))} />
            </label>
            <label>
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
            <label>
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
          </div>
        </details>

        <div className="inline-options">
          <div className="choice-group">
            <span>Auto-Anchor</span>
            <button className={autoAnchor === null ? "active" : ""} onClick={() => setAutoAnchor(null)} type="button">Auto</button>
            <button className={autoAnchor === true ? "active" : ""} onClick={() => setAutoAnchor(true)} type="button">Force On</button>
            <button className={autoAnchor === false ? "active" : ""} onClick={() => setAutoAnchor(false)} type="button">Force Off</button>
          </div>
          <label className="check-line">
            <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
            Verbose CLI logs
          </label>
          <label className="check-line">
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
          <button onClick={handlePreflight} disabled={busy !== null || !canSubmit}>
            {busy === "preflight" ? "Checking..." : "Preflight"}
          </button>
          <button onClick={handleAutoAnchorPreview} disabled={busy !== null || !canSubmit}>
            {busy === "anchor" ? "Building..." : "Auto-Anchor Preview"}
          </button>
          <button className="primary" onClick={handleSubmit} disabled={busy !== null || !canSubmit}>
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
            <div><dt>Auto-Anchor Effective</dt><dd>{String(preflightResult.auto_anchor_effective)}</dd></div>
            <div><dt>Anchor Required</dt><dd>{String(preflightResult.anchor_required)}</dd></div>
            <div><dt>Frame Range</dt><dd>{preflightResult.frame_start}..{preflightResult.frame_end}</dd></div>
            <div><dt>Refine</dt><dd>{String(preflightResult.refine_enabled)}</dd></div>
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
          <h3>Anchor Preview</h3>
          <p>Generated anchor mask, source frame, and overlay. Useful for catching white/empty starts early.</p>
        </div>
        {!anchorPreview ? (
          <p className="muted">Use “Auto-Anchor Preview” to generate and inspect the initial subject mask before starting the job.</p>
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
      </section>

      <PathBrowserModal
        open={picker !== null}
        mode={picker?.mode ?? "file"}
        title={picker?.title ?? "Browse"}
        initialPath={picker && typeof form[picker.target] === "string" ? String(form[picker.target]) : ""}
        onClose={() => setPicker(null)}
        onPick={(path) => {
          if (picker) {
            update(picker.target, path);
          }
          setPicker(null);
        }}
      />
    </div>
  );
}
