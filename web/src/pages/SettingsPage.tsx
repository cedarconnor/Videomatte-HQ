import { useEffect, useState } from "react";
import type { UiPreferences } from "../types";

interface Props {
  prefs: UiPreferences;
  onSave: (prefs: UiPreferences) => void;
}

export function SettingsPage({ prefs, onSave }: Props) {
  const [draft, setDraft] = useState<UiPreferences>(prefs);
  const [status, setStatus] = useState<string>("Preferences are stored locally in your browser (localStorage).");

  useEffect(() => {
    setDraft(prefs);
  }, [prefs]);

  function update<K extends keyof UiPreferences>(key: K, value: UiPreferences[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="panel-grid">
      <section className="panel panel-main">
        <div className="panel-head">
          <h2>Settings</h2>
          <p>Saved locally for this browser. These defaults pre-fill the Run tab on next open.</p>
        </div>

        <div className="field-grid">
          <label>
            Default Output Directory
            <input value={draft.default_output_dir} onChange={(e) => update("default_output_dir", e.target.value)} />
          </label>
          <label>
            Default Device
            <select value={draft.default_device} onChange={(e) => update("default_device", e.target.value)}>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label>
            Default Precision
            <select value={draft.default_precision} onChange={(e) => update("default_precision", e.target.value)}>
              <option value="fp16">FP16</option>
              <option value="fp32">FP32</option>
            </select>
          </label>
          <label>
            Default MEMatte Repo Dir
            <input value={draft.default_mematte_repo_dir} onChange={(e) => update("default_mematte_repo_dir", e.target.value)} />
          </label>
          <label className="panel-span">
            Default MEMatte Checkpoint
            <input
              value={draft.default_mematte_checkpoint}
              onChange={(e) => update("default_mematte_checkpoint", e.target.value)}
            />
          </label>
        </div>

        <div className="action-row">
          <button
            className="primary"
            type="button"
            onClick={() => {
              onSave(draft);
              setStatus("Saved UI preferences to localStorage.");
            }}
          >
            Save Preferences
          </button>
          <button
            type="button"
            onClick={() => {
              setDraft(prefs);
              setStatus("Reverted unsaved changes.");
            }}
          >
            Revert
          </button>
        </div>

        <div className="status-strip">
          <span>{status}</span>
        </div>
      </section>

      <aside className="panel">
        <div className="panel-head compact">
          <h3>Notes</h3>
        </div>
        <div className="note-list">
          <div className="note-card">
            <h3>Local-only backend</h3>
            <p>
              The backend is intended for local use on <code>127.0.0.1</code>. It wraps the CLI and stores job metadata in
              SQLite under <code>logs/web_jobs</code>.
            </p>
          </div>
          <div className="note-card">
            <h3>MEMatte path policy</h3>
            <p>
              MEMatte repo/checkpoint must remain inside this repository. The backend uses the hardened CLI preflight checks.
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}
