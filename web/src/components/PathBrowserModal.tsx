import { useEffect, useState } from "react";
import { browseFs } from "../api";
import type { BrowseEntry } from "../types";

interface Props {
  open: boolean;
  mode: "file" | "dir";
  title: string;
  initialPath: string;
  onClose: () => void;
  onPick: (path: string) => void;
}

export function PathBrowserModal({ open, mode, title, initialPath, onClose, onPick }: Props) {
  const [currentPath, setCurrentPath] = useState<string>(initialPath || "");
  const [resolvedPath, setResolvedPath] = useState<string>("");
  const [parent, setParent] = useState<string | null>(null);
  const [roots, setRoots] = useState<BrowseEntry[]>([]);
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function load(path: string | null) {
    setBusy(true);
    setError(null);
    try {
      const res = await browseFs(path, mode);
      setResolvedPath(res.current);
      setCurrentPath(res.current);
      setParent(res.parent);
      setRoots(res.roots);
      setEntries(res.entries);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!open) return;
    void load(initialPath || null);
  }, [open, initialPath, mode]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-shell" onClick={(e) => e.stopPropagation()}>
        <div className="panel-head compact">
          <h3>{title}</h3>
          <button type="button" onClick={onClose}>Close</button>
        </div>

        <div className="path-browser-toolbar">
          <input
            value={currentPath}
            onChange={(e) => setCurrentPath(e.target.value)}
            placeholder="Enter a directory path"
          />
          <button type="button" onClick={() => void load(currentPath)} disabled={busy}>
            {busy ? "Loading..." : "Go"}
          </button>
          {parent && (
            <button type="button" onClick={() => void load(parent)} disabled={busy}>
              Up
            </button>
          )}
        </div>

        <div className="path-browser-layout">
          <div className="path-browser-roots">
            <div className="subhead"><span>Roots</span></div>
            {roots.map((r) => (
              <button key={r.path} type="button" className="path-entry" onClick={() => void load(r.path)}>
                {r.path}
              </button>
            ))}
          </div>

          <div className="path-browser-entries">
            <div className="subhead">
              <span>{resolvedPath || currentPath || "Browse"}</span>
              {mode === "dir" && resolvedPath && (
                <button type="button" className="primary" onClick={() => onPick(resolvedPath)}>
                  Use This Folder
                </button>
              )}
            </div>
            {error && <div className="status-strip"><span className="status-error">{error}</span></div>}
            <div className="entry-list">
              {entries.map((entry) => (
                <div key={entry.path} className="entry-row">
                  <button
                    type="button"
                    className={`path-entry ${entry.is_dir ? "dir" : "file"}`}
                    onClick={() => {
                      if (entry.is_dir) {
                        void load(entry.path);
                      } else if (mode === "file") {
                        onPick(entry.path);
                      }
                    }}
                  >
                    <span className="entry-icon">{entry.is_dir ? "DIR" : "FILE"}</span>
                    <span className="entry-name">{entry.name || entry.path}</span>
                  </button>
                  {mode === "dir" && entry.is_dir && (
                    <button type="button" onClick={() => onPick(entry.path)}>Select</button>
                  )}
                  {mode === "file" && entry.is_file && (
                    <button type="button" onClick={() => onPick(entry.path)}>Select</button>
                  )}
                </div>
              ))}
              {!busy && entries.length === 0 && <p className="muted">No entries found.</p>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
