import { useEffect, useMemo, useRef, useState } from "react";
import { cancelJob, getQcInfo, qcPreviewUrl } from "../api";
import { WipeCompare } from "../components/WipeCompare";
import type { JobRecord, QcInfoResponse } from "../types";

interface Props {
  jobs: JobRecord[];
  selectedJobId: string | null;
  onSelectJob: (jobId: string | null) => void;
}

export function QCPage({ jobs, selectedJobId, onSelectJob }: Props) {
  const [qcInfo, setQcInfo] = useState<QcInfoResponse | null>(null);
  const [frame, setFrame] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const [stride, setStride] = useState(1);
  const [zoom, setZoom] = useState(1);

  const selectedJob = useMemo(
    () => jobs.find((j) => j.id === (qcInfo?.job_id ?? selectedJobId)) ?? null,
    [jobs, qcInfo?.job_id, selectedJobId],
  );
  const isJobRunning = selectedJob?.status === "running";

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const info = await getQcInfo(selectedJobId);
        if (!cancelled) {
          setQcInfo(info);
          setError(null);
          if ((info.output.frame_start ?? null) !== null) {
            setFrame((prev) => {
              const lo = Number(info.output.frame_start);
              const hi = Number(info.output.frame_end ?? lo);
              return Math.max(lo, Math.min(prev || lo, hi));
            });
          }
        }
      } catch (e) {
        if (!cancelled) setError(String((e as Error).message || e));
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [selectedJobId, jobs.length]);

  /* Live polling while job is running */
  const lastCountRef = useRef<number | null>(null);
  useEffect(() => {
    if (!isJobRunning || !selectedJobId) return;
    const id = setInterval(async () => {
      try {
        const info = await getQcInfo(selectedJobId);
        setQcInfo(info);
        setError(null);
        const count = info.output.count ?? 0;
        if (lastCountRef.current !== null && count > lastCountRef.current) {
          setNonce((n) => n + 1);
        }
        lastCountRef.current = count;
        /* expand slider range without resetting user scrub position */
        if ((info.output.frame_start ?? null) !== null) {
          setFrame((prev) => {
            const lo = Number(info.output.frame_start);
            const hi = Number(info.output.frame_end ?? lo);
            if (prev < lo) return lo;
            if (prev > hi) return hi;
            return prev;
          });
        }
      } catch {
        /* swallow polling errors silently */
      }
    }, 4000);
    return () => clearInterval(id);
  }, [isJobRunning, selectedJobId]);

  const range = useMemo(() => {
    const start = qcInfo?.output.frame_start;
    const end = qcInfo?.output.frame_end;
    if (typeof start === "number" && typeof end === "number") {
      return { start, end, valid: end >= start };
    }
    return { start: 0, end: 0, valid: false };
  }, [qcInfo]);

  const jobId = qcInfo?.job_id ?? selectedJobId;
  const cacheBust = `v=${nonce}&t=${Date.now()}`;
  const inputSrc = jobId ? `${qcPreviewUrl(jobId, frame, "input")}&${cacheBust}` : "";
  const alphaSrc = jobId ? `${qcPreviewUrl(jobId, frame, "alpha")}&${cacheBust}` : "";
  const trimapSrc = jobId ? `${qcPreviewUrl(jobId, frame, "trimap")}&${cacheBust}` : "";
  const trimapAvailable = Boolean(qcInfo?.output.trimap_available);

  return (
    <div className="panel-grid">
      <section className="panel panel-main">
        <div className="panel-head">
          <h2>Quality Control</h2>
          <p>Scrub frames, inspect alpha output, and monitor running jobs. Use stride to skip frames and zoom to inspect detail.</p>
        </div>
        <div className="inline-options">
          <label>
            Job
            <select
              value={selectedJobId ?? ""}
              onChange={(e) => onSelectJob(e.target.value || null)}
              className="compact-select"
            >
              <option value="">Latest</option>
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.id.slice(0, 8)} · {j.status}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => setNonce((n) => n + 1)}>Refresh Images</button>
          {isJobRunning && selectedJob && (
            <button className="danger" type="button" onClick={() => void cancelJob(selectedJob.id)}>
              Cancel Job
            </button>
          )}
        </div>

        {isJobRunning && qcInfo && (
          <div className="status-strip">
            <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}>
              <span className="dot" />
              Recording... {qcInfo.output.count ?? 0}
              {selectedJob?.progress_total ? ` / ${selectedJob.progress_total}` : ""} frames
            </span>
          </div>
        )}

        {error && <div className="status-strip"><span className="status-error">{error}</span></div>}

        {!jobId || !range.valid ? (
          <p className="muted">
            Select a completed job with written alpha frames. QC previews appear after at least one frame is rendered.
          </p>
        ) : (
          <>
            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "0.6rem" }}>
              <div className="choice-group">
                <span>Stride</span>
                {[1, 2, 5, 10, 25, 50].map((s) => (
                  <button
                    key={s}
                    className={stride === s ? "active" : ""}
                    onClick={() => {
                      setStride(s);
                      setFrame((f) => {
                        const aligned = range.start + Math.round((f - range.start) / s) * s;
                        return Math.max(range.start, Math.min(range.end, aligned));
                      });
                    }}
                  >
                    {s === 1 ? "1 (all)" : String(s)}
                  </button>
                ))}
              </div>
              <div className="choice-group">
                <span>Zoom</span>
                {[1, 2, 4].map((z) => (
                  <button key={z} className={zoom === z ? "active" : ""} onClick={() => setZoom(z)}>
                    {z === 1 ? "Fit" : `${z}x`}
                  </button>
                ))}
              </div>
              {stride > 1 && (
                <span className="muted" style={{ alignSelf: "center" }}>
                  Showing every {stride}{stride === 2 ? "nd" : stride === 3 ? "rd" : "th"} frame
                  ({Math.floor((range.end - range.start) / stride) + 1} of {range.end - range.start + 1})
                </span>
              )}
            </div>
            <div className="scrub-row">
              <button type="button" onClick={() => setFrame(Math.max(range.start, frame - stride))}>Prev</button>
              <input
                type="range"
                min={range.start}
                max={range.end}
                step={stride}
                value={frame}
                onChange={(e) => setFrame(Number(e.target.value))}
              />
              <button type="button" onClick={() => setFrame(Math.min(range.end, frame + stride))}>Next</button>
              <label>
                Frame
                <input
                  type="number"
                  value={frame}
                  onChange={(e) => setFrame(Number(e.target.value))}
                  min={range.start}
                  max={range.end}
                  step={stride}
                  className="frame-input"
                />
              </label>
            </div>
            <WipeCompare leftSrc={inputSrc} rightSrc={alphaSrc} leftLabel="Input RGB" rightLabel="Alpha" zoom={zoom} />
            <p className="muted">
              Tip: drag the vertical divider in the wipe preview to reveal the alpha side if it looks fully dark at first.
            </p>
            {trimapAvailable && (
              <>
                <WipeCompare leftSrc={inputSrc} rightSrc={trimapSrc} leftLabel="Input RGB" rightLabel="Trimap" zoom={zoom} />
                <p className="muted">
                  Trimap preview shows Stage-2 inputs: black = background, white = foreground, gray = MEMatte unknown band.
                </p>
              </>
            )}
            <div className="preview-grid two">
              <figure>
                <div style={zoom > 1 ? { overflow: "auto", maxHeight: "320px" } : undefined}>
                  <img
                    src={inputSrc}
                    alt="input frame"
                    style={zoom > 1 ? { maxHeight: "none", width: `${zoom * 100}%`, transform: "none" } : undefined}
                  />
                </div>
                <figcaption>Input Frame</figcaption>
              </figure>
              <figure>
                <div style={zoom > 1 ? { overflow: "auto", maxHeight: "320px" } : undefined}>
                  <img
                    src={alphaSrc}
                    alt="alpha frame"
                    style={zoom > 1 ? { maxHeight: "none", width: `${zoom * 100}%`, transform: "none" } : undefined}
                  />
                </div>
                <figcaption>Alpha Preview</figcaption>
              </figure>
              {trimapAvailable && (
                <figure>
                  <div style={zoom > 1 ? { overflow: "auto", maxHeight: "320px" } : undefined}>
                    <img
                      src={trimapSrc}
                      alt="trimap frame"
                      style={zoom > 1 ? { maxHeight: "none", width: `${zoom * 100}%`, transform: "none" } : undefined}
                    />
                  </div>
                  <figcaption>Trimap Preview</figcaption>
                </figure>
              )}
            </div>
          </>
        )}
      </section>

      <aside className="panel">
        <div className="panel-head compact">
          <h3>QC Info</h3>
        </div>
        {qcInfo ? (
          <dl className="kv-list">
            <div><dt>Input</dt><dd className="small">{qcInfo.input.source || "-"}</dd></div>
            <div><dt>Input Range</dt><dd>{qcInfo.input.frame_start ?? "-"}..{qcInfo.input.frame_end ?? "-"}</dd></div>
            <div><dt>Output Range</dt><dd>{qcInfo.output.frame_start ?? "-"}..{qcInfo.output.frame_end ?? "-"}</dd></div>
            <div><dt>Output Count</dt><dd>{qcInfo.output.count ?? 0}</dd></div>
            <div><dt>Alpha Format</dt><dd>{qcInfo.output.alpha_format || "-"}</dd></div>
            <div><dt>Trimap Preview</dt><dd>{String(Boolean(qcInfo.output.trimap_available))}</dd></div>
          </dl>
        ) : (
          <p className="muted">No QC metadata loaded yet.</p>
        )}
      </aside>
    </div>
  );
}
