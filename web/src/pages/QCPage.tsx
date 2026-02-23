import { useEffect, useMemo, useState } from "react";
import { getQcInfo, qcPreviewUrl } from "../api";
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

  const range = useMemo(() => {
    const start = qcInfo?.output.frame_start;
    const end = qcInfo?.output.frame_end;
    if (typeof start === "number" && typeof end === "number") {
      return { start, end, valid: end >= start };
    }
    return { start: 0, end: 0, valid: false };
  }, [qcInfo]);

  const jobId = qcInfo?.job_id ?? selectedJobId;
  const inputSrc = jobId ? `${qcPreviewUrl(jobId, frame, "input")}&v=${nonce}` : "";
  const alphaSrc = jobId ? `${qcPreviewUrl(jobId, frame, "alpha")}&v=${nonce}` : "";
  const trimapSrc = jobId ? `${qcPreviewUrl(jobId, frame, "trimap")}&v=${nonce}` : "";
  const trimapAvailable = Boolean(qcInfo?.output.trimap_available);

  return (
    <div className="panel-grid">
      <section className="panel panel-main">
        <div className="panel-head">
          <h2>Quality Control</h2>
          <p>Scrub frame pairs and inspect alpha output against the source frame.</p>
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
        </div>

        {error && <div className="status-strip"><span className="status-error">{error}</span></div>}

        {!jobId || !range.valid ? (
          <p className="muted">
            Select a completed job with written alpha frames. QC previews appear after at least one frame is rendered.
          </p>
        ) : (
          <>
            <div className="scrub-row">
              <button type="button" onClick={() => setFrame(Math.max(range.start, frame - 1))}>Prev</button>
              <input
                type="range"
                min={range.start}
                max={range.end}
                value={frame}
                onChange={(e) => setFrame(Number(e.target.value))}
              />
              <button type="button" onClick={() => setFrame(Math.min(range.end, frame + 1))}>Next</button>
              <label>
                Frame
                <input
                  type="number"
                  value={frame}
                  onChange={(e) => setFrame(Number(e.target.value))}
                  min={range.start}
                  max={range.end}
                  className="frame-input"
                />
              </label>
            </div>
            <WipeCompare leftSrc={inputSrc} rightSrc={alphaSrc} leftLabel="Input RGB" rightLabel="Alpha" />
            <p className="muted">
              Tip: drag the vertical divider in the wipe preview to reveal the alpha side if it looks fully dark at first.
            </p>
            {trimapAvailable && (
              <>
                <WipeCompare leftSrc={inputSrc} rightSrc={trimapSrc} leftLabel="Input RGB" rightLabel="Trimap" />
                <p className="muted">
                  Trimap preview shows Stage-2 inputs: black = background, white = foreground, gray = MEMatte unknown band.
                </p>
              </>
            )}
            <div className="preview-grid two">
              <figure>
                <img src={inputSrc} alt="input frame" />
                <figcaption>Input Frame</figcaption>
              </figure>
              <figure>
                <img src={alphaSrc} alt="alpha frame" />
                <figcaption>Alpha Preview</figcaption>
              </figure>
              {trimapAvailable && (
                <figure>
                  <img src={trimapSrc} alt="trimap frame" />
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
