import { useEffect, useState } from "react";
import { cancelJob, getJobLogs } from "../api";
import type { JobRecord } from "../types";

interface Props {
  jobs: JobRecord[];
  selectedJobId: string | null;
  onSelectJob: (jobId: string | null) => void;
}

export function JobsPage({ jobs, selectedJobId, onSelectJob }: Props) {
  const selected = jobs.find((j) => j.id === selectedJobId) ?? null;
  const [logs, setLogs] = useState("");
  const [logError, setLogError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadLogs() {
      if (!selectedJobId) {
        setLogs("");
        return;
      }
      try {
        const text = await getJobLogs(selectedJobId);
        if (!cancelled) {
          setLogs(text);
          setLogError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setLogError(String((e as Error).message || e));
        }
      }
    }
    void loadLogs();
    const t = window.setInterval(loadLogs, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [selectedJobId]);

  const mematteNoTileFailureHint = (() => {
    if (!selected || selected.status !== "failed") return false;
    const needle = "mematte did not execute on any tiles";
    const errorText = (selected.error ?? "").toLowerCase();
    const logText = (logs ?? "").toLowerCase();
    return errorText.includes(needle) || logText.includes(needle);
  })();

  return (
    <div className="jobs-layout">
      <section className="panel">
        <div className="panel-head compact">
          <h2>Jobs</h2>
          <p>Queued and completed CLI runs.</p>
        </div>
        <div className="job-list">
          {jobs.length === 0 && <p className="muted">No jobs yet.</p>}
          {jobs.map((job) => (
            <button
              type="button"
              key={job.id}
              onClick={() => onSelectJob(job.id)}
              className={`job-row ${selectedJobId === job.id ? "selected" : ""}`}
            >
              <div className="job-row-top">
                <span className={`badge badge-${job.status}`}>{job.status}</span>
                <span className="job-time">{new Date(job.created_at).toLocaleTimeString()}</span>
              </div>
              <div className="job-id">{job.id}</div>
              <div className="job-meta">
                <span>{job.frame_start ?? "?"}..{job.frame_end ?? "?"}</span>
                <span>{job.refine_enabled ? "refine:on" : "refine:off"}</span>
                {typeof job.progress_percent === "number" && (
                  <span>{job.progress_percent.toFixed(0)}%</span>
                )}
              </div>
              {typeof job.progress_percent === "number" && (
                <div className="progress-wrap" aria-label="job progress">
                  <div className="progress-bar" style={{ width: `${Math.max(0, Math.min(100, job.progress_percent))}%` }} />
                </div>
              )}
            </button>
          ))}
        </div>
      </section>

      <section className="panel panel-main">
        <div className="panel-head compact">
          <h3>Job Detail</h3>
          {selected && selected.status === "running" && (
            <button className="danger" type="button" onClick={() => void cancelJob(selected.id)}>
              Cancel
            </button>
          )}
        </div>
        {!selected ? (
          <p className="muted">Select a job to inspect status and logs.</p>
        ) : (
          <>
            <dl className="kv-list">
              <div><dt>Status</dt><dd><span className={`badge badge-${selected.status}`}>{selected.status}</span></dd></div>
              <div><dt>Input</dt><dd className="small">{selected.input}</dd></div>
              <div><dt>Output</dt><dd className="small">{selected.output_dir}</dd></div>
              <div><dt>Return Code</dt><dd>{selected.return_code ?? "-"}</dd></div>
              <div><dt>Progress</dt><dd>{typeof selected.progress_percent === "number" ? `${selected.progress_percent.toFixed(1)}%` : "-"}</dd></div>
              <div><dt>Stage</dt><dd>{selected.progress_stage ?? "-"}</dd></div>
              <div><dt>Error</dt><dd className="small">{selected.error ?? "-"}</dd></div>
            </dl>
            {mematteNoTileFailureHint && (
              <div className="hint-box">
                <strong>MEMatte did not receive an unknown band to refine</strong>
                <p className="muted">
                  Stage 1/trimap thresholds produced an empty gray unknown region. Increase the <code>Trimap Refine Band</code> on the Run page (try <code>Wider</code> or <code>Maximum</code>), then re-run and inspect the QC trimap preview.
                </p>
                <p className="muted">
                  If the subject is very small or distant, also check the anchor mask and try a shorter frame range first.
                </p>
              </div>
            )}
            <div className="log-box-wrap">
              <div className="subhead">
                <span>Live Logs</span>
                {logError && <span className="status-error">{logError}</span>}
              </div>
              {selected.progress_message && <div className="muted">{selected.progress_message}</div>}
              <pre className="log-box">{logs || "(no logs yet)"}</pre>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
