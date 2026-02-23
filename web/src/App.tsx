import { useEffect, useMemo, useState } from "react";
import { listJobs } from "./api";
import { JobsPage } from "./pages/JobsPage";
import { QCPage } from "./pages/QCPage";
import { RunPage } from "./pages/RunPage";
import { SettingsPage } from "./pages/SettingsPage";
import type { JobRecord, UiPreferences } from "./types";
import { loadUiPreferences, saveUiPreferences } from "./uiPrefs";

type TabId = "run" | "jobs" | "qc" | "settings";

export default function App() {
  const [tab, setTab] = useState<TabId>("run");
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uiPrefs, setUiPrefs] = useState<UiPreferences>(() => loadUiPreferences());

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const data = await listJobs();
        if (!cancelled) {
          setJobs(data);
          setError(null);
          if (!selectedJobId && data.length > 0) {
            setSelectedJobId(data[0].id);
          }
        }
      } catch (e) {
        if (!cancelled) setError(String((e as Error).message || e));
      }
    }
    void refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedJobId]);

  const runningJob = useMemo(() => jobs.find((j) => j.status === "running") ?? null, [jobs]);

  function handleSavePrefs(next: UiPreferences) {
    setUiPrefs(next);
    saveUiPreferences(next);
  }

  const frontendUrl = typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:5173";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">VH</div>
          <div>
            <h1>Videomatte-HQ v2</h1>
            <p>SAM3 + MEMatte local UI</p>
          </div>
        </div>
        <div className="header-meta">
          {runningJob ? (
            <div className="run-pill">
              <span className="dot" />
              Running: {runningJob.id.slice(0, 8)}
              {typeof runningJob.progress_percent === "number" && (
                <span>· {runningJob.progress_percent.toFixed(0)}%</span>
              )}
            </div>
          ) : (
            <div className="run-pill idle">No active job</div>
          )}
          {error && <div className="error-pill">{error}</div>}
        </div>
      </header>

      <div className="app-main">
        <aside className="sidebar">
          <nav>
            <TabButton id="run" tab={tab} setTab={setTab} label="Run" />
            <TabButton id="jobs" tab={tab} setTab={setTab} label={`Jobs (${jobs.length})`} />
            <TabButton id="qc" tab={tab} setTab={setTab} label="QC" />
            <TabButton id="settings" tab={tab} setTab={setTab} label="Settings" />
          </nav>
          <div className="sidebar-foot">
            <p>Backend API: <code>127.0.0.1:8000</code></p>
            <p>Frontend: <code>{frontendUrl}</code></p>
          </div>
        </aside>

        <main className="content">
          {tab === "run" && (
            <RunPage
              uiPrefs={uiPrefs}
              onJobQueued={(jobId) => {
                setSelectedJobId(jobId);
                setTab("jobs");
              }}
            />
          )}
          {tab === "jobs" && (
            <JobsPage jobs={jobs} selectedJobId={selectedJobId} onSelectJob={setSelectedJobId} />
          )}
          {tab === "qc" && (
            <QCPage jobs={jobs} selectedJobId={selectedJobId} onSelectJob={setSelectedJobId} />
          )}
          {tab === "settings" && <SettingsPage prefs={uiPrefs} onSave={handleSavePrefs} />}
        </main>
      </div>
    </div>
  );
}

function TabButton({
  id,
  tab,
  setTab,
  label
}: {
  id: TabId;
  tab: TabId;
  setTab: (v: TabId) => void;
  label: string;
}) {
  return (
    <button type="button" className={`tab-btn ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>
      {label}
    </button>
  );
}
