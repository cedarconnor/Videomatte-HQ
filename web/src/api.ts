import type {
  BrowseResponse,
  JobRecord,
  PointPromptPreviewResponse,
  PreflightResponse,
  PreviewResponse,
  QcInfoResponse,
  VideoInfoResponse,
  VideoMatteConfigForm
} from "./types";

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = String(body.detail ?? body.error ?? detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function getInputSuggestions(): Promise<string[]> {
  const res = await fetch("/api/fs/input-suggestions");
  const data = await parseJson<{ status: string; paths: string[] }>(res);
  return Array.isArray(data.paths) ? data.paths : [];
}

export async function pickNative(
  mode: "file" | "dir",
  title: string,
  initialDir = "",
  fileTypes?: [string, string][]
): Promise<string | null> {
  const res = await fetch("/api/fs/pick-native", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, title, initial_dir: initialDir, file_types: fileTypes ?? null })
  });
  const data = await parseJson<{ status: string; path: string }>(res);
  return data.status === "ok" && data.path ? data.path : null;
}

export async function browseFs(path: string | null, mode: "any" | "file" | "dir"): Promise<BrowseResponse> {
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  params.set("mode", mode);
  const res = await fetch(`/api/fs/browse?${params.toString()}`);
  return parseJson<BrowseResponse>(res);
}

export async function preflight(
  config: Partial<VideoMatteConfigForm>,
  auto_anchor: boolean | null,
  allow_external_paths = false
): Promise<PreflightResponse> {
  const res = await fetch("/api/preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, auto_anchor, allow_external_paths })
  });
  return parseJson<PreflightResponse>(res);
}

export async function previewAutoAnchor(config: Partial<VideoMatteConfigForm>): Promise<PreviewResponse> {
  const res = await fetch("/api/anchor/auto-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config })
  });
  return parseJson<PreviewResponse>(res);
}

export async function submitJob(
  config: Partial<VideoMatteConfigForm>,
  opts: { auto_anchor: boolean | null; allow_external_paths?: boolean; verbose: boolean }
): Promise<{ status: string; id: string }> {
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, ...opts })
  });
  return parseJson<{ status: string; id: string }>(res);
}

export async function listJobs(): Promise<JobRecord[]> {
  const res = await fetch("/api/jobs");
  return parseJson<JobRecord[]>(res);
}

export async function getJob(jobId: string): Promise<JobRecord> {
  const res = await fetch(`/api/jobs/${jobId}`);
  return parseJson<JobRecord>(res);
}

export async function getJobLogs(jobId: string, tailChars = 12000): Promise<string> {
  const params = new URLSearchParams({ tail_chars: String(tailChars) });
  const res = await fetch(`/api/jobs/${jobId}/logs?${params.toString()}`);
  const data = await parseJson<{ logs: string }>(res);
  return data.logs ?? "";
}

export async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  await parseJson<{ status: string }>(res);
}

export async function getQcInfo(jobId?: string | null): Promise<QcInfoResponse> {
  const qs = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
  const res = await fetch(`/api/qc/info${qs}`);
  return parseJson<QcInfoResponse>(res);
}

export function qcPreviewUrl(jobId: string, frame: number, kind: "input" | "alpha" | "trimap"): string {
  const params = new URLSearchParams({
    job_id: jobId,
    frame: String(frame),
    kind
  });
  return `/api/qc/frame-preview?${params.toString()}`;
}

export async function getVideoInfo(inputPath: string): Promise<VideoInfoResponse> {
  const params = new URLSearchParams({ input_path: inputPath });
  const res = await fetch(`/api/video/info?${params.toString()}`);
  return parseJson<VideoInfoResponse>(res);
}

export function pointPickerFrameUrl(inputPath: string, frame: number, frameStart = 0, maxLongSide = 1280): string {
  const params = new URLSearchParams({
    input_path: inputPath,
    frame: String(frame),
    frame_start: String(frameStart),
    max_long_side: String(maxLongSide),
  });
  return `/api/point-picker/frame?${params.toString()}`;
}

export async function previewPointPrompt(
  config: Partial<VideoMatteConfigForm>,
  frameIndex: number,
  positivePoints: [number, number][],
  negativePoints: [number, number][],
): Promise<PointPromptPreviewResponse> {
  const res = await fetch("/api/point-prompt/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config,
      frame_index: frameIndex,
      positive_points: positivePoints,
      negative_points: negativePoints,
    }),
  });
  return parseJson<PointPromptPreviewResponse>(res);
}
