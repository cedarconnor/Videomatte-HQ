export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobRecord {
  id: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  return_code?: number | null;
  log_file?: string | null;
  config_file?: string | null;
  output_dir?: string;
  input?: string;
  frame_start?: number;
  frame_end?: number;
  refine_enabled?: boolean;
  progress_stage?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
  progress_percent?: number | null;
  progress_message?: string | null;
}

export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
  is_file: boolean;
}

export interface BrowseResponse {
  status: "ok";
  cwd: string;
  current: string;
  parent: string | null;
  mode: "any" | "file" | "dir";
  roots: BrowseEntry[];
  entries: BrowseEntry[];
}

export interface PreviewResponse {
  status: "ok";
  mask_path: string;
  method: string;
  probe_frame: number;
  requested_frame_start: number;
  effective_frame_start: number;
  mask_coverage: number;
  mask_preview_data_url: string;
  frame_preview_data_url: string;
  overlay_preview_data_url: string;
}

export interface PreflightResponse {
  status: "ok";
  input: string;
  output_dir: string;
  is_video_input: boolean;
  auto_anchor_effective: boolean;
  anchor_missing: boolean;
  anchor_required: boolean;
  frame_start: number;
  frame_end: number;
  refine_enabled: boolean;
  mematte_repo_dir: string;
  mematte_checkpoint: string;
  allow_external_paths?: boolean;
}

export interface QcInfoResponse {
  status: "ok";
  job_id: string | null;
  input: {
    source?: string;
    frame_start?: number;
    frame_end?: number;
    is_video?: boolean;
  };
  output: {
    output_dir?: string;
    alpha_pattern?: string;
    trimap_pattern?: string;
    alpha_format?: string;
    frame_start?: number | null;
    frame_end?: number | null;
    count?: number;
    trimap_available?: boolean;
  };
}

export interface VideoMatteConfigForm {
  input: string;
  output_dir: string;
  output_alpha: string;
  frame_start: number;
  frame_end: number;
  anchor_mask: string;
  segment_backend: string;
  sam3_model: string;
  chunk_size: number;
  chunk_overlap: number;
  refine_enabled: boolean;
  mematte_repo_dir: string;
  mematte_checkpoint: string;
  tile_size: number;
  tile_overlap: number;
  trimap_mode: string;
  trimap_erosion_px: number;
  trimap_dilation_px: number;
  trimap_fg_threshold: number;
  trimap_bg_threshold: number;
  trimap_fallback_band_px: number;
  device: string;
  precision: string;
  prompt_mode: "mask" | "points";
  point_prompts: Record<string, { positive: [number, number][]; negative: [number, number][] }>;
}

export interface VideoInfoResponse {
  status: "ok";
  frame_count: number;
  width: number;
  height: number;
  fps: number;
}

export interface PointPromptPreviewResponse {
  status: "ok";
  frame_index: number;
  mask_coverage: number;
  mask_preview_data_url: string;
  overlay_preview_data_url: string;
}

export interface UiPreferences {
  default_output_dir: string;
  default_device: string;
  default_precision: string;
  default_mematte_repo_dir: string;
  default_mematte_checkpoint: string;
}
