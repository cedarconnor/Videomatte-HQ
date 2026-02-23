import type { UiPreferences } from "./types";

const KEY = "videomatte_hq_v2_ui_prefs";

export const DEFAULT_UI_PREFERENCES: UiPreferences = {
  default_output_dir: "output_ui",
  default_device: "cuda",
  default_precision: "fp16",
  default_mematte_repo_dir: "third_party/MEMatte",
  default_mematte_checkpoint: "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth"
};

export function loadUiPreferences(): UiPreferences {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_UI_PREFERENCES };
    const parsed = JSON.parse(raw) as Partial<UiPreferences>;
    return { ...DEFAULT_UI_PREFERENCES, ...parsed };
  } catch {
    return { ...DEFAULT_UI_PREFERENCES };
  }
}

export function saveUiPreferences(prefs: UiPreferences): void {
  window.localStorage.setItem(KEY, JSON.stringify(prefs));
}
