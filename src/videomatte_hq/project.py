"""Project state and keyframe mask asset helpers for Option B."""

from __future__ import annotations

import bisect
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from videomatte_hq.config import VideoMatteConfig


PROJECT_VERSION = 1
PROJECT_FILENAME = "project.vmhqproj"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class KeyframeAssignment(ProjectBaseModel):
    frame: int
    mask_asset: str
    source: str = "user"
    kind: Literal["initial", "correction"] = "initial"
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)


class ProjectState(ProjectBaseModel):
    version: int = PROJECT_VERSION
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)
    input_path: str
    frame_start: int
    frame_end: int
    output_dir: str
    keyframes: list[KeyframeAssignment] = Field(default_factory=list)

    def get_assignment(self, frame: int) -> KeyframeAssignment | None:
        for item in self.keyframes:
            if item.frame == frame:
                return item
        return None

    def upsert_assignment(self, assignment: KeyframeAssignment) -> None:
        for idx, item in enumerate(self.keyframes):
            if item.frame == assignment.frame:
                self.keyframes[idx] = assignment
                self.updated_at = _utc_now_iso()
                return
        self.keyframes.append(assignment)
        self.keyframes.sort(key=lambda x: x.frame)
        self.updated_at = _utc_now_iso()


def resolve_project_path(cfg: VideoMatteConfig) -> Path:
    """Resolve project file path from config."""

    if cfg.project.path:
        return Path(cfg.project.path)
    return Path(cfg.io.output_dir) / PROJECT_FILENAME


def create_project(cfg: VideoMatteConfig) -> ProjectState:
    """Create initial project state from runtime config."""

    return ProjectState(
        input_path=cfg.io.input,
        frame_start=cfg.io.frame_start,
        frame_end=cfg.io.frame_end,
        output_dir=cfg.io.output_dir,
    )


def load_project(project_path: Path) -> ProjectState | None:
    """Load a project if it exists."""

    if not project_path.exists():
        return None
    import json

    data = json.loads(project_path.read_text(encoding="utf-8"))
    return ProjectState(**data)


def save_project(project_path: Path, project: ProjectState) -> None:
    """Persist project state."""

    import json

    project.updated_at = _utc_now_iso()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    project_path.write_text(
        json.dumps(project.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def ensure_project(cfg: VideoMatteConfig) -> tuple[Path, ProjectState]:
    """Load project if present, otherwise create and save it."""

    project_path = resolve_project_path(cfg)
    project = load_project(project_path)
    if project is None:
        project = create_project(cfg)
        if cfg.project.autosave:
            save_project(project_path, project)
    return project_path, project


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize arbitrary mask input to float32 alpha in [0, 1]."""

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[..., 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    if mask.dtype == np.uint8:
        out = mask.astype(np.float32) / 255.0
    elif mask.dtype == np.uint16:
        out = mask.astype(np.float32) / 65535.0
    else:
        out = mask.astype(np.float32)
        if out.max() > 1.0:
            out = out / max(out.max(), 1.0)

    return np.clip(out, 0.0, 1.0)


def _mask_asset_rel_path(cfg: VideoMatteConfig, frame: int) -> str:
    return str(Path(cfg.project.masks_dir) / f"keyframe_{frame:06d}.png")


def import_keyframe_mask(
    cfg: VideoMatteConfig,
    project_path: Path,
    project: ProjectState,
    frame: int,
    mask_path: Path,
    source: str = "user",
    kind: Literal["initial", "correction"] = "initial",
) -> KeyframeAssignment:
    """Import/copy a keyframe mask into project assets and upsert assignment."""

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask image: {mask_path}")

    if mask.ndim == 3:
        # OpenCV returns BGR(A) for regular image loads.
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    alpha = _normalize_mask(mask)

    mask_rel = _mask_asset_rel_path(cfg, frame)
    mask_abs = project_path.parent / mask_rel
    mask_abs.parent.mkdir(parents=True, exist_ok=True)

    # Store canonical assignment mask as 16-bit grayscale PNG.
    alpha_u16 = (alpha * 65535.0).round().astype(np.uint16)
    ok = cv2.imwrite(str(mask_abs), alpha_u16)
    if not ok:
        raise IOError(f"Failed to write project mask asset: {mask_abs}")

    existing = project.get_assignment(frame)
    assignment = KeyframeAssignment(
        frame=frame,
        mask_asset=mask_rel,
        source=source,
        kind=kind,
        created_at=(existing.created_at if existing else _utc_now_iso()),
        updated_at=_utc_now_iso(),
    )
    project.upsert_assignment(assignment)

    if cfg.project.autosave:
        save_project(project_path, project)

    return assignment


def load_keyframe_masks(
    project_path: Path,
    project: ProjectState,
    target_shape: tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Load all keyframe masks as normalized float alpha arrays."""

    masks: dict[int, np.ndarray] = {}
    for item in project.keyframes:
        abs_path = project_path.parent / item.mask_asset
        if not abs_path.exists():
            continue
        raw = cv2.imread(str(abs_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        alpha = _normalize_mask(raw)
        if target_shape is not None and alpha.shape[:2] != target_shape:
            w = target_shape[1]
            h = target_shape[0]
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)
        masks[item.frame] = alpha.astype(np.float32)
    return masks


def suggest_reprocess_range(
    project: ProjectState,
    anchor_frame: int,
    memory_window: int,
    clip_start: int = 0,
    clip_end: int = -1,
) -> tuple[int, int]:
    """Suggest a reprocess range for a newly added correction anchor.

    Strategy:
    - Use neighboring keyframes (midpoint split) to bound the likely impacted span.
    - Expand toward clip edges by half the memory window when a neighbor is missing.
    """
    frames = sorted({int(item.frame) for item in project.keyframes})
    if not frames:
        return max(clip_start, anchor_frame), clip_end

    if anchor_frame in frames:
        idx = frames.index(anchor_frame)
    else:
        idx = bisect.bisect_left(frames, anchor_frame)
        frames.insert(idx, anchor_frame)

    prev_kf = frames[idx - 1] if idx > 0 else None
    next_kf = frames[idx + 1] if idx + 1 < len(frames) else None
    half_window = max(1, int(memory_window) // 2)

    if prev_kf is None:
        start = anchor_frame - half_window
    else:
        start = (prev_kf + anchor_frame) // 2

    if next_kf is None:
        end = anchor_frame + half_window if clip_end >= 0 else -1
    else:
        end = (anchor_frame + next_kf) // 2

    start = max(int(clip_start), int(start))
    if clip_end >= 0 and end >= 0:
        end = min(int(clip_end), int(end))
    if end >= 0 and end < start:
        end = start
    return int(start), int(end)
