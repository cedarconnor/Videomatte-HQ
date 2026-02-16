"""FastAPI Server definition."""

import base64
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.reader import FrameSource
from videomatte_hq.mask_builder import build_prompt_mask_grabcut
from videomatte_hq.prompt_mask_range import build_prompt_masks_range
from videomatte_hq.propagation_assist import (
    SUPPORTED_PROPAGATION_BACKENDS,
    propagate_masks_assist,
    select_propagation_frames,
)
from videomatte_hq.prompt_boxes import suggest_prompt_boxes
from videomatte_hq.sam_builder import build_prompt_mask_sam, DEFAULT_SAM_MODEL_ID
from videomatte_hq.utils.image import frame_to_rgb_u8
from videomatte_hq.project import (
    ensure_project,
    import_keyframe_mask,
    load_keyframe_masks,
    suggest_reprocess_range,
    upsert_keyframe_alpha,
)
from videomatte_hq_web.jobs import JobManager, JobStatus

logger = logging.getLogger(__name__)

job_manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await job_manager.start()
    yield
    # Shutdown



# Custom Log Filter to suppress polling noise
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /api/jobs") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(title="VideoMatte-HQ Web API", lifespan=lifespan)


# Allow CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobSubmitRequest(BaseModel):
    config: dict  # Full config dict matching VideoMatteConfig schema


class ProjectStateRequest(BaseModel):
    config: dict


class ImportAssignmentRequest(BaseModel):
    config: dict
    frame: int = 0
    mask_path: str = Field(min_length=1)
    source: str = "ui"
    kind: str = "initial"


class SuggestReprocessRangeRequest(BaseModel):
    config: dict
    frame: int = 0


class PromptPoint(BaseModel):
    x: float
    y: float


class PromptBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class AssignmentFramePreviewRequest(BaseModel):
    config: dict
    frame: int = 0


class BuildAssignmentMaskRequest(BaseModel):
    config: dict
    frame: int = 0
    box: PromptBox
    fg_points: list[PromptPoint] = Field(default_factory=list)
    bg_points: list[PromptPoint] = Field(default_factory=list)
    backend: str = "grabcut"
    point_radius: int = Field(default=8, ge=1, le=128)
    iter_count: int = Field(default=5, ge=1, le=20)
    sam_model_id: str = DEFAULT_SAM_MODEL_ID
    sam_local_files_only: bool = True
    sam_fallback_to_grabcut: bool = True
    source: str = "ui_builder"
    kind: str = "initial"


class BuildAssignmentMaskRangeRequest(BaseModel):
    config: dict
    anchor_frame: int = 0
    frame_start: int | None = None
    frame_end: int | None = None
    box: PromptBox
    fg_points: list[PromptPoint] = Field(default_factory=list)
    bg_points: list[PromptPoint] = Field(default_factory=list)
    backend: str = "sam"
    point_radius: int = Field(default=8, ge=1, le=128)
    iter_count: int = Field(default=5, ge=1, le=20)
    sam_model_id: str = DEFAULT_SAM_MODEL_ID
    sam_local_files_only: bool = True
    sam_fallback_to_grabcut: bool = False
    samurai_model_cfg: str = ""
    samurai_checkpoint: str = ""
    samurai_offload_video_to_cpu: bool = False
    samurai_offload_state_to_cpu: bool = False
    track_prompts_with_flow: bool = False
    track_bg_points_with_flow: bool = False
    flow_downscale: float = Field(default=0.5, ge=0.15, le=1.0)
    save_stride: int = Field(default=1, ge=1, le=300)
    kind: str = "correction"
    source: str = "ui_builder_range"
    overwrite_existing: bool = False


class SuggestAssignmentBoxesRequest(BaseModel):
    config: dict
    frame: int = 0
    prompt: str = Field(default="person", min_length=1)
    max_candidates: int = Field(default=5, ge=1, le=12)


class PropagateAssignmentMasksRequest(BaseModel):
    config: dict
    anchor_frame: int = 0
    frame_start: int | None = None
    frame_end: int | None = None
    backend: str = "flow"
    fallback_to_flow: bool = True
    stride: int = Field(default=8, ge=1, le=300)
    max_new_keyframes: int = Field(default=24, ge=1, le=1000)
    flow_downscale: float = Field(default=0.5, ge=0.15, le=1.0)
    flow_min_coverage: float = Field(default=0.002, ge=0.0, le=1.0)
    flow_max_coverage: float = Field(default=0.98, ge=0.0, le=1.0)
    flow_feather_px: int = Field(default=1, ge=0, le=32)
    samurai_model_cfg: str = ""
    samurai_checkpoint: str = ""
    samurai_offload_video_to_cpu: bool = False
    samurai_offload_state_to_cpu: bool = False
    kind: str = "correction"
    source: str = "ui_propagate"
    overwrite_existing: bool = False


def _build_project_summary(cfg: VideoMatteConfig) -> dict:
    project_path, project = ensure_project(cfg)
    keyframes = [
        {
            "frame": item.frame,
            "mask_asset": item.mask_asset,
            "source": item.source,
            "kind": item.kind,
            "updated_at": item.updated_at,
        }
        for item in project.keyframes
    ]
    return {
        "project_path": str(project_path),
        "keyframe_count": len(keyframes),
        "keyframes": keyframes,
        "require_assignment": bool(cfg.assignment.require_assignment),
    }


def _resolve_local_frame_index(cfg: VideoMatteConfig, source: FrameSource, requested_frame: int) -> int:
    """Map a requested absolute frame number to a local source index."""
    num_frames = int(source.num_frames)
    if num_frames <= 0:
        raise ValueError("Input has no frames.")

    start = int(getattr(cfg.io, "frame_start", 0))
    candidates = [int(requested_frame) - start, int(requested_frame)]
    seen: set[int] = set()
    for idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)
        if 0 <= idx < num_frames:
            return idx
    raise ValueError(
        f"Requested frame {requested_frame} is outside loaded range "
        f"[{start}:{start + num_frames - 1}]"
    )


def _load_input_frame_rgb_u8(cfg: VideoMatteConfig, requested_frame: int) -> tuple[np.ndarray, int]:
    source = FrameSource(
        pattern=cfg.io.input,
        frame_start=cfg.io.frame_start,
        frame_end=cfg.io.frame_end,
        prefetch_workers=0,
    )
    try:
        local_idx = _resolve_local_frame_index(cfg=cfg, source=source, requested_frame=requested_frame)
        rgb_u8 = _frame_to_rgb_u8(source[local_idx], error_context="mask builder")
        return rgb_u8, local_idx
    finally:
        source.close()


def _frame_to_rgb_u8(frame: np.ndarray, error_context: str = "frame IO") -> np.ndarray:
    return frame_to_rgb_u8(frame, error_context=error_context)


def _png_data_url_from_gray_u8(gray: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise ValueError("Failed to encode grayscale preview PNG.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _png_data_url_from_rgb_u8(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to encode frame preview PNG.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@app.get("/api/jobs")
async def list_jobs():
    jobs = job_manager.list_jobs()
    return [
        {
            "id": j.id,
            "status": j.status,
            "created_at": j.created_at,
            "started_at": j.started_at,
            "completed_at": j.completed_at,
            "error": j.error,
        }
        for j in jobs
    ]


@app.post("/api/jobs")
async def submit_job(req: JobSubmitRequest):
    # Parse config
    try:
        # Pydantic validation
        cfg = VideoMatteConfig(**req.config)
        job_id = await job_manager.submit(cfg)
        return {"id": job_id, "status": "queued"}
    except Exception as e:
        logger.exception("Failed to submit job")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/project/state")
async def project_state(req: ProjectStateRequest):
    """Resolve project and list current keyframe assignments."""
    try:
        cfg = VideoMatteConfig(**req.config)
        return _build_project_summary(cfg)
    except Exception as e:
        logger.exception("Failed to resolve project state")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/import")
async def import_assignment(req: ImportAssignmentRequest):
    """Import a keyframe mask from filesystem into project assets."""
    try:
        cfg = VideoMatteConfig(**req.config)
        project_path, project = ensure_project(cfg)
        assignment = import_keyframe_mask(
            cfg=cfg,
            project_path=project_path,
            project=project,
            frame=req.frame,
            mask_path=Path(req.mask_path),
            source=req.source,
            kind=req.kind if req.kind in ("initial", "correction") else "initial",
        )
        suggested_start, suggested_end = suggest_reprocess_range(
            project=project,
            anchor_frame=req.frame,
            memory_window=cfg.memory.window,
            clip_start=cfg.io.frame_start,
            clip_end=cfg.io.frame_end,
        )
        summary = _build_project_summary(cfg)
        return {
            "status": "ok",
            "assignment": {
                "frame": assignment.frame,
                "mask_asset": assignment.mask_asset,
                "source": assignment.source,
                "kind": assignment.kind,
                "updated_at": assignment.updated_at,
            },
            "suggested_reprocess_range": {
                "frame_start": suggested_start,
                "frame_end": suggested_end,
                "reason": "neighbor_midpoint_window",
            },
            **summary,
        }
    except Exception as e:
        logger.exception("Failed to import assignment")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/suggest-range")
async def suggest_assignment_range(req: SuggestReprocessRangeRequest):
    """Suggest a reprocess span for a correction anchor frame."""
    try:
        cfg = VideoMatteConfig(**req.config)
        _project_path, project = ensure_project(cfg)
        start, end = suggest_reprocess_range(
            project=project,
            anchor_frame=req.frame,
            memory_window=cfg.memory.window,
            clip_start=cfg.io.frame_start,
            clip_end=cfg.io.frame_end,
        )
        return {
            "status": "ok",
            "frame_start": start,
            "frame_end": end,
            "reason": "neighbor_midpoint_window",
        }
    except Exception as e:
        logger.exception("Failed to suggest assignment range")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/frame-preview")
async def assignment_frame_preview(req: AssignmentFramePreviewRequest):
    """Load a frame for interactive assignment mask building."""
    try:
        cfg = VideoMatteConfig(**req.config)
        frame_rgb_u8, local_idx = _load_input_frame_rgb_u8(cfg=cfg, requested_frame=req.frame)
        h, w = frame_rgb_u8.shape[:2]
        return {
            "status": "ok",
            "frame": int(req.frame),
            "local_index": int(local_idx),
            "width": int(w),
            "height": int(h),
            "data_url": _png_data_url_from_rgb_u8(frame_rgb_u8),
        }
    except Exception as e:
        logger.exception("Failed to load assignment frame preview")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/build-mask")
async def build_assignment_mask(req: BuildAssignmentMaskRequest):
    """Build and import a keyframe mask from box/point prompts."""
    try:
        cfg = VideoMatteConfig(**req.config)
        frame_rgb_u8, local_idx = _load_input_frame_rgb_u8(cfg=cfg, requested_frame=req.frame)
        fg_points = [(p.x, p.y) for p in req.fg_points]
        bg_points = [(p.x, p.y) for p in req.bg_points]
        box_xyxy = (req.box.x0, req.box.y0, req.box.x1, req.box.y1)

        backend_requested = str(req.backend).strip().lower() or "grabcut"
        backend_used = backend_requested
        builder_note: str | None = None
        if backend_requested in {"grabcut", "classic"}:
            alpha = build_prompt_mask_grabcut(
                frame_rgb_u8=frame_rgb_u8,
                box_xyxy=box_xyxy,
                fg_points=fg_points,
                bg_points=bg_points,
                point_radius=req.point_radius,
                iter_count=req.iter_count,
            )
            backend_used = "grabcut"
        elif backend_requested in {"sam", "sam_hq", "segment_anything"}:
            try:
                alpha = build_prompt_mask_sam(
                    frame_rgb_u8=frame_rgb_u8,
                    box_xyxy=box_xyxy,
                    fg_points=fg_points,
                    bg_points=bg_points,
                    model_id=req.sam_model_id or DEFAULT_SAM_MODEL_ID,
                    local_files_only=req.sam_local_files_only,
                    device_hint=cfg.runtime.device,
                    point_radius=req.point_radius,
                )
                backend_used = "sam"
            except Exception as sam_exc:
                if not req.sam_fallback_to_grabcut:
                    raise
                logger.warning("SAM backend unavailable, falling back to GrabCut: %s", sam_exc)
                alpha = build_prompt_mask_grabcut(
                    frame_rgb_u8=frame_rgb_u8,
                    box_xyxy=box_xyxy,
                    fg_points=fg_points,
                    bg_points=bg_points,
                    point_radius=req.point_radius,
                    iter_count=req.iter_count,
                )
                backend_used = "grabcut_fallback"
                builder_note = f"SAM unavailable: {sam_exc}. Used GrabCut fallback."
        else:
            raise ValueError(f"Unsupported mask builder backend: {req.backend}")

        project_path, project = ensure_project(cfg)
        assignment = upsert_keyframe_alpha(
            cfg=cfg,
            project_path=project_path,
            project=project,
            frame=req.frame,
            alpha=alpha,
            source=req.source,
            kind=req.kind if req.kind in ("initial", "correction") else "initial",
        )
        suggested_start, suggested_end = suggest_reprocess_range(
            project=project,
            anchor_frame=req.frame,
            memory_window=cfg.memory.window,
            clip_start=cfg.io.frame_start,
            clip_end=cfg.io.frame_end,
        )
        summary = _build_project_summary(cfg)

        mask_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        coverage = float(mask_u8.mean()) / 255.0
        return {
            "status": "ok",
            "frame": int(req.frame),
            "local_index": int(local_idx),
            "coverage": coverage,
            "backend_requested": backend_requested,
            "backend_used": backend_used,
            "builder_note": builder_note,
            "assignment": {
                "frame": assignment.frame,
                "mask_asset": assignment.mask_asset,
                "source": assignment.source,
                "kind": assignment.kind,
                "updated_at": assignment.updated_at,
            },
            "suggested_reprocess_range": {
                "frame_start": suggested_start,
                "frame_end": suggested_end,
                "reason": "neighbor_midpoint_window",
            },
            "mask_preview_data_url": _png_data_url_from_gray_u8(mask_u8),
            **summary,
        }
    except Exception as e:
        logger.exception("Failed to build assignment mask")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/build-mask-range")
async def build_assignment_mask_range(req: BuildAssignmentMaskRangeRequest):
    """Build prompt masks across a frame range and import as keyframe assignments."""
    try:
        cfg = VideoMatteConfig(**req.config)
        project_path, project = ensure_project(cfg)

        source = FrameSource(
            pattern=cfg.io.input,
            frame_start=cfg.io.frame_start,
            frame_end=cfg.io.frame_end,
            prefetch_workers=0,
        )
        try:
            num_frames = int(source.num_frames)
            if num_frames <= 0:
                raise ValueError("Input has no frames for range mask building.")

            clip_abs_start = int(cfg.io.frame_start)
            clip_abs_end = int(cfg.io.frame_end) if int(cfg.io.frame_end) >= 0 else (clip_abs_start + num_frames - 1)

            anchor_local = _resolve_local_frame_index(cfg=cfg, source=source, requested_frame=req.anchor_frame)
            anchor_abs = int(clip_abs_start + anchor_local)

            start_abs = clip_abs_start if (req.frame_start is None or int(req.frame_start) < 0) else int(req.frame_start)
            end_abs = clip_abs_end if (req.frame_end is None or int(req.frame_end) < 0) else int(req.frame_end)
            start_abs = max(clip_abs_start, min(clip_abs_end, start_abs))
            end_abs = max(clip_abs_start, min(clip_abs_end, end_abs))
            if end_abs < start_abs:
                raise ValueError(f"Invalid range: {start_abs}..{end_abs}")

            local_start = int(start_abs - clip_abs_start)
            local_end = int(end_abs - clip_abs_start)
            if anchor_local < local_start or anchor_local > local_end:
                raise ValueError(
                    f"Anchor frame {anchor_abs} is outside selected range {start_abs}..{end_abs}."
                )

            def _load_local_rgb_u8(local_idx: int) -> np.ndarray:
                if local_idx < local_start or local_idx > local_end:
                    raise ValueError(f"Local frame {local_idx} outside selected range {local_start}..{local_end}.")
                return _frame_to_rgb_u8(source[int(local_idx)], error_context="range mask builder")

            fg_points = [(p.x, p.y) for p in req.fg_points]
            bg_points = [(p.x, p.y) for p in req.bg_points]
            box_xyxy = (req.box.x0, req.box.y0, req.box.x1, req.box.y1)

            result = build_prompt_masks_range(
                frame_loader=_load_local_rgb_u8,
                frame_start=local_start,
                frame_end=local_end,
                anchor_frame=anchor_local,
                box_xyxy=box_xyxy,
                fg_points=fg_points,
                bg_points=bg_points,
                backend=req.backend,
                point_radius=int(req.point_radius),
                iter_count=int(req.iter_count),
                sam_model_id=req.sam_model_id or DEFAULT_SAM_MODEL_ID,
                sam_local_files_only=bool(req.sam_local_files_only),
                sam_fallback_to_grabcut=bool(req.sam_fallback_to_grabcut),
                samurai_model_cfg=str(req.samurai_model_cfg or ""),
                samurai_checkpoint=str(req.samurai_checkpoint or ""),
                samurai_offload_video_to_cpu=bool(req.samurai_offload_video_to_cpu),
                samurai_offload_state_to_cpu=bool(req.samurai_offload_state_to_cpu),
                track_prompts_with_flow=bool(req.track_prompts_with_flow),
                track_bg_points_with_flow=bool(req.track_bg_points_with_flow),
                flow_downscale=float(req.flow_downscale),
                device_hint=cfg.runtime.device,
            )

            stride = max(1, int(req.save_stride))
            selected_local = [idx for idx in range(local_start, local_end + 1) if (idx - local_start) % stride == 0]
            if anchor_local not in selected_local:
                selected_local.append(anchor_local)
            if local_end not in selected_local:
                selected_local.append(local_end)
            selected_local = sorted(set(selected_local))

            normalized_kind = req.kind if req.kind in ("initial", "correction") else "correction"
            existing_frames = {int(item.frame) for item in project.keyframes}
            inserted_frames: list[int] = []
            skipped_existing_frames: list[int] = []
            inserted_coverage: list[float] = []

            for local_idx in selected_local:
                alpha = result.masks.get(int(local_idx))
                if alpha is None:
                    continue
                abs_frame = int(clip_abs_start + int(local_idx))
                if abs_frame in existing_frames and not bool(req.overwrite_existing):
                    skipped_existing_frames.append(abs_frame)
                    continue
                alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
                coverage = float(alpha.mean())
                assignment = upsert_keyframe_alpha(
                    cfg=cfg,
                    project_path=project_path,
                    project=project,
                    frame=abs_frame,
                    alpha=alpha,
                    source=f"{req.source}:{result.backend_used}",
                    kind=normalized_kind,  # type: ignore[arg-type]
                )
                inserted_frames.append(int(assignment.frame))
                inserted_coverage.append(coverage)
                existing_frames.add(abs_frame)

            if inserted_frames:
                suggested_start = int(max(clip_abs_start, min(inserted_frames)))
                suggested_end = int(min(clip_abs_end, max(inserted_frames)))
                suggestion_reason = "prompt_mask_range_span"
            else:
                suggested_start, suggested_end = suggest_reprocess_range(
                    project=project,
                    anchor_frame=anchor_abs,
                    memory_window=cfg.memory.window,
                    clip_start=cfg.io.frame_start,
                    clip_end=cfg.io.frame_end,
                )
                suggestion_reason = "neighbor_midpoint_window"

            summary = _build_project_summary(cfg)
            return {
                "status": "ok",
                "anchor_frame": int(anchor_abs),
                "frame_start": int(start_abs),
                "frame_end": int(end_abs),
                "backend_requested": str(req.backend).strip().lower() or "sam",
                "backend_used": result.backend_used,
                "builder_note": result.note,
                "track_prompts_with_flow": bool(req.track_prompts_with_flow),
                "track_bg_points_with_flow": bool(req.track_bg_points_with_flow),
                "save_stride": int(stride),
                "selected_local_frames": [int(x) for x in selected_local],
                "selected_frames": [int(clip_abs_start + int(x)) for x in selected_local],
                "inserted_frames": [int(x) for x in inserted_frames],
                "skipped_existing_frames": [int(x) for x in skipped_existing_frames],
                "inserted_count": len(inserted_frames),
                "mean_inserted_coverage": (
                    float(np.mean(inserted_coverage)) if inserted_coverage else 0.0
                ),
                "suggested_reprocess_range": {
                    "frame_start": int(suggested_start),
                    "frame_end": int(suggested_end),
                    "reason": suggestion_reason,
                },
                **summary,
            }
        finally:
            source.close()
    except Exception as e:
        logger.exception("Failed to build assignment mask range")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/suggest-boxes")
async def suggest_assignment_boxes(req: SuggestAssignmentBoxesRequest):
    """Suggest assignment boxes from a text prompt on the selected frame."""
    try:
        cfg = VideoMatteConfig(**req.config)
        frame_rgb_u8, local_idx = _load_input_frame_rgb_u8(cfg=cfg, requested_frame=req.frame)
        h, w = frame_rgb_u8.shape[:2]
        cands = suggest_prompt_boxes(
            frame_rgb_u8=frame_rgb_u8,
            prompt=req.prompt,
            max_candidates=req.max_candidates,
        )
        return {
            "status": "ok",
            "frame": int(req.frame),
            "local_index": int(local_idx),
            "width": int(w),
            "height": int(h),
            "prompt": req.prompt,
            "candidates": [c.as_dict() for c in cands],
        }
    except Exception as e:
        logger.exception("Failed to suggest assignment boxes")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/assignments/propagate")
async def propagate_assignment_masks(req: PropagateAssignmentMasksRequest):
    """Create additional keyframe masks by propagating from one anchor assignment."""
    try:
        cfg = VideoMatteConfig(**req.config)
        project_path, project = ensure_project(cfg)
        if not project.keyframes:
            raise ValueError("No keyframe assignments exist. Import or build one keyframe first.")

        source = FrameSource(
            pattern=cfg.io.input,
            frame_start=cfg.io.frame_start,
            frame_end=cfg.io.frame_end,
            prefetch_workers=0,
        )
        try:
            num_frames = int(source.num_frames)
            if num_frames <= 0:
                raise ValueError("Input has no frames.")

            clip_abs_start = int(cfg.io.frame_start)
            clip_abs_end = int(cfg.io.frame_end) if int(cfg.io.frame_end) >= 0 else (clip_abs_start + num_frames - 1)

            anchor_local = _resolve_local_frame_index(cfg=cfg, source=source, requested_frame=req.anchor_frame)
            anchor_abs_candidates = [int(req.anchor_frame), clip_abs_start + int(anchor_local)]
            anchor_abs_frame: int | None = None
            for candidate in anchor_abs_candidates:
                if project.get_assignment(candidate) is not None:
                    anchor_abs_frame = int(candidate)
                    break
            if anchor_abs_frame is None:
                raise ValueError(
                    f"Anchor frame {req.anchor_frame} has no imported assignment. "
                    "Use Import Mask or Build + Import Mask first."
                )

            start_abs = clip_abs_start if req.frame_start is None else int(req.frame_start)
            end_abs = clip_abs_end if req.frame_end is None else int(req.frame_end)
            start_abs = max(clip_abs_start, min(clip_abs_end, start_abs))
            end_abs = max(clip_abs_start, min(clip_abs_end, end_abs))
            if end_abs < start_abs:
                raise ValueError(f"Invalid propagation range: {start_abs}..{end_abs}")

            local_start = int(start_abs - clip_abs_start)
            local_end = int(end_abs - clip_abs_start)
            if anchor_local < local_start or anchor_local > local_end:
                raise ValueError(
                    f"Anchor frame {anchor_abs_frame} is outside propagation range {start_abs}..{end_abs}."
                )

            anchor_frame_rgb_u8 = _frame_to_rgb_u8(source[int(anchor_local)], error_context="propagation assist")
            h, w = anchor_frame_rgb_u8.shape[:2]
            keyframe_masks = load_keyframe_masks(project_path, project, target_shape=(h, w))
            anchor_alpha = keyframe_masks.get(int(anchor_abs_frame))
            if anchor_alpha is None:
                raise ValueError(f"Failed to load anchor assignment mask for frame {anchor_abs_frame}.")

            def _load_local_rgb_u8(local_idx: int) -> np.ndarray:
                if local_idx < local_start or local_idx > local_end:
                    raise ValueError(f"Local frame {local_idx} is outside range {local_start}..{local_end}.")
                return _frame_to_rgb_u8(source[int(local_idx)], error_context="propagation assist")

            prop_result = propagate_masks_assist(
                frame_loader=_load_local_rgb_u8,
                frame_start=local_start,
                frame_end=local_end,
                anchor_frame=int(anchor_local),
                anchor_mask=anchor_alpha,
                backend=req.backend,
                fallback_to_flow=bool(req.fallback_to_flow),
                flow_downscale=float(req.flow_downscale),
                flow_min_coverage=float(req.flow_min_coverage),
                flow_max_coverage=float(req.flow_max_coverage),
                flow_feather_px=int(req.flow_feather_px),
                samurai_model_cfg=str(req.samurai_model_cfg or ""),
                samurai_checkpoint=str(req.samurai_checkpoint or ""),
                samurai_offload_video_to_cpu=bool(req.samurai_offload_video_to_cpu),
                samurai_offload_state_to_cpu=bool(req.samurai_offload_state_to_cpu),
                device_hint=str(cfg.runtime.device or "cuda"),
            )

            selected_local_frames = select_propagation_frames(
                frame_start=local_start,
                frame_end=local_end,
                anchor_frame=int(anchor_local),
                stride=int(req.stride),
                max_new_keyframes=int(req.max_new_keyframes),
            )

            normalized_kind = req.kind if req.kind in ("initial", "correction") else "correction"
            existing_frames = {int(item.frame) for item in project.keyframes}
            inserted_frames: list[int] = []
            skipped_existing_frames: list[int] = []
            inserted_coverage: list[float] = []

            for local_idx in selected_local_frames:
                abs_frame = int(clip_abs_start + int(local_idx))
                if abs_frame == int(anchor_abs_frame):
                    continue
                if abs_frame in existing_frames and not bool(req.overwrite_existing):
                    skipped_existing_frames.append(abs_frame)
                    continue

                alpha = prop_result.masks.get(int(local_idx))
                if alpha is None:
                    continue
                alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
                coverage = float(alpha.mean())
                if coverage < float(req.flow_min_coverage):
                    continue

                assignment = upsert_keyframe_alpha(
                    cfg=cfg,
                    project_path=project_path,
                    project=project,
                    frame=abs_frame,
                    alpha=alpha,
                    source=f"{req.source}:{prop_result.backend_used}",
                    kind=normalized_kind,  # type: ignore[arg-type]
                )
                inserted_frames.append(int(assignment.frame))
                inserted_coverage.append(coverage)
                existing_frames.add(abs_frame)

            if inserted_frames:
                all_frames = [int(anchor_abs_frame)] + [int(f) for f in inserted_frames]
                suggested_start = max(clip_abs_start, min(all_frames))
                suggested_end = min(clip_abs_end, max(all_frames))
                suggestion_reason = "phase4_propagation_span"
            else:
                suggested_start, suggested_end = suggest_reprocess_range(
                    project=project,
                    anchor_frame=int(anchor_abs_frame),
                    memory_window=cfg.memory.window,
                    clip_start=cfg.io.frame_start,
                    clip_end=cfg.io.frame_end,
                )
                suggestion_reason = "neighbor_midpoint_window"

            summary = _build_project_summary(cfg)
            return {
                "status": "ok",
                "anchor_frame": int(anchor_abs_frame),
                "frame_start": int(start_abs),
                "frame_end": int(end_abs),
                "backend_requested": str(req.backend).strip().lower() or "flow",
                "backend_used": prop_result.backend_used,
                "builder_note": prop_result.note,
                "supported_backends": list(SUPPORTED_PROPAGATION_BACKENDS),
                "selected_local_frames": [int(x) for x in selected_local_frames],
                "selected_frames": [int(clip_abs_start + int(x)) for x in selected_local_frames],
                "inserted_frames": [int(x) for x in inserted_frames],
                "skipped_existing_frames": [int(x) for x in skipped_existing_frames],
                "inserted_count": len(inserted_frames),
                "mean_inserted_coverage": (
                    float(np.mean(inserted_coverage)) if inserted_coverage else 0.0
                ),
                "suggested_reprocess_range": {
                    "frame_start": int(suggested_start),
                    "frame_end": int(suggested_end),
                    "reason": suggestion_reason,
                },
                **summary,
            }
        finally:
            source.close()
    except Exception as e:
        logger.exception("Failed to propagate assignment masks")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "config": job.config.model_dump(),
        "log_file": str(job.log_file) if job.log_file else None,
        "error": job.error,
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    await job_manager.cancel(job_id)
    return {"status": "cancelled"}


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    job = job_manager.get_job(job_id)
    if not job or not job.log_file or not job.log_file.exists():
        return {"logs": ""}
    return {"logs": job.log_file.read_text()}


@app.get("/api/qc/info")
async def qc_info():
    """Return available input/output frame info for QC tab."""
    result = {
        "input": {
            "pattern": None,
            "count": 0,
            "padding": 5,
            "prefix": "frame_",
            "ext": "png",
            "dir": "input_frames",
        },
        "output": {
            "pattern": None,
            "count": 0,
            "padding": 6,
            "prefix": "",
            "ext": "png",
            "dir": "out/alpha",
        },
    }

    root = Path(".").resolve()
    image_exts = {".png", ".exr", ".jpg", ".jpeg", ".tif", ".tiff"}

    def _pattern_hint(pattern: str) -> tuple[Path, str | None, int | None, str | None]:
        p = Path(pattern)
        directory = p.parent if str(p.parent) not in ("", ".") else Path(".")
        name = p.name
        m = re.match(r"^(.*)%0?(\d+)d\.(\w+)$", name)
        if m:
            return directory, m.group(1), int(m.group(2)), m.group(3)
        return directory, None, None, None

    def _scan(directory: Path, prefix: str | None, padding: int | None, ext: str | None) -> dict | None:
        target = directory if directory.is_absolute() else (root / directory)
        if not target.exists() or not target.is_dir():
            return None

        files = [f for f in target.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
        if ext is not None:
            files = [f for f in files if f.suffix.lower() == f".{ext.lower()}"]
        if prefix is not None:
            files = [f for f in files if f.name.startswith(prefix)]
        if not files:
            return None

        files = sorted(files)
        first = files[0]
        resolved_prefix = prefix
        resolved_padding = padding
        resolved_ext = ext

        if resolved_prefix is None or resolved_padding is None or resolved_ext is None:
            m = re.match(r"^(.*?)(\d+)\.(\w+)$", first.name)
            if m:
                resolved_prefix = m.group(1)
                resolved_padding = len(m.group(2))
                resolved_ext = m.group(3)
            else:
                resolved_prefix = first.stem
                resolved_padding = 0
                resolved_ext = first.suffix.lstrip(".")

        try:
            rel_dir = target.resolve().relative_to(root).as_posix()
        except Exception:
            return None

        return {
            "count": len(files),
            "prefix": resolved_prefix,
            "padding": resolved_padding,
            "ext": resolved_ext,
            "dir": rel_dir,
            "pattern": first.name,
        }

    input_patterns: list[str] = []
    output_patterns: list[str] = []
    for job in sorted(job_manager.list_jobs(), key=lambda j: j.created_at, reverse=True):
        cfg = job.config
        if cfg.io.input:
            input_patterns.append(cfg.io.input)
        if cfg.io.output_alpha:
            output_patterns.append(str(Path(cfg.io.output_dir) / cfg.io.output_alpha))

    input_patterns.extend(
        [
            "input_frames/%05d.png",
            "input_frames/frame_%05d.png",
            "input_frames/*.png",
        ]
    )
    output_patterns.extend(
        [
            "out/alpha/%06d.png",
            "output/alpha/frame_%05d.png",
            "output/alpha/%06d.png",
        ]
    )

    for pattern in input_patterns:
        directory, prefix, padding, ext = _pattern_hint(pattern)
        scanned = _scan(directory, prefix, padding, ext)
        if scanned:
            result["input"].update(scanned)
            break

    for pattern in output_patterns:
        directory, prefix, padding, ext = _pattern_hint(pattern)
        scanned = _scan(directory, prefix, padding, ext)
        if scanned:
            result["output"].update(scanned)
            break

    return result


# Serve files from project root for preview
# WARNING: This exposes the entire project directory. Safe for local use only.
logger.warning("Mounting project root at /files for local preview. Do not expose this server to the public internet.")
app.mount("/files", StaticFiles(directory="."), name="files")

# Static files (Frontend will go here later)
# app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
