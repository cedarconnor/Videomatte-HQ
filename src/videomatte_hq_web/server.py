"""FastAPI Server definition."""

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.project import ensure_project, import_keyframe_mask, suggest_reprocess_range
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
