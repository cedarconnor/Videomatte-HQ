"""FastAPI Server definition."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq_web.jobs import JobManager, JobStatus

logger = logging.getLogger(__name__)

job_manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await job_manager.start()
    yield
    # Shutdown


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


# Serve files from project root for preview
# WARNING: This exposes the entire project directory. Safe for local use only.
logger.warning("Mounting project root at /files for local preview. Do not expose this server to the public internet.")
app.mount("/files", StaticFiles(directory="."), name="files")

# Static files (Frontend will go here later)
# app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
