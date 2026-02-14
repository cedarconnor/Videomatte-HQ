"""Simple Job Queue for managing background pipeline tasks."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from videomatte_hq.config import VideoMatteConfig

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    config: VideoMatteConfig
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    log_file: Optional[Path] = None
    process: Optional[asyncio.subprocess.Process] = None


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.current_job: Optional[Job] = None
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def submit(self, config: VideoMatteConfig) -> str:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, config=config)
        self.jobs[job_id] = job
        await self.queue.put(job_id)
        return job_id

    async def cancel(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return

        if job.status == JobStatus.RUNNING and job.process:
            job.process.terminate()
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
        elif job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self.jobs.values())

    async def _worker_loop(self):
        while True:
            job_id = await self.queue.get()
            job = self.jobs[job_id]

            if job.status == JobStatus.CANCELLED:
                self.queue.task_done()
                continue

            self.current_job = job
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Setup logging
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            job.log_file = log_dir / f"{job_id}.log"

            try:
                # Construct command from config
                # We'll write a temp config file and point the CLI to it
                temp_config_path = log_dir / f"{job_id}.yaml"
                job.config.to_yaml(temp_config_path)

                cmd = [
                    "python", "-m", "videomatte_hq.cli",
                    "--config", str(temp_config_path),
                ]

                # Run process
                with open(job.log_file, "w") as log_f:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=log_f,
                        stderr=log_f
                    )
                    job.process = process
                    await process.wait()

                if job.status == JobStatus.CANCELLED:
                    pass  # Status already set in cancel()
                elif process.returncode == 0:
                    job.status = JobStatus.COMPLETED
                else:
                    job.status = JobStatus.FAILED
                    job.error = f"Process exited with code {process.returncode}"

            except Exception as e:
                logger.exception(f"Job {job_id} failed")
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                job.completed_at = datetime.now()
                self.current_job = None
                self.queue.task_done()
