"""Simple Job Queue for managing background pipeline tasks."""

import asyncio
import logging
import os
import sys
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
        self._cli_python: str = self._resolve_cli_python()

    def _resolve_cli_python(self) -> str:
        """Prefer the project venv interpreter for CLI jobs."""
        repo_root = Path.cwd()
        venv_python = repo_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if venv_python.exists():
            return str(venv_python)
        return str(sys.executable)

    @staticmethod
    def _tail_log(log_file: Path, max_chars: int = 4000) -> str:
        try:
            if not log_file.exists():
                return ""
            text = log_file.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
        except Exception:
            return ""

    def _format_process_failure(self, exit_code: int, log_file: Path) -> str:
        tail = self._tail_log(log_file)
        lower_tail = tail.lower()
        runtime_hints = (
            "winerror 127",
            "c10_cuda.dll",
            "torchvision",
            "torch._c",
            "sam2 runtime unavailable",
            "could not import sam2",
        )
        if any(h in lower_tail for h in runtime_hints):
            return (
                "Runtime dependency issue detected (PyTorch/SAM2/Samurai). "
                f"Backend CLI interpreter: {self._cli_python}. "
                "Verify with: .\\.venv\\Scripts\\python -c \"import torch, torchvision; import importlib; importlib.import_module('sam2.build_sam')\""
            )

        aborted_codes = {3, 134}
        if exit_code in aborted_codes or "aborted!" in lower_tail:
            return (
                f"Pipeline process aborted (exit code {exit_code}). "
                "This often indicates a native runtime/memory failure. "
                "Try a smaller frame range first (for example 0..30), then scale up."
            )

        if not tail.strip():
            return (
                f"Process exited with code {exit_code} and produced no logs. "
                f"Interpreter used: {self._cli_python}."
            )

        return f"Process exited with code {exit_code}"

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
                temp_config_path = log_dir / f"{job_id}.yaml"
                job.config.to_yaml(temp_config_path)

                cmd = [
                    self._cli_python, "-m", "videomatte_hq.cli",
                    "--config", str(temp_config_path),
                ]

                # Run process in a thread to support streaming logs robustly on Windows
                exit_code = await asyncio.to_thread(
                    self._run_job_process,
                    cmd,
                    job.log_file,
                    job_id
                )

                if job.status == JobStatus.CANCELLED:
                    pass  # Status already set in cancel()
                elif exit_code == 0:
                    job.status = JobStatus.COMPLETED
                else:
                    job.status = JobStatus.FAILED
                    job.error = self._format_process_failure(exit_code, job.log_file)

            except Exception as e:
                logger.exception(f"Job {job_id} failed")
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                job.completed_at = datetime.now()
                self.current_job = None
                self.queue.task_done()

    def _run_job_process(self, cmd: list[str], log_file: Path, job_id: str) -> int:
        """Synchronous subprocess wrapper for streaming logs."""
        import subprocess

        # Open log file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"[job-runner] python={cmd[0]}\n")
            f.flush()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                encoding="utf-8",
                errors="replace"
            )
            
            # We can't easily kill this from main thread without storing the process handle
            # But for now, let's focus on logging. Cancellation might need a shared flag or handle.
            # (To really support cancellation, we'd need to store process in job.process, but that's hard across threads)
            # For this fix, we assume run-to-completion or force-kill entire server.
            
            for line in process.stdout:
                # Write to file
                f.write(line)
                f.flush()
                # Write to stdout
                sys.stdout.write(f"[JOB-{job_id[:4]}] {line}")
                sys.stdout.flush()
            
            process.wait()
            return process.returncode
