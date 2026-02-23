"""Background job queue for the local web UI."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from videomatte_hq.config import VideoMatteConfig


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Job:
    id: str
    config: VideoMatteConfig
    cli_flags: list[str] = field(default_factory=list)
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    return_code: Optional[int] = None
    log_file: Optional[Path] = None
    config_file: Optional[Path] = None
    process: Optional[asyncio.subprocess.Process] = None
    cancel_requested: bool = False
    progress_stage: Optional[str] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    progress_percent: Optional[float] = None
    progress_message: Optional[str] = None


class JobManager:
    """Serial job runner that executes the existing CLI in subprocesses."""

    _RE_FRAME_PROGRESS = re.compile(r"\b(?:frame|frames?)\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
    _RE_SIMPLE_RATIO = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
    _RE_SEGMENT_CHUNK = re.compile(r"\bSegment chunk\s+(\d+)\.\.(\d+)\s+processed\b", re.IGNORECASE)

    def __init__(self, *, db_path: Path | None = None) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.current_job: Optional[Job] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._cli_python: str = self._resolve_cli_python()
        self._stop_event = asyncio.Event()
        self._db_path = Path(db_path) if db_path is not None else (self._log_dir() / "jobs.sqlite3")
        self._init_db()
        self._load_jobs_from_db()

    @staticmethod
    def _resolve_cli_python() -> str:
        repo_root = Path.cwd()
        venv_python = repo_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if venv_python.exists():
            return str(venv_python)
        return str(sys.executable)

    @staticmethod
    def _log_dir() -> Path:
        path = Path("logs") / "web_jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _db_connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._db_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    cli_flags_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    return_code INTEGER,
                    log_file TEXT,
                    config_file TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    progress_stage TEXT,
                    progress_current INTEGER,
                    progress_total INTEGER,
                    progress_percent REAL,
                    progress_message TEXT
                )
                """
            )

    @staticmethod
    def _dt_from_iso(raw: str | None) -> Optional[datetime]:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def _persist_job(self, job: Job) -> None:
        with self._db_connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, config_json, cli_flags_json, status, created_at, started_at, completed_at, error,
                    return_code, log_file, config_file, cancel_requested, progress_stage, progress_current,
                    progress_total, progress_percent, progress_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    config_json=excluded.config_json,
                    cli_flags_json=excluded.cli_flags_json,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    error=excluded.error,
                    return_code=excluded.return_code,
                    log_file=excluded.log_file,
                    config_file=excluded.config_file,
                    cancel_requested=excluded.cancel_requested,
                    progress_stage=excluded.progress_stage,
                    progress_current=excluded.progress_current,
                    progress_total=excluded.progress_total,
                    progress_percent=excluded.progress_percent,
                    progress_message=excluded.progress_message
                """,
                (
                    job.id,
                    json.dumps(job.config.to_dict(), ensure_ascii=True),
                    json.dumps(list(job.cli_flags), ensure_ascii=True),
                    str(job.status.value),
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.error,
                    job.return_code,
                    str(job.log_file) if job.log_file else None,
                    str(job.config_file) if job.config_file else None,
                    1 if job.cancel_requested else 0,
                    job.progress_stage,
                    job.progress_current,
                    job.progress_total,
                    job.progress_percent,
                    job.progress_message,
                ),
            )

    def _load_jobs_from_db(self) -> None:
        try:
            with self._db_connect() as conn:
                rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        except Exception:
            return
        for row in rows:
            try:
                cfg = VideoMatteConfig.from_dict(json.loads(str(row["config_json"])))
                cli_flags = json.loads(str(row["cli_flags_json"]))
                if not isinstance(cli_flags, list):
                    cli_flags = []
                raw_status = str(row["status"])
                try:
                    parsed_status = JobStatus(raw_status)
                except Exception:
                    suffix = raw_status.split(".", 1)[-1].lower()
                    parsed_status = JobStatus(suffix) if suffix in {s.value for s in JobStatus} else JobStatus.FAILED

                job = Job(
                    id=str(row["id"]),
                    config=cfg,
                    cli_flags=[str(v) for v in cli_flags],
                    status=parsed_status,
                    created_at=self._dt_from_iso(row["created_at"]) or datetime.now(),
                    started_at=self._dt_from_iso(row["started_at"]),
                    completed_at=self._dt_from_iso(row["completed_at"]),
                    error=row["error"],
                    return_code=row["return_code"],
                    log_file=Path(str(row["log_file"])) if row["log_file"] else None,
                    config_file=Path(str(row["config_file"])) if row["config_file"] else None,
                    cancel_requested=bool(int(row["cancel_requested"] or 0)),
                    progress_stage=row["progress_stage"],
                    progress_current=row["progress_current"],
                    progress_total=row["progress_total"],
                    progress_percent=row["progress_percent"],
                    progress_message=row["progress_message"],
                )
                # Any previously "running" jobs must be treated as failed after restart.
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    if not job.error:
                        job.error = "Server restarted while job was running."
                    job.completed_at = job.completed_at or datetime.now()
                self.jobs[job.id] = job
            except Exception:
                continue

    async def start(self) -> None:
        if self._worker_task is None:
            self._stop_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self.current_job and self.current_job.process and self.current_job.process.returncode is None:
            self.current_job.cancel_requested = True
            self.current_job.process.terminate()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def submit(self, config: VideoMatteConfig, *, cli_flags: list[str] | None = None) -> str:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, config=config, cli_flags=list(cli_flags or []))
        self.jobs[job_id] = job
        self._persist_job(job)
        await self.queue.put(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        if job.status == JobStatus.QUEUED:
            job.cancel_requested = True
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._persist_job(job)
            return
        if job.status == JobStatus.RUNNING and job.process and job.process.returncode is None:
            job.cancel_requested = True
            job.process.terminate()
            self._persist_job(job)

    def _job_paths(self, job_id: str) -> tuple[Path, Path]:
        base = self._log_dir()
        return (base / f"{job_id}.json", base / f"{job_id}.log")

    def _build_command(self, job: Job) -> list[str]:
        if job.config_file is None:
            raise RuntimeError("job.config_file must be set before building command")
        return [
            self._cli_python,
            "-m",
            "videomatte_hq.cli",
            "--config",
            str(job.config_file),
            *job.cli_flags,
        ]

    @staticmethod
    def _tail_log(log_file: Path, max_chars: int = 4000) -> str:
        try:
            if not log_file.exists():
                return ""
            max_chars = max(64, int(max_chars))
            max_bytes = max_chars * 4 + 256
            with log_file.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size <= 0:
                    return ""
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            return text[-max_chars:]
        except Exception:
            return ""

    @staticmethod
    def _job_total_frames(job: Job) -> int | None:
        start = int(job.config.frame_start)
        end = int(job.config.frame_end)
        if end < start:
            return None
        return max(0, end - start + 1)

    def _set_progress(
        self,
        job: Job,
        *,
        stage: str | None = None,
        current: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        changed = False
        if stage != job.progress_stage:
            job.progress_stage = stage
            changed = True
        if current != job.progress_current:
            job.progress_current = current
            changed = True
        if total != job.progress_total:
            job.progress_total = total
            changed = True
        if message is not None and message != job.progress_message:
            job.progress_message = message
            changed = True

        percent: float | None = None
        if total is not None and total > 0 and current is not None:
            percent = max(0.0, min(100.0, (float(current) / float(total)) * 100.0))
        if percent != job.progress_percent:
            job.progress_percent = percent
            changed = True

        if changed:
            self._persist_job(job)

    def _update_progress_from_log_line(self, job: Job, line: str) -> None:
        text = str(line).strip()
        if not text:
            return

        total_frames = self._job_total_frames(job)
        lower = text.lower()
        if "segment chunk" in lower:
            m = self._RE_SEGMENT_CHUNK.search(text)
            if m:
                end_idx = int(m.group(2))
                current = end_idx + 1
                if total_frames is not None:
                    current = min(current, total_frames)
                self._set_progress(job, stage="segment", current=current, total=total_frames, message=text)
                return

        m = self._RE_FRAME_PROGRESS.search(text) or self._RE_SIMPLE_RATIO.search(text)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            stage = "refine" if "refine" in lower else ("segment" if "segment" in lower else "running")
            self._set_progress(job, stage=stage, current=current, total=total, message=text)
            return

        if "auto-anchor" in lower:
            self._set_progress(job, stage="anchor", message=text)
            return

        if text.startswith("[job]"):
            self._set_progress(job, stage="starting", message=text)

    def _format_process_failure(self, exit_code: int, log_file: Path) -> str:
        tail = self._tail_log(log_file)
        lower = tail.lower()
        if any(h in lower for h in ("winerror 127", "c10_cuda.dll", "torchvision", "torch._c")):
            return (
                "Runtime dependency issue detected (PyTorch/CUDA/SAM). "
                f"CLI interpreter: {self._cli_python}. "
                "Check the project venv and GPU runtime."
            )
        if not tail.strip():
            return f"Process exited with code {exit_code} and produced no logs."
        return f"Process exited with code {exit_code}"

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                self.queue.task_done()
                continue
            if job.status == JobStatus.CANCELLED:
                self.queue.task_done()
                continue

            self.current_job = job
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._set_progress(job, stage="starting", current=0, total=self._job_total_frames(job), message="Queued job started.")
            cfg_path, log_path = self._job_paths(job.id)
            job.config_file = cfg_path
            job.log_file = log_path
            self._persist_job(job)

            try:
                job.config.to_file(cfg_path)
                cmd = self._build_command(job)
                exit_code = await self._run_subprocess(job, cmd, log_path)
                job.return_code = exit_code
                if job.cancel_requested or job.status == JobStatus.CANCELLED:
                    job.status = JobStatus.CANCELLED
                elif exit_code == 0:
                    job.status = JobStatus.COMPLETED
                    total = self._job_total_frames(job)
                    self._set_progress(
                        job,
                        stage="completed",
                        current=total if total is not None else job.progress_current,
                        total=total if total is not None else job.progress_total,
                        message="Completed.",
                    )
                else:
                    job.status = JobStatus.FAILED
                    job.error = self._format_process_failure(exit_code, log_path)
                    self._set_progress(job, stage="failed", message=job.error or f"Failed (exit {exit_code}).")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                self._set_progress(job, stage="failed", message=str(exc))
            finally:
                job.completed_at = datetime.now()
                job.process = None
                self.current_job = None
                self._persist_job(job)
                self.queue.task_done()

    async def _run_subprocess(self, job: Job, cmd: list[str], log_file: Path) -> int:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            f.write(f"[job] started_at={datetime.now().isoformat()}\n")
            f.write(f"[job] cmd={' '.join(cmd)}\n")
            f.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            job.process = proc
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                f.write(text)
                f.flush()
                self._update_progress_from_log_line(job, text)
            return int(await proc.wait())


__all__ = ["Job", "JobManager", "JobStatus"]
