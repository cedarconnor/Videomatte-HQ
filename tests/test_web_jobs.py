import asyncio
from pathlib import Path

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq_web.jobs import Job, JobManager


def test_job_manager_prefers_repo_venv_python(monkeypatch, tmp_path):
    repo = tmp_path
    venv_python = repo / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.chdir(repo)
    assert JobManager._resolve_cli_python() == str(venv_python)


def test_job_manager_builds_cli_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = JobManager()
    cfg = VideoMatteConfig(input="clip.mp4", output_dir="out")
    job = Job(id="job-1", config=cfg, cli_flags=["--auto-anchor", "--verbose"])
    cfg_path = Path("logs") / "web_jobs" / "job-1.json"
    job.config_file = cfg_path

    cmd = manager._build_command(job)

    assert cmd[:3] == [manager._cli_python, "-m", "videomatte_hq.cli"]
    assert "--config" in cmd
    assert str(cfg_path) in cmd
    assert cmd[-2:] == ["--auto-anchor", "--verbose"]


def test_job_manager_tail_log_returns_only_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "tail.log"
    log_file.write_text("A" * 10000 + "\nEND\n", encoding="utf-8")

    tail = JobManager._tail_log(log_file, max_chars=64)

    assert len(tail) <= 64
    assert tail.replace("\r\n", "\n").endswith("END\n")


def test_job_manager_persists_jobs_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = JobManager()
    cfg = VideoMatteConfig(input="clip.mp4", output_dir="out")

    job_id = asyncio.run(manager.submit(cfg))

    manager2 = JobManager()
    restored = manager2.get_job(job_id)

    assert restored is not None
    assert restored.id == job_id
    assert str(restored.config.input) == "clip.mp4"
    assert restored.status.value == "queued"


def test_job_manager_parses_segment_chunk_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = JobManager()
    cfg = VideoMatteConfig(input="clip.mp4", output_dir="out", frame_start=0, frame_end=99)
    job = Job(id="job-p", config=cfg)

    manager._update_progress_from_log_line(job, "Segment chunk 0..19 processed (reanchors=0).")

    assert job.progress_stage == "segment"
    assert job.progress_current == 20
    assert job.progress_total == 100
    assert job.progress_percent is not None and job.progress_percent > 0
