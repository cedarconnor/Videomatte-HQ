"""Local FastAPI backend runtime for the Videomatte-HQ v2 UI."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import logging
from contextlib import asynccontextmanager
import os
from pathlib import Path
import string
from typing import Any, Literal

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
import numpy as np
from pydantic import BaseModel, Field

from videomatte_hq.cli import _looks_like_video_input, _run_preflight_checks
from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.reader import FrameSource
from videomatte_hq.prompts.auto_anchor import build_auto_anchor_mask_for_video
from videomatte_hq.utils.image import frame_to_rgb_u8
from videomatte_hq_web.jobs import JobManager, JobStatus

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
FRONTEND_DEV_URL_FILE = Path("logs") / "web_frontend_dev_url.txt"
QC_TRIMAP_DIR = "qc"
QC_TRIMAP_PATTERN = "trimap.%06d.png"

job_manager = JobManager()


class ConfigRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class PreflightRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    auto_anchor: bool | None = None
    allow_external_paths: bool = False


class AutoAnchorPreviewRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None


class JobSubmitRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    auto_anchor: bool | None = None
    allow_external_paths: bool = False
    verbose: bool = False


class PathInfoRequest(BaseModel):
    path: str = Field(min_length=1)


class PreviewCache:
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max(16, int(max_items))
        self._items: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> bytes | None:
        val = self._items.get(key)
        if val is None:
            return None
        self._items.move_to_end(key)
        return val

    def put(self, key: tuple[Any, ...], value: bytes) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


_preview_cache = PreviewCache(max_items=96)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await job_manager.start()
    try:
        yield
    finally:
        await job_manager.shutdown()


def _config_from_payload(payload: dict[str, Any]) -> VideoMatteConfig:
    return VideoMatteConfig.from_dict(payload)


def _png_bytes_from_gray(gray_u8: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", np.asarray(gray_u8, dtype=np.uint8))
    if not ok:
        raise RuntimeError("Failed to encode preview PNG.")
    return bytes(enc.tobytes())


def _png_bytes_from_rgb(rgb_u8: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(np.asarray(rgb_u8, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode preview PNG.")
    return bytes(enc.tobytes())


def _png_data_url_from_gray(gray_u8: np.ndarray) -> str:
    payload = base64.b64encode(_png_bytes_from_gray(gray_u8)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _png_data_url_from_rgb(rgb_u8: np.ndarray) -> str:
    payload = base64.b64encode(_png_bytes_from_rgb(rgb_u8)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _normalize_gray_preview_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.clip(arr.astype(np.float32) * (255.0 / 65535.0), 0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        denom = float(np.iinfo(arr.dtype).max)
        return np.clip(arr.astype(np.float32) * (255.0 / max(1.0, denom)), 0, 255).astype(np.uint8)
    arr_f = arr.astype(np.float32)
    if arr_f.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if float(arr_f.max()) <= 1.0:
        arr_f = arr_f * 255.0
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def _load_video_frame_rgb_u8(cfg: VideoMatteConfig, frame_abs: int) -> np.ndarray:
    source = FrameSource(
        pattern=cfg.input,
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        prefetch_workers=0,
    )
    try:
        local_idx = int(frame_abs) - int(cfg.frame_start)
        if local_idx < 0 or local_idx >= len(source):
            raise ValueError(
                f"Requested frame {frame_abs} is outside loaded range "
                f"[{cfg.frame_start}:{int(cfg.frame_start) + len(source) - 1}]"
            )
        return frame_to_rgb_u8(source[local_idx], error_context="web preview")
    finally:
        source.close()


def _load_video_frame_rgb_u8_by_args(input_path: str, frame_start: int, frame_end: int, frame_abs: int) -> np.ndarray:
    cfg = VideoMatteConfig(input=input_path, frame_start=int(frame_start), frame_end=int(frame_end))
    return _load_video_frame_rgb_u8(cfg, int(frame_abs))


def _alpha_output_frame_path(cfg: VideoMatteConfig, frame_idx: int) -> Path:
    try:
        rel = str(cfg.output_alpha) % int(frame_idx)
    except TypeError:
        rel = str(cfg.output_alpha).format(int(frame_idx))
    return Path(str(cfg.output_dir)) / rel


def _trimap_output_frame_path(cfg: VideoMatteConfig, frame_idx: int) -> Path:
    return Path(str(cfg.output_dir)) / QC_TRIMAP_DIR / (QC_TRIMAP_PATTERN % int(frame_idx))


def _job_cli_flags_from_request(req: JobSubmitRequest) -> list[str]:
    flags: list[str] = []
    if req.auto_anchor is True:
        flags.append("--auto-anchor")
    elif req.auto_anchor is False:
        flags.append("--no-auto-anchor")
    if req.verbose:
        flags.append("--verbose")
    if bool(req.allow_external_paths):
        flags.append("--allow-external-paths")
    return flags


def _job_to_json(job) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": str(getattr(job.status, "value", job.status)),
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "return_code": job.return_code,
        "log_file": str(job.log_file) if job.log_file else None,
        "config_file": str(job.config_file) if job.config_file else None,
        "output_dir": str(job.config.output_dir),
        "input": str(job.config.input),
        "frame_start": int(job.config.frame_start),
        "frame_end": int(job.config.frame_end),
        "refine_enabled": bool(job.config.refine_enabled),
        "progress_stage": job.progress_stage,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "progress_percent": job.progress_percent,
        "progress_message": job.progress_message,
    }


def _path_to_json(p: Path) -> dict[str, Any]:
    return {
        "name": p.name or str(p),
        "path": str(p.resolve()),
        "is_dir": p.is_dir(),
        "is_file": p.is_file(),
    }


def _filesystem_roots() -> list[Path]:
    if os.name == "nt":
        roots: list[Path] = []
        for letter in string.ascii_uppercase:
            candidate = Path(f"{letter}:\\")
            if candidate.exists():
                roots.append(candidate)
        if roots:
            return roots
        anchor = Path.cwd().anchor
        return [Path(anchor)] if anchor else [Path.cwd()]
    return [Path("/")]


def _browse_path(path: str | None, *, mode: str = "any", limit: int = 400) -> dict[str, Any]:
    if path:
        current = Path(path).expanduser()
        if not current.is_absolute():
            current = (Path.cwd() / current).resolve()
    else:
        current = Path.cwd()

    if not current.exists():
        raise FileNotFoundError(f"Browse path not found: {current}")
    if current.is_file():
        current = current.parent
    if not current.is_dir():
        raise ValueError(f"Browse path is not a directory: {current}")

    entries: list[dict[str, Any]] = []
    for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:limit]:
        if mode == "file" and not child.is_dir() and not child.is_file():
            continue
        if mode == "dir" and not child.is_dir():
            continue
        entries.append(_path_to_json(child))

    roots = [_path_to_json(r) for r in _filesystem_roots()]
    parent = current.parent if current.parent != current else None
    return {
        "status": "ok",
        "cwd": str(Path.cwd().resolve()),
        "current": str(current.resolve()),
        "parent": str(parent.resolve()) if parent is not None else None,
        "mode": mode,
        "roots": roots,
        "entries": entries,
    }


def _cached_qc_input_preview_png(cfg: VideoMatteConfig, frame: int) -> bytes:
    key = ("qc_input", str(cfg.input), int(cfg.frame_start), int(cfg.frame_end), int(frame))
    cached = _preview_cache.get(key)
    if cached is not None:
        return cached
    rgb_u8 = _load_video_frame_rgb_u8_by_args(str(cfg.input), int(cfg.frame_start), int(cfg.frame_end), int(frame))
    data = _png_bytes_from_rgb(rgb_u8)
    _preview_cache.put(key, data)
    return data


def _cached_qc_alpha_preview_png(alpha_path: Path) -> bytes:
    path = alpha_path.resolve()
    stat = path.stat()
    key = ("qc_alpha", str(path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _preview_cache.get(key)
    if cached is not None:
        return cached
    alpha_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if alpha_img is None:
        raise RuntimeError(f"Failed to read alpha frame: {path}")
    alpha_u8 = _normalize_gray_preview_u8(alpha_img)
    data = _png_bytes_from_gray(alpha_u8)
    _preview_cache.put(key, data)
    return data


def _cached_qc_trimap_preview_png(trimap_path: Path) -> bytes:
    path = trimap_path.resolve()
    stat = path.stat()
    key = ("qc_trimap", str(path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _preview_cache.get(key)
    if cached is not None:
        return cached
    tri_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if tri_img is None:
        raise RuntimeError(f"Failed to read trimap preview frame: {path}")
    tri_u8 = _normalize_gray_preview_u8(tri_img)
    data = _png_bytes_from_gray(tri_u8)
    _preview_cache.put(key, data)
    return data


def _recent_output_frame_indices(cfg: VideoMatteConfig, *, max_scan: int = 2000) -> tuple[int | None, int | None, int]:
    start = int(cfg.frame_start)
    end_cfg = int(cfg.frame_end)
    if end_cfg >= start and end_cfg >= 0:
        scan_end = end_cfg
    else:
        scan_end = start + max_scan - 1

    existing: list[int] = []
    for idx in range(start, scan_end + 1):
        if _alpha_output_frame_path(cfg, idx).exists():
            existing.append(idx)
    if not existing:
        return None, None, 0
    return min(existing), max(existing), len(existing)


def _frontend_dev_url_hint() -> str | None:
    try:
        if FRONTEND_DEV_URL_FILE.exists():
            text = FRONTEND_DEV_URL_FILE.read_text(encoding="utf-8", errors="replace").strip()
            return text or None
    except Exception:
        return None
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="Videomatte-HQ v2 Web API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "cli_python": job_manager._cli_python, "frontend_dev_url_hint": _frontend_dev_url_hint()}

    @app.get("/api/fs/input-suggestions")
    async def input_suggestions() -> dict[str, Any]:
        root = Path.cwd() / "TestFiles"
        if not root.exists() or not root.is_dir():
            return {"status": "ok", "paths": []}
        allowed = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mxf"}
        paths = [
            str(p.resolve())
            for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in allowed
        ]
        return {"status": "ok", "paths": paths}

    @app.post("/api/fs/path-info")
    async def path_info(req: PathInfoRequest) -> dict[str, Any]:
        p = Path(req.path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        exists = p.exists()
        return {
            "status": "ok",
            "resolved_path": str(p),
            "exists": bool(exists),
            "is_file": bool(exists and p.is_file()),
            "is_dir": bool(exists and p.is_dir()),
        }

    @app.get("/api/fs/browse")
    async def fs_browse(
        path: str | None = Query(None),
        mode: Literal["any", "file", "dir"] = Query("any"),
        limit: int = Query(400, ge=1, le=2000),
    ) -> dict[str, Any]:
        try:
            return _browse_path(path, mode=str(mode), limit=int(limit))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/preflight")
    async def preflight(req: PreflightRequest) -> dict[str, Any]:
        try:
            cfg = _config_from_payload(req.config)
            _run_preflight_checks(cfg, allow_external_paths=bool(req.allow_external_paths))
            is_video = _looks_like_video_input(str(cfg.input))
            auto_anchor_default = is_video
            auto_anchor_effective = req.auto_anchor if req.auto_anchor is not None else auto_anchor_default
            anchor_missing = not str(cfg.anchor_mask).strip()
            anchor_required = bool(anchor_missing and not auto_anchor_effective)
            return {
                "status": "ok",
                "input": str(cfg.input),
                "output_dir": str(cfg.output_dir),
                "is_video_input": bool(is_video),
                "auto_anchor_effective": bool(auto_anchor_effective),
                "anchor_missing": bool(anchor_missing),
                "anchor_required": bool(anchor_required),
                "frame_start": int(cfg.frame_start),
                "frame_end": int(cfg.frame_end),
                "refine_enabled": bool(cfg.refine_enabled),
                "mematte_repo_dir": str(cfg.mematte_repo_dir),
                "mematte_checkpoint": str(cfg.mematte_checkpoint),
                "allow_external_paths": bool(req.allow_external_paths),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/anchor/auto-preview")
    async def auto_anchor_preview(req: AutoAnchorPreviewRequest) -> dict[str, Any]:
        try:
            cfg = _config_from_payload(req.config)
            if not _looks_like_video_input(str(cfg.input)):
                raise ValueError("Auto-anchor preview currently supports video file inputs only.")
            out_path = Path(req.output_path) if req.output_path else (Path(str(cfg.output_dir)) / "anchor_mask.auto.png")
            result = await asyncio.to_thread(
                build_auto_anchor_mask_for_video,
                cfg.input,
                out_path,
                device=str(cfg.device),
                frame_start=int(cfg.frame_start),
            )
            frame_rgb = await asyncio.to_thread(_load_video_frame_rgb_u8, cfg, int(result.probe_frame))
            mask_img = cv2.imread(str(result.mask_path), cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                raise RuntimeError(f"Failed to read generated anchor mask: {result.mask_path}")
            mask_u8 = _normalize_gray_preview_u8(mask_img)
            overlay = frame_rgb.copy()
            green = np.zeros_like(overlay)
            green[..., 1] = 255
            alpha = (mask_u8.astype(np.float32) / 255.0)[..., None] * 0.45
            overlay = np.clip((1.0 - alpha) * overlay + alpha * green, 0, 255).astype(np.uint8)
            return {
                "status": "ok",
                "mask_path": str(Path(result.mask_path).resolve()),
                "method": str(result.method),
                "probe_frame": int(result.probe_frame),
                "requested_frame_start": int(cfg.frame_start),
                "effective_frame_start": int(max(int(cfg.frame_start), int(result.probe_frame))),
                "mask_coverage": float(mask_u8.mean()) / 255.0,
                "mask_preview_data_url": _png_data_url_from_gray(mask_u8),
                "frame_preview_data_url": _png_data_url_from_rgb(frame_rgb),
                "overlay_preview_data_url": _png_data_url_from_rgb(overlay),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        return [_job_to_json(j) for j in job_manager.list_jobs()]

    @app.post("/api/jobs")
    async def submit_job(req: JobSubmitRequest) -> dict[str, Any]:
        try:
            cfg = _config_from_payload(req.config)
            if not bool(cfg.refine_enabled):
                raise ValueError(
                    "MEMatte refinement is mandatory for this tool. "
                    "Disable-preview/no-refine runs are not supported."
                )
            job_id = await job_manager.submit(cfg, cli_flags=_job_cli_flags_from_request(req))
            return {"status": "queued", "id": job_id}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_json(job)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        if job_manager.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")
        await job_manager.cancel(job_id)
        return {"status": "cancel_requested", "id": job_id}

    @app.get("/api/jobs/{job_id}/logs")
    async def get_job_logs(job_id: str, tail_chars: int = Query(12000, ge=256, le=200000)) -> dict[str, Any]:
        job = job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.log_file or not job.log_file.exists():
            return {"logs": ""}
        return {"logs": job_manager._tail_log(job.log_file, max_chars=int(tail_chars))}

    @app.get("/api/jobs/{job_id}/artifacts")
    async def get_job_artifacts(job_id: str) -> dict[str, Any]:
        job = job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        cfg = job.config
        out_start, out_end, out_count = _recent_output_frame_indices(cfg)
        return {
            "status": "ok",
            "job_id": job_id,
            "input": str(cfg.input),
            "output_dir": str(Path(str(cfg.output_dir)).resolve()),
            "output_alpha_pattern": str(cfg.output_alpha),
            "frame_start": int(cfg.frame_start),
            "frame_end": int(cfg.frame_end),
            "output_frame_start": out_start,
            "output_frame_end": out_end,
            "output_frame_count": out_count,
            "run_summary_path": str((Path(str(cfg.output_dir)) / "run_summary.json").resolve()),
            "config_used_path": str((Path(str(cfg.output_dir)) / "config_used.json").resolve()),
        }

    @app.get("/api/qc/info")
    async def qc_info(job_id: str | None = None) -> dict[str, Any]:
        jobs = job_manager.list_jobs()
        job = None
        if job_id:
            job = job_manager.get_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
        elif jobs:
            job = jobs[0]
        if job is None:
            return {"status": "ok", "job_id": None, "input": {}, "output": {}}
        cfg = job.config
        out_start, out_end, out_count = _recent_output_frame_indices(cfg)
        return {
            "status": "ok",
            "job_id": job.id,
            "input": {
                "source": str(cfg.input),
                "frame_start": int(cfg.frame_start),
                "frame_end": int(cfg.frame_end),
                "is_video": bool(_looks_like_video_input(str(cfg.input))),
            },
            "output": {
                "output_dir": str(Path(str(cfg.output_dir)).resolve()),
                "alpha_pattern": str(cfg.output_alpha),
                "trimap_pattern": f"{QC_TRIMAP_DIR}/{QC_TRIMAP_PATTERN}",
                "alpha_format": str(cfg.alpha_format),
                "frame_start": out_start,
                "frame_end": out_end,
                "count": out_count,
                "trimap_available": bool(
                    out_start is not None and _trimap_output_frame_path(cfg, int(out_start)).exists()
                ),
            },
        }

    @app.get("/api/qc/frame-preview")
    async def qc_frame_preview(
        job_id: str,
        frame: int = Query(..., ge=0),
        kind: Literal["input", "alpha", "trimap"] = Query("alpha"),
    ) -> Response:
        job = job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        cfg = job.config
        try:
            if kind == "input":
                png_bytes = await asyncio.to_thread(_cached_qc_input_preview_png, cfg, int(frame))
                return Response(content=png_bytes, media_type="image/png")
            if kind == "trimap":
                trimap_path = _trimap_output_frame_path(cfg, int(frame))
                if not trimap_path.exists():
                    raise FileNotFoundError(f"Trimap preview frame not found: {trimap_path}")
                png_bytes = await asyncio.to_thread(_cached_qc_trimap_preview_png, trimap_path)
                return Response(content=png_bytes, media_type="image/png")

            alpha_path = _alpha_output_frame_path(cfg, int(frame))
            if not alpha_path.exists():
                raise FileNotFoundError(f"Alpha frame not found: {alpha_path}")
            png_bytes = await asyncio.to_thread(_cached_qc_alpha_preview_png, alpha_path)
            return Response(content=png_bytes, media_type="image/png")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/")
    async def root() -> Response:
        if WEB_DIST_DIR.exists():
            index_html = WEB_DIST_DIR / "index.html"
            if index_html.exists():
                return FileResponse(index_html)
        dev_hint = _frontend_dev_url_hint()
        msg = (
            "Videomatte-HQ v2 Web API is running.\n"
            "Frontend build not found. Start the Vite dev server in ./web or build web/dist.\n"
        )
        if dev_hint:
            msg += f"Frontend dev URL hint: {dev_hint}\n"
        msg += "API endpoints are available under /api/.\n"
        return PlainTextResponse(
            msg
        )

    if WEB_DIST_DIR.exists():
        try:
            from fastapi.staticfiles import StaticFiles

            app.mount("/assets", StaticFiles(directory=str(WEB_DIST_DIR / "assets")), name="assets")
        except Exception:
            logger.exception("Failed to mount built frontend assets")

    return app


app = create_app()
