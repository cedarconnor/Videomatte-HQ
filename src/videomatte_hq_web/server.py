"""Thin entrypoint wrapper for the local Videomatte-HQ v2 web backend."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Any

RUNTIME_MODULE = "videomatte_hq_web.server_runtime"
_RUNTIME_IMPORT_ERROR: Exception | None = None

try:
    _runtime = import_module(RUNTIME_MODULE)
except Exception as exc:  # pragma: no cover - optional dependency path
    _runtime = None
    _RUNTIME_IMPORT_ERROR = exc


def create_app() -> Any:
    if _runtime is None:
        raise RuntimeError(
            "Web backend dependencies are not installed. Install with `pip install -e .[web]`."
        ) from _RUNTIME_IMPORT_ERROR
    return _runtime.create_app()


if _runtime is not None:
    try:
        app = _runtime.app
    except Exception:  # pragma: no cover - keep import-time behavior tolerant
        app = None
else:  # pragma: no cover - optional dependency path
    app = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Videomatte-HQ v2 web UI backend.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development only)")
    parser.add_argument("--log-level", default="warning", help="Uvicorn log level (default: warning)")
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable Uvicorn access logs (disabled by default to reduce UI polling noise)",
    )
    args = parser.parse_args(argv)

    if _runtime is None:
        raise RuntimeError(
            "Running the web backend requires optional dependencies. Install with `pip install -e .[web]`."
        ) from _RUNTIME_IMPORT_ERROR

    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError(
            "Running the web backend requires optional dependencies. Install with `pip install -e .[web]`."
        ) from exc

    uvicorn.run(
        "videomatte_hq_web.server_runtime:app",
        host=args.host,
        port=int(args.port),
        reload=bool(args.reload),
        log_level=str(args.log_level),
        access_log=bool(args.access_log),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
