@echo off
setlocal

set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo [ERROR] .venv Python not found: %VENV_PYTHON%
    echo Create/install the environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -e .
    pause
    exit /b 1
)

:: Runtime preflight for SAM2/Samurai dependencies.
"%VENV_PYTHON%" -c "import torch, torchvision, importlib; importlib.import_module('sam2.build_sam')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch/SAM2 runtime check failed in .venv.
    echo.
    echo Run this to verify:
    echo   .\.venv\Scripts\python -c "import torch, torchvision; import importlib; importlib.import_module('sam2.build_sam')"
    echo.
    echo If it fails, reinstall matching wheels:
    echo   .\.venv\Scripts\pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)

:: Start Backend (Background)
start /B "VideoMatte Backend" "%VENV_PYTHON%" -m uvicorn videomatte_hq_web.server:app --reload --port 8000

:: Start Frontend (Dev Server)
cd /d "%~dp0web"
start /B "VideoMatte Frontend" npm run dev

echo Web UI starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C to stop servers.
pause
