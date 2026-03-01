@echo off
setlocal enabledelayedexpansion

:: ============================================================================
::  Videomatte-HQ v2 — Full Installer (Windows)
::
::  Installs Python venv, PyTorch with CUDA, all dependencies, web frontend,
::  and downloads required model checkpoints.
::
::  Usage:   install.bat              (interactive, prompts for CUDA version)
::           install.bat --cpu        (CPU-only PyTorch, no CUDA)
::           install.bat --cuda 12.4  (skip prompt, use CUDA 12.4)
::           install.bat --cuda 12.1  (skip prompt, use CUDA 12.1)
::           install.bat --cuda 11.8  (skip prompt, use CUDA 11.8)
:: ============================================================================

echo.
echo  =============================================
echo   Videomatte-HQ v2 Installer
echo  =============================================
echo.

:: ── Step 0: Check prerequisites ──

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    exit /b 1
)

:: Check Python version (need 3.10+)
for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PY_VER=%%V"
echo [INFO] Found Python %PY_VER%

for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

if %PY_MAJOR% LSS 3 (
    echo [ERROR] Python 3.10+ required, found %PY_VER%
    exit /b 1
)
if %PY_MAJOR%==3 if %PY_MINOR% LSS 10 (
    echo [ERROR] Python 3.10+ required, found %PY_VER%
    exit /b 1
)

echo [OK]   Python version is compatible.

:: ── Step 1: Parse arguments ──

set "TORCH_MODE="
set "CUDA_VER="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--cpu" (
    set "TORCH_MODE=cpu"
    shift
    goto parse_args
)
if /i "%~1"=="--cuda" (
    set "TORCH_MODE=cuda"
    set "CUDA_VER=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:args_done

:: ── Step 2: CUDA selection ──

if "%TORCH_MODE%"=="" (
    echo.
    echo  Select PyTorch compute backend:
    echo.
    echo    1) CUDA 12.4  (recommended for modern NVIDIA GPUs)
    echo    2) CUDA 12.1
    echo    3) CUDA 11.8  (older GPUs / drivers)
    echo    4) CPU only   (no GPU acceleration, very slow)
    echo.
    set /p "CUDA_CHOICE=  Enter choice [1]: "
    if "!CUDA_CHOICE!"=="" set "CUDA_CHOICE=1"
    if "!CUDA_CHOICE!"=="1" (
        set "TORCH_MODE=cuda"
        set "CUDA_VER=12.4"
    ) else if "!CUDA_CHOICE!"=="2" (
        set "TORCH_MODE=cuda"
        set "CUDA_VER=12.1"
    ) else if "!CUDA_CHOICE!"=="3" (
        set "TORCH_MODE=cuda"
        set "CUDA_VER=11.8"
    ) else if "!CUDA_CHOICE!"=="4" (
        set "TORCH_MODE=cpu"
    ) else (
        echo [ERROR] Invalid choice.
        exit /b 1
    )
)

if "%TORCH_MODE%"=="cpu" (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    echo [INFO] Using CPU-only PyTorch.
) else (
    if "%CUDA_VER%"=="12.4" (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
    ) else if "%CUDA_VER%"=="12.1" (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    ) else if "%CUDA_VER%"=="11.8" (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
    ) else (
        echo [ERROR] Unsupported CUDA version: %CUDA_VER%
        echo         Supported: 12.4, 12.1, 11.8
        exit /b 1
    )
    echo [INFO] Using CUDA %CUDA_VER% PyTorch.
)

:: ── Step 3: Create virtual environment ──

echo.
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment already exists at .venv
) else (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK]   Virtual environment created.
)

:: Activate venv
call .venv\Scripts\activate.bat
echo [OK]   Virtual environment activated.

:: ── Step 4: Upgrade pip ──

echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK]   pip upgraded.

:: ── Step 5: Install PyTorch ──

echo.
echo [INFO] Installing PyTorch and torchvision (%TORCH_MODE%)...
echo        Index: %TORCH_INDEX%
echo        This may take several minutes on first install.
pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    echo         Check your internet connection and try again.
    exit /b 1
)
echo [OK]   PyTorch installed.

:: Verify CUDA if requested
if "%TORCH_MODE%"=="cuda" (
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'[OK]   CUDA available: {torch.cuda.get_device_name(0)}')" 2>nul
    if errorlevel 1 (
        echo [WARN] CUDA is not available. PyTorch will fall back to CPU.
        echo        Check your NVIDIA driver and CUDA toolkit installation.
    )
)

:: ── Step 6: Install project + web extras ──

echo.
echo [INFO] Installing Videomatte-HQ v2 and web UI dependencies...
pip install -e ".[web,dev]"
if errorlevel 1 (
    echo [ERROR] Project installation failed.
    exit /b 1
)
echo [OK]   Project installed.

:: ── Step 7: Install additional runtime dependencies ──

echo.
echo [INFO] Installing additional runtime dependencies (einops, timm)...
pip install einops timm >nul 2>&1
echo [OK]   Additional dependencies installed.

:: ── Step 8: Frontend (npm) ──

echo.
where npm >nul 2>&1
if errorlevel 1 (
    echo [WARN] npm not found on PATH. Skipping web frontend build.
    echo        The web UI will still work if web/dist/ exists from a prior build.
    echo        Install Node.js from https://nodejs.org/ to enable frontend dev mode.
    goto skip_npm
)

echo [INFO] Installing web frontend dependencies...
pushd web
call npm install
if errorlevel 1 (
    echo [WARN] npm install failed. Web frontend may not work in dev mode.
    popd
    goto skip_npm
)

echo [INFO] Building web frontend...
call npm run build
if errorlevel 1 (
    echo [WARN] Frontend build failed. Dev mode (npm run dev) will still work.
)
popd
echo [OK]   Web frontend ready.

:skip_npm

:: ── Step 9: Download SAM model ──

echo.
if exist "sam2_l.pt" (
    echo [OK]   SAM model already exists: sam2_l.pt
) else (
    echo [INFO] Downloading SAM2-Large model (sam2_l.pt)...
    echo        This is ~450 MB and may take a few minutes.
    python -c "from ultralytics import SAM; SAM('sam2_l.pt')" 2>nul
    if exist "sam2_l.pt" (
        echo [OK]   SAM model downloaded.
    ) else (
        echo [WARN] SAM model auto-download may have failed.
        echo        It will be downloaded automatically on first run.
    )
)

:: ── Step 10: Verify MEMatte assets ──

echo.
set "MEMATTE_OK=1"

if exist "third_party\MEMatte\inference.py" (
    echo [OK]   MEMatte repo found: third_party\MEMatte
) else (
    echo [WARN] MEMatte repo NOT found at third_party\MEMatte
    echo        Clone it: git clone https://github.com/AcademicFuworker/MEMatte third_party\MEMatte
    set "MEMATTE_OK=0"
)

if exist "third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth" (
    echo [OK]   MEMatte checkpoint found: MEMatte_ViTS_DIM.pth
) else (
    echo [WARN] MEMatte checkpoint NOT found.
    echo        Download from the MEMatte GitHub releases page and place at:
    echo        third_party\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth
    set "MEMATTE_OK=0"
)

:: ── Step 11: Run tests ──

echo.
echo [INFO] Running quick test suite...
python -m pytest tests/ -x -q --tb=short 2>&1
if errorlevel 1 (
    echo [WARN] Some tests failed. This may be expected if MEMatte assets are missing.
) else (
    echo [OK]   All tests passed.
)

:: ── Step 12: Verify CLI ──

echo.
echo [INFO] Verifying CLI entry point...
python -m videomatte_hq.cli --help >nul 2>&1
if errorlevel 1 (
    echo [WARN] CLI entry point verification failed.
) else (
    echo [OK]   CLI is working: videomatte-hq-v2 --help
)

:: ── Summary ──

echo.
echo  =============================================
echo   Installation Complete
echo  =============================================
echo.
echo  Activate the environment:
echo    .venv\Scripts\activate
echo.
echo  Start the web UI:
echo    run_web.bat
echo.
echo  Run from CLI:
echo    videomatte-hq-v2 --input video.mp4 --output-dir output --frame-end 29
echo.

if "%MEMATTE_OK%"=="0" (
    echo  [ACTION REQUIRED] MEMatte assets are missing.
    echo  See BEGINNER_GUIDE.md for download instructions.
    echo.
)

if "%TORCH_MODE%"=="cpu" (
    echo  [NOTE] CPU-only mode. Processing will be slow.
    echo  For GPU acceleration, reinstall with: install.bat --cuda 12.4
    echo.
)

endlocal
