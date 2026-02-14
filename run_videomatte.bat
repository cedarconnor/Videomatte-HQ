@echo off
title VideoMatte-HQ Launcher
color 0A

echo ============================================================
echo          VideoMatte-HQ  --  8K People Video Matting
echo ============================================================
echo.

:: ---- Configuration ----
:: Edit these paths to match your setup
set INPUT=TestFiles\6138680-uhd_3840_2160_24fps.mp4
set OUTPUT=out\alpha\%%06d.png
set ALPHA_FORMAT=png16
set SHOT_TYPE=locked_off
set DEVICE=cuda
set PRECISION=fp16
set FRAME_START=
set FRAME_END=

:: Fast preset (1=on, 0=off)
:: Speeds up iteration by lowering intermediate cost, reducing ROI detection frequency,
:: disabling temporal flow, and disabling preview rendering.
set FAST_PRESET=1

:: ---- Locate venv Python ----
set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo [INFO] .venv not found. Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv. Make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo [INFO] Installing PyTorch with CUDA 12.8...
    .venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed.
        pause
        exit /b 1
    )
    echo [INFO] Installing videomatte-hq...
    .venv\Scripts\pip.exe install -e .
    if errorlevel 1 (
        echo [ERROR] Package installation failed.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Installation complete.
    echo.
)

:: ---- Check package is installed ----
"%VENV_PYTHON%" -c "import videomatte_hq" >nul 2>&1
if errorlevel 1 (
    echo [INFO] videomatte-hq not installed in .venv. Installing now...
    .venv\Scripts\pip.exe install -e .
    if errorlevel 1 (
        echo [ERROR] Installation failed. Check the output above.
        pause
        exit /b 1
    )
)

:: ---- Handle drag-and-drop ----
if not "%~1"=="" (
    set INPUT=%~1
    echo [INFO] Using dropped file: %INPUT%
    echo.
)

:: ---- Check input exists ----
if not exist "%INPUT%" (
    echo [ERROR] Input not found: %INPUT%
    echo.
    echo Usage: Edit the INPUT variable in this .bat file, or drag-and-drop
    echo        a video file onto the script.
    pause
    exit /b 1
)

:: ---- Create output directory ----
if not exist "out\alpha" mkdir "out\alpha"

:: ---- Run pipeline ----
echo [START] Processing: %INPUT%
echo [CONFIG] Format=%ALPHA_FORMAT%  Shot=%SHOT_TYPE%  Device=%DEVICE%  Precision=%PRECISION%
echo [CONFIG] FrameRange=%FRAME_START%..%FRAME_END%  FastPreset=%FAST_PRESET%
echo.

set EXTRA_ARGS=
if not "%FRAME_START%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --start %FRAME_START%
if not "%FRAME_END%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --end %FRAME_END%
if "%FAST_PRESET%"=="1" set EXTRA_ARGS=%EXTRA_ARGS% --temporal none --no-preview --intermediate-long-side 3072 --roi-detect-every 30

"%VENV_PYTHON%" -m videomatte_hq.cli ^
    --input "%INPUT%" ^
    --out "%OUTPUT%" ^
    --alpha-format %ALPHA_FORMAT% ^
    --shot-type %SHOT_TYPE% ^
    --device %DEVICE% ^
    --precision %PRECISION% ^
    --preview ^
    --preview-modes checker,alpha,white,flicker ^
    --resume ^
    %EXTRA_ARGS%

echo.
if errorlevel 1 (
    echo [ERROR] Pipeline failed. Check the output above for details.
) else (
    echo ============================================================
    echo [DONE] Output written to: out\alpha\
    echo        Preview: out\preview\live_preview.mp4
    echo        QC Report: out\qc\report.html
    echo ============================================================
)

echo.
pause
