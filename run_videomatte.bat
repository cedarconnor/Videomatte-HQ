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
set PROJECT=out\project.vmhqproj
set ASSIGN_MASK=
set ASSIGN_FRAME=0
set REQUIRE_ASSIGNMENT=1
set FRAME_START=
set FRAME_END=
set QC_ENABLE=1
set QC_FAIL_ON_REGRESSION=1
set QC_SAMPLE_OUTPUT_FRAMES=3
set QC_MAX_OUTPUT_ROUNDTRIP_MAE=0.002
set QC_MAX_P95_FLICKER=0.005
set QC_MAX_P95_EDGE_FLICKER=0.02
set QC_MIN_MEAN_EDGE_CONFIDENCE=0.22
set QC_BAND_SPIKE_RATIO=1.8
set QC_MAX_BAND_SPIKE_FRAMES=3

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

:: ---- Validate assignment config ----
if not "%ASSIGN_MASK%"=="" if not exist "%ASSIGN_MASK%" (
    echo [ERROR] ASSIGN_MASK file not found: %ASSIGN_MASK%
    echo.
    echo Fix this line near the top of the launcher:
    echo   set ASSIGN_MASK=path\to\your_mask.png
    pause
    exit /b 1
)

if "%REQUIRE_ASSIGNMENT%"=="1" if "%ASSIGN_MASK%"=="" (
    echo [ERROR] REQUIRE_ASSIGNMENT is enabled, but ASSIGN_MASK is empty.
    echo.
    echo Set a keyframe mask path near the top of this file, for example:
    echo   set ASSIGN_MASK=masks\mask_00000.png
    echo   set ASSIGN_FRAME=0
    echo.
    echo Then run this launcher again.
    pause
    exit /b 1
)

:: ---- Create output directory ----
if not exist "out\alpha" mkdir "out\alpha"

:: ---- Run pipeline ----
echo [START] Processing: %INPUT%
echo [CONFIG] Format=%ALPHA_FORMAT%  Shot=%SHOT_TYPE%  Device=%DEVICE%  Precision=%PRECISION%
echo [CONFIG] Project=%PROJECT%  AssignMask=%ASSIGN_MASK%  RequireAssignment=%REQUIRE_ASSIGNMENT%
echo [CONFIG] FrameRange=%FRAME_START%..%FRAME_END%
echo [CONFIG] QC=%QC_ENABLE%  QCFailOnRegression=%QC_FAIL_ON_REGRESSION%
echo [CONFIG] QCTuned maxFlicker=%QC_MAX_P95_FLICKER% maxEdgeFlicker=%QC_MAX_P95_EDGE_FLICKER% minEdgeConf=%QC_MIN_MEAN_EDGE_CONFIDENCE%
echo.

set EXTRA_ARGS=
if not "%FRAME_START%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --start %FRAME_START%
if not "%FRAME_END%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --end %FRAME_END%
if not "%PROJECT%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --project "%PROJECT%"
if not "%ASSIGN_MASK%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --assign-mask "%ASSIGN_MASK%" --assign-frame %ASSIGN_FRAME%
if "%REQUIRE_ASSIGNMENT%"=="0" set EXTRA_ARGS=%EXTRA_ARGS% --allow-empty-assignment
if "%QC_ENABLE%"=="1" set EXTRA_ARGS=%EXTRA_ARGS% --qc
if "%QC_ENABLE%"=="0" set EXTRA_ARGS=%EXTRA_ARGS% --no-qc
if "%QC_FAIL_ON_REGRESSION%"=="1" set EXTRA_ARGS=%EXTRA_ARGS% --qc-fail-on-regression
if "%QC_FAIL_ON_REGRESSION%"=="0" set EXTRA_ARGS=%EXTRA_ARGS% --no-qc-fail-on-regression
if not "%QC_SAMPLE_OUTPUT_FRAMES%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-sample-output-frames %QC_SAMPLE_OUTPUT_FRAMES%
if not "%QC_MAX_OUTPUT_ROUNDTRIP_MAE%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-max-output-roundtrip-mae %QC_MAX_OUTPUT_ROUNDTRIP_MAE%
if not "%QC_MAX_P95_FLICKER%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-max-p95-flicker %QC_MAX_P95_FLICKER%
if not "%QC_MAX_P95_EDGE_FLICKER%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-max-p95-edge-flicker %QC_MAX_P95_EDGE_FLICKER%
if not "%QC_MIN_MEAN_EDGE_CONFIDENCE%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-min-mean-edge-confidence %QC_MIN_MEAN_EDGE_CONFIDENCE%
if not "%QC_BAND_SPIKE_RATIO%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-band-spike-ratio %QC_BAND_SPIKE_RATIO%
if not "%QC_MAX_BAND_SPIKE_FRAMES%"=="" set EXTRA_ARGS=%EXTRA_ARGS% --qc-max-band-spike-frames %QC_MAX_BAND_SPIKE_FRAMES%

"%VENV_PYTHON%" -m videomatte_hq.cli ^
    --input "%INPUT%" ^
    --out "%OUTPUT%" ^
    --alpha-format %ALPHA_FORMAT% ^
    --shot-type %SHOT_TYPE% ^
    --device %DEVICE% ^
    --precision %PRECISION% ^
    --resume ^
    %EXTRA_ARGS%

echo.
if errorlevel 1 (
    echo [ERROR] Pipeline failed. Check the output above for details.
) else (
    echo ============================================================
    echo [DONE] Output written to: out\alpha\
    echo        Project: %PROJECT%
    echo ============================================================
)

echo.
pause
