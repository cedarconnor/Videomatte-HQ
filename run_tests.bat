@echo off
setlocal

set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo [ERROR] .venv not found at %~dp0.venv
    echo Run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -e .
    exit /b 1
)

"%VENV_PYTHON%" -m pytest -q %*
exit /b %ERRORLEVEL%

