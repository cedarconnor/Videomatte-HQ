@echo off
setlocal

:: Setup environment
call .venv\Scripts\activate.bat

:: Start Backend (Background)
start /B "VideoMatte Backend" python -m uvicorn videomatte_hq_web.server:app --reload --port 8000

:: Start Frontend (Dev Server)
cd web
start /B "VideoMatte Frontend" npm run dev

echo Web UI starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C to stop servers.
pause
