@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [run_web] Project venv not found at .venv\Scripts\python.exe
  echo [run_web] Create/activate the venv and install optional web deps:
  echo          pip install -e .[web]
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [run_web] npm was not found on PATH.
  echo [run_web] Install Node.js with npm, then re-run this script.
  exit /b 1
)

set "FRONTEND_PORT="
for /f %%P in ('powershell -NoProfile -Command "$ports=5173..5199; foreach($p in $ports){ try { $c=[System.Net.Sockets.TcpClient]::new('127.0.0.1', $p); $c.Dispose() } catch { Write-Output $p; break } }"') do (
  set "FRONTEND_PORT=%%P"
)
if not defined FRONTEND_PORT set "FRONTEND_PORT=5173"
set "FRONTEND_URL=http://127.0.0.1:%FRONTEND_PORT%"

if not exist "logs" mkdir logs >nul 2>&1
> "logs\web_frontend_dev_url.txt" echo %FRONTEND_URL%

echo [run_web] Starting FastAPI backend on http://127.0.0.1:8000
start "Videomatte-HQ2 Backend" cmd /k ".venv\Scripts\python.exe -m videomatte_hq_web --host 127.0.0.1 --port 8000"

if exist "web\package.json" (
  echo [run_web] Starting Vite frontend on %FRONTEND_URL%
  if not exist "web\node_modules" echo [run_web] web\node_modules missing - the frontend window will run npm install first.
  start "Videomatte-HQ2 Frontend" cmd /k "cd /d web && (if not exist node_modules npm install) && npm run dev -- --host 127.0.0.1 --port %FRONTEND_PORT% --strictPort"
) else (
  echo [run_web] web\package.json not found. Frontend scaffold missing.
)

echo [run_web] Frontend URL: %FRONTEND_URL%
echo [run_web] Backend URL:  http://127.0.0.1:8000
echo [run_web] If the frontend does not start, run:
echo          cd web ^&^& npm install ^&^& npm run dev -- --host 127.0.0.1 --port %FRONTEND_PORT% --strictPort
endlocal
