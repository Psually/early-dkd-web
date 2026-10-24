@echo off
setlocal
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "DKD_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DKD_PYTHON%" set "DKD_PYTHON=python"
echo Open http://127.0.0.1:8765/ . Press Ctrl+C to stop.
cd /d "%~dp0"
"%DKD_PYTHON%" -m app.serve
pause
