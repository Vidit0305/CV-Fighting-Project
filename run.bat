@echo off
REM Runner script for CV Fighter on Windows

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Starting CV Fighter using virtual environment (.venv)...
    .venv\Scripts\python.exe main.py %*
) else if exist "venv\Scripts\python.exe" (
    echo Starting CV Fighter using virtual environment (venv)...
    venv\Scripts\python.exe main.py %*
) else (
    echo Running with system python...
    python main.py %*
)

pause
