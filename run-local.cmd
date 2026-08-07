@echo off
setlocal
cd /d "%~dp0"

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    py -3 scripts\local_launcher.py %*
    exit /b
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    python scripts\local_launcher.py %*
    exit /b
)

echo Python 3.10 or newer was not found.
echo Install Python, then run run-local.cmd again.
exit /b 1
