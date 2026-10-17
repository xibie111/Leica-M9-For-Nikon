@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Local Python environment is missing.
    echo Run setup_nef.ps1 in PowerShell first. See docs\NEF_GUIDE.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m m9.gui
if errorlevel 1 (
    echo The NEF window could not start. See the error above.
    pause
    exit /b 1
)
