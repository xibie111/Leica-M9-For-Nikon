@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Local Python environment is missing.
    echo Run setup_nef.ps1 in PowerShell first. See docs\NEF_GUIDE.md.
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Usage: process_nef.cmd "D:\Photos\DSC_0001.NEF" --output-dir "D:\Photos\M9_output"
    echo Options: --method dng^|native --resolution original^|phone --device auto^|cuda^|cpu
    exit /b 2
)
".venv\Scripts\python.exe" -m m9.nef %*
exit /b %errorlevel%
