@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "runtime\python.exe" (
    echo Bundled Python is missing. Extract the complete ZIP before starting.
    pause
    exit /b 1
)
"runtime\python.exe" -B -u -m m9.portable
set "M9_EXIT=%ERRORLEVEL%"
if not "%M9_EXIT%"=="0" (
    echo Failed. See errors above and logs in %%LOCALAPPDATA%%\LeicaM9ForNikon\logs.
    pause
)
exit /b %M9_EXIT%
