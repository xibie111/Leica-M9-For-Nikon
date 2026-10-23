@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "runtime\python.exe" (
    echo 包内 Python 缺失。请完整解压 ZIP 后重试。
    pause
    exit /b 1
)
"runtime\python.exe" -B -u -m m9.portable --diagnose
set "M9_EXIT=%ERRORLEVEL%"
echo 诊断退出码：%M9_EXIT%
pause
exit /b %M9_EXIT%
