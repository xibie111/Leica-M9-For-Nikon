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
"runtime\python.exe" -B -u -m m9.portable
set "M9_EXIT=%ERRORLEVEL%"
if not "%M9_EXIT%"=="0" (
    echo 程序失败。请查看以上错误和 %%LOCALAPPDATA%%\LeicaM9ForNikon\logs 内的日志。
    pause
)
exit /b %M9_EXIT%
