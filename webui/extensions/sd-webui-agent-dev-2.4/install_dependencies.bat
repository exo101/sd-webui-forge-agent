@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%\..\..\..\..\system\python\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%ROOT%scripts\install_dependencies.py"
if errorlevel 1 (
  echo.
  echo Agent dependency installation failed. Please check the error above.
  exit /b 1
)
echo Agent dependencies are ready.
