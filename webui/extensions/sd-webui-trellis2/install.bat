@echo off
echo ========================================
echo TRELLIS.2 Extension Installation Script
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python environment...
if exist "..\..\python\python.exe" (
    set PYTHON=..\..\python\python.exe
    echo Using portable Python: %PYTHON%
) else (
    set PYTHON=python
    echo Using system Python
)
echo.

echo [2/3] Checking and installing dependencies...
echo This will skip already installed packages.
echo.
%PYTHON% install.py
echo.

echo [3/3] Checking model files...
if exist "..\..\models\trellis2\TRELLIS.2-4B\pipeline.json" (
    echo ✓ Model files already exist
) else (
    echo ⚠ Model not found, downloading...
    %PYTHON% download_model.py
)
echo.

echo ========================================
echo Installation complete!
echo ========================================
echo.
echo NOTE: The following compiled modules should already be installed:
echo   - nvdiffrast
echo   - nvdiffrec
echo   - CuMesh
echo   - FlexGEMM
echo   - utils3d
echo   - o-voxel
echo.
pause
