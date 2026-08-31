"""
SAM Matting auto-install script
Copyright (C) 2024

Automatically installs SAM Matting dependencies on WebUI startup.

Required dependencies:
- rembg: Background removal tool
- onnxruntime-gpu: GPU-accelerated ONNX runtime
- litelama: Lightweight image inpainting model
- segment-anything: Meta's Segment Anything Model

Usage:
Place this script in the extension's scripts folder; it runs automatically on WebUI startup.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_python_executable():
    """Get the current Python executable path"""
    python_exe = sys.executable
    if not python_exe:
        # Try common fallback paths
        venv_dir = Path(__file__).parent.parent.parent / "venv"
        if venv_dir.exists():
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = "python"
    return python_exe


def is_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def install_package(package_name, display_name=None):
    """Install a package using pip"""
    if display_name is None:
        display_name = package_name
    
    print(f"\n{'='*60}")
    print(f"[PKG] Installing {display_name}...")
    print(f"{'='*60}")
    
    python_exe = get_python_executable()
    
    try:
        # Install using the current Python environment
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", package_name, "--upgrade"],
            check=True,
            capture_output=False,
            encoding="utf-8"
        )
        
        if result.returncode == 0:
            print(f"[OK] {display_name} installed successfully!")
            return True
        else:
            print(f"[FAIL] {display_name} installation failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error installing {display_name}: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error installing {display_name}: {e}")
        return False


def install_dependencies():
    """Install all required dependency packages"""
    print("\n" + "="*60)
    print("[SETUP] SAM Matting Dependency Installer")
    print("="*60)
    
    # Define packages to install with display names
    packages = [
        ("rembg", "rembg (Background Removal)"),
        ("onnxruntime-gpu", "onnxruntime-gpu (GPU ONNX Runtime)"),
        ("litelama", "litelama (Lightweight Inpainting)"),
        ("segment-anything", "segment-anything (Meta SAM)"),
    ]
    
    installed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for package, display_name in packages:
        # Check if already installed
        if is_package_installed(package.split('[')[0]):  # Handle extras
            print(f"\n[OK] {display_name} already installed, skipping")
            skipped_count += 1
            continue
        
        # Install the package
        if install_package(package, display_name):
            installed_count += 1
        else:
            failed_count += 1
            print(f"\n[WARN] {display_name} installation failed. You can install manually:")
            print(f"   python -m pip install {package}")
    
    # Print summary
    print("\n" + "="*60)
    print("[STATS] Installation Summary")
    print("="*60)
    print(f"[OK] Newly installed: {installed_count}")
    print(f"[SKIP] Already present: {skipped_count}")
    print(f"[FAIL] Failed: {failed_count}")
    
    if failed_count > 0:
        print(f"\n[WARN] {failed_count} package(s) failed to install. Please check:")
        print("  1. Network connectivity")
        print("  2. pip mirror accessibility (you can configure a domestic mirror)")
        print("  3. Python version compatibility")
        print("\nManual install commands:")
        for package, display_name in packages:
            if not is_package_installed(package.split('[')[0]):
                print(f"  python -m pip install {package}")
    else:
        print("\n[DONE] All dependencies installed successfully!")
        print("\n[TIP] If this is a first-time install, restart WebUI to ensure all modules load correctly")
    
    print("="*60 + "\n")
    
    return failed_count == 0


if __name__ == "__main__":
    print("\n" + "="*60)
    print("[START] SAM Matting Plugin - Auto Install Script")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print("="*60 + "\n")
    
    # Execute installation
    success = install_dependencies()
    
    if success:
        print("[DONE] Installation completed successfully, starting WebUI...\n")
    else:
        print("[WARN] Some dependencies failed to install, but WebUI will continue...\n")
        print("You can manually install the failed packages in the terminal after WebUI starts\n")