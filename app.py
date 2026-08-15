#!/usr/bin/env python3
"""ModelScope Studio entry point for sd-webui-forge-neo-v3"""

import os
import sys
import subprocess

# Path to the webui directory
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sd-webui-forge-neo-v3', 'webui')


def check_torch_cuda():
    """Check if the currently installed PyTorch has CUDA support."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() == "True"
    except Exception:
        return False


def main():
    os.chdir(WEBUI_DIR)
    sys.path.insert(0, WEBUI_DIR)

    torch_has_cuda = check_torch_cuda()
    print(f"[app.py] PyTorch CUDA available: {torch_has_cuda}", flush=True)

    # Inject command-line args for the ModelScope environment
    sys.argv = [
        sys.argv[0],
        "--skip-python-version-check",
        "--no-half-vae",
        "--api",
    ]
    if not torch_has_cuda:
        print("[app.py] WARNING: GPU base image not detected. Installing CUDA PyTorch...", flush=True)
        sys.argv.append("--reinstall-torch")
        sys.argv.append("--skip-torch-cuda-test")

    # Prepare environment and start webui
    from modules import launch_utils
    launch_utils.prepare_environment()
    launch_utils.start()


if __name__ == '__main__':
    main()