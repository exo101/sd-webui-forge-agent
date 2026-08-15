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
    # Change to webui directory
    os.chdir(WEBUI_DIR)
    sys.path.insert(0, WEBUI_DIR)

    # Check if the pre-installed PyTorch already has CUDA support.
    # The ModelScope environment has CUDA 12.8 drivers and may already provide
    # a CUDA-enabled PyTorch, making a reinstall unnecessary.
    torch_has_cuda = check_torch_cuda()
    print(f"[app.py] PyTorch CUDA available: {torch_has_cuda}", flush=True)

    if not torch_has_cuda:
        # PyTorch is CPU-only or missing. We need to install a CUDA-enabled build.
        # The official PyTorch index (download.pytorch.org) is unreachable from
        # within ModelScope, and the Aliyun mirror blocks direct wheel downloads.
        # Strategy: use --extra-index-url with the Aliyun PyTorch mirror.
        if "TORCH_INDEX_URL" not in os.environ:
            os.environ["TORCH_INDEX_URL"] = "https://mirrors.aliyun.com/pytorch-wheels/cu128"
        if "TORCH_COMMAND" not in os.environ:
            os.environ["TORCH_COMMAND"] = (
                "pip install torch==2.10.0+cu128 torchvision==0.22.0+cu128 "
                f"--extra-index-url {os.environ['TORCH_INDEX_URL']}"
            )

    # Inject command-line args for the ModelScope environment
    sys.argv = [
        sys.argv[0],
        "--skip-python-version-check",
        "--no-half-vae",
        "--api",
    ]
    if not torch_has_cuda:
        # Torch is CPU-only: reinstall with CUDA support and skip the CUDA test
        # to avoid failure during the transition.
        sys.argv.append("--reinstall-torch")
        sys.argv.append("--skip-torch-cuda-test")

    # Prepare environment
    from modules import launch_utils
    launch_utils.prepare_environment()

    # Start webui
    launch_utils.start()


if __name__ == '__main__':
    main()