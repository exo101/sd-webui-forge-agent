#!/usr/bin/env python3
"""ModelScope Studio entry point for sd-webui-forge-neo-v3"""

import os
import sys

# Path to the webui directory
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sd-webui-forge-neo-v3', 'webui')


def main():
    # Change to webui directory
    os.chdir(WEBUI_DIR)
    sys.path.insert(0, WEBUI_DIR)

    # The ModelScope environment has CUDA 12.8.1 drivers but PyTorch is CPU-only.
    # Use the official PyTorch index to install CUDA-enabled builds.
    # The Aliyun mirror's PEP 503 index does not properly expose +cu128 versions.
    if "TORCH_INDEX_URL" not in os.environ:
        os.environ["TORCH_INDEX_URL"] = "https://download.pytorch.org/whl/cu128"
    if "TORCH_COMMAND" not in os.environ:
        os.environ["TORCH_COMMAND"] = (
            "pip install torch==2.10.0+cu128 torchvision==0.22.0+cu128 "
            f"--extra-index-url {os.environ['TORCH_INDEX_URL']}"
        )

    # Inject command-line args for the ModelScope environment
    sys.argv = [
        sys.argv[0],
        "--skip-python-version-check",
        "--skip-torch-cuda-test",
        "--reinstall-torch",
        "--no-half-vae",
        "--api",
    ]

    # Prepare environment
    from modules import launch_utils
    launch_utils.prepare_environment()

    # Start webui
    launch_utils.start()


if __name__ == '__main__':
    main()