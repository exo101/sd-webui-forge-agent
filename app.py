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
    # Force reinstall of CUDA-enabled PyTorch via Aliyun mirror (reachable from within ModelScope).
    # Note: --extra-index-url is required (not --find-links) so pip searches the mirror as an
    # index and can resolve local version identifiers like +cu128. torchvision is installed
    # without +cu128 suffix because the Aliyun mirror lacks cp312 torchvision+cu128 wheels.
    if "TORCH_COMMAND" not in os.environ:
        os.environ["TORCH_COMMAND"] = (
            "pip install torch==2.10.0+cu128 torchvision==0.22.0 "
            "--force-reinstall "
            "--extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cu128/"
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