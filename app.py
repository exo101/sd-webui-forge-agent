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
    # Install CUDA-enabled PyTorch via direct wheel URL from the Aliyun mirror.
    # The PEP 503 index at the mirror does not properly expose +cu128 versions to pip,
    # so we bypass index resolution by specifying the exact wheel URL.
    # torchvision is installed from the main PyPI index (without +cu128 suffix, which
    # is unavailable for cp312 on the Aliyun mirror).
    if "TORCH_COMMAND" not in os.environ:
        TORCH_WHEEL = (
            "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
            "torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
        )
        os.environ["TORCH_COMMAND"] = (
            f"pip install {TORCH_WHEEL} torchvision==0.22.0 "
            "--force-reinstall"
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