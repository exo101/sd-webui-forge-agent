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
        # PyTorch is CPU-only or missing. Install CUDA-enabled build by downloading
        # the wheel directly from the Aliyun mirror, then installing with pip.
        # This bypasses the PEP 503 index resolution issue with +cu128 versions.
        ALIYUAN_MIRROR = "https://mirrors.aliyun.com/pytorch-wheels/cu128"
        # Download the wheel file using Python's urllib (avoids pip's 403 issue)
        import urllib.request
        wheel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_wheel.whl")
        print(f"[app.py] Downloading torch wheel from Aliyun mirror ...", flush=True)
        # Try with custom headers to bypass Aliyun mirror hotlink protection
        urls_to_try = [
            (f"{ALIYUAN_MIRROR}/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl", "%2B"),
            (f"{ALIYUAN_MIRROR}/torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl", "+"),
        ]
        downloaded = False
        for url, label in urls_to_try:
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; ModelScope)",
                        "Referer": "https://mirrors.aliyun.com/pytorch-wheels/cu128/",
                    }
                )
                with urllib.request.urlopen(req, timeout=300) as response:
                    with open(wheel_path, "wb") as f:
                        f.write(response.read())
                print(f"[app.py] Downloaded ({label}): {wheel_path}", flush=True)
                downloaded = True
                break
            except Exception as e:
                print(f"[app.py] Download failed ({label}): {e}", flush=True)
        if not downloaded:
            wheel_path = None

        if wheel_path and os.path.exists(wheel_path):
            os.environ["TORCH_COMMAND"] = (
                f"pip install {wheel_path} torchvision==0.22.0 --force-reinstall"
            )
        else:
            # Fallback: try --extra-index-url
            if "TORCH_INDEX_URL" not in os.environ:
                os.environ["TORCH_INDEX_URL"] = ALIYUAN_MIRROR
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