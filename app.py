#!/usr/bin/env python3
"""ModelScope Studio entry point for sd-webui-forge-neo-v3"""

import os
import sys
import subprocess
import urllib.request
import time
import signal
import stat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_DIR = os.path.join(BASE_DIR, 'llama.cpp')
MODELS_DIR = os.path.join(LLAMA_DIR, 'models')
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_BIN = os.path.join(LLAMA_DIR, 'llama-server')
LLAMA_SERVER_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/latest/download/"
    "llama-server-linux-x64"
)
MODEL_NAME = "Qwen3.5-2B-Q6_K.gguf"
MMPROJ_NAME = "Qwen3.5-2B-mmproj-BF16.gguf"
WEBUI_DIR = os.path.join(BASE_DIR, 'webui')

llama_proc = None


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress indicator."""
    print(f"[app.py] {desc} ...", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"[app.py] {desc} done: {dest_path}", flush=True)


def ensure_llama_server():
    """Download Linux llama-server binary if not present."""
    if os.path.exists(LLAMA_SERVER_BIN):
        return
    print("[app.py] llama-server binary not found, downloading Linux version...", flush=True)
    download_file(LLAMA_SERVER_URL, LLAMA_SERVER_BIN, "Downloading llama-server")
    os.chmod(LLAMA_SERVER_BIN, os.stat(LLAMA_SERVER_BIN).st_mode | stat.S_IEXEC)
    print("[app.py] llama-server binary ready.", flush=True)


def ensure_models():
    """Check model files exist."""
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    mmproj_path = os.path.join(MODELS_DIR, MMPROJ_NAME)
    if not os.path.exists(model_path) or not os.path.exists(mmproj_path):
        print(f"[app.py] ERROR: Models not found in {MODELS_DIR}", flush=True)
        for f in os.listdir(MODELS_DIR):
            print(f"  - {f}", flush=True)
        return False
    print(f"[app.py] Models found: {MODEL_NAME}, {MMPROJ_NAME}", flush=True)
    return True


def start_llama_server():
    """Start llama.cpp server as background process."""
    global llama_proc

    if not os.path.isdir(LLAMA_DIR):
        print("[app.py] llama.cpp directory not found, skipping llama-server startup", flush=True)
        return None

    if not ensure_models():
        print("[app.py] WARNING: Models not available, skipping llama-server startup", flush=True)
        return None

    try:
        ensure_llama_server()
    except Exception as e:
        print(f"[app.py] WARNING: Failed to download llama-server: {e}", flush=True)
        return None

    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    mmproj_path = os.path.join(MODELS_DIR, MMPROJ_NAME)

    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--mmproj", mmproj_path,
        "--host", "0.0.0.0",
        "--port", str(LLAMA_SERVER_PORT),
        "-ngl", "100",
    ]

    print(f"[app.py] Starting llama-server on port {LLAMA_SERVER_PORT}...", flush=True)
    print(f"[app.py] Command: {' '.join(cmd)}", flush=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=LLAMA_DIR,
        )
    except Exception as e:
        print(f"[app.py] Failed to start llama-server: {e}", flush=True)
        return None

    # Wait for server to be ready
    for i in range(60):
        time.sleep(2)
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=5)
            if req.status == 200:
                print(f"[app.py] llama-server ready on port {LLAMA_SERVER_PORT}!", flush=True)
                llama_proc = proc
                return proc
        except Exception:
            pass
        print(f"[app.py] Waiting for llama-server... ({i+1}/60)", flush=True)

    print("[app.py] WARNING: llama-server did not become ready in time", flush=True)
    llama_proc = proc
    return proc


def stop_llama_server():
    """Stop the llama.cpp server."""
    global llama_proc
    if llama_proc is None:
        return
    print("[app.py] Stopping llama-server...", flush=True)
    llama_proc.terminate()
    try:
        llama_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        llama_proc.kill()
        llama_proc.wait()
    print("[app.py] llama-server stopped.", flush=True)
    llama_proc = None


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
        "--api",
    ]
    if not torch_has_cuda:
        print("[app.py] WARNING: GPU base image not detected. Installing CUDA PyTorch...", flush=True)
        sys.argv.append("--reinstall-torch")
        sys.argv.append("--skip-torch-cuda-test")

    # Start llama.cpp server as background process
    start_llama_server()

    # Prepare environment and start webui
    from modules import launch_utils
    try:
        launch_utils.prepare_environment()
        launch_utils.start()
    finally:
        stop_llama_server()


if __name__ == '__main__':
    main()
