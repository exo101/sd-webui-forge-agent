from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

# Intentionally do NOT install or upgrade torch / torchvision / numpy.
# Forge owns those packages and replacing them can break the whole WebUI.
REQUIRED = {
    "safetensors": "safetensors",
    "tqdm": "tqdm",
    "psutil": "psutil",
    "einops": "einops",
    "omegaconf": "omegaconf>=2.3.0",
    "diffusers": "diffusers>=0.33.1",
    "peft": "peft>=0.17.0",
    "rotary_embedding_torch": "rotary-embedding-torch>=0.5.3",
    "cv2": "opencv-python",
    "gguf": "gguf",
    "matplotlib": "matplotlib",
}

ROOT = Path(__file__).resolve().parent


def is_installed(module_name: str) -> bool:
    # Avoid importing heavyweight packages (peft/diffusers) during Forge startup.
    # Discovery is enough to decide whether a dependency is present.
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def pip_install(pkg_name: str) -> None:
    print(f"[sd-webui-seedvr2] Installing missing dependency: {pkg_name}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])


def install_dependencies() -> None:
    for module_name, pkg_name in REQUIRED.items():
        if not is_installed(module_name):
            pip_install(pkg_name)

    # Forge-Neo pins some packages itself. Report the important host versions but
    # never auto-upgrade them behind Forge's back.
    for dist_name in ("omegaconf", "diffusers", "transformers", "torch"):
        try:
            version = importlib.metadata.version(dist_name)
            print(f"[sd-webui-seedvr2] Host {dist_name}={version}")
        except importlib.metadata.PackageNotFoundError:
            pass


def install_vendor_source() -> None:
    # Import by adding extension root, because install.py may be executed directly.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from seedvr2_forge.bootstrap import SEEDVR2_VERSION, install_seedvr2_source
    from seedvr2_forge.config import vendor_seedvr2_root

    try:
        changed = install_seedvr2_source(vendor_seedvr2_root())
        if changed:
            print(f"[sd-webui-seedvr2] Installed official SeedVR2 source {SEEDVR2_VERSION}")
        else:
            print(f"[sd-webui-seedvr2] SeedVR2 source already installed ({SEEDVR2_VERSION})")
    except Exception as e:
        # Do not make the entire Forge startup fail because GitHub is temporarily unreachable.
        print(f"[sd-webui-seedvr2] WARNING: Could not bootstrap SeedVR2 source: {e}")
        print("[sd-webui-seedvr2] You may manually place the official repo in vendor/seedvr2")


if __name__ == "__main__":
    install_dependencies()
    install_vendor_source()
