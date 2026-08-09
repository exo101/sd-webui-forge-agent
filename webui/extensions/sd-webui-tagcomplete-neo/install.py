"""
install.py — Dependency installer for Tag Autocomplete Neo.

Runs automatically when Forge Neo loads the extension.
Only PyYAML is a hard requirement beyond the Forge Neo baseline.
"""

import importlib
import subprocess
import sys

REQUIRED_PACKAGES = [
    ("yaml", "pyyaml"),
]


def install(package: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--quiet"],
        stdout=subprocess.DEVNULL,
    )


def check_and_install() -> None:
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"[Tag Autocomplete Neo] Installing missing dependency: {pip_name}")
            install(pip_name)
            print(f"[Tag Autocomplete Neo] {pip_name} installed successfully.")


check_and_install()
