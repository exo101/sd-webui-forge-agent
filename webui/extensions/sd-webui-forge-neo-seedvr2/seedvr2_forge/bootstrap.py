from __future__ import annotations

import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Pin to a tagged release so upstream changes do not silently break the Forge adapter.
SEEDVR2_VERSION = "v2.5.23"
SEEDVR2_VERSION_NUMBER = SEEDVR2_VERSION.lstrip("v")
SEEDVR2_ARCHIVE_URLS = [
    (
        "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/"
        f"archive/refs/tags/{SEEDVR2_VERSION}.zip"
    ),
    (
        "https://codeload.github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/"
        f"zip/refs/tags/{SEEDVR2_VERSION}"
    ),
]


def installed_version(root: Path) -> Optional[str]:
    constants = Path(root) / "src" / "utils" / "constants.py"
    if not constants.exists():
        return None
    try:
        text = constants.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        return match.group(1) if match else None
    except Exception:
        return None


def source_is_compatible(root: Path) -> bool:
    root = Path(root)
    required = [
        root / "src" / "core" / "generation_utils.py",
        root / "src" / "core" / "generation_phases.py",
        root / "src" / "utils" / "debug.py",
        root / "pos_emb.pt",
        root / "neg_emb.pt",
    ]
    return all(p.exists() for p in required) and installed_version(root) == SEEDVR2_VERSION_NUMBER


def install_seedvr2_source(vendor_root: Path, force: bool = False) -> bool:
    """Download the pinned official SeedVR2 source into vendor_root.

    Returns True if files were installed, False if the exact pinned source was kept.
    """
    vendor_root = Path(vendor_root)
    if source_is_compatible(vendor_root) and not force:
        return False

    vendor_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="seedvr2_forge_") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "seedvr2.zip"

        last_error = None
        for url in SEEDVR2_ARCHIVE_URLS:
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "sd-webui-seedvr2-forge-neo"},
                )
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as f:
                    shutil.copyfileobj(response, f)
                last_error = None
                break
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise RuntimeError(f"Failed to download SeedVR2 source: {last_error}") from last_error

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(tmp_path / "extract")

        candidates = [p for p in (tmp_path / "extract").iterdir() if p.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(f"Unexpected SeedVR2 archive layout: {candidates}")
        extracted = candidates[0]

        if vendor_root.exists():
            shutil.rmtree(vendor_root)
        shutil.copytree(extracted, vendor_root)

    if not source_is_compatible(vendor_root):
        found = installed_version(vendor_root)
        raise RuntimeError(
            f"SeedVR2 source installation finished, but version validation failed "
            f"(expected {SEEDVR2_VERSION_NUMBER}, found {found!r})"
        )
    return True
