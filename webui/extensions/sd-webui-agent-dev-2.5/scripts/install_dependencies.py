"""Agent extension dependency bootstrap.

Only standard-library modules are imported here, so this module can run before
agent_chat/agent_ui are imported. It installs missing packages into the Python
interpreter that is running Forge (sys.executable), not the system Python.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parent.parent
_REQUIREMENTS = _EXT_DIR / "requirements.txt"
# Distribution name -> import name. Optional document/audio helpers are included
# in the same bootstrap so their features do not fail later on first use.
_REQUIRED = {
    "openai": "openai",
    "requests": "requests",
    "httpx": "httpx",
    "mutagen": "mutagen",
    "python-docx": "docx",
    "pypdf": "pypdf",
}


def _missing():
    return [dist for dist, module in _REQUIRED.items()
            if importlib.util.find_spec(module) is None]


def ensure_dependencies() -> bool:
    """Install missing Agent dependencies once, using Forge's interpreter."""
    missing = _missing()
    if not missing:
        return True

    print(f"[Agent] 检测到缺少依赖: {', '.join(missing)}")
    print(f"[Agent] 正在使用 Forge Python 自动安装: {sys.executable}")
    if not _REQUIREMENTS.is_file():
        print(f"[Agent] ❌ 找不到依赖清单: {_REQUIREMENTS}")
        return False

    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "-r", str(_REQUIREMENTS)]
    try:
        result = subprocess.run(command, cwd=str(_EXT_DIR), check=False)
        if result.returncode != 0:
            print(f"[Agent] ❌ 依赖安装失败，pip 返回码: {result.returncode}")
            print("[Agent] 可手动执行：" + " ".join(command))
            return False
    except Exception as exc:
        print(f"[Agent] ❌ 无法启动 pip: {exc}")
        print("[Agent] 可手动执行：" + " ".join(command))
        return False

    still_missing = _missing()
    if still_missing:
        print(f"[Agent] ❌ 安装后仍缺少: {', '.join(still_missing)}")
        return False
    print("[Agent] ✅ Agent 依赖安装完成")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if ensure_dependencies() else 1)

