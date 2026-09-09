from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXTENSION_ROOT / "data"
CONFIG_PATH = DATA_DIR / "config.json"
LORA_PRESETS_PATH = DATA_DIR / "lora_presets.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "backend_mode": "managed",
    "comfy_url": "http://127.0.0.1:8189",
    "comfy_path": "",
    "python_executable": "",
    "port": 8189,
    "extra_args": "--preview-method auto",
    "auto_start_on_tab": True,
    "startup_timeout": 180,
    "request_timeout": 30,
    "output_prefix": "video/Forge_H3_Studio",
    "minimax_api_key": "",
    "minimax_api_base": "https://api.minimaxi.com",
}

_lock = threading.RLock()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def load_config() -> dict[str, Any]:
    with _lock:
        config = deepcopy(DEFAULT_CONFIG)
        if CONFIG_PATH.is_file():
            try:
                saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    config.update(saved)
            except (OSError, ValueError, TypeError):
                pass
        return config


def save_config(changes: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULT_CONFIG)
    with _lock:
        config = load_config()
        for key, value in changes.items():
            if key in allowed:
                config[key] = value
        _atomic_write(CONFIG_PATH, config)
        return config


def load_lora_presets() -> list[dict[str, Any]]:
    with _lock:
        if not LORA_PRESETS_PATH.is_file():
            return []
        try:
            data = json.loads(LORA_PRESETS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError, TypeError):
            return []


def save_lora_presets(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with _lock:
        cleaned = presets[:100]
        _atomic_write(LORA_PRESETS_PATH, cleaned)
        return cleaned
