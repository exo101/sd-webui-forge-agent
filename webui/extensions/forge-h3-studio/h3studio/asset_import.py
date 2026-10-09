from __future__ import annotations

from pathlib import Path
from typing import Any

from .cloud_client import CLOUD_UPLOAD_DIR
from .comfy_client import ComfyClient
from .config import load_config

_EXTERNAL_ASSET_QUEUE: list[dict[str, Any]] = []
ASSET_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg",
}
API_ROOT = "/h3studio/api"


def _asset_url(item: dict[str, Any]) -> str:
    from urllib.parse import quote
    return (f"{API_ROOT}/media?filename={quote(str(item.get('filename') or item.get('name') or ''))}"
            f"&subfolder={quote(str(item.get('subfolder') or ''))}&type={quote(str(item.get('type') or 'input'))}")


def import_local_asset(path: str | Path) -> dict[str, Any]:
    """Import a locally generated media file through the same path as browser uploads."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"文件不存在：{source}")
    suffix = source.suffix.lower()
    if suffix not in ASSET_EXTENSIONS:
        raise ValueError(f"不支持的素材格式：{suffix or '(无扩展名)'}")

    if load_config().get("backend_mode") in {"api", "local"}:
        CLOUD_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = CLOUD_UPLOAD_DIR / f"{source.stem}_{source.stat().st_mtime_ns}{suffix}"
        target.write_bytes(source.read_bytes())
        return {
            "name": target.name,
            "subfolder": "",
            "type": "input",
            "file": target.name,
            "url": _asset_url({"name": target.name, "type": "input"}),
        }

    with source.open("rb") as handle:
        result = ComfyClient().upload(handle, source.name, None)
    result["file"] = "/".join(part for part in (result.get("subfolder"), result.get("name")) if part)
    result["url"] = _asset_url(result)
    return result


def queue_external_asset(item: dict[str, Any]) -> dict[str, Any]:
    _EXTERNAL_ASSET_QUEUE.append(dict(item))
    return item


def drain_external_assets() -> list[dict[str, Any]]:
    items = list(_EXTERNAL_ASSET_QUEUE)
    _EXTERNAL_ASSET_QUEUE.clear()
    return items
