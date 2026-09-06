"""ModelScope Gradio API adapter for See-Through."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.agent_tools_registry import agent_tool

SEE_THROUGH_URL = "https://ljsabc-see-through.ms.show"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "see_through_remote"


def configure_space_url(url: str, token: str | None = None) -> str:
    value = (url or "").strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("Gradio API 地址必须以 http:// 或 https:// 开头")
    global SEE_THROUGH_URL
    SEE_THROUGH_URL = value
    return value


def check_space(url: str, token: str = "", expected_tool: str = "inference") -> dict[str, Any]:
    """检查 Gradio API 页面，不再执行 MCP 握手。"""
    try:
        import httpx
        response = httpx.get(url.rstrip("/"), timeout=30)
        if response.status_code >= 400:
            return {"status": "error", "message": f"Gradio API 返回 HTTP {response.status_code}", "url": url}
        return {"status": "connected", "message": f"Gradio API 可访问：{expected_tool or '自定义 API'}", "url": url}
    except Exception as exc:
        return {"status": "error", "message": f"Gradio API 检测失败：{exc}", "url": url}


def _copy_result(value: Any, job_dir: Path, files: list[str]) -> None:
    if isinstance(value, dict):
        path = value.get("path")
        if path and os.path.isfile(path):
            target = job_dir / Path(path).name
            shutil.copy2(path, target)
            files.append(str(target))
        for item in value.values():
            _copy_result(item, job_dir, files)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _copy_result(item, job_dir, files)


@agent_tool(
    name="see_through_remote",
    description="通过 ModelScope Gradio API 调用 See-Through，将上传的动漫图片分解为分层 PSD。",
    parameters={
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "输入图片路径，由助手自动注入"},
            "resolution": {"type": "integer", "default": 1024},
            "seed": {"type": "integer", "default": 42},
            "tblr_split": {"type": "boolean", "default": False},
        },
        "required": ["image"],
    },
)
def see_through_remote(image: str, resolution: int = 1024, seed: int = 42, tblr_split: bool = False):
    if not image or not os.path.isfile(image):
        return None, "找不到输入图片，请先上传一张图片。"
    try:
        from gradio_client import Client, handle_file
        client = Client(SEE_THROUGH_URL, verbose=False)
        result = client.predict(
            image=handle_file(image),
            resolution=max(768, min(1280, int(resolution))),
            seed=max(0, min(9999, int(seed))),
            tblr_split=bool(tblr_split),
            api_name="/inference",
        )
        job_dir = OUTPUT_DIR / f"run_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        _copy_result(result, job_dir, files)
        psd = next((p for p in files if p.lower().endswith(".psd")), None)
        previews = [p for p in files if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        return previews[:12], {"status": "success", "backend": "ModelScope Gradio API", "psd_path": psd, "files": files, "result": result}
    except Exception as exc:
        return None, f"调用 See-Through Gradio API 失败：{exc}"
