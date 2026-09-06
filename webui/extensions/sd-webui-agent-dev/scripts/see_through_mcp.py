"""Remote See-through connector for the DreamArt Agent.

The ModelScope Space exposes a Gradio MCP endpoint, but currently protects
the MCP handshake with 403 while leaving its public Gradio API available.
This connector uses the same public inference tool through gradio_client;
the function signature mirrors the MCP tool exactly.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.agent_tools_registry import agent_tool


SPACE_URL = os.getenv(
    "SEE_THROUGH_SPACE_URL",
    "https://ljsabc-see-through.ms.show",
).rstrip("/")
SPACE_TOKEN = os.getenv("SEE_THROUGH_SPACE_TOKEN", "").strip()
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "see_through_remote"


def configure_space_url(url: str, token: str | None = None) -> str:
    """Set the active Space URL for the current WebUI process."""
    global SPACE_URL, SPACE_TOKEN
    value = (url or "").strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("MCP/Space 地址必须以 http:// 或 https:// 开头")
    SPACE_URL = value
    if token is not None:
        SPACE_TOKEN = token.strip()
    return SPACE_URL


def check_space(url: str, token: str = "") -> dict:
    """Check the Space public API and report its available endpoint."""
    import httpx

    value = configure_space_url(url, token)
    headers = {"Authorization": f"Bearer {SPACE_TOKEN}"} if SPACE_TOKEN else None
    response = httpx.get(f"{value}/gradio_api/info", headers=headers, timeout=20)
    if response.status_code == 403:
        return {"status": "forbidden", "message": "空间返回 403，请填写有效的魔搭访问令牌或确认空间权限", "url": value}
    response.raise_for_status()
    info = response.json()
    endpoint = info.get("named_endpoints", {}).get("/inference")
    if not endpoint:
        return {"status": "error", "message": "未找到 /inference 接口"}
    params = [p.get("parameter_name") for p in endpoint.get("parameters", [])]
    return {
        "status": "connected",
        "message": f"已连接，可用接口 /inference，参数：{', '.join(params)}",
        "url": value,
    }


def _find_path(value: Any) -> list[str]:
    """Collect file paths returned by Gradio from nested result data."""
    found: list[str] = []
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if os.path.isfile(text):
            found.append(text)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_find_path(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_find_path(item))
    return found


@agent_tool(
    name="see_through_remote",
    description=(
        "调用魔搭 See-Through 空间进行远程动漫角色图层分离，输出分层 PSD。"
        "当用户要求图层分离、拆分角色图层或生成分层 PSD 时使用；需要用户上传图片。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "输入图片路径；通常由助手自动注入用户上传的图片",
            },
            "resolution": {
                "type": "integer",
                "description": "推理分辨率，范围 768-1600，默认 1024",
                "default": 1024,
            },
            "seed": {
                "type": "integer",
                "description": "随机种子，范围 0-9999，默认 42",
                "default": 42,
            },
            "tblr_split": {
                "type": "boolean",
                "description": "是否将左右手臂与腿拆成独立图层",
                "default": False,
            },
        },
        "required": ["image"],
    },
)
def see_through_remote(
    image: str,
    resolution: int = 1024,
    seed: int = 42,
    tblr_split: bool = False,
):
    """Run the public ModelScope See-through Space inference endpoint."""
    if not image or not os.path.isfile(image):
        return None, "找不到输入图片，请先上传一张图片。"

    resolution = max(768, min(1600, int(resolution)))
    seed = max(0, min(9999, int(seed)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from gradio_client import Client

        print(f"[See-through] Calling remote Space: {SPACE_URL}")
        headers = {"Authorization": f"Bearer {SPACE_TOKEN}"} if SPACE_TOKEN else None
        client = Client(SPACE_URL, headers=headers)
        result = client.predict(
            image=image,
            resolution=resolution,
            seed=seed,
            tblr_split=bool(tblr_split),
            api_name="/inference",
        )
        returned = _find_path(result)
        if not returned:
            return None, f"See-Through 已返回，但没有找到输出文件：{result!r}"

        job_dir = OUTPUT_DIR / f"run_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        local_files: list[str] = []
        for source in returned:
            target = job_dir / Path(source).name
            if Path(source).resolve() != target.resolve():
                shutil.copy2(source, target)
            local_files.append(str(target))

        psd = next((p for p in local_files if p.lower().endswith(".psd")), None)
        previews = [p for p in local_files if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        info = {
            "status": "success",
            "backend": "ModelScope See-through",
            "space": SPACE_URL,
            "resolution": resolution,
            "seed": seed,
            "tblr_split": bool(tblr_split),
            "psd_path": psd,
            "files": local_files,
        }
        return previews[:12], info
    except Exception as exc:
        print(f"[See-through] Remote inference failed: {exc}")
        return None, f"调用魔搭 See-Through 失败：{exc}"
