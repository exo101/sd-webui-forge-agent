"""ModelScope Gradio API adapter for xmccln/TRELLIS.2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts.agent_tools_registry import agent_tool

TRELLIS_URL = "https://xmccln-trellis-2.ms.show"


def configure_token(token: str) -> None:
    """保留统一 MCP 设置接口；公开 ms.show API 不需要额外 Token。"""
    return None


def check_space() -> dict[str, Any]:
    """检查 TRELLIS.2 的 Gradio API 页面；该 Space 不提供 MCP 端点。"""
    import httpx
    try:
        response = httpx.get(TRELLIS_URL, timeout=30)
        if response.status_code < 400:
            return {"status": "connected", "message": "TRELLIS.2 Gradio API 可访问，已发现 /image_to_3d"}
        return {"status": "error", "message": f"TRELLIS.2 Gradio API 返回 HTTP {response.status_code}"}
    except Exception as exc:
        return {"status": "error", "message": f"TRELLIS.2 API 检测失败：{exc}"}


@agent_tool(
    name="trellis_remote",
    description="通过 ModelScope Gradio API 调用 TRELLIS.2，将单张图片生成 3D 模型。",
    parameters={
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "输入图片路径，由助手自动注入"},
            "seed": {"type": "integer", "default": 0},
            "resolution": {"type": "string", "enum": ["512", "1024"], "default": "1024"},
        },
        "required": ["image"],
    },
)
def trellis_remote(image: str, seed: int = 0, resolution: str = "1024"):
    if not image or not os.path.isfile(image):
        return None, "找不到输入图片，请先上传一张图片。"
    try:
        from gradio_client import Client, handle_file

        client = Client(TRELLIS_URL)
        client.predict(api_name="/start_session")
        prepared = client.predict(
            image=handle_file(image), api_name="/preprocess_image_1"
        )
        result = client.predict(
            image=prepared,
            seed=int(seed),
            resolution=str(resolution),
            ss_guidance_strength=7.5,
            ss_guidance_rescale=0.7,
            ss_sampling_steps=12,
            ss_rescale_t=5,
            shape_slat_guidance_strength=7.5,
            shape_slat_guidance_rescale=0.5,
            shape_slat_sampling_steps=12,
            shape_slat_rescale_t=3,
            tex_slat_guidance_strength=1,
            tex_slat_guidance_rescale=0,
            tex_slat_sampling_steps=12,
            tex_slat_rescale_t=3,
            api_name="/image_to_3d",
        )
        return [], {"status": "success", "backend": "ModelScope Gradio API", "project": "TRELLIS.2", "result": result}
    except Exception as exc:
        return None, f"调用 TRELLIS.2 失败：{exc}"
