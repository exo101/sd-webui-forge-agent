"""ModelScope MCP adapter for VAST-AI-Research/TripoSplat-Demo."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from scripts.agent_tools_registry import agent_tool

TRIPOSPLAT_URL = "https://studio-vast-ai-research-triposplat-demo.api-inference.modelscope.net"
MCP_PATH = "/gradio_api/mcp/"
TRIPOSPLAT_TOKEN = os.getenv("TRIPOSPLAT_MCP_TOKEN", "").strip()


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TRIPOSPLAT_TOKEN}"} if TRIPOSPLAT_TOKEN else {}


def configure_token(token: str) -> None:
    global TRIPOSPLAT_TOKEN
    TRIPOSPLAT_TOKEN = (token or "").strip()


def _image_data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


async def _call_generate(args: dict[str, Any]) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        f"{TRIPOSPLAT_URL}{MCP_PATH}", headers=_headers(), timeout=60, sse_read_timeout=900
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool("generate", args)


def _run(coro):
    return asyncio.run(coro)


def _content_text(result: Any) -> str:
    values = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            values.append(block.text)
    return "\n".join(values) or "TripoSplat 未返回文本结果。"


@agent_tool(
    name="triposplat_remote",
    description="通过 ModelScope MCP 调用 TripoSplat，将上传的单张图片生成 3D Gaussian Splat 模型。",
    parameters={
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "输入图片路径，由助手自动注入"},
            "seed": {"type": "integer", "default": 42},
            "steps": {"type": "number", "default": 20},
            "guidance_scale": {"type": "number", "default": 3.0},
            "num_gaussians": {"type": "string", "enum": ["32768", "65536", "131072", "262144"], "default": "262144"},
            "output_format": {"type": "string", "enum": ["ply", "splat"], "default": "ply"},
        },
        "required": ["image"],
    },
)
def triposplat_remote(image: str, seed: int = 42, steps: float = 20, guidance_scale: float = 3.0, num_gaussians: str = "262144", output_format: str = "ply"):
    if not image or not os.path.isfile(image):
        return None, "找不到输入图片，请先上传一张图片。"
    try:
        result = _run(_call_generate({
            "prepared": _image_data_url(image),
            "seed": int(seed),
            "steps": steps,
            "guidance_scale": guidance_scale,
            "num_gaussians": str(num_gaussians),
            "output_format": output_format,
        }))
        text = _content_text(result)
        if getattr(result, "isError", False):
            return None, {"status": "error", "backend": "ModelScope MCP", "error": text}
        return [], {"status": "success", "backend": "ModelScope MCP", "tool": "generate", "result": text}
    except Exception as exc:
        return None, f"调用 TripoSplat MCP 失败：{exc}"
