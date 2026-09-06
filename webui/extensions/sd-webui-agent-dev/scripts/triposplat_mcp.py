"""ModelScope MCP adapter for VAST-AI-Research/TripoSplat-Demo."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import urllib.parse
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


def _upload_file_to_gradio(path: str) -> str:
    """等价实现 Gradio 官方 upload-mcp 的 upload_file_to_gradio。"""
    import httpx

    upload_url = f"{TRIPOSPLAT_URL}/gradio_api/upload"
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        response = httpx.post(
            upload_url,
            headers=_headers(),
            files={"files": (Path(path).name, handle, mime)},
            timeout=120,
        )
    response.raise_for_status()
    values = response.json()
    if not isinstance(values, list) or not values or not isinstance(values[0], str):
        raise RuntimeError(f"upload_file_to_gradio 返回格式异常：{values}")
    remote_path = values[0]
    # 官方 upload-mcp 使用 /gradio_api/file=，不是 /file=。
    return f"{TRIPOSPLAT_URL}/gradio_api/file={urllib.parse.quote(remote_path, safe='/')}"


async def _call_tool(tool_name: str, args: dict[str, Any]) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        f"{TRIPOSPLAT_URL}{MCP_PATH}", headers=_headers(), timeout=60, sse_read_timeout=900
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(tool_name, args)


def _run(coro):
    return asyncio.run(coro)


def _content_text(result: Any) -> str:
    values = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            values.append(block.text)
    return "\n".join(values) or "TripoSplat 未返回文本结果。"


def _prepared_data_url(result: Any) -> str | None:
    """把 on_image_change 返回的 MCP 图片块转换为下一步可接受的 Data URL。"""
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "image" and getattr(block, "data", None):
            mime = getattr(block, "mimeType", None) or "image/webp"
            return f"data:{mime};base64,{block.data}"
    return None


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
        # TripoSplat 的 generate 必须使用 on_image_change 产生的 prepared 图，
        # 不能把原图直接传给 generate。
        prepared_result = _run(_call_tool("on_image_change", {
            "image": _image_data_url(image),
        }))
        if getattr(prepared_result, "isError", False):
            return None, {"status": "error", "backend": "ModelScope MCP", "stage": "on_image_change", "error": _content_text(prepared_result)}
        prepared = _prepared_data_url(prepared_result)
        if not prepared:
            return None, {"status": "error", "backend": "ModelScope MCP", "stage": "on_image_change", "error": "未返回 prepared 预处理图。"}
        result = _run(_call_tool("generate", {
            "prepared": prepared,
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
