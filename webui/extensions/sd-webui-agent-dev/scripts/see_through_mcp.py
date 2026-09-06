"""ModelScope See-through MCP client for the DreamArt Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from scripts.agent_tools_registry import agent_tool

API_SPACE_URL = "https://studio-ljsabc-see-through.api-inference.modelscope.net"
LEGACY_SPACE_URL = "https://ljsabc-see-through.ms.show"
SPACE_URL = os.getenv("SEE_THROUGH_SPACE_URL", API_SPACE_URL).rstrip("/")
MCP_PATH = "/gradio_api/mcp/"
SPACE_TOKEN = os.getenv("SEE_THROUGH_SPACE_TOKEN", "").strip()
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "see_through_remote"


def configure_space_url(url: str, token: str | None = None) -> str:
    global SPACE_URL, SPACE_TOKEN
    value = (url or "").strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ValueError("MCP/Space 地址必须以 http:// 或 https:// 开头")
    # 普通 .ms.show 地址会返回 403，不能用于 SDK Token 的 MCP 访问。
    # 即使旧配置仍存在，也必须在真正发起请求前强制改成 API 专用地址。
    if value == LEGACY_SPACE_URL or value.startswith(f"{LEGACY_SPACE_URL}/"):
        value = API_SPACE_URL
        print(f"[See-through MCP] 已将旧地址 {LEGACY_SPACE_URL} 强制迁移为 {API_SPACE_URL}")
    SPACE_URL = value
    if token is not None:
        SPACE_TOKEN = token.strip()
    return SPACE_URL


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SPACE_TOKEN}"} if SPACE_TOKEN else {}


async def _list_mcp_tools() -> list[Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(f"{SPACE_URL}{MCP_PATH}", headers=_headers(), timeout=60, sse_read_timeout=300) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return list((await session.list_tools()).tools)


async def _call_mcp_tool(args: dict) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(f"{SPACE_URL}{MCP_PATH}", headers=_headers(), timeout=60, sse_read_timeout=900) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool("inference", args)


def _run(coro):
    return asyncio.run(coro)


def _exception_text(exc: BaseException) -> str:
    """展开 ExceptionGroup，避免 MCP 只显示笼统的 TaskGroup 错误。"""
    parts: list[str] = []
    nested = getattr(exc, "exceptions", None)
    if nested:
        for item in nested:
            text = _exception_text(item)
            if text:
                parts.append(text)
    direct = str(exc).strip()
    if direct and direct not in parts:
        parts.insert(0, direct)
    return "；".join(dict.fromkeys(parts)) or exc.__class__.__name__


def _mcp_error_message(exc: BaseException) -> str:
    detail = _exception_text(exc)
    lower = detail.lower()
    if "403" in lower:
        if LEGACY_SPACE_URL in SPACE_URL:
            return (
                "MCP 返回 HTTP 403：当前普通 Space 地址不支持 SDK Token，"
                "请使用 ModelScope API 专用地址："
                f"{API_SPACE_URL}"
            )
        return f"MCP 返回 HTTP 403：API 专用地址已连接，但服务端拒绝了本次请求。原始信息：{detail}"
    if "401" in lower or "authentication failed" in lower or "invalid token" in lower:
        return "MCP 返回 HTTP 401：请填写有效的 ModelScope 访问令牌。"
    return f"MCP 握手失败：{detail}"


def _mcp_result_error(result: Any) -> str:
    """提取 call_tool 返回的 isError 内容，保留服务端真实错误。"""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            value = str(getattr(block, "text", "") or "").strip()
            if value:
                try:
                    parsed = json.loads(value)
                    parts.append(json.dumps(parsed, ensure_ascii=False))
                except Exception:
                    parts.append(value)
        elif block_type == "resource":
            resource = getattr(block, "resource", None)
            value = getattr(resource, "text", None) or getattr(resource, "uri", None)
            if value:
                parts.append(str(value))
    return "；".join(dict.fromkeys(parts)) or "服务端未返回具体错误信息"


def check_space(url: str, token: str = "") -> dict:
    """Perform a real MCP initialize + tools/list handshake."""
    configure_space_url(url, token)
    try:
        tools = _run(_list_mcp_tools())
        names = [tool.name for tool in tools]
        if "inference" not in names:
            return {"status": "error", "message": f"MCP 已连接，但没有 inference 工具：{names}"}
        return {"status": "connected", "message": f"MCP 已连接，已发现 inference 工具（共 {len(names)} 个）", "url": f"{SPACE_URL}{MCP_PATH}", "tools": names}
    except Exception as exc:
        detail = _mcp_error_message(exc)
        status = "forbidden" if "HTTP 403" in detail else ("unauthorized" if "HTTP 401" in detail else "error")
        return {"status": status, "message": detail, "url": SPACE_URL}


def _image_data(path: str) -> dict:
    raw = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return {"path": None, "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", "size": len(raw), "orig_name": Path(path).name, "mime_type": mime, "is_stream": False, "meta": {"_type": "gradio.FileData"}}


def _upload_file(path: str) -> str:
    """上传到 Gradio，再返回 MCP inference 要求的公网文件 URL。"""
    import httpx

    upload_url = f"{SPACE_URL}/gradio_api/upload"
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        response = httpx.post(
            upload_url,
            headers=_headers(),
            files={"files": (Path(path).name, handle, mime)},
            timeout=120,
        )
    response.raise_for_status()
    uploaded = response.json()
    if isinstance(uploaded, list):
        uploaded = uploaded[0] if uploaded else ""
    if not isinstance(uploaded, str) or not uploaded:
        raise RuntimeError(f"Gradio 上传返回格式异常：{uploaded}")
    if uploaded.startswith(("http://", "https://")):
        return uploaded
    # Gradio 文件服务格式：/file=/tmp/gradio/...
    return f"{SPACE_URL}/file={urllib.parse.quote(uploaded, safe='/')}"


def _save_mcp_content(result: Any, job_dir: Path) -> tuple[list[str], str]:
    files: list[str] = []
    texts: list[str] = []

    def visit(value: Any):
        if value is None:
            return
        if isinstance(value, str):
            texts.append(value)
            return
        if isinstance(value, dict):
            if value.get("path") and os.path.isfile(value["path"]):
                target = job_dir / Path(value["path"]).name
                target.write_bytes(Path(value["path"]).read_bytes())
                files.append(str(target))
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            try:
                visit(json.loads(text))
            except Exception:
                visit(text)
        elif getattr(block, "type", None) == "image":
            target = job_dir / f"preview_{len(files)}.png"
            target.write_bytes(base64.b64decode(block.data))
            files.append(str(target))
        elif getattr(block, "type", None) == "resource":
            visit(getattr(block, "resource", None))
    return list(dict.fromkeys(files)), "\n".join(texts)


@agent_tool(
    name="see_through_remote",
    description="通过 ModelScope MCP 调用 See-through 的 inference 工具，将上传的动漫图片分解为分层 PSD。",
    parameters={
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "输入图片路径，由助手自动注入"},
            "resolution": {"type": "integer", "description": "768-1600，默认 1024", "default": 1024},
            "seed": {"type": "integer", "description": "0-9999，默认 42", "default": 42},
            "tblr_split": {"type": "boolean", "description": "拆分左右手臂与腿", "default": False},
        },
        "required": ["image"],
    },
)
def see_through_remote(image: str, resolution: int = 1024, seed: int = 42, tblr_split: bool = False):
    if not image or not os.path.isfile(image):
        return None, "找不到输入图片，请先上传一张图片。"
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        job_dir = OUTPUT_DIR / f"run_{int(time.time())}"
        job_dir.mkdir(parents=True, exist_ok=True)
        image_url = _upload_file(image)
        result = _run(_call_mcp_tool({"image": image_url, "resolution": max(768, min(1600, int(resolution))), "seed": max(0, min(9999, int(seed))), "tblr_split": bool(tblr_split)}))
        if getattr(result, "isError", False):
            detail = _mcp_result_error(result)
            return None, {"status": "error", "backend": "ModelScope MCP", "mcp_url": f"{SPACE_URL}{MCP_PATH}", "error": f"See-through MCP 工具执行失败：{detail}"}
        files, text = _save_mcp_content(result, job_dir)
        previews = [p for p in files if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        psd = next((p for p in files if p.lower().endswith(".psd")), None)
        return previews[:12], {"status": "success", "backend": "ModelScope MCP", "mcp_url": f"{SPACE_URL}{MCP_PATH}", "psd_path": psd, "files": files, "message": text}
    except Exception as exc:
        detail = _exception_text(exc)
        print(f"[See-through MCP] Tool call failed: {detail}")
        return None, f"调用 See-through MCP 失败：{_mcp_error_message(exc)}"
