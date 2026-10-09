"""Forge WebUI MCP 服务器（代理绘梦智能体）

直接复用绘梦智能体的完整工具实现，MCP 只做协议转换。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_WEBUI_PORT = 7860


def _http_get(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, payload: dict[str, Any], timeout: float = 300.0) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AgentClient:
    """绘梦智能体 API 客户端。"""

    def __init__(self, webui_url: str):
        self.webui_url = webui_url.rstrip("/")

    def list_tools(self) -> list[dict[str, Any]]:
        result = _http_get(f"{self.webui_url}/agent/api/tools")
        return result.get("tools", [])

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any],
                     image_path: str | None = None) -> dict[str, Any]:
        payload = {
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
        if image_path:
            payload["image_path"] = image_path
        return _http_post(f"{self.webui_url}/agent/api/execute_tool", payload)


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _convert_tool_schema(agent_tool: dict[str, Any]) -> dict[str, Any]:
    """将绘梦智能体的 tool schema 转换为 MCP 格式。

    绘梦智能体使用 OpenAI function calling 格式:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    func = agent_tool.get("function", agent_tool)
    return {
        "name": func["name"],
        "description": func.get("description", ""),
        "inputSchema": func.get("parameters", {"type": "object", "properties": {}}),
    }


DEFAULT_WEBUI_PORT = 7860
# 常见 WebUI 端口列表，按优先级排序
COMMON_PORTS = [7860, 7861, 7862, 7863, 7864, 7865, 8080, 8081, 8082]


def _detect_webui_port(preferred: int | None = None) -> int:
    """自动检测 WebUI 端口。

    优先级：命令行参数 > launcher_config.json > 扫描常见端口
    """
    # 1. 命令行指定的端口
    if preferred:
        return preferred

    # 2. 从 launcher_config.json 读取
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "launcher_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            port = config.get("port")
            if isinstance(port, int) and port > 0:
                return port
        except Exception:
            pass

    # 3. 扫描常见端口，找到可用的 WebUI
    for port in COMMON_PORTS:
        try:
            url = f"http://127.0.0.1:{port}/agent/api/tools"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return port
        except Exception:
            continue

    # 4. 都没找到，返回默认端口
    return DEFAULT_WEBUI_PORT


def main():
    parser = argparse.ArgumentParser(description="Forge WebUI MCP 服务器（代理绘梦智能体）")
    parser.add_argument("--webui-port", type=int, default=None, help="WebUI 端口（不指定则自动检测）")
    args = parser.parse_args()

    # 自动检测端口
    webui_port = _detect_webui_port(args.webui_port)
    webui_url = f"http://127.0.0.1:{webui_port}"
    client = AgentClient(webui_url)

    # 启动时获取工具列表
    try:
        agent_tools = client.list_tools()
        tools = [_convert_tool_schema(t) for t in agent_tools]
        sys.stderr.write(f"[Forge MCP] 已连接绘梦智能体，加载 {len(tools)} 个工具\n")
    except Exception as e:
        sys.stderr.write(f"[Forge MCP] 警告: 无法获取工具列表 ({e})，将在首次调用时重试\n")
        tools = []

    sys.stderr.write(f"[Forge MCP] WebUI: {webui_url}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "forge-webui-mcp", "version": "3.0.0"},
                },
            }
        elif method == "tools/list":
            # 每次都重新获取，确保工具列表最新
            if not tools:
                try:
                    agent_tools = client.list_tools()
                    tools = [_convert_tool_schema(t) for t in agent_tools]
                except Exception:
                    pass
            response = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}
        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            # 提取图片路径参数（绘梦智能体的工具用 image 参数）
            image_path = tool_args.pop("image", None) or tool_args.pop("image_path", None)

            try:
                result = client.execute_tool(tool_name, tool_args, image_path=image_path)
                result_str = result.get("result", "")
                images = result.get("images", [])

                parts = []
                if result_str:
                    parts.append({"type": "text", "text": result_str})
                for i, img_path in enumerate(images):
                    if img_path and isinstance(img_path, str):
                        # 读取图片并转 base64，用 MCP image 类型返回
                        try:
                            img_file = Path(img_path)
                            if img_file.is_file():
                                suffix = img_file.suffix.lower()
                                mime = {
                                    ".png": "image/png",
                                    ".jpg": "image/jpeg",
                                    ".jpeg": "image/jpeg",
                                    ".webp": "image/webp",
                                    ".gif": "image/gif",
                                }.get(suffix, "image/png")
                                img_data = base64.b64encode(img_file.read_bytes()).decode("ascii")
                                parts.append({
                                    "type": "image",
                                    "data": img_data,
                                    "mimeType": mime,
                                })
                                parts.append({"type": "text", "text": f"[图片 {i+1}] {img_path}"})
                            else:
                                parts.append({"type": "text", "text": f"[图片 {i+1}]\n{img_path}"})
                        except Exception:
                            parts.append({"type": "text", "text": f"[图片 {i+1}]\n{img_path}"})
                if not parts:
                    parts.append({"type": "text", "text": "工具执行完成"})

                response = {"jsonrpc": "2.0", "id": msg_id, "result": {"content": parts}}
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                response = {
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"HTTP {e.code}: {body[:500]}"}], "isError": True},
                }
            except Exception as e:
                response = {
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"错误: {e}"}], "isError": True},
                }
        else:
            continue

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
