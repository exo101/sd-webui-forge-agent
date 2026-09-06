"""Import a Gradio Python API example and register it as an Agent tool."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import httpx

from scripts.agent_tools_registry import agent_tool


def _literal(value: str) -> Any:
    try:
        return ast.literal_eval(value.strip())
    except Exception:
        return None


def parse_example(example: str) -> dict[str, Any]:
    text = str(example or "")
    url_match = re.search(r"(?:Client|client)\s*\(\s*['\"](https?://[^'\"]+)", text)
    api_matches = list(re.finditer(r"api_name\s*=\s*['\"](/[^'\"]+)['\"]", text))
    if not url_match:
        raise ValueError("没有找到 Client('https://...') 的 Space 地址")
    if not api_matches:
        raise ValueError("没有找到 api_name='/...' 接口名称")
    api_match = api_matches[-1]
    call_start = text.rfind("predict(", 0, api_match.start())
    call_text = text[call_start:api_match.start()] if call_start >= 0 else ""
    params: dict[str, Any] = {}
    for match in re.finditer(r"(\w+)\s*=\s*([^,\n\)]+)", call_text):
        key, raw = match.group(1), match.group(2).strip()
        if key in {"self", "api_name"} or "handle_file" in raw:
            continue
        value = _literal(raw)
        if value is not None:
            params[key] = value
    image_param = None
    image_match = re.search(r"(\w+)\s*=\s*handle_file\s*\(", call_text)
    if image_match:
        image_param = image_match.group(1)
    host = re.sub(r"^https?://", "", url_match.group(1)).split("/", 1)[0]
    return {"url": url_match.group(1).rstrip("/"), "api_name": api_match.group(1), "params": params, "image_param": image_param, "name": f"{host}{api_match.group(1).replace('/', '_')}"}


def register_project(project: dict[str, Any]) -> str:
    tool_name = "gradio_api_" + re.sub(r"[^a-zA-Z0-9_]+", "_", project["name"]).strip("_").lower()
    params = dict(project.get("params") or {})
    image_param = project.get("image_param")
    properties = {}
    if image_param:
        properties["image"] = {"type": "string", "description": "输入图片路径，由助手自动注入"}
    for key, value in params.items():
        if isinstance(value, bool): kind = "boolean"
        elif isinstance(value, int): kind = "integer"
        elif isinstance(value, float): kind = "number"
        elif isinstance(value, list): kind = "array"
        else: kind = "string"
        properties[key] = {"type": kind, "default": value, "description": f"Gradio 参数 {key}"}
    schema = {"type": "object", "properties": properties}
    if image_param:
        schema["required"] = ["image"]

    def dynamic_tool(image: str = None, **kwargs):
        from gradio_client import Client, handle_file
        client = Client(project["url"], verbose=False)
        call_args = dict(params)
        if image_param:
            call_args[image_param] = handle_file(image)
        for key, value in kwargs.items():
            if value is not None:
                call_args[key] = value
        result = client.predict(api_name=project["api_name"], **call_args)
        return [], {"status": "success", "backend": "Gradio API", "project": project["name"], "api_name": project["api_name"], "result": result}

    dynamic_tool.__name__ = tool_name
    dynamic_tool.__doc__ = f"调用 Gradio API：{project['url']}{project['api_name']}"
    agent_tool(tool_name, dynamic_tool.__doc__, schema)(dynamic_tool)
    return tool_name


def import_and_register(example: str) -> tuple[dict[str, Any], str]:
    project = parse_example(example)
    project["tool_name"] = register_project(project)
    return project, project["tool_name"]


def check_project(project: dict[str, Any]) -> dict[str, Any]:
    """Check the Space page before saving it as an imported Gradio project."""
    url = str(project.get("url") or "").rstrip("/")
    if not url:
        return {"status": "error", "message": "Space 地址为空"}
    try:
        response = httpx.get(url, timeout=20, follow_redirects=True)
        if response.status_code in (401, 403):
            return {"status": "reachable", "message": f"地址可达，但 Space 首页返回 HTTP {response.status_code}；将由 Gradio API 调用验证接口"}
        if response.status_code >= 400:
            return {"status": "error", "message": f"HTTP {response.status_code}：{url}"}
        return {"status": "connected", "message": f"Gradio API 页面可访问：{url}"}
    except Exception as exc:
        return {"status": "error", "message": f"无法访问 Gradio API 页面：{exc}"}
