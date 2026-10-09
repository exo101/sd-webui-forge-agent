"""
统一 LLM 后端管理模块
支持 llama.cpp / Ollama / LM Studio 三种后端，自动检测可用端口。

所有后端均使用 OpenAI 兼容 API（/v1/chat/completions）：
- llama.cpp: 默认端口 8080
- LM Studio: 默认端口 1234
- Ollama:   默认端口 11434（v0.1.38+ 支持 OpenAI 兼容接口）
"""

import socket
import requests
import logging
import base64
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ==================== 后端默认配置 ====================

BACKEND_CONFIG = {
    "llamacpp": {
        "label": "🦙 llama.cpp",
        "default_ports": [8080, 8079, 8081],
        "default_host": "localhost",
    },
    "lmstudio": {
        "label": "🏠 LM Studio",
        "default_ports": [1234, 1235, 1236],
        "default_host": "localhost",
    },
    "ollama": {
        "label": "🦫 Ollama",
        "default_ports": [11434],
        "default_host": "localhost",
    },
}

# 缓存已检测到的后端（避免每次调用都扫描端口）
_detected_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamp: float = 0.0
_CACHE_TTL = 30.0  # 缓存 30 秒


def _is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """快速检测端口是否开放（TCP 层）"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _probe_backend_api(base_url: str, backend_type: str) -> bool:
    """探测后端是否真正提供视觉/聊天 API（HTTP 层验证）"""
    endpoints = [
        f"{base_url.rstrip('/')}/v1/models",
    ]
    if backend_type == "ollama":
        endpoints.append(f"{base_url.rstrip('/')}/api/tags")

    for ep in endpoints:
        try:
            resp = requests.get(ep, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                # 只要返回了 JSON 且包含模型列表字段即认为可用
                if any(k in data for k in ("data", "models", "model")):
                    return True
                # ollama /api/tags 返回 {"models": [...]}
                if isinstance(data, dict) and "models" in data:
                    return True
        except Exception:
            continue
    return False


def detect_backends(force: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    自动检测所有可用的后端服务。

    Returns:
        dict: {backend_type: {"base_url": ..., "label": ..., "available": bool}}
              其中 available=True 的后端是实际可连接的。
    """
    global _detected_cache, _cache_timestamp

    now = time.time()
    if not force and _detected_cache and (now - _cache_timestamp) < _CACHE_TTL:
        return _detected_cache

    result: Dict[str, Dict[str, Any]] = {}

    for backend_type, cfg in BACKEND_CONFIG.items():
        host = cfg["default_host"]
        found_url: Optional[str] = None

        # 1. 先扫默认端口
        for port in cfg["default_ports"]:
            if not _is_port_open(host, port):
                continue
            url = f"http://{host}:{port}"
            if _probe_backend_api(url, backend_type):
                found_url = url
                break

        # 2. 兜底：再扫一圈相邻端口（防止用户改过端口，仅扫 ±5）
        if found_url is None:
            base = cfg["default_ports"][0]
            for offset in range(-5, 6):
                if offset == 0:
                    continue
                port = base + offset
                if port in cfg["default_ports"] or port < 1024:
                    continue
                if not _is_port_open(host, port):
                    continue
                url = f"http://{host}:{port}"
                if _probe_backend_api(url, backend_type):
                    found_url = url
                    break

        if found_url:
            result[backend_type] = {
                "base_url": found_url,
                "label": cfg["label"],
                "available": True,
            }
            logger.info(f"✅ 检测到 {cfg['label']} 服务: {found_url}")
        else:
            result[backend_type] = {
                "base_url": f"http://{cfg['default_host']}:{cfg['default_ports'][0]}",
                "label": cfg["label"],
                "available": False,
            }

    _detected_cache = result
    _cache_timestamp = now
    return result


def get_available_backends() -> List[str]:
    """返回当前可用的后端类型列表"""
    backends = detect_backends()
    return [k for k, v in backends.items() if v.get("available")]


def get_backend_url(backend_type: str) -> Optional[str]:
    """获取指定后端的 base_url，若不可用返回 None"""
    backends = detect_backends()
    info = backends.get(backend_type)
    if info and info.get("available"):
        return info["base_url"]
    return None


def get_first_available_backend() -> Optional[str]:
    """返回第一个可用的后端类型，用于自动选择"""
    for bt in ("llamacpp", "lmstudio", "ollama"):
        if get_backend_url(bt):
            return bt
    return None


def get_models(backend_type: str, base_url: Optional[str] = None) -> List[str]:
    """
    获取指定后端上加载的模型列表。

    Args:
        backend_type: llamacpp / lmstudio / ollama
        base_url: 可选，手动指定服务地址

    Returns:
        模型 ID 列表
    """
    if base_url is None:
        base_url = get_backend_url(backend_type)
    if base_url is None:
        return []

    models: List[str] = []

    # llama.cpp / LM Studio: /v1/models
    if backend_type in ("llamacpp", "lmstudio"):
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("data", []):
                if isinstance(m, dict) and "id" in m:
                    models.append(m["id"])
        except Exception as e:
            logger.debug(f"获取 {backend_type} 模型列表失败: {e}")

    # Ollama: /api/tags
    elif backend_type == "ollama":
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("models", []):
                if isinstance(m, dict) and "name" in m:
                    models.append(m["name"])
                elif isinstance(m, str):
                    models.append(m)
        except Exception as e:
            logger.debug(f"获取 ollama 模型列表失败: {e}")

    # 去重
    return list(dict.fromkeys(models))


def analyze_image(
    image_path: str,
    prompt: str,
    model: str,
    backend_type: str,
    base_url: Optional[str] = None,
    timeout: int = 300,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    统一图像分析接口（支持 llama.cpp / LM Studio / Ollama）。

    三种后端统一使用 OpenAI 兼容的 /v1/chat/completions 接口。

    Args:
        image_path: 图片路径
        prompt: 分析提示词
        model: 模型名称
        backend_type: llamacpp / lmstudio / ollama
        base_url: 可选，手动指定服务地址
        timeout: 请求超时（秒）
        max_tokens: 最大输出 token
        temperature: 采样温度

    Returns:
        {"success": bool, "analysis": str, "model": str, "image_path": str}
    """
    if base_url is None:
        base_url = get_backend_url(backend_type)
    if base_url is None:
        return {
            "success": False,
            "analysis": f"❌ 后端 {BACKEND_CONFIG.get(backend_type, {}).get('label', backend_type)} 未检测到运行中的服务",
        }

    # 编码图片
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return {"success": False, "analysis": f"❌ 图片编码失败: {e}"}

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ],
    }]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # 兼容推理模型（reasoning_content）
        if not content or not content.strip():
            reasoning = data.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
            if reasoning and reasoning.strip():
                content = reasoning.strip()

        if not content:
            return {
                "success": False,
                "analysis": "⚠️ 模型返回内容为空",
            }

        return {
            "success": True,
            "analysis": content,
            "model": model,
            "image_path": image_path,
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "analysis": (
                f"❌ 无法连接到 {BACKEND_CONFIG.get(backend_type, {}).get('label', backend_type)} 服务 ({base_url})\n"
                f"请确认服务已启动。"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "analysis": f"❌ 请求超时（{timeout}秒），请检查模型是否已加载或使用更小的图片",
        }
    except Exception as e:
        return {"success": False, "analysis": f"❌ 分析出错: {e}"}


def test_connection(backend_type: str, base_url: Optional[str] = None) -> Tuple[bool, str]:
    """测试指定后端的连接状态"""
    if base_url is None:
        base_url = get_backend_url(backend_type)
    if base_url is None:
        return False, f"❌ 未检测到 {BACKEND_CONFIG.get(backend_type, {}).get('label', backend_type)} 服务"

    if backend_type == "ollama":
        test_url = f"{base_url.rstrip('/')}/api/tags"
    else:
        test_url = f"{base_url.rstrip('/')}/v1/models"

    try:
        resp = requests.get(test_url, timeout=5)
        if resp.status_code == 200:
            models = get_models(backend_type, base_url)
            if models:
                return True, f"✅ 连接成功，已加载 {len(models)} 个模型: {', '.join(models[:5])}"
            return True, "✅ 连接成功，但未检测到模型"
        return False, f"❌ 响应异常: HTTP {resp.status_code}"
    except Exception as e:
        return False, f"❌ 无法连接: {e}"
