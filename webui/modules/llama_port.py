"""
读取启动器配置的 llama.cpp 端口

插件通过此模块读取启动器写入的 llama_port.json，自动使用与启动器相同的端口。
无需手动修改配置。

当配置文件不存在时，自动探测常见端口（8079, 8080）上运行的 llama.cpp 服务。
"""
import json
import os
import sys
import socket

# 推断启动器目录（webui 的父目录的 launcher 子目录）
_webui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_launcher_dir = os.path.join(os.path.dirname(_webui_dir), "launcher")
LLAMA_PORT_FILE = os.path.join(_launcher_dir, "llama_port.json")

# 常见 llama.cpp 端口列表（按优先级排序）
_COMMON_PORTS = [8079, 8080]


def _is_port_open(port: int, host: str = "localhost") -> bool:
    """快速检测指定端口是否开放"""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


def _detect_llama_port() -> int:
    """探测实际运行的 llama.cpp 端口（兜底方案）"""
    for port in _COMMON_PORTS:
        if _is_port_open(port):
            return port
    return 8080


def get_llama_port() -> int:
    """获取启动器当前启动的 llama.cpp 端口（默认 8080）

    优先级：
    1. 启动器写入的 llama_port.json 配置文件
    2. 自动探测常见端口上运行的 llama.cpp 服务
    3. 回退到 8080
    """
    # 优先读取启动器配置文件
    try:
        if os.path.exists(LLAMA_PORT_FILE):
            with open(LLAMA_PORT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("port", 8080))
    except Exception:
        pass

    # 配置文件不存在时自动探测
    detected = _detect_llama_port()
    if detected != 8080:
        return detected

    return 8080


def get_llama_url() -> str:
    """获取启动器当前启动的 llama.cpp 服务地址（例如 http://localhost:8080）"""
    port = get_llama_port()
    return f"http://localhost:{port}"


def get_llama_host() -> str:
    """获取 llama 监听地址（默认 0.0.0.0）"""
    try:
        if os.path.exists(LLAMA_PORT_FILE):
            with open(LLAMA_PORT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("host", "0.0.0.0")
    except Exception:
        pass
    return "0.0.0.0"