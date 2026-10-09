"""
Forge WebUI MCP 一键安装脚本
自动检测系统上的 MCP 客户端并写入 forge-webui 配置
使用方法: python install_mcp.py
"""

import json
import os
import sys
import shutil
from pathlib import Path


def get_python_path():
    """获取 WebUI 自带的 Python 路径"""
    # 优先使用 WebUI 自带的 python
    webui_python = Path(__file__).parent.parent / "system" / "python" / "python.exe"
    if webui_python.exists():
        return str(webui_python)
    # 回退到系统 python
    return sys.executable


def get_server_dir():
    """MCP 服务器目录"""
    return str(Path(__file__).parent)


def get_server_script():
    """MCP 服务器脚本路径"""
    return str(Path(__file__).parent / "server.py")


def get_start_bat():
    """启动脚本路径"""
    return str(Path(__file__).parent / "start.bat")


def get_mcp_config():
    """生成 forge-webui MCP 配置"""
    return {
        "name": "forge-webui",
        "description": "Forge WebUI 全能工具：生图、生视频、生语音、图层拆分",
        "enabled": True,
        "transport": "stdio",
        "command": get_python_path(),
        "args": [get_server_script()],
        "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        "cwd": get_server_dir()
    }


def install_qwenpaw():
    """安装到 QwenPaw"""
    config_path = Path.home() / ".qwenpaw" / "workspaces" / "default" / "agent.json"
    if not config_path.exists():
        return False, "未找到 QwenPaw 配置"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "mcp" not in config:
            config["mcp"] = {"clients": {}, "migration_version": 0}
        if "clients" not in config["mcp"]:
            config["mcp"]["clients"] = {}

        config["mcp"]["clients"]["forge-webui"] = get_mcp_config()
        # 重置迁移版本，让 QwenPaw 重新加载 MCP 配置
        config["mcp"]["migration_version"] = 0

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True, f"已写入 QwenPaw 配置: {config_path}"
    except Exception as e:
        return False, f"QwenPaw 配置失败: {e}"


def install_codex():
    """安装到 OpenAI Codex"""
    config_path = Path.home() / ".codex" / "config.json"
    if not config_path.exists():
        return False, "未找到 Codex 配置"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["forge-webui"] = {
            "command": get_python_path(),
            "args": [get_server_script()],
            "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True, f"已写入 Codex 配置: {config_path}"
    except Exception as e:
        return False, f"Codex 配置失败: {e}"


def install_claude_code():
    """安装到 Claude Code"""
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return False, "未找到 Claude Code 配置"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["forge-webui"] = {
            "command": get_python_path(),
            "args": [get_server_script()],
            "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True, f"已写入 Claude Code 配置: {config_path}"
    except Exception as e:
        return False, f"Claude Code 配置失败: {e}"


def install_qwen_code():
    """安装到 Qwen Code"""
    # Qwen Code 配置可能在多个位置
    possible_paths = [
        Path.home() / ".qwen" / "settings.json",
        Path.home() / ".config" / "qwen-code" / "settings.json",
    ]

    config_path = None
    for p in possible_paths:
        if p.exists():
            config_path = p
            break

    if not config_path:
        return False, "未找到 Qwen Code 配置"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["forge-webui"] = {
            "command": get_python_path(),
            "args": [get_server_script()],
            "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True, f"已写入 Qwen Code 配置: {config_path}"
    except Exception as e:
        return False, f"Qwen Code 配置失败: {e}"


def ensure_start_bat():
    """确保 start.bat 存在且内容正确"""
    start_bat = Path(get_start_bat())
    python_path = get_python_path()
    server_script = get_server_script()

    # 注意：不要加 chcp 命令，会输出非 UTF-8 内容干扰 MCP stdio
    content = f'@echo off\nset PYTHONUTF8=1\nset PYTHONIOENCODING=utf-8\n"{python_path}" "{server_script}" %*\n'

    try:
        # 用 utf-8-sig 去掉 BOM，确保纯文本
        with open(start_bat, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"[错误] 无法创建 start.bat: {e}")
        return False


def main():
    print("=" * 60)
    print("  Forge WebUI MCP 一键安装")
    print("=" * 60)
    print()

    # 1. 确保 start.bat 存在
    print("[1/2] 生成启动脚本...")
    if not ensure_start_bat():
        print("启动脚本生成失败，退出。")
        return
    print(f"  ✓ 启动脚本: {get_start_bat()}")
    print(f"  ✓ Python: {get_python_path()}")
    print()

    # 2. 检测并安装到各 MCP 客户端
    print("[2/2] 检测 MCP 客户端并写入配置...")
    print()

    installers = [
        ("QwenPaw", install_qwenpaw),
        ("OpenAI Codex", install_codex),
        ("Claude Code", install_claude_code),
        ("Qwen Code", install_qwen_code),
    ]

    installed = []
    not_found = []

    for name, installer in installers:
        ok, msg = installer()
        if ok:
            print(f"  ✓ {name}: {msg}")
            installed.append(name)
        else:
            print(f"  - {name}: {msg}")
            if "未找到" in msg:
                not_found.append(name)
            else:
                print(f"    [错误] {msg}")

    print()
    print("=" * 60)
    if installed:
        print(f"安装成功！已配置: {', '.join(installed)}")
        print()
        print("使用说明:")
        print("  1. 重启对应的 MCP 客户端")
        print("  2. 确保 Forge WebUI 已启动")
        print("  3. 对 AI 说:'帮我画一只猫' 即可调用生图")
    else:
        print("未检测到支持的 MCP 客户端。")
        print("支持的客户端: QwenPaw, Codex, Claude Code, Qwen Code")
        print("请先安装其中一个客户端，再运行此脚本。")
    print("=" * 60)


if __name__ == "__main__":
    main()
    try:
        input("\n按回车键退出...")
    except EOFError:
        pass
