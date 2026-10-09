from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import gradio as gr
from modules import script_callbacks

from h3studio.api import register_api

# 本插件 scripts 目录（多媒体子模块所在位置）
_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module_from_file(module_name: str, file_name: str):
    """按绝对文件路径加载模块，不依赖 sys.path。"""
    file_path = _SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _mount_multimedia_tabs():
    """把多媒体功能（音乐生成 / 语音合成 / SCAIL-2 角色动画）作为子标签页直接挂入工作台。"""

    # 1. ACE-Step 音乐生成
    try:
        ace = _load_module_from_file("h3_forge_ace_step_ui", "ace_step_ui.py")
        with gr.TabItem("🎵 音乐生成"):
            ace.create_ace_step_ui(asset_bridge_id="h3studio-asset-bridge-ace")
    except Exception as e:
        with gr.TabItem("🎵 音乐生成"):
            gr.Markdown(f"❌ 模块初始化错误：{e}")

    # 2. Breeze-TTS-2 语音合成
    try:
        btts = _load_module_from_file("breeze_tts_ui", "breeze_tts_ui.py")
        with gr.TabItem("🔊 语音合成"):
            btts.create_breeze_tts_ui(asset_bridge_id="h3studio-asset-bridge-breeze")
    except Exception as e:
        with gr.TabItem("🔊 语音合成"):
            gr.Markdown(f"❌ 模块初始化错误：{e}")

    # 3. SCAIL-2 角色动画/替换
    try:
        scail2 = _load_module_from_file("h3_forge_scail2_ui", "scail2_ui.py")
        with gr.TabItem("🎭 SCAIL-2 角色动画"):
            scail2.create_scail2_ui()
    except Exception as e:
        with gr.TabItem("🎭 SCAIL-2 角色动画"):
            gr.Markdown(f"❌ SCAIL-2 模块初始化错误：{e}")


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as studio:
        with gr.Tabs(elem_id="forge-h3-studio-tabs"):
            # 主工作台（SPA）
            with gr.TabItem("🎬 H3 工作台", id="h3-workbench"):
                gr.HTML('<div id="forge-h3-studio-root" class="h3s-boot">正在载入 MiniMax H3 工作台…</div>')
            # 多媒体功能（直接移植）
            _mount_multimedia_tabs()
    return [(studio, "多模态视频创作工作台", "forge_h3_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs, name="h3_studio_tab")
script_callbacks.on_app_started(register_api, name="h3_studio_api")
