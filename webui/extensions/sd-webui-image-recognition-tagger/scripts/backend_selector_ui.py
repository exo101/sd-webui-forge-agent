"""
后端选择器 UI 组件（可被三个模块复用）。
提供：后端选择下拉框、模型列表下拉框、刷新按钮、状态文本。
"""

import gradio as gr
import logging
import sys
from pathlib import Path

# This module can be imported directly by another WebUI script before
# main.py has initialized the extension path.  Make the sibling backend
# module resolvable regardless of extension load order.
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

logger = logging.getLogger(__name__)

try:
    from llm_backend import (
        detect_backends,
        get_available_backends,
        get_models,
        test_connection,
        BACKEND_CONFIG,
    )
    LLM_BACKEND_AVAILABLE = True
except Exception as e:
    LLM_BACKEND_AVAILABLE = False
    logger.error(f"llm_backend 模块不可用: {e}")


def build_backend_selector(default_backend: str = None):
    """
    构建后端选择器 UI 组件组。

    Returns:
        dict: 包含 backend_dropdown, model_dropdown, refresh_btn, status_text, model_status_text
    """
    available = get_available_backends() if LLM_BACKEND_AVAILABLE else []
    all_choices = [(v["label"], k) for k, v in BACKEND_CONFIG.items()] if LLM_BACKEND_AVAILABLE else []

    # 默认选择第一个可用后端，否则取第一个选项
    if default_backend and any(c[1] == default_backend for c in all_choices):
        default_value = default_backend
    elif available:
        default_value = available[0]
    elif all_choices:
        default_value = all_choices[0][1]
    else:
        default_value = None

    with gr.Row():
        backend_dropdown = gr.Dropdown(
            choices=all_choices,
            value=default_value,
            label="🔌 后端服务（自动检测）",
            interactive=True,
            scale=2,
        )
        refresh_btn = gr.Button("🔄 重新检测", scale=0, min_width=100)

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=[],
            value="",
            label="🤖 模型",
            interactive=True,
            allow_custom_value=True,
            scale=4,
        )
        refresh_models_btn = gr.Button("🔄 刷新模型", scale=0, min_width=100)

    status_text = gr.Textbox(
        label="连接状态",
        value=_build_initial_status(available),
        interactive=False,
        lines=1,
        show_label=False,
    )

    # 事件绑定
    def on_backend_change(backend):
        if not backend:
            return gr.update(choices=[], value=""), "❌ 请选择后端"
        models = get_models(backend) if LLM_BACKEND_AVAILABLE else []
        ok, msg = test_connection(backend) if LLM_BACKEND_AVAILABLE else (False, "llm_backend 不可用")
        if models:
            return gr.update(choices=models, value=models[0]), msg
        return gr.update(choices=[], value=""), msg

    def on_refresh_backends():
        available = get_available_backends() if LLM_BACKEND_AVAILABLE else []
        status = _build_initial_status(available)
        # 不改变当前选择，只刷新状态
        return status

    def on_refresh_models(backend):
        if not backend:
            return gr.update(choices=[], value=""), "❌ 请先选择后端"
        models = get_models(backend) if LLM_BACKEND_AVAILABLE else []
        if models:
            return gr.update(choices=models, value=models[0]), f"✅ 发现 {len(models)} 个模型"
        return gr.update(choices=[], value=""), "⚠️ 未检测到模型，请确认模型已加载"

    backend_dropdown.change(
        fn=on_backend_change,
        inputs=[backend_dropdown],
        outputs=[model_dropdown, status_text],
    )
    refresh_btn.click(
        fn=on_refresh_backends,
        inputs=[],
        outputs=[status_text],
    )
    refresh_models_btn.click(
        fn=on_refresh_models,
        inputs=[backend_dropdown],
        outputs=[model_dropdown, status_text],
    )

    # 初始化时加载一次模型列表
    if default_value:
        initial_models = get_models(default_value) if LLM_BACKEND_AVAILABLE else []
        if initial_models:
            model_dropdown.choices = initial_models
            model_dropdown.value = initial_models[0]

    return {
        "backend_dropdown": backend_dropdown,
        "model_dropdown": model_dropdown,
        "refresh_btn": refresh_btn,
        "refresh_models_btn": refresh_models_btn,
        "status_text": status_text,
    }


def _build_initial_status(available):
    if not available:
        return "⚠️ 未检测到任何后端服务，请启动 llama.cpp / LM Studio / Ollama 后点击「重新检测」"
    labels = []
    for bt in available:
        info = BACKEND_CONFIG.get(bt, {})
        labels.append(info.get("label", bt))
    return f"✅ 已检测到: {', '.join(labels)}"
