from __future__ import annotations

import gradio as gr
from modules import script_callbacks

from h3studio.api import register_api


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as studio:
        gr.HTML('<div id="forge-h3-studio-root" class="h3s-boot">正在载入 MiniMax H3 工作台…</div>')
    return [(studio, "MiniMax H3 工作台", "forge_h3_studio")]


script_callbacks.on_ui_tabs(on_ui_tabs, name="h3_studio_tab")
script_callbacks.on_app_started(register_api, name="h3_studio_api")

