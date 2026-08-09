"""Forge 兼容：原始 SD WebUI 的 generation_parameters_copypaste 模块"""
import gradio as gr


def create_buttons(buttons: list, **kwargs) -> list:
    return [gr.Button(b, elem_id=f"{b}_button") for b in buttons]


def bind_buttons(buttons, tabname, fn):
    """将按钮绑定到输出组件（Forge 兼容桩）"""
    pass