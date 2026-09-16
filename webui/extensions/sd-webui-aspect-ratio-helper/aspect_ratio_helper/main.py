from __future__ import annotations

import gradio as gr
from modules import script_callbacks
from modules import scripts

import aspect_ratio_helper._constants as _constants
import aspect_ratio_helper._settings as _settings


class AspectRatioStepScript(scripts.Script):
    """Aspect Ratio Helper（融合版）

    独立 "Aspect Ratio Helper" 折叠块已移除。全部尺寸工具
    （图形比例按钮 / 百分比缩放 / 最大最小尺寸缩放）通过
    on_after_component 钩子内联注入到 Width/Height 滑块正下方
    （txt2img 与 img2img 各自尺寸列内），与滑块组成紧凑的
    "尺寸设置簇"，减少独立 UI 块对页面的占用。

    JS 比例锁（Off / 🔒 / ️ 循环按钮）仍位于 ⇅ 按钮旁的
    size_toolbox 中，行为不变。
    """

    def __init__(self):
        self.t2i_w: gr.components.Slider | None = None
        self.t2i_h: gr.components.Slider | None = None
        self.i2i_w: gr.components.Slider | None = None
        self.i2i_h: gr.components.Slider | None = None
        self.wc: gr.components.Slider | None = None
        self.hc: gr.components.Slider | None = None
        self.max_dimension: float = 1024.0
        self._hooks_registered: set = set()
        self._rendered_pages: set = set()

    def title(self) -> str:
        return _constants.EXTENSION_NAME

    def show(self, is_img2img) -> scripts.AlwaysVisible:
        # 必须在此阶段（组件创建之前）注册内联注入钩子
        self._ensure_hook(is_img2img)
        return scripts.AlwaysVisible

    # ------------------------------------------------------------------
    # 内联注入：渲染到 Width/Height 滑块正下方
    # ------------------------------------------------------------------

    def _ensure_hook(self, is_img2img: bool):
        anchor = 'img2img_height' if is_img2img else 'txt2img_height'
        if anchor in self._hooks_registered:
            return
        self._hooks_registered.add(anchor)
        self.on_after_component(
            lambda event, _i2i=is_img2img: self._on_height_created(event, _i2i),
            elem_id=anchor,
        )

    def _sync_wc_hc(self, is_img2img: bool):
        if is_img2img:
            self.wc, self.hc = self.i2i_w, self.i2i_h
        else:
            self.wc, self.hc = self.t2i_w, self.t2i_h

    def _on_height_created(self, event, is_img2img: bool):
        # on_after_component 回调先于脚本自身的 after_component 触发，
        # 这里先记录 height 滑块引用
        if is_img2img:
            self.i2i_h = event.component
        else:
            self.t2i_h = event.component
        self._sync_wc_hc(is_img2img)

        mark = 'img2img' if is_img2img else 'txt2img'
        if mark in self._rendered_pages or self.wc is None or self.hc is None:
            return
        self._rendered_pages.add(mark)
        # 此时 Gradio 上下文正处于该页的尺寸列（column_size）内，
        # 渲染的组件会落在 Height 滑块正下方
        self._render_inline_tools(mark)

    def _render_inline_tools(self, mark: str):
        components = _settings.sort_components_by_keys(
            [component(self) for component in _settings.COMPONENTS],
        )
        if not any(c.should_show() for c in components):
            return

        with gr.Group(
                elem_id=f'arh_inline_tools_{mark}',
                elem_classes=['arh-inline-tools'],
        ):
            # 刻意不检查 should_show()：需要调用 render 实例化组件，
            # 各组件用自己的 visible 属性控制显隐（与原逻辑一致）
            for component in components:
                component.render()

    def ui(self, is_img2img):
        # 主渲染路径是内联注入（见 _on_height_created）。
        # 此处仅做兜底：若 WebUI 升级导致锚点 elem_id 变化、
        # 钩子未触发，则在脚本区渲染工具，避免功能完全丢失。
        self._ensure_hook(is_img2img)
        self._sync_wc_hc(is_img2img)

        mark = 'img2img' if is_img2img else 'txt2img'
        if mark in self._rendered_pages:
            return None  # 已内联注入到宽高滑块旁
        if self.wc is None or self.hc is None:
            return None  # 引用未就绪（罕见），不渲染
        self._render_inline_tools(mark)
        return None

    def after_component(self, component: gr.components.Component, **kwargs):
        element_id = kwargs.get('elem_id')

        if isinstance(component, gr.components.Slider):
            if element_id == 'txt2img_width':
                self.t2i_w: gr.components.Slider = component
            elif element_id == 'txt2img_height':
                self.t2i_h: gr.components.Slider = component
            elif element_id == 'img2img_width':
                self.i2i_w: gr.components.Slider = component
            elif element_id == 'img2img_height':
                self.i2i_h: gr.components.Slider = component


script_callbacks.on_ui_settings(_settings.on_ui_settings)
