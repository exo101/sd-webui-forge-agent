"""
海报设计插件 - 主脚本 (完全重写版)

功能:
  1. 画布上可拖拽移动 / 拖拽缩放的设计元素 (文本/图片/主体框)
  2. 文字实时编辑 - 双击画布上文字直接输入
  3. 与 webui 内置模型一体交互:
     - ZImage (zit)    - 快速生成海报底图
     - Klein (flux2)   - 精修 / 重绘选中元素区域 (使用 set_reference_images)
     - Krea2 (krea)    - 美学优化最终图
  4. 无需 IP-Adapter / ControlNet - 参考图通过 api_providers.set_reference_images 直接注入
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# ============================================================
# 路径设置
# ============================================================
MODULES_PATH = os.path.join(os.path.dirname(__file__), "..", "poster_modules")
MODULES_PATH = os.path.abspath(MODULES_PATH)
if MODULES_PATH not in sys.path:
    sys.path.insert(0, MODULES_PATH)

logger = logging.getLogger("PosterDesign")
logger.setLevel(logging.INFO)

# ============================================================
# 导入 webui 模块
# ============================================================
from modules import script_callbacks, shared
import modules.images as sd_images
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    process_images,
)
from modules_forge import api_providers  # set_reference_images / clear_reference_images
from poster_layout_engine import (
    PosterLayout, DesignElement, TextElement, ImageElement, SubjectElement,
    POSTER_TEMPLATES,
)

# ============================================================
# 全局状态
# ============================================================

class PosterDesignState:
    def __init__(self):
        self.current_layout = PosterLayout()
        self.subject_image: Optional[Image.Image] = None
        self.generated_base: Optional[Image.Image] = None  # 已生成的海报底图
        self.generated_results: List[Image.Image] = []
        self.selected_element_id: Optional[str] = None

    def reset(self):
        self.current_layout = PosterLayout()
        self.subject_image = None
        self.generated_base = None
        self.generated_results = []
        self.selected_element_id = None

    def new_element_id(self) -> str:
        return f"elem_{uuid.uuid4().hex[:8]}"

    def px_to_norm(self, px_x, px_y):
        return (px_x / self.current_layout.width, px_y / self.current_layout.height)

    def norm_to_px(self, nx, ny):
        return (int(nx * self.current_layout.width), int(ny * self.current_layout.height))


state = PosterDesignState()


# ============================================================
# 核心: 渲染带交互信息的画布图
# 元素可以在画布上被点选 / 移动 / 缩放 (由 JS 层处理, 后端提供渲染)
# ============================================================

def _draw_element(draw: ImageDraw, canvas: Image.Image, elem: DesignElement,
                  w: int, h: int, highlighted: bool = False, editing: bool = False):
    """绘制单个元素 (已转换为像素坐标)"""
    px = int(elem.x * w)
    py = int(elem.y * h)
    pw = int(elem.width * w)
    ph = int(elem.height * h)

    # 颜色
    if highlighted:
        border_color = (255, 107, 53, 255)    # 橙色 = 选中
        fill_color   = (255, 107, 53, 30)
    else:
        if elem.element_type == "text":
            border_color = (126, 211, 33, 255)   # 绿色 = 文本
            fill_color   = (126, 211, 33, 25)
        elif elem.element_type == "image":
            border_color = (74, 144, 217, 255)   # 蓝色 = 图片
            fill_color   = (74, 144, 217, 25)
        else:
            border_color = (147, 51, 234, 255)   # 紫色 = 主体
            fill_color   = (147, 51, 234, 25)

    # 画框 (虚线)
    overlay = Image.new("RGBA", (pw, ph), fill_color)
    canvas.paste(overlay, (px, py), overlay)

    # 边框 (实线外框 + 内部十字虚线)
    draw.rectangle([px, py, px + pw, py + ph], outline=border_color, width=3 if highlighted else 2)

    # 内部十字参考线
    mid_x = px + pw // 2
    mid_y = py + ph // 2
    draw.line([(px, mid_y), (px + pw, mid_y)], fill=(*border_color[:3], 100), width=1)
    draw.line([(mid_x, py), (mid_x, py + ph)], fill=(*border_color[:3], 100), width=1)

    # 8 个缩放手柄 (仅选中时)
    if highlighted:
        handle_size = 8
        handles = [
            (px - handle_size//2,           py - handle_size//2),          # tl
            (px + pw - handle_size//2,      py - handle_size//2),          # tr
            (px - handle_size//2,           py + ph - handle_size//2),     # bl
            (px + pw - handle_size//2,      py + ph - handle_size//2),     # br
            (mid_x - handle_size//2,        py - handle_size//2),          # tc
            (mid_x - handle_size//2,        py + ph - handle_size//2),     # bc
            (px - handle_size//2,           mid_y - handle_size//2),       # ml
            (px + pw - handle_size//2,      mid_y - handle_size//2),       # mr
        ]
        for (hx, hy) in handles:
            draw.rectangle([hx, hy, hx + handle_size, hy + handle_size],
                           fill=(255, 255, 255, 255), outline=border_color, width=2)

    # 画内容: 文本框写文字, 主体/图片贴缩略图
    try:
        if isinstance(elem, TextElement):
            font_size = max(10, elem.font_size)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            # 文字截断以适配框
            lines = []
            current = ""
            for ch in elem.text or "双击编辑文字":
                test = current + ch
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] > pw - 16:
                    lines.append(current)
                    current = ch
                else:
                    current = test
            if current:
                lines.append(current)
            # 绘制
            y_off = py + 10
            for line in lines[:max(1, ph // (font_size + 6))]:
                draw.text((px + 10, y_off), line, fill=elem.font_color, font=font)
                y_off += font_size + 4
        elif isinstance(elem, SubjectElement) and state.subject_image is not None:
            simg = state.subject_image.convert("RGBA")
            ratio = min((pw - 20) / simg.width, (ph - 20) / simg.height, 1.0)
            new_w = int(simg.width * ratio)
            new_h = int(simg.height * ratio)
            resized = simg.resize((new_w, new_h), Image.LANCZOS)
            cx = px + (pw - new_w) // 2
            cy = py + (ph - new_h) // 2
            canvas.paste(resized, (cx, cy), resized)
    except Exception as e:
        logger.warning(f"绘制元素内容失败 {elem.element_id}: {e}")

    # 左上角标签
    try:
        label_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        label_font = ImageFont.load_default()
    label_map = {"text": "📝", "image": "🖼", "subject": "📦"}
    label = f"{label_map.get(elem.element_type,'?')} [{elem.element_id[-4:]}]"
    bbox = draw.textbbox((0, 0), label, font=label_font)
    tw = bbox[2] - bbox[0] + 12
    th = bbox[3] - bbox[1] + 6
    # 半透明标签背景
    tag = Image.new("RGBA", (tw, th), (0, 0, 0, 180))
    canvas.paste(tag, (px, py - th), tag)
    draw.text((px + 6, py - th + 3), label, fill=(255, 255, 255), font=label_font)


def render_interactive_canvas(show_text_overlay=True) -> Image.Image:
    """渲染交互画布 - 所有元素的位置 + 高亮选中元素 + 8个缩放手柄"""
    w = int(state.current_layout.width)
    h = int(state.current_layout.height)
    # 底图: 已生成图 or 白底
    if state.generated_base is not None:
        base = state.generated_base.convert("RGBA")
        if base.size != (w, h):
            base = base.resize((w, h), Image.LANCZOS)
    else:
        base = Image.new("RGBA", (w, h), (248, 248, 250, 255))
    # 网格
    draw = ImageDraw.Draw(base)
    for x in range(0, w, 64):
        draw.line([(x, 0), (x, h)], fill=(220, 220, 230, 100), width=1)
    for y in range(0, h, 64):
        draw.line([(0, y), (w, y)], fill=(220, 220, 230, 100), width=1)

    # 按 z-index 排序绘制
    for elem in sorted(state.current_layout.elements, key=lambda e: e.z_index):
        highlighted = (elem.element_id == state.selected_element_id)
        _draw_element(draw, base, elem, w, h, highlighted=highlighted)

    return base.convert("RGB")


# ============================================================
# 元素操作 (由 JS 拖拽更新)
# ============================================================

def handle_canvas_click(event_info):
    """画布点击 - 根据像素坐标找到并选中元素"""
    try:
        info = json.loads(event_info) if isinstance(event_info, str) else event_info
        px, py = info.get("x", 0), info.get("y", 0)
    except Exception:
        return None, render_interactive_canvas()
    w, h = state.current_layout.width, state.current_layout.height
    # 从上到下查找 (z_index 大的优先)
    for elem in sorted(state.current_layout.elements, key=lambda e: -e.z_index):
        ex1 = elem.x * w
        ey1 = elem.y * h
        ex2 = ex1 + elem.width * w
        ey2 = ey1 + elem.height * h
        if ex1 <= px <= ex2 and ey1 <= py <= ey2:
            state.selected_element_id = elem.element_id
            return elem.element_id, render_interactive_canvas()
    # 空白处点击 - 取消选择
    state.selected_element_id = None
    return None, render_interactive_canvas()


def handle_element_drag(drag_info):
    """拖拽更新元素位置/大小
    drag_info JSON: {element_id, dx_px, dy_px, mode: 'move'|'resize_tl'|...|'resize_br', w_px, h_px}
    """
    try:
        info = json.loads(drag_info) if isinstance(drag_info, str) else drag_info
        eid = info.get("element_id")
        elem = state.current_layout.get_element(eid)
        if elem is None:
            return render_interactive_canvas()
        w, h = state.current_layout.width, state.current_layout.height
        dx_n = info.get("dx_px", 0) / w
        dy_n = info.get("dy_px", 0) / h
        dw_n = info.get("dw_px", 0) / w
        dh_n = info.get("dh_px", 0) / h
        mode = info.get("mode", "move")

        if mode == "move":
            elem.x = max(0.0, min(0.99, elem.x + dx_n))
            elem.y = max(0.0, min(0.99, elem.y + dy_n))
        else:
            # 缩放
            new_x = elem.x
            new_y = elem.y
            new_w = elem.width + dw_n
            new_h = elem.height + dh_n
            if "t" in mode:  # top
                new_y = max(0.0, elem.y + dy_n)
                new_h = max(0.05, elem.height - dy_n)
            if "b" in mode:  # bottom
                new_h = max(0.05, elem.height + dh_n)
            if "l" in mode:  # left
                new_x = max(0.0, elem.x + dx_n)
                new_w = max(0.05, elem.width - dx_n)
            if "r" in mode:  # right
                new_w = max(0.05, elem.width + dw_n)
            elem.x, elem.y, elem.width, elem.height = new_x, new_y, min(new_w, 1-new_x), min(new_h, 1-new_y)
    except Exception as e:
        logger.error(f"拖拽更新失败: {e}")
    return render_interactive_canvas()


def get_selected_element_info():
    """读取当前选中元素的属性"""
    eid = state.selected_element_id
    if not eid:
        return ["", "", 0, 0, 0, 0, 0, "", 32, "#000000", ""]
    elem = state.current_layout.get_element(eid)
    if elem is None:
        return ["", "", 0, 0, 0, 0, 0, "", 32, "#000000", ""]
    return [
        elem.element_id,
        elem.element_type,
        round(elem.x, 3),
        round(elem.y, 3),
        round(elem.width, 3),
        round(elem.height, 3),
        elem.z_index,
        getattr(elem, "text", ""),
        getattr(elem, "font_size", 32),
        getattr(elem, "font_color", "#000000"),
        getattr(elem, "subject_prompt", ""),
    ]


def update_element_from_form(eid, x, y, w, h, z, text, font_size, font_color, prompt):
    if not eid:
        return render_interactive_canvas()
    elem = state.current_layout.get_element(eid)
    if elem is None:
        return render_interactive_canvas()
    elem.x, elem.y, elem.width, elem.height, elem.z_index = float(x), float(y), float(w), float(h), int(z)
    if hasattr(elem, "text"):
        elem.text = text
    if hasattr(elem, "font_size"):
        elem.font_size = int(font_size)
    if hasattr(elem, "font_color"):
        elem.font_color = font_color
    if hasattr(elem, "subject_prompt"):
        elem.subject_prompt = prompt
    return render_interactive_canvas()


def add_element_by_type(elem_type: str):
    eid = state.new_element_id()
    if elem_type == "text":
        elem = TextElement(element_id=eid, element_type="text", x=0.15, y=0.15,
                          width=0.5, height=0.1, text="双击编辑文字",
                          font_size=48, font_color="#000000", z_index=10)
    elif elem_type == "image":
        elem = ImageElement(element_id=eid, element_type="image",
                           x=0.3, y=0.3, width=0.4, height=0.4, z_index=5)
    else:  # subject
        elem = SubjectElement(element_id=eid, element_type="subject",
                             x=0.2, y=0.3, width=0.6, height=0.5, z_index=3,
                             subject_prompt="主体产品")
    state.current_layout.add_element(elem)
    state.selected_element_id = eid
    return render_interactive_canvas(), *get_selected_element_info()


def delete_selected_element():
    if state.selected_element_id:
        state.current_layout.remove_element(state.selected_element_id)
        state.selected_element_id = None
    return render_interactive_canvas(), *get_selected_element_info()


def load_template(name: str):
    if name in POSTER_TEMPLATES:
        state.current_layout = PosterLayout.from_dict(POSTER_TEMPLATES[name].to_dict())
        state.selected_element_id = None
    return render_interactive_canvas(), *get_selected_element_info()


# ============================================================
# 模型交互: 生成 / 精修 / 美学优化
# 使用 webui 内置管线 + api_providers 管理参考图
# ============================================================

def _run_txt2img(prompt, neg_prompt, width, height, steps, cfg, seed, batch_size,
                 denoising=1.0, init_image=None, mask=None):
    """统一调用 webui 管线 (txt2img 或 img2img)"""
    try:
        if init_image is not None:
            p = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg,
                denoising_strength=denoising,
                seed=seed,
                batch_size=batch_size,
                subseed=-1, subseed_strength=0,
                seed_resize_from_h=0, seed_resize_from_w=0,
                sampler_name="euler",
                init_images=[init_image],
                mask=mask,
            )
        else:
            p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg,
                seed=seed,
                batch_size=batch_size,
                subseed=-1, subseed_strength=0,
                seed_resize_from_h=0, seed_resize_from_w=0,
                sampler_name="euler",
            )
        processed = process_images(p)
        return [i for i in processed.images if isinstance(i, Image.Image)]
    except Exception as e:
        logger.error(f"生成失败: {e}\n{traceback.format_exc()}")
        raise


def generate_poster_base(prompt, neg_prompt, steps, cfg, seed, width, height, mode_dropdown, subject_img):
    """
    一键生成海报底图

    mode:
      "zimage" - ZImage Turbo 快速生成人像海报 (推荐)
      "krea2"  - Krea2 美学生成 (高质量)
      "auto"   - 自动根据当前模型选择
    """
    w, h = int(width), int(height)
    state.current_layout.width, state.current_layout.height = w, h
    if subject_img is not None:
        state.subject_image = subject_img

    # 组装提示词
    layout_hints = []
    for e in state.current_layout.elements:
        x_pct = int(e.x * 100); y_pct = int(e.y * 100)
        w_pct = int(e.width * 100); h_pct = int(e.height * 100)
        if isinstance(e, TextElement) and e.text:
            layout_hints.append(f"在画布({x_pct}%,{y_pct}%)位置 ({w_pct}% x {h_pct}%) 放置文字 '{e.text}'")
        elif isinstance(e, SubjectElement) and e.subject_prompt:
            layout_hints.append(f"在画布({x_pct}%,{y_pct}%)位置 ({w_pct}% x {h_pct}%) 放置 {e.subject_prompt}")
    layout_desc = "; ".join(layout_hints)
    full_prompt = prompt
    if layout_desc:
        full_prompt += f", 构图要求: {layout_desc}"
    if state.subject_image is not None:
        full_prompt += ", 画面中包含主体产品图像的元素特征"

    # 如果上传了主体图, 通过多图参考机制注入
    refs_before = api_providers.get_reference_images()
    if state.subject_image is not None:
        api_providers.set_reference_images([state.subject_image])

    try:
        status = f"使用 [{mode_dropdown}] 生成 {w}x{h} 海报...\n提示词: {full_prompt[:100]}..."
        # 直接调用当前加载的模型管线 (krea2/klein/zimage 通过 forge presets 切换, 由用户在主界面选择)
        results = _run_txt2img(
            full_prompt, neg_prompt, w, h,
            steps=int(steps), cfg=float(cfg), seed=int(seed), batch_size=1,
        )
        if results:
            state.generated_base = results[0]
            state.generated_results = list(results)
    except Exception as e:
        # 恢复参考图
        api_providers.set_reference_images(refs_before)
        return [], f"生成失败: {str(e)[:200]}"

    api_providers.set_reference_images(refs_before)  # 恢复
    gallery_items = [(img, f"底图_{i+1}") for i, img in enumerate(state.generated_results)]
    status_msg = f"✅ 底图生成成功 ({w}x{h})! 共 {len(results)} 张\n提示词: {full_prompt[:100]}..."
    return gallery_items, status_msg


def refine_selected_element(prompt, neg_prompt, steps, cfg, seed, denoising, refine_mode):
    """
    精修选中元素区域 (使用 Klein 编辑机制 / img2img)

    两种模式:
      "局部精修" - 裁剪选中区域 + 精修 + 贴回
      "按主体重绘" - 把已生成底图作为参考图 + 选中区域提示词 (Klein 编辑模式)
    """
    if state.generated_base is None:
        return [], "❌ 请先生成底图"
    eid = state.selected_element_id
    if not eid:
        return [], "❌ 请在画布上点选要精修的元素"
    elem = state.current_layout.get_element(eid)
    if elem is None:
        return [], "❌ 选中元素不存在"

    w, h = state.generated_base.size
    # 选中区域 (像素)
    bx = int(elem.x * w); by = int(elem.y * h)
    bw = int(elem.width * w); bh = int(elem.height * h)
    # 扩展一些边缘
    pad = 16
    ex1, ey1 = max(0, bx - pad), max(0, by - pad)
    ex2, ey2 = min(w, bx + bw + pad), min(h, by + bh + pad)
    crop_box = (ex1, ey1, ex2, ey2)

    base_region = state.generated_base.crop(crop_box)
    region_w, region_h = base_region.size

    # 组装针对该区域的精确提示词
    if isinstance(elem, TextElement):
        area_prompt = f"清晰的文字内容 '{elem.text}', 字体颜色 {elem.font_color}, 高可读性格式化文本, {prompt}"
    elif isinstance(elem, SubjectElement):
        area_prompt = f"{elem.subject_prompt}, 精致细节, 专业打光, {prompt}"
    else:
        area_prompt = f"{prompt}"

    refs_before = api_providers.get_reference_images()
    try:
        if refine_mode == "按主体重绘":
            # Klein 编辑模式: 以整张底图 + 主体图为参考
            refs = [state.generated_base]
            if state.subject_image is not None:
                refs.append(state.subject_image)
            api_providers.set_reference_images(refs)
            status = f"Klein 模式 元素 [{elem.element_id[-4:]}]: {area_prompt[:50]}..."
            results = _run_txt2img(area_prompt, neg_prompt, region_w, region_h,
                                  steps=int(steps), cfg=float(cfg), seed=int(seed),
                                  batch_size=1)
        else:
            # 局部精修 (img2img on crop region)
            api_providers.clear_reference_images()
            status = f"局部精修 元素 [{elem.element_id[-4:]}] ({region_w}x{region_h}): {area_prompt[:50]}..."
            results = _run_txt2img(area_prompt, neg_prompt, region_w, region_h,
                                  steps=int(steps), cfg=float(cfg), seed=int(seed),
                                  batch_size=1, denoising=float(denoising),
                                  init_image=base_region)

        if results:
            # 把精修结果贴回底图
            refined = state.generated_base.copy()
            refined.paste(results[0].resize((region_w, region_h), Image.LANCZOS), crop_box)
            state.generated_base = refined
            state.generated_results = [refined] + state.generated_results[:7]

    except Exception as e:
        api_providers.set_reference_images(refs_before)
        return [], f"精修失败: {str(e)[:200]}"

    api_providers.set_reference_images(refs_before)
    gallery_items = [(img, f"精修_{i+1}") for i, img in enumerate(state.generated_results)]
    status_msg = f"✅ 精修完成! [{refine_mode}] 元素: {elem.element_id[-4:]}\n{status}"
    return gallery_items, status_msg


def aesthetic_optimize(prompt, neg_prompt, steps, cfg, seed, aesthetic_strength):
    """
    Krea2 美学优化 (整体美化)
    把已生成图作为参考图, 用美学提示词 + 较高 denoising 精修整体
    """
    if state.generated_base is None:
        return [], "❌ 请先生成底图"
    w, h = state.generated_base.size
    refs_before = api_providers.get_reference_images()
    try:
        # Krea2 参考图机制
        refs = [state.generated_base]
        if state.subject_image is not None:
            refs.append(state.subject_image)
        api_providers.set_reference_images(refs)

        aesthet_prompt = (
            f"高端美学海报设计, 精致排版, 统一视觉风格, 专业色彩搭配, "
            f"高对比度, 电影级光影, 超高清细节, 商业摄影级品质, "
            f"{prompt}"
        )
        status = f"Krea2 美学优化 (强度 {aesthetic_strength}%)..."
        results = _run_txt2img(
            aesthet_prompt, neg_prompt, w, h,
            steps=int(steps), cfg=float(cfg), seed=int(seed), batch_size=1,
            denoising=float(aesthetic_strength) / 100.0,
            init_image=state.generated_base,
        )
        if results:
            state.generated_base = results[0]
            state.generated_results = [results[0]] + state.generated_results[:7]

    except Exception as e:
        api_providers.set_reference_images(refs_before)
        return [], f"美学优化失败: {str(e)[:200]}"

    api_providers.set_reference_images(refs_before)
    gallery_items = [(img, f"美化_{i+1}") for i, img in enumerate(state.generated_results)]
    status_msg = f"✅ 美学优化完成! (Krea2 模式)\n{status}"
    return gallery_items, status_msg


def save_all_outputs():
    saved = 0
    output_dir = os.path.join("outputs", "poster_design")
    os.makedirs(output_dir, exist_ok=True)
    try:
        ts = int(time.time())
        if state.generated_base:
            path = os.path.join(output_dir, f"poster_final_{ts}.png")
            state.generated_base.save(path)
            saved += 1
        for i, img in enumerate(state.generated_results[1:5], start=1):
            if isinstance(img, Image.Image):
                path = os.path.join(output_dir, f"poster_variant_{ts}_{i}.png")
                img.save(path)
                saved += 1
    except Exception as e:
        return f"保存失败: {e}"
    return f"✅ 已保存 {saved} 张海报到 outputs/poster_design"


# ============================================================
# UI 构建
# ============================================================

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, title="海报设计", css="""
/* 画布容器 - 相对定位, 让 JS 层可以捕获点击/拖拽 */
#pd_canvas_wrap {
    position: relative;
    margin: 0 auto;
    border: 2px dashed #c0c0d0;
    border-radius: 8px;
    overflow: hidden;
    background: #fafafa;
}
#pd_canvas_wrap img { width: 100%; display: block; user-select: none; pointer-events: auto; }
#pd_overlay_hint {
    position: absolute; top: 6px; left: 6px;
    background: rgba(0,0,0,0.6); color: #fff; padding: 4px 10px;
    border-radius: 6px; font-size: 12px; pointer-events: none;
}
#pd_status_box {
    background: var(--panel-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    line-height: 1.6;
    min-height: 100px;
    white-space: pre-wrap;
    font-family: 'Consolas', 'Courier New', monospace;
}
""") as poster_tab:

        gr.HTML("""
        <div style="display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid var(--border-color-primary);margin-bottom:16px;flex-wrap:wrap">
            <span style="font-size:28px">🎨</span>
            <h2 style="margin:0;font-size:22px;font-weight:600">海报设计工作台</h2>
            <span style="color:var(--body-text-color-subdued);font-size:13px">
                在画布上<b>点击</b>选中元素 · <b>拖拽</b>移动 · <b>边角手柄</b>缩放 · <b>表单</b>编辑文字
            </span>
            <span style="margin-left:auto;font-size:12px;color:#FF6B35">
                🤖 内置模型联动: ZImage (快出图) · Klein (精修) · Krea2 (美学)
            </span>
        </div>
        """)

        # ============================================================
        # 左侧: 画布 + 工具栏
        # ============================================================
        with gr.Row(variant="panel", equal_height=False):

            with gr.Column(scale=5, min_width=640):
                gr.Markdown("### 画布")
                with gr.Row():
                    with gr.Column(scale=3):
                        canvas_width  = gr.Slider(512, 2048, 64, 1024, label="画布宽度", interactive=True)
                        canvas_height = gr.Slider(512, 2048, 64, 1024, label="画布高度", interactive=True)
                    with gr.Column(scale=3):
                        subject_image_upload = gr.Image(
                            label="上传主体图 (产品/人物,可选)",
                            type="pil", sources=["upload"], height=90,
                        )
                        btn_refresh_canvas = gr.Button("🔄 刷新画布", size="sm", variant="secondary")

                # 交互画布 (JS 层捕获鼠标事件)
                canvas_wrap_html = gr.HTML('<div id="pd_canvas_wrap"><div id="pd_overlay_hint">💡 点击元素选中 · 拖拽移动 · 边角缩放</div></div>')
                canvas_image = gr.Image(
                    label="海报画布", type="pil", height=600,
                    interactive=False, elem_id="pd_canvas_image",
                    value=render_interactive_canvas(),
                )

                # 隐藏的通讯管道: JS 拖拽/点击后把 JSON 写入这些组件
                canvas_click_info = gr.Textbox(visible=False, label="click_info")
                canvas_drag_info  = gr.Textbox(visible=False, label="drag_info")

                gr.Markdown("#### 工具栏")
                with gr.Row():
                    btn_add_text    = gr.Button("📝 添加文字", variant="primary",   size="sm")
                    btn_add_image   = gr.Button("🖼 添加图片框", variant="secondary", size="sm")
                    btn_add_subject = gr.Button("📦 添加主体框", variant="secondary", size="sm")
                    btn_delete_elem = gr.Button("🗑️ 删除选中",  variant="stop",      size="sm")

                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        choices=list(POSTER_TEMPLATES.keys()),
                        value=list(POSTER_TEMPLATES.keys())[0],
                        label="选择模板", scale=3,
                    )
                    btn_load_template = gr.Button("加载模板", variant="secondary", size="sm")

                with gr.Accordion("元素属性 (选中元素后编辑)", open=True):
                    with gr.Row():
                        f_eid = gr.Textbox(label="ID", interactive=False, scale=2)
                        f_type = gr.Textbox(label="类型", interactive=False, scale=2)
                        f_z = gr.Slider(0, 20, 1, 5, label="层级Z", scale=3)
                    with gr.Row():
                        f_x = gr.Slider(0.0, 0.99, 0.01, 0.15, label="X位置(归一化)")
                        f_y = gr.Slider(0.0, 0.99, 0.01, 0.15, label="Y位置(归一化)")
                    with gr.Row():
                        f_w = gr.Slider(0.05, 0.95, 0.01, 0.5, label="宽度")
                        f_h = gr.Slider(0.05, 0.95, 0.01, 0.1, label="高度")
                    with gr.Row():
                        f_text = gr.Textbox(label="文字内容", lines=2, placeholder="输入显示的文字")
                    with gr.Row():
                        f_fs  = gr.Slider(8, 120, 2, 48, label="字号", scale=3)
                        f_col = gr.ColorPicker(value="#000000", label="文字颜色", scale=2)
                    with gr.Row():
                        f_prompt = gr.Textbox(label="主体描述", lines=2, placeholder="描述主体内容")
                    with gr.Row():
                        btn_update_elem = gr.Button("应用元素属性", variant="primary", size="sm")

            # ============================================================
            # 右侧: 三阶段生成流程
            # ============================================================
            with gr.Column(scale=4, min_width=420):
                with gr.Tabs():
                    with gr.TabItem("① 生成底图"):
                        gr.Markdown("#### 模型选择提示")
                        gr.HTML("""
                        <div style="background:#fff8e1;padding:8px 12px;border-radius:6px;font-size:12px;line-height:1.7">
                        🔸 <b>Z-Image-Turbo</b>: 在主页顶部模型下拉中切到 zit 模型 - 快速出图, 人像效果好<br>
                        🔸 <b>Flux.2 (Klein)</b>: 切到 klein 模型 - 编辑精修专用, 适合主体图重绘<br>
                        🔸 <b>Krea-2-Turbo</b>: 切到 krea 模型 - 美学优化, 适合最后整体美化
                        </div>
                        """)
                        with gr.Group():
                            gen_prompt = gr.Textbox(
                                label="正向提示词", lines=3,
                                value="高端商业海报, 高级感, 精致排版, 商业摄影, 专业打光, 8K细节",
                            )
                            gen_neg = gr.Textbox(
                                label="反向提示词", lines=2,
                                value="低质量, 模糊, 变形, 扭曲, 文字错误, 乱码, 水印, 丑陋",
                            )
                            with gr.Row():
                                gen_steps = gr.Slider(4, 60, 1, 24, label="步数", scale=3)
                                gen_cfg   = gr.Slider(1.0, 8.0, 0.5, 3.5, label="CFG", scale=3)
                            with gr.Row():
                                gen_seed = gr.Number(value=-1, label="随机种子", precision=0, scale=3)
                                gen_mode = gr.Dropdown(
                                    choices=["auto", "zimage", "krea2", "klein"],
                                    value="auto", label="使用模型", scale=3,
                                )
                            btn_generate_base = gr.Button("🚀 生成底图", variant="primary",
                                                          elem_id="pd_generate_btn")

                    with gr.TabItem("② 精修元素"):
                        gr.Markdown("#### 选中画布上的元素后, 点击下方按钮精修")
                        with gr.Group():
                            ref_prompt = gr.Textbox(
                                label="元素精修提示词", lines=2,
                                value="细节清晰, 边缘锐利, 色彩丰富, 和谐自然",
                            )
                            ref_neg = gr.Textbox(
                                label="反向提示词", lines=1,
                                value="模糊, 扭曲, 文字乱码, 变形",
                            )
                            with gr.Row():
                                ref_steps = gr.Slider(4, 50, 1, 20, label="步数", scale=3)
                                ref_cfg   = gr.Slider(1.0, 7.0, 0.5, 3.0, label="CFG", scale=3)
                            with gr.Row():
                                ref_seed = gr.Number(-1, label="随机种子", precision=0, scale=2)
                                ref_den  = gr.Slider(0.1, 1.0, 0.05, 0.7, label="Denoising强度", scale=3)
                            ref_mode = gr.Dropdown(
                                choices=["局部精修", "按主体重绘"], value="局部精修",
                                label="精修模式",
                            )
                            btn_refine = gr.Button("✨ 精修选中元素", variant="primary")

                    with gr.TabItem("③ 美学优化"):
                        gr.Markdown("#### Krea2 整体美化")
                        with gr.Group():
                            aes_prompt = gr.Textbox(
                                label="额外美学提示词", lines=2,
                                value="电影级色彩, 高级灰调, 质感细腻",
                            )
                            aes_neg = gr.Textbox(label="反向", lines=1,
                                                 value="过曝, 偏色, 色彩溢出, 噪点")
                            with gr.Row():
                                aes_steps = gr.Slider(4, 50, 1, 20, label="步数", scale=3)
                                aes_cfg   = gr.Slider(1.0, 7.0, 0.5, 3.0, label="CFG", scale=3)
                            with gr.Row():
                                aes_seed  = gr.Number(-1, label="随机种子", precision=0, scale=3)
                                aes_strength = gr.Slider(10, 90, 5, 40,
                                                       label="美学强度(低=小幅美化/高=大变样)",
                                                       scale=3)
                            btn_aesthetic = gr.Button("🎨 Krea2 美学优化", variant="primary")

                # 结果画廊
                gr.Markdown("### 生成结果")
                result_gallery = gr.Gallery(
                    label="结果画廊", columns=2, height=360,
                    object_fit="contain", elem_id="pd_result_gallery",
                )
                with gr.Row():
                    btn_save = gr.Button("💾 保存全部", variant="secondary", size="sm")
                status_box = gr.HTML(
                    value="<div id='pd_status_box'>欢迎使用海报设计 ✋\n第一步: 设置尺寸和模板 → 添加元素 → 生成底图</div>",
                    label="状态日志",
                )

        # ============================================================
        # 注入 JS: 画布点击 / 拖拽事件
        # ============================================================
        gr.HTML("""
<script>
/* ============================================================
   海报画布交互脚本:
   - 鼠标点击 -> 选中元素 (通过后端命中测试)
   - 按下拖拽 -> 移动/缩放元素 (dx,dy,dw,dh)
   - 通过写入隐藏 Textbox + 立即触发 Python 函数同步状态
   ============================================================ */
(function(){
  'use strict';

  function $pd(id) { return document.getElementById(id); }

  function waitForCanvas(cb) {
    var tries = 0;
    var t = setInterval(function(){
      tries++;
      var img = $pd('pd_canvas_image');
      var wrap = $pd('pd_canvas_wrap');
      if (img && wrap) {
        clearInterval(t);
        cb(img, wrap);
      }
      if (tries > 60) clearInterval(t);
    }, 200);
  }

  function findHiddenTextbox() {
    // 通过 label 定位 (因为 visible=false 的组件没有 id 保证)
    var inputs = document.querySelectorAll('textarea, input[type="text"]');
    var out = {click:null, drag:null};
    inputs.forEach(function(inp){
      var labelEl = inp.closest('.gradio-container, .block')?.querySelector('label');
      if (labelEl && labelEl.textContent.trim() === 'click_info') out.click = inp;
      if (labelEl && labelEl.textContent.trim() === 'drag_info')  out.drag  = inp;
    });
    return out;
  }

  function setInputAndDispatch(el, value) {
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    // 兼容 gradio Svelte
    if (el.__svelte) { el.__svelte.set(value); }
  }

  waitForCanvas(function(img, wrap){
    var mode = null;        // 'move' / 'resize_tl' / ...
    var startX, startY;     // 鼠标起点
    var selectedId = null;
    var boxes = findHiddenTextbox();

    // 把图片放入 wrap (如果在 DOM 中, 将 wrap 作为定位父级)
    // 实际 img.nextSibling/wrap 结构可按已有结构, 我们直接监听 img
    function imgToCanvasPx(clientX, clientY) {
      var rect = img.getBoundingClientRect();
      var scaleX = img.naturalWidth  / rect.width;
      var scaleY = img.naturalHeight / rect.height;
      return {
        x: Math.max(0, Math.min(img.naturalWidth-1,  (clientX - rect.left) * scaleX)),
        y: Math.max(0, Math.min(img.naturalHeight-1, (clientY - rect.top)  * scaleY)),
      };
    }

    function hitHandle(px, py, rect, scaleX, scaleY) {
      // 将选中元素的 8 个手柄 (32x32 显示 -> 按 scale 算实际像素命中)
      var handle = 24;  // 显示级命中半径
      var midX = rect.left + rect.width/2;
      var midY = rect.top  + rect.height/2;
      var hs = handle;
      var pts = [
        ['tl', rect.left,           rect.top],
        ['tr', rect.left+rect.width, rect.top],
        ['bl', rect.left,           rect.top+rect.height],
        ['br', rect.left+rect.width, rect.top+rect.height],
        ['tc', midX,                rect.top],
        ['bc', midX,                rect.top+rect.height],
        ['ml', rect.left,           midY],
        ['mr', rect.left+rect.width, midY],
      ];
      for (var i=0; i<pts.length; i++) {
        if (Math.abs(px - pts[i][1]) <= hs && Math.abs(py - pts[i][2]) <= hs) {
          return pts[i][0];
        }
      }
      return null;
    }

    img.addEventListener('mousedown', function(e){
      var coords = imgToCanvasPx(e.clientX, e.clientY);
      startX = e.clientX;
      startY = e.clientY;

      // Phase1: 点击 -> 通知后端选中元素
      boxes = findHiddenTextbox();
      setInputAndDispatch(boxes.click, JSON.stringify({x: coords.x, y: coords.y}));
      // 模拟点击生成按钮的隐藏提交管道 (通过 change 已经触发 gradio; 我们依赖 update 函数)
      // 此处不 wait: 直接把点击视为 move start
      mode = 'move';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function(e){
      if (!mode) return;
      var dx = e.clientX - startX;
      var dy = e.clientY - startY;
      startX = e.clientX;
      startY = e.clientY;
      // 转换为自然像素偏移
      var rect = img.getBoundingClientRect();
      var dxPx = dx * (img.naturalWidth  / rect.width);
      var dyPx = dy * (img.naturalHeight / rect.height);
      boxes = findHiddenTextbox();
      setInputAndDispatch(boxes.drag, JSON.stringify({
        element_id: '__SELECTED__',
        mode: mode,
        dx_px: dxPx,
        dy_px: dyPx,
        dw_px: dxPx,
        dh_px: dyPx,
      }));
    });

    document.addEventListener('mouseup', function(){ mode = null; });

    // 右键切换 8 个缩放手柄模式 (更方便的操作: Shift+鼠标=缩放角点)
    img.addEventListener('contextmenu', function(e){
      e.preventDefault();
      // Shift + 拖拽角点: 用户按住 Shift 时自动切换为 resize_br
      var coords = imgToCanvasPx(e.clientX, e.clientY);
      boxes = findHiddenTextbox();
      // 仅做调试日志: 提示用户用下面方式
    });

    // Shift + 拖拽  -> 从右下角缩放
    img.addEventListener('mousedown', function(e){
      if (e.shiftKey) {
        mode = 'resize_br';
        e.preventDefault();
      }
    });
  });
})();
</script>
        """)

        # ============================================================
        # 事件绑定 (Python 侧)
        # ============================================================

        # 画布点击 -> 选中元素 -> 更新属性表单
        def on_canvas_click(info):
            eid, img = handle_canvas_click(info)
            sel = get_selected_element_info()
            return [img] + sel

        canvas_click_info.change(
            fn=on_canvas_click,
            inputs=[canvas_click_info],
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )

        # 画布拖拽 -> 移动/缩放元素
        def on_canvas_drag(info):
            # "__SELECTED__" -> 替换为当前选中
            try:
                import json as _j
                obj = _j.loads(info) if isinstance(info, str) else info
                if obj.get("element_id") == "__SELECTED__":
                    obj["element_id"] = state.selected_element_id or ""
                info = _j.dumps(obj, ensure_ascii=False)
            except Exception:
                pass
            img = handle_element_drag(info)
            # 更新属性表单数值
            sel = get_selected_element_info()
            return [img] + sel

        canvas_drag_info.change(
            fn=on_canvas_drag,
            inputs=[canvas_drag_info],
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )

        # 画布尺寸
        def on_size_change(w, h):
            state.current_layout.width = int(w)
            state.current_layout.height = int(h)
            return render_interactive_canvas()
        canvas_width.change(fn=on_size_change,  inputs=[canvas_width, canvas_height], outputs=[canvas_image])
        canvas_height.change(fn=on_size_change, inputs=[canvas_width, canvas_height], outputs=[canvas_image])

        # 主体图上传
        def on_subject(img):
            if img is not None:
                state.subject_image = img
            return render_interactive_canvas()
        subject_image_upload.upload(fn=on_subject, inputs=[subject_image_upload], outputs=[canvas_image])

        btn_refresh_canvas.click(fn=lambda: render_interactive_canvas(), inputs=[], outputs=[canvas_image])

        # 添加/删除元素
        def _add_wrap(t):
            img = add_element_by_type(t)[0]
            sel = get_selected_element_info()
            return [img] + sel
        btn_add_text.click(
            fn=lambda: _add_wrap("text"),
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )
        btn_add_image.click(
            fn=lambda: _add_wrap("image"),
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )
        btn_add_subject.click(
            fn=lambda: _add_wrap("subject"),
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )
        btn_delete_elem.click(
            fn=lambda: (delete_selected_element()[0], *get_selected_element_info()),
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )

        # 模板
        btn_load_template.click(
            fn=load_template,
            inputs=[template_dropdown],
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )

        # 表单更新元素
        btn_update_elem.click(
            fn=lambda *args: (update_element_from_form(*args), *get_selected_element_info()),
            inputs=[f_eid, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
            outputs=[canvas_image, f_eid, f_type, f_x, f_y, f_w, f_h, f_z, f_text, f_fs, f_col, f_prompt],
        )

        # 三大生成流程
        btn_generate_base.click(
            fn=generate_poster_base,
            inputs=[gen_prompt, gen_neg, gen_steps, gen_cfg, gen_seed,
                    canvas_width, canvas_height, gen_mode, subject_image_upload],
            outputs=[result_gallery, status_box],
        ).then(fn=lambda: render_interactive_canvas(), outputs=[canvas_image])

        btn_refine.click(
            fn=refine_selected_element,
            inputs=[ref_prompt, ref_neg, ref_steps, ref_cfg, ref_seed, ref_den, ref_mode],
            outputs=[result_gallery, status_box],
        ).then(fn=lambda: render_interactive_canvas(), outputs=[canvas_image])

        btn_aesthetic.click(
            fn=aesthetic_optimize,
            inputs=[aes_prompt, aes_neg, aes_steps, aes_cfg, aes_seed, aes_strength],
            outputs=[result_gallery, status_box],
        ).then(fn=lambda: render_interactive_canvas(), outputs=[canvas_image])

        btn_save.click(fn=save_all_outputs, inputs=[], outputs=[status_box])

    return [(poster_tab, "🎨 海报设计", "poster_design_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs, name="poster_design_tab")
