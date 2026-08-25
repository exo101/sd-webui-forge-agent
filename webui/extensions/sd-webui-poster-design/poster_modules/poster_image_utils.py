"""
图像处理工具 - 参考 CreatiDesign 的图像处理流程
"""

from __future__ import annotations
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
from typing import Optional, Tuple, Dict

from poster_layout_engine import (
    PosterLayout, TextElement, ImageElement, SubjectElement, DesignElement,
)


def create_subject_mask(image: Image.Image, threshold: int = 200) -> Image.Image:
    """
    创建主体遮罩 - 类似 CreatiDesign 的 subject_mask
    用于在生成时指定主体区域
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = image.split()[3]
    if alpha.getextrema() != (255, 255):
        return alpha

    gray = image.convert("L")
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge = edge.point(lambda x: 255 if x > threshold else 0)
    edge = edge.filter(ImageFilter.MaxFilter(5))
    return edge


def prepare_condition_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    background_color: str = "gray",
) -> Image.Image:
    """
    准备条件图像 - 参考 CreatiDesign 的 condition_img 处理
    将主体图放在中性背景上，用于 IP-Adapter 条件注入
    """
    img = image.convert("RGBA")

    if background_color == "white":
        bg = Image.new("RGBA", target_size, (255, 255, 255, 255))
    elif background_color == "black":
        bg = Image.new("RGBA", target_size, (0, 0, 0, 255))
    else:  # gray
        bg = Image.new("RGBA", target_size, (128, 128, 128, 255))

    ratio = min(target_size[0] * 0.8 / max(1, img.width),
                target_size[1] * 0.8 / max(1, img.height))
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    x = (target_size[0] - new_w) // 2
    y = (target_size[1] - new_h) // 2
    bg.paste(img_resized, (x, y), img_resized)

    return bg.convert("RGB")


def create_layout_control_image(
    layout_elements: list,
    canvas_size: Tuple[int, int],
    box_color: str = "#FFFFFF",
    bg_color: str = "#000000",
) -> Image.Image:
    """
    创建布局控制图 - 参考 CreatiDesign 的 layout 条件
    将布局元素绘制为控制图，用于 ControlNet 布局引导
    """
    canvas = Image.new("RGB", canvas_size, bg_color)
    draw = ImageDraw.Draw(canvas)

    for elem in layout_elements:
        x = int(elem.x * canvas_size[0])
        y = int(elem.y * canvas_size[1])
        w = max(1, int(elem.width * canvas_size[0]))
        h = max(1, int(elem.height * canvas_size[1]))

        if elem.element_type == "text":
            draw.rectangle([x, y, x + w, y + h], fill=box_color, outline=None)
            line_y = y + h // 3
            draw.line([(x + 10, line_y), (x + w - 10, line_y)], fill="#888888", width=2)
            line_y2 = y + h * 2 // 3
            draw.line([(x + 10, line_y2), (x + w - 10, line_y2)], fill="#888888", width=2)

        elif elem.element_type == "image":
            draw.rectangle([x, y, x + w, y + h], fill="#666666", outline=None)
            draw.line([(x, y), (x + w, y + h)], fill="#999999", width=1)
            draw.line([(x + w, y), (x, y + h)], fill="#999999", width=1)

        elif elem.element_type == "subject":
            draw.rectangle([x, y, x + w, y + h], fill="#DDDDDD", outline=None)

    return canvas


def create_attention_mask_map(
    layout_elements: list,
    canvas_size: Tuple[int, int],
    element_id: str,
) -> np.ndarray:
    """
    创建注意力掩码图 - 参考 CreatiDesign 的 attention_mask 机制
    为指定元素生成其在画布上的位置掩码
    """
    mask = np.zeros(canvas_size, dtype=np.float32)

    for elem in layout_elements:
        if elem.element_id == element_id:
            x = int(elem.x * canvas_size[0])
            y = int(elem.y * canvas_size[1])
            w = max(1, int(elem.width * canvas_size[0]))
            h = max(1, int(elem.height * canvas_size[1]))
            x_end = min(canvas_size[0], x + w)
            y_end = min(canvas_size[1], y + h)
            x = max(0, x)
            y = max(0, y)
            mask[y:y_end, x:x_end] = 1.0
            break

    return mask


def create_composite_preview(
    layout_elements: list,
    canvas_size: Tuple[int, int],
    subject_image: Optional[Image.Image] = None,
    element_images: Optional[Dict[str, Image.Image]] = None,
) -> Image.Image:
    """创建合成预览图 - 将所有元素放在一起预览"""
    if element_images is None:
        element_images = {}

    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))

    for elem in sorted(layout_elements, key=lambda e: e.z_index):
        x = int(elem.x * canvas_size[0])
        y = int(elem.y * canvas_size[1])
        w = max(1, int(elem.width * canvas_size[0]))
        h = max(1, int(elem.height * canvas_size[1]))

        if elem.element_type == "subject" and subject_image is not None:
            img = subject_image.convert("RGBA")
            ratio = min(w / max(1, img.width), h / max(1, img.height))
            new_w = max(1, int(img.width * ratio))
            new_h = max(1, int(img.height * ratio))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            cx = x + (w - new_w) // 2
            cy = y + (h - new_h) // 2
            canvas.paste(resized, (cx, cy), resized)

        elif elem.element_type == "image" and elem.element_id in element_images:
            img = element_images[elem.element_id].convert("RGBA")
            img_resized = img.resize((w, h), Image.LANCZOS)
            canvas.paste(img_resized, (x, y), img_resized)

        elif elem.element_type == "text":
            draw = ImageDraw.Draw(canvas)
            font = None
            font_candidates = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/arial.ttf",
            ]
            for fp in font_candidates:
                try:
                    font = ImageFont.truetype(fp, getattr(elem, "font_size", 24))
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()
            fc = getattr(elem, "font_color", "#000000")
            draw.text((x + 10, y + 10), getattr(elem, "text", ""), fill=fc, font=font)

    return canvas.convert("RGB")
