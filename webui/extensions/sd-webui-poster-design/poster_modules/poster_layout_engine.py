"""
海报布局引擎
参考 CreatiDesign 的布局概念，将海报设计抽象为多个设计元素（文本框、图片框）的布局管理
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import json


# ============================================================
# 设计元素数据模型
# ============================================================

@dataclass
class DesignElement:
    """单个设计元素的基类"""
    element_id: str
    element_type: str          # "text" | "image" | "subject"
    x: float                   # 归一化坐标 0-1
    y: float
    width: float
    height: float
    z_index: int = 0           # 层级

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "z_index": self.z_index,
        }


@dataclass
class TextElement(DesignElement):
    """文本框元素"""
    text: str = ""
    font_size: int = 24
    font_color: str = "#000000"
    font_family: str = "Arial"
    text_align: str = "left"           # left | center | right
    background_color: str = "transparent"
    opacity: float = 1.0
    rotation: float = 0.0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "text": self.text,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "font_family": self.font_family,
            "text_align": self.text_align,
            "background_color": self.background_color,
            "opacity": self.opacity,
            "rotation": self.rotation,
        })
        return base


@dataclass
class ImageElement(DesignElement):
    """图片框元素（装饰图、LOGO等）"""
    image_path: str = ""
    fit_mode: str = "cover"            # cover | contain | fill
    opacity: float = 1.0
    rotation: float = 0.0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "image_path": self.image_path,
            "fit_mode": self.fit_mode,
            "opacity": self.opacity,
            "rotation": self.rotation,
        })
        return base


@dataclass
class SubjectElement(DesignElement):
    """主体元素（产品图 - 类似 CreatiDesign 的 subject）"""
    image_path: str = ""
    subject_prompt: str = ""
    fit_mode: str = "contain"
    shadow: bool = False
    shadow_offset: Tuple[int, int] = (5, 5)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "image_path": self.image_path,
            "subject_prompt": self.subject_prompt,
            "fit_mode": self.fit_mode,
            "shadow": self.shadow,
            "shadow_offset": list(self.shadow_offset),
        })
        return base


# ============================================================
# 海报布局
# ============================================================

@dataclass
class PosterLayout:
    """完整海报布局"""
    width: int = 1024
    height: int = 1024
    background_prompt: str = "干净简约的背景, 柔和的渐变, 高级感"
    background_color: str = "#FFFFFF"
    elements: List[DesignElement] = field(default_factory=list)
    global_prompt: str = "高质量海报设计, 商业级, 精致排版"
    negative_prompt: str = "低质量, 模糊, 扭曲, 变形, 文字错误"

    def add_element(self, element: DesignElement):
        self.elements.append(element)
        self.elements.sort(key=lambda e: e.z_index)

    def remove_element(self, element_id: str):
        self.elements = [e for e in self.elements if e.element_id != element_id]

    def get_element(self, element_id: str) -> Optional[DesignElement]:
        for e in self.elements:
            if e.element_id == element_id:
                return e
        return None

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "background_prompt": self.background_prompt,
            "background_color": self.background_color,
            "global_prompt": self.global_prompt,
            "negative_prompt": self.negative_prompt,
            "elements": [e.to_dict() for e in self.elements],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "PosterLayout":
        layout = cls(
            width=data.get("width", 1024),
            height=data.get("height", 1024),
            background_prompt=data.get("background_prompt", ""),
            background_color=data.get("background_color", "#FFFFFF"),
            global_prompt=data.get("global_prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
        )
        for elem_data in data.get("elements", []):
            elem_type = elem_data.get("element_type")
            if elem_type == "text":
                valid_fields = {k: v for k, v in elem_data.items() if k in TextElement.__dataclass_fields__}
                element = TextElement(**valid_fields)
            elif elem_type == "image":
                valid_fields = {k: v for k, v in elem_data.items() if k in ImageElement.__dataclass_fields__}
                element = ImageElement(**valid_fields)
            elif elem_type == "subject":
                valid_fields = {k: v for k, v in elem_data.items() if k in SubjectElement.__dataclass_fields__}
                element = SubjectElement(**valid_fields)
            else:
                continue
            layout.elements.append(element)
        return layout


# ============================================================
# 布局预览生成
# ============================================================

def _load_font(size: int = 14) -> ImageFont.ImageFont:
    """加载字体，优先使用系统中文字体"""
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",       # 宋体
        "C:/Windows/Fonts/arial.ttf",        # Arial
        "arial.ttf",
    ]
    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_grid(draw: ImageDraw.ImageDraw, w: int, h: int, spacing: int = 64):
    """绘制辅助网格"""
    grid_color = (200, 200, 200, 40)
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: str = "#FFFFFF"):
    """在元素左上角绘制标签"""
    font = _load_font(14)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 4
    label_y = max(th, y)
    draw.rectangle([x, label_y - th, x + tw, label_y], fill="#333333DD")
    draw.text((x + 4, label_y - th + 2), text, fill=color, font=font)


def _resize_image_to_fit(img: Image.Image, target_w: int, target_h: int, mode: str = "cover") -> Image.Image:
    """调整图片适配目标区域"""
    img = img.convert("RGBA")
    if mode == "cover":
        ratio = max(target_w / img.width, target_h / img.height)
        new_w = max(1, int(img.width * ratio))
        new_h = max(1, int(img.height * ratio))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return resized.crop((max(0, left), max(0, top), left + target_w, top + target_h))
    elif mode == "contain":
        ratio = min(target_w / img.width, target_h / img.height)
        new_w = max(1, int(img.width * ratio))
        new_h = max(1, int(img.height * ratio))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas.paste(resized, (x, y), resized)
        return canvas
    else:  # fill
        return img.resize((target_w, target_h), Image.LANCZOS)


def render_layout_preview(
    layout: PosterLayout,
    subject_image: Optional[Image.Image] = None,
    element_images: Optional[dict] = None,
) -> Image.Image:
    """渲染布局预览图 - 在画布上绘制所有元素的位置框及内容"""
    if element_images is None:
        element_images = {}

    w = max(64, int(layout.width))
    h = max(64, int(layout.height))

    canvas = Image.new("RGBA", (w, h), layout.background_color)
    draw = ImageDraw.Draw(canvas)
    _draw_grid(draw, w, h)

    for element in layout.elements:
        px = max(0, int(element.x * w))
        py = max(0, int(element.y * h))
        pw = max(1, int(element.width * w))
        ph = max(1, int(element.height * h))

        if element.element_type == "subject" and subject_image is not None:
            try:
                resized = _resize_image_to_fit(subject_image, pw, ph, element.fit_mode)
                canvas.paste(resized, (px, py), resized if resized.mode == "RGBA" else None)
            except Exception:
                pass
            draw.rectangle([px, py, px + pw, py + ph], outline="#FF6B35", width=3)
            _draw_label(draw, px, py, f"主体 [{element.element_id}]")

        elif element.element_type == "image" and element_images and element.element_id in element_images:
            try:
                img = element_images[element.element_id]
                resized = _resize_image_to_fit(img, pw, ph, element.fit_mode)
                canvas.paste(resized, (px, py), resized if resized.mode == "RGBA" else None)
            except Exception:
                pass
            draw.rectangle([px, py, px + pw, py + ph], outline="#4A90D9", width=2)
            _draw_label(draw, px, py, f"图片 [{element.element_id}]")

        elif element.element_type == "text":
            draw.rectangle([px, py, px + pw, py + ph], outline="#7ED321", width=2)
            overlay = Image.new("RGBA", (pw, ph), (126, 211, 33, 30))
            canvas.paste(overlay, (px, py), overlay)
            display_text = element.text[:20] + "..." if len(element.text) > 20 else element.text
            _draw_label(draw, px, py, f"文字: {display_text}")

        else:
            draw.rectangle([px, py, px + pw, py + ph], outline="#888888", width=1, fill=(200, 200, 200, 60))
            _draw_label(draw, px, py, f"[{element.element_type}] {element.element_id}")

    return canvas.convert("RGB")


# ============================================================
# 预设海报模板
# ============================================================

POSTER_TEMPLATES = {
    "电商产品海报": PosterLayout(
        width=1024,
        height=1280,
        background_prompt="纯净白色背景, 柔和光影, 高级感商业摄影背景",
        global_prompt="高端电商产品海报, 极简风格, 精致光影, 商业摄影级品质, 8K",
        elements=[
            TextElement(element_id="title", element_type="text", x=0.05, y=0.03, width=0.9, height=0.08,
                        text="新品首发", font_size=48, font_color="#1a1a1a", text_align="center", z_index=10),
            TextElement(element_id="subtitle", element_type="text", x=0.05, y=0.10, width=0.9, height=0.05,
                        text="2025 春季限定系列", font_size=28, font_color="#666666", text_align="center", z_index=9),
            SubjectElement(element_id="product", element_type="subject", x=0.1, y=0.18, width=0.8, height=0.55,
                           subject_prompt="主体产品, 精致细节, 专业打光", z_index=5),
            TextElement(element_id="price", element_type="text", x=0.3, y=0.78, width=0.4, height=0.06,
                        text="¥ 299", font_size=36, font_color="#FF6B35", text_align="center", z_index=8),
            TextElement(element_id="cta", element_type="text", x=0.25, y=0.87, width=0.5, height=0.06,
                        text="立即购买", font_size=24, font_color="#FFFFFF", text_align="center",
                        background_color="#FF6B35", z_index=7),
            TextElement(element_id="footer", element_type="text", x=0.05, y=0.94, width=0.9, height=0.04,
                        text="* 限时优惠, 最终解释权归品牌所有", font_size=14, font_color="#999999",
                        text_align="center", z_index=6),
        ],
    ),
    "电影海报": PosterLayout(
        width=1024,
        height=1536,
        background_prompt="史诗感暗色背景, 戏剧性光影, 电影级质感",
        global_prompt="电影海报, 史诗级, 戏剧性光影, 电影级质感, 震撼视觉",
        elements=[
            SubjectElement(element_id="hero", element_type="subject", x=0.0, y=0.0, width=1.0, height=0.7,
                           subject_prompt="主角人物, 特写, 光影对比", z_index=1),
            TextElement(element_id="title", element_type="text", x=0.05, y=0.72, width=0.9, height=0.1,
                        text="星际迷航", font_size=64, font_color="#FFFFFF", text_align="center", z_index=10),
            TextElement(element_id="tagline", element_type="text", x=0.05, y=0.82, width=0.9, height=0.05,
                        text="探索未知 超越极限", font_size=28, font_color="#CCCCCC", text_align="center", z_index=9),
            TextElement(element_id="date", element_type="text", x=0.05, y=0.90, width=0.9, height=0.04,
                        text="2025年8月 全国上映", font_size=20, font_color="#FFD700", text_align="center", z_index=8),
            TextElement(element_id="credits", element_type="text", x=0.05, y=0.95, width=0.9, height=0.04,
                        text="导演: 某某某 | 主演: 某某某", font_size=14, font_color="#888888",
                        text_align="center", z_index=7),
        ],
    ),
    "社交媒体卡片": PosterLayout(
        width=1024,
        height=1024,
        background_prompt="柔和渐变背景, 清新自然, 简洁",
        global_prompt="社交媒体海报, 清新风格, 简洁大方, 高级感",
        elements=[
            TextElement(element_id="title", element_type="text", x=0.05, y=0.05, width=0.9, height=0.08,
                        text="每日灵感", font_size=52, font_color="#2C3E50", text_align="center", z_index=10),
            TextElement(element_id="subtitle", element_type="text", x=0.05, y=0.13, width=0.9, height=0.05,
                        text="DESIGN INSPIRATION", font_size=18, font_color="#95A5A6", text_align="center", z_index=9),
            SubjectElement(element_id="visual", element_type="subject", x=0.1, y=0.22, width=0.8, height=0.55,
                           subject_prompt="精美的设计作品, 艺术感, 创意", z_index=5),
            TextElement(element_id="quote", element_type="text", x=0.05, y=0.80, width=0.9, height=0.08,
                        text="设计不止于视觉, 更是思维的延伸", font_size=24, font_color="#34495E",
                        text_align="center", z_index=8),
            TextElement(element_id="footer", element_type="text", x=0.05, y=0.92, width=0.9, height=0.05,
                        text="#设计灵感 #创意生活", font_size=16, font_color="#7F8C8D", text_align="center", z_index=7),
        ],
    ),
    "简约品牌海报": PosterLayout(
        width=1024,
        height=1024,
        background_prompt="纯色背景, 极简, 高级感, 留白",
        global_prompt="品牌海报, 极简主义, 高级感, 精致排版, 商业设计",
        elements=[
            TextElement(element_id="brand", element_type="text", x=0.1, y=0.08, width=0.8, height=0.12,
                        text="BRAND", font_size=72, font_color="#000000", text_align="center", z_index=10),
            TextElement(element_id="slogan", element_type="text", x=0.1, y=0.20, width=0.8, height=0.06,
                        text="简约 · 优雅 · 品质", font_size=24, font_color="#999999", text_align="center", z_index=9),
            SubjectElement(element_id="product", element_type="subject", x=0.2, y=0.35, width=0.6, height=0.45,
                           subject_prompt="产品展示, 简洁, 精致", z_index=5),
            TextElement(element_id="desc", element_type="text", x=0.1, y=0.85, width=0.8, height=0.08,
                        text="探索更多", font_size=20, font_color="#000000", text_align="center", z_index=8),
        ],
    ),
}
