"""
头部比例参考线生成插件 for Stable Diffusion WebUI
功能：
- 根据选择的头身比（2-9头身）生成人体轮廓参考图
- AI视觉分析：通过llama.cpp调用视觉模型分析参考图，提取身材比例和风格关键词
- 自动生成对应的Prompt关键词（正向+负向）
- 参考图可直接用于ControlNet草图/线稿控制
"""

import os
import math
import io
import json
import base64
import re
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from modules import scripts, scripts_postprocessing, shared
from modules.processing import StableDiffusionProcessing
import gradio as gr

logger = logging.getLogger(__name__)

# 确保 scripts 目录在 Python 路径中（必须在导入 llm_backend 之前）
import sys
from pathlib import Path
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# 统一后端模块
try:
    import llm_backend
    from backend_selector_ui import build_backend_selector
    LLM_BACKEND_AVAILABLE = True
except Exception as e:
    LLM_BACKEND_AVAILABLE = False

# 打光分析模块（用于图片打光分析 tab）
try:
    import lighting_assistant_module
    LIGHTING_ANALYSIS_AVAILABLE = True
except Exception as e:
    LIGHTING_ANALYSIS_AVAILABLE = False
    logger.warning(f"lighting_assistant_module 不可用：{e}")


class ProportionGuideScript(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "人物场景设计分析"

    def show(self, is_img2img):
        return False

    def ui(self, is_img2img):
        with gr.Accordion("✨ 人体头身比例场景构图分析", open=False, elem_id="proportion-guide-accordion"):
            # 左右切换的 Tab 页
            with gr.Tabs():
                # ============================================================
                # Tab 1: 人体头身比例分析
                # ============================================================
                with gr.Tab("🧍 人体头身比例"):
                    # 流程1: AI 视觉分析
                    gr.Markdown("### 🖼️ AI视觉分析（从参考图提取身材与风格）")

                    with gr.Row():
                        analysis_image = gr.Image(
                            label="上传参考图（角色全身图）",
                            type="pil",
                            height=300
                        )
                        analysis_grid_preview = gr.Image(
                            label="参考线叠加预览（将发送给视觉模型分析）",
                            interactive=False,
                            height=300
                        )

                    with gr.Row():
                        analyze_status = gr.Textbox(
                            label="状态",
                            value="等待分析...",
                            interactive=False
                        )
                        analyze_btn = gr.Button("🔍 分析图片", variant="primary", scale=0, min_width=120)

                    with gr.Row():
                        analyzed_raw = gr.Textbox(
                            label="📝 原始图像描述（未过滤，可编辑）",
                            placeholder="等待分析结果...",
                            lines=4,
                            interactive=True
                        )

                    with gr.Row():
                        analyzed_proportion = gr.Textbox(
                            label="📏 头身比/身材描述（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        analyzed_negative = gr.Textbox(
                            label="🚫 风格关键词 - 负向（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    gr.Markdown("---")
                    gr.Markdown("### 🎨 设计元素分析（同图提取色彩、风格、排版与视觉构成）")

                    with gr.Row():
                        design_color = gr.Textbox(
                            label="🎨 色彩方案（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        design_category = gr.Textbox(
                            label="🏷️ 设计风格类别（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        design_typography = gr.Textbox(
                            label="📝 排版布局（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        design_hierarchy = gr.Textbox(
                            label="📐 视觉层次与构成（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    # ============================================================
                # Tab 2: 场景构图分析
                # ============================================================
                with gr.Tab("🎬 场景构图分析"):
                    gr.Markdown("### 🎬 场景构图分析（从参考图提取构图与镜头语言）")

                    with gr.Row():
                        scene_analysis_image = gr.Image(
                            label="上传场景参考图",
                            type="pil",
                            height=300
                        )
                        scene_grid_preview = gr.Image(
                            label="构图参考线叠加预览（将发送给视觉模型分析）",
                            interactive=False,
                            height=300
                        )

                    with gr.Row():
                        scene_analyze_status = gr.Textbox(
                            label="状态",
                            value="等待分析...",
                            interactive=False
                        )
                        scene_analyze_btn = gr.Button("🔍 分析场景构图", variant="primary", scale=0, min_width=140)

                    with gr.Row():
                        scene_raw_desc = gr.Textbox(
                            label="📝 场景原始描述（未过滤，可编辑）",
                            placeholder="等待分析结果...",
                            lines=4,
                            interactive=True
                        )

                    with gr.Row():
                        scene_composition = gr.Textbox(
                            label="📐 构图方式（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        scene_camera = gr.Textbox(
                            label="📷 镜头角度与景别（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        scene_lighting = gr.Textbox(
                            label="💡 光影氛围（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        scene_positive = gr.Textbox(
                            label="🎨 场景正向Prompt（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        scene_negative = gr.Textbox(
                            label="🚫 场景负向Prompt（过滤后，可编辑）",
                            placeholder="等待分析结果...",
                            lines=2,
                            interactive=True
                        )

                    with gr.Row():
                        gr.Markdown("""
                        **使用说明：**
                        上传场景参考图 → 点击「分析场景构图」→ 自动叠加三分法/对角线构图参考线 → llama.cpp视觉模型分析 → 自动填入场景描述、构图方式、镜头角度、光影氛围和正/负向Prompt（均可自由编辑）
                        """)

            # 分析按钮事件（人体比例 + 设计元素）
            analyze_btn.click(
                fn=self.analyze_reference_image,
                inputs=[analysis_image],
                outputs=[analyze_status, analysis_grid_preview, analyzed_raw,
                         analyzed_proportion, analyzed_negative,
                         design_color, design_category, design_typography, design_hierarchy]
            )

            # 场景构图分析按钮事件
            scene_analyze_btn.click(
                fn=self.analyze_scene_image,
                inputs=[scene_analysis_image],
                outputs=[scene_analyze_status, scene_grid_preview, scene_raw_desc,
                         scene_composition, scene_camera, scene_lighting,
                         scene_positive, scene_negative]
            )

            return [
                analysis_image, analysis_grid_preview,
                analyzed_raw, analyzed_proportion, analyzed_negative,
                analyze_btn, analyze_status,
                design_color, design_category, design_typography, design_hierarchy,
                scene_analysis_image, scene_grid_preview,
                scene_analyze_status, scene_raw_desc, scene_composition,
                scene_camera, scene_lighting, scene_positive, scene_negative,
                scene_analyze_btn
            ]

    # ------------------------------------------------------------------
    # 方法1: 在图片上叠加头身比参考线
    # ------------------------------------------------------------------
    def overlay_proportion_grid(self, pil_image):
        """在用户上传的参考图上自动绘制头身比参考线（固定9头身网格），返回叠加后的图片"""
        img = pil_image.convert("RGBA")
        w, h = img.size

        n_heads = 9
        head_size = h / n_heads

        # 先压暗原图，让参考线更突出
        dim = Image.new("RGBA", (w, h), (0, 0, 0, 40))
        base = Image.alpha_composite(img, dim)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 每隔一段交替着色，让视觉模型更容易区分"头"的段落
        colors = [(255, 215, 0, 35), (255, 255, 255, 0)]  # 浅金色半透明交替
        for i in range(n_heads):
            ys = int(i * head_size)
            ye = int((i + 1) * head_size)
            draw.rectangle([0, ys, w, ye], fill=colors[i % 2])

        # 右侧加粗红色计数条（每个头一段）
        bar_w = 18
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = None
        for i in range(n_heads):
            ys = int(i * head_size)
            ye = int((i + 1) * head_size)
            draw.rectangle([w - bar_w - 4, ys, w - 4, ye], fill=(255, 60, 60, 200))
            # 白色大号数字
            cy = (ys + ye) // 2
            draw.text((w - bar_w - 2, cy - 10), f"{i+1}", fill=(255, 255, 255, 255), font=font)

        # 左侧文字标签 + 粗亮蓝色水平线
        try:
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_small = None
        for i in range(n_heads + 1):
            y = int(i * head_size)
            draw.line([(0, y), (w, y)], fill=(0, 180, 255, 220), width=4)
            lx, ly = 4, y + 2
            label = f"{i}头"
            # 黑色描边+白色文字
            draw.text((lx, ly), label, fill=(0, 0, 0, 230), font=font_small)
            draw.text((lx - 1, ly - 1), label, fill=(255, 255, 255, 255), font=font_small)

        # 最外边框
        draw.rectangle([0, 0, w - 1, h - 1], outline=(0, 180, 255, 220), width=3)

        combined = Image.alpha_composite(base, overlay).convert("RGB")
        return combined

    # ------------------------------------------------------------------
    # 方法2: 调用 llama.cpp 视觉模型分析图片（带参考线叠加）
    # ------------------------------------------------------------------
    def analyze_reference_image(self, pil_image, backend="llamacpp", model=""):
        """
        将叠加了参考线的图片发送到视觉模型分析，
        返回（状态, 叠加预览图, 原始描述, 头身比, 正向风格, 负向风格,
               色彩方案, 设计风格类别, 排版布局, 视觉层次）
        """
        if pil_image is None:
            return "⚠️ 请先上传参考图", None, "", "", "", "", "", "", ""

        # ---- 先自动叠加头身比参考线（固定9头身） ----
        try:
            overlaid = self.overlay_proportion_grid(pil_image)
        except Exception as e:
            return f"❌ 叠加参考线失败: {str(e)}", None, "", "", "", "", "", "", ""

        # ---- 保存到临时文件 ----
        try:
            import tempfile
            img_resized = overlaid.copy()
            max_side = 768
            w, h = img_resized.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img_resized = img_resized.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img_resized.save(temp_file.name, format="JPEG", quality=85)
            temp_path = temp_file.name
        except Exception as e:
            return f"❌ 图片编码失败: {str(e)}", overlaid, "", "", "", "", "", "", ""

        # ---- 构建分析 Prompt ----
        system_prompt = (
            "分析图片中的角色，严格按照以下格式输出七行，只输出这七行：\n"
            "RAW: 用完整段落详细描述角色的外貌、五官、发型、服装、姿态、体型，风格等所有视觉细节。\n"
            "PROPORTION: 根据头部大小计算出的人物头身比例，格式为「X头身比例」并描述角色身材特点」\n"
            "NEGATIVE: 负向关键词，逗号分隔\n"
            "COLOR: 图片的色彩方案，包括主色调、辅助色、点缀色，以及色彩搭配关系。\n"
            "STYLE_TYPE: 设计和艺术风格类别，如扁平化、拟物化、极简主义、赛博朋克、日式、北欧风格等。\n"
            "TYPOGRAPHY: 排版与布局特点，如字体风格、对齐方式、留白运用、元素空间分布等。\n"
            "HIERARCHY: 视觉层次与构成，包括主体位置、视觉重心、元素大小对比、空间关系等。\n\n"
            "注意：RAW、PROPORTION、NEGATIVE、COLOR、STYLE_TYPE、TYPOGRAPHY、HIERARCHY 这七个标签必须使用中文标签。不要添加任何额外文字、解释或Markdown格式。"
        )

        try:
            import requests
            # 调用统一后端
            result = llm_backend.analyze_image(
                image_path=temp_path,
                prompt=system_prompt,
                model=model,
                backend_type=backend,
                max_tokens=8192,
            )
            if not result.get("success"):
                return result.get("analysis", "分析失败"), overlaid, "", "", "", "", "", "", ""

            content = result["analysis"]

            # ---- 清理模型输出 ----
            content = re.sub(r'</?think>', '', content, flags=re.DOTALL)
            content = content.strip()

            # ---- 解析返回的八个字段（支持中英文标签、多行内容） ----
            raw_desc = ""
            proportion = ""
            style = ""
            negative = ""
            color = ""
            style_type = ""
            typography = ""
            hierarchy = ""

            # 定义标签映射：英文标签（大写）→ 中文标签
            label_map = {
                "RAW": ["RAW", "原始描述", "原始图像描述", "原始"],
                "PROPORTION": ["PROPORTION", "头身比", "身材描述", "比例", "身材"],
                "STYLE": ["STYLE", "风格关键词", "正向关键词", "风格", "正向"],
                "NEGATIVE": ["NEGATIVE", "负向关键词", "负向"],
                "COLOR": ["COLOR", "色彩方案", "色彩", "颜色"],
                "STYLE_TYPE": ["STYLE_TYPE", "设计风格类别", "设计风格", "风格类别"],
                "TYPOGRAPHY": ["TYPOGRAPHY", "排版布局", "排版", "字体"],
                "HIERARCHY": ["HIERARCHY", "视觉层次", "层次", "构成", "视觉"],
            }

            # 方法1：按行解析（处理单行标签格式）
            for line in content.split("\n"):
                line_stripped = line.strip().lstrip("*#-").strip().replace("：", ":")
                upper_line = line_stripped.upper()

                if upper_line.startswith("RAW:"):
                    raw_desc = line_stripped[len("RAW:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("PROPORTION:"):
                    proportion = line_stripped[len("PROPORTION:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("STYLE:"):
                    style = line_stripped[len("STYLE:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("NEGATIVE:"):
                    negative = line_stripped[len("NEGATIVE:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("COLOR:"):
                    color = line_stripped[len("COLOR:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("STYLE_TYPE:"):
                    style_type = line_stripped[len("STYLE_TYPE:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("TYPOGRAPHY:"):
                    typography = line_stripped[len("TYPOGRAPHY:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("HIERARCHY:"):
                    hierarchy = line_stripped[len("HIERARCHY:"):].strip().lstrip("*#-").strip()

            # 方法2：如果按行解析没命中，尝试用正则从全文提取区块
            if not any([raw_desc, proportion, style, negative, color, style_type, typography, hierarchy]):
                found_any = False

                # 用正则提取每个字段：匹配 "标签: 内容（可跨行）" 直到下一个标签或结尾
                all_labels = [l for group in label_map.values() for l in group]
                escaped_labels = [re.escape(l) for l in all_labels]
                label_pattern = "|".join(escaped_labels)

                for label_en, label_variants in label_map.items():
                    # 匹配当前标签后跟冒号，然后捕获内容直到下一个标签或结尾
                    escaped_variants = [re.escape(v) for v in label_variants]
                    pattern = rf'(?:{"|".join(escaped_variants)})\s*[:：]\s*(.+?)(?=\n(?:{label_pattern})\s*[:：]|\Z)'
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        val = match.group(1).strip().rstrip("*#-").strip()
                        if val:
                            if label_en == "RAW":
                                raw_desc = val
                            elif label_en == "PROPORTION":
                                proportion = val
                            elif label_en == "STYLE":
                                style = val
                            elif label_en == "NEGATIVE":
                                negative = val
                            elif label_en == "COLOR":
                                color = val
                            elif label_en == "STYLE_TYPE":
                                style_type = val
                            elif label_en == "TYPOGRAPHY":
                                typography = val
                            elif label_en == "HIERARCHY":
                                hierarchy = val
                            found_any = True

                if not found_any:
                    # 方法3：完全没匹配到任何标签，全文放入 raw_desc
                    raw_desc = content.strip()
                    # 截取前200字符用于调试
                    content_preview = content[:200].replace("\n", "↵")
                    # 尝试从头身比关键词提取
                    m = re.search(r'(\d+)\s*头身', content)
                    if m:
                        proportion = f"{m.group(1)}头身"
                    # 尝试提取风格关键词（逗号分隔的中文词组）

                    # 在状态中显示原始输出预览，帮助调试
                    return (
                        f"⚠️ 未识别到标准标签，原始输出已填入原始描述框（预览: {content_preview}）",
                        overlaid, raw_desc, proportion, negative,
                        color, style_type, typography, hierarchy
                    )

            return "✅ 分析完成！原始描述和过滤后关键词均已填入", overlaid, raw_desc, proportion, negative, color, style_type, typography, hierarchy

        except ImportError:
            return "❌ 错误: 缺少必要模块", overlaid, "", "", "", "", "", "", ""
        except Exception as e:
            return f"❌ 分析失败: {str(e)}", overlaid, "", "", "", "", "", "", ""

    # ------------------------------------------------------------------
    # 方法4: 在场景图上叠加构图参考线（三分法+对角线）
    # ------------------------------------------------------------------
    def overlay_composition_grid(self, pil_image):
        """在用户上传的场景图上叠加构图参考线（三分法+对角线），返回叠加后的图片"""
        img = pil_image.convert("RGBA")
        w, h = img.size

        # 先压暗原图，让参考线更突出
        dim = Image.new("RGBA", (w, h), (0, 0, 0, 40))
        base = Image.alpha_composite(img, dim)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 三分法参考线（黄色）
        line_color = (255, 215, 0, 200)
        line_width = 3

        # 垂直三分线
        for i in range(1, 3):
            x = int(w * i / 3)
            draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)

        # 水平三分线
        for i in range(1, 3):
            y = int(h * i / 3)
            draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

        # 对角线参考线（蓝色）
        diagonal_color = (0, 180, 255, 180)
        # 主对角线
        draw.line([(0, 0), (w, h)], fill=diagonal_color, width=2)
        draw.line([(w, 0), (0, h)], fill=diagonal_color, width=2)

        # 中心点标记
        center_x, center_y = w // 2, h // 2
        marker_size = 15
        draw.ellipse([
            center_x - marker_size, center_y - marker_size,
            center_x + marker_size, center_y + marker_size
        ], outline=(255, 60, 60, 220), width=3)

        # 四个三分交点标记
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = None

        intersection_points = [
            (int(w / 3), int(h / 3), "左上"),
            (int(w * 2 / 3), int(h / 3), "右上"),
            (int(w / 3), int(h * 2 / 3), "左下"),
            (int(w * 2 / 3), int(h * 2 / 3), "右下"),
        ]

        for px, py, label in intersection_points:
            # 小圆点
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 60, 60, 200))
            # 标签
            draw.text((px + 10, py - 8), label, fill=(255, 255, 255, 255), font=font)
            draw.text((px + 9, py - 9), label, fill=(0, 0, 0, 230), font=font)

        # 最外边框
        draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 215, 0, 220), width=3)

        combined = Image.alpha_composite(base, overlay).convert("RGB")
        return combined

    # ------------------------------------------------------------------
    # 方法5: 调用 llama.cpp 视觉模型分析场景构图
    # ------------------------------------------------------------------
    def analyze_scene_image(self, pil_image, backend="llamacpp", model=""):
        """
        将叠加了构图参考线的场景图发送到视觉模型分析，
        返回（状态, 叠加预览图, 原始描述, 构图方式, 镜头角度, 光影氛围, 正向Prompt, 负向Prompt）
        """
        if pil_image is None:
            return "⚠️ 请先上传场景参考图", None, "", "", "", "", "", ""

        # ---- 先自动叠加构图参考线 ----
        try:
            overlaid = self.overlay_composition_grid(pil_image)
        except Exception as e:
            return f"❌ 叠加构图参考线失败: {str(e)}", None, "", "", "", "", "", ""

        # ---- 保存到临时文件 ----
        try:
            import tempfile
            img_resized = overlaid.copy()
            max_side = 768
            w, h = img_resized.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img_resized = img_resized.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img_resized.save(temp_file.name, format="JPEG", quality=85)
            temp_path = temp_file.name
        except Exception as e:
            return f"❌ 图片编码失败: {str(e)}", overlaid, "", "", "", "", "", ""

        # ---- 构建场景构图分析 Prompt ----
        system_prompt = (
            "分析图片中的场景构图，严格按照以下格式输出六行，只输出这六行：\n"
            "RAW: 用完整段落详细描述场景的整体氛围、环境元素、空间布局、色彩基调等所有视觉细节。\n"
            "COMPOSITION: 构图方式，如三分法构图、对角线构图、中心构图、框架式构图、引导线构图等，并说明主体位置。\n"
            "CAMERA: 镜头角度与景别，如俯视角度、仰视角度、平视角度、特写、近景、中景、全景、远景等。\n"
            "LIGHTING: 光影氛围，如自然光、逆光、侧光、顶光、暖色调、冷色调、高对比度、柔光等。\n"
            "POSITIVE: 场景正向Prompt关键词，逗号分隔，包含构图、镜头、光影、氛围等。\n"
            "NEGATIVE: 场景负向Prompt关键词，逗号分隔。\n\n"
            "注意：RAW、COMPOSITION、CAMERA、LIGHTING、POSITIVE、NEGATIVE 这六个标签必须使用中文标签。不要添加任何额外文字、解释或Markdown格式。"
        )

        try:
            import requests
            # 调用统一后端
            result = llm_backend.analyze_image(
                image_path=temp_path,
                prompt=system_prompt,
                model=model,
                backend_type=backend,
                max_tokens=8192,
            )
            if not result.get("success"):
                return result.get("analysis", "分析失败"), overlaid, "", "", "", "", "", ""

            content = result["analysis"]

            # ---- 清理模型输出 ----
            content = re.sub(r'</?think>', '', content, flags=re.DOTALL)
            content = content.strip()

            # ---- 解析返回的六个字段 ----
            raw_desc = ""
            composition = ""
            camera = ""
            lighting = ""
            positive = ""
            negative = ""

            # 定义标签映射
            label_map = {
                "RAW": ["RAW", "原始描述", "场景描述", "原始"],
                "COMPOSITION": ["COMPOSITION", "构图方式", "构图"],
                "CAMERA": ["CAMERA", "镜头角度", "景别", "镜头"],
                "LIGHTING": ["LIGHTING", "光影氛围", "光影", "光照"],
                "POSITIVE": ["POSITIVE", "正向关键词", "正向Prompt", "正向"],
                "NEGATIVE": ["NEGATIVE", "负向关键词", "负向Prompt", "负向"],
            }

            # 方法1：按行解析
            for line in content.split("\n"):
                line_stripped = line.strip().lstrip("*#-").strip().replace("：", ":")
                upper_line = line_stripped.upper()

                if upper_line.startswith("RAW:"):
                    raw_desc = line_stripped[len("RAW:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("COMPOSITION:"):
                    composition = line_stripped[len("COMPOSITION:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("CAMERA:"):
                    camera = line_stripped[len("CAMERA:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("LIGHTING:"):
                    lighting = line_stripped[len("LIGHTING:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("POSITIVE:"):
                    positive = line_stripped[len("POSITIVE:"):].strip().lstrip("*#-").strip()
                elif upper_line.startswith("NEGATIVE:"):
                    negative = line_stripped[len("NEGATIVE:"):].strip().lstrip("*#-").strip()

            # 方法2：如果按行解析没命中，尝试用正则从全文提取区块
            if not any([raw_desc, composition, camera, lighting, positive, negative]):
                found_any = False

                all_labels = [l for group in label_map.values() for l in group]
                escaped_labels = [re.escape(l) for l in all_labels]
                label_pattern = "|".join(escaped_labels)

                for label_en, label_variants in label_map.items():
                    escaped_variants = [re.escape(v) for v in label_variants]
                    pattern = rf'(?:{"|".join(escaped_variants)})\s*[:：]\s*(.+?)(?=\n(?:{label_pattern})\s*[:：]|\Z)'
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        val = match.group(1).strip().rstrip("*#-").strip()
                        if val:
                            if label_en == "RAW":
                                raw_desc = val
                            elif label_en == "COMPOSITION":
                                composition = val
                            elif label_en == "CAMERA":
                                camera = val
                            elif label_en == "LIGHTING":
                                lighting = val
                            elif label_en == "POSITIVE":
                                positive = val
                            elif label_en == "NEGATIVE":
                                negative = val
                            found_any = True

                if not found_any:
                    # 方法3：完全没匹配到任何标签，全文放入 raw_desc
                    raw_desc = content.strip()
                    content_preview = content[:200].replace("\n", "↵")

                    return (
                        f"⚠️ 未识别到标准标签，原始输出已填入原始描述框（预览: {content_preview}）",
                        overlaid, raw_desc, "", "", "", "", ""
                    )

            return "✅ 场景构图分析完成！", overlaid, raw_desc, composition, camera, lighting, positive, negative

        except ImportError:
            return "❌ 错误: 缺少必要模块", overlaid, "", "", "", "", "", ""
        except Exception as e:
            return f"❌ 分析失败: {str(e)}", overlaid, "", "", "", "", "", ""


def create_proportion_guide_ui():
    """创建人体比例与构图 UI（供美学提升插件调用）"""
    pg = ProportionGuideScript()

    # 共享后端选择器（两个 Tab 共用）
    gr.Markdown("### 🔌 后端服务（自动检测 llama.cpp / LM Studio / Ollama）")
    selector = build_backend_selector() if LLM_BACKEND_AVAILABLE else {}
    shared_backend = selector.get("backend_dropdown")
    shared_model = selector.get("model_dropdown")

    with gr.Tabs():
        # ============================================================
        # Tab 1: 人体头身比例分析
        # ============================================================
        with gr.Tab("🧍 人体头身比例"):
            # 流程1: AI 视觉分析
            gr.Markdown("### 🖼️ AI视觉分析（从参考图提取身材与风格）")

            with gr.Row():
                analysis_image = gr.Image(
                    label="上传参考图（角色全身图）",
                    type="pil",
                    height=300
                )
                analysis_grid_preview = gr.Image(
                    label="参考线叠加预览（将发送给视觉模型分析）",
                    interactive=False,
                    height=300
                )

            with gr.Row():
                analyze_status = gr.Textbox(
                    label="状态",
                    value="等待分析...",
                    interactive=False
                )
                analyze_btn = gr.Button("🔍 分析图片", variant="primary", scale=0, min_width=120)

            with gr.Row():
                analyzed_raw = gr.Textbox(
                    label="📝 原始图像描述（未过滤，可编辑）",
                    placeholder="等待分析结果...",
                    lines=4,
                    interactive=True
                )

            with gr.Row():
                analyzed_proportion = gr.Textbox(
                    label="📏 头身比/身材描述（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                analyzed_negative = gr.Textbox(
                    label="🚫 风格关键词 - 负向（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            gr.Markdown("---")
            gr.Markdown("### 🎨 设计元素分析（同图提取色彩、风格、排版与视觉构成）")

            with gr.Row():
                design_color = gr.Textbox(
                    label="🎨 色彩方案（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                design_category = gr.Textbox(
                    label="🏷️ 设计风格类别（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                design_typography = gr.Textbox(
                    label="📝 排版布局（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                design_hierarchy = gr.Textbox(
                    label="📐 视觉层次与构成（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            # ============================================================
        # Tab 2: 场景构图分析
        # ============================================================
        with gr.Tab("🎬 场景构图分析"):
            gr.Markdown("### 🎬 场景构图分析（从参考图提取构图与镜头语言）")

            with gr.Row():
                scene_analysis_image = gr.Image(
                    label="上传场景参考图",
                    type="pil",
                    height=300
                )
                scene_grid_preview = gr.Image(
                    label="构图参考线叠加预览（将发送给视觉模型分析）",
                    interactive=False,
                    height=300
                )

            with gr.Row():
                scene_analyze_status = gr.Textbox(
                    label="状态",
                    value="等待分析...",
                    interactive=False
                )
                scene_analyze_btn = gr.Button("🔍 分析场景构图", variant="primary", scale=0, min_width=140)

            with gr.Row():
                scene_raw_desc = gr.Textbox(
                    label="📝 场景原始描述（未过滤，可编辑）",
                    placeholder="等待分析结果...",
                    lines=4,
                    interactive=True
                )

            with gr.Row():
                scene_composition = gr.Textbox(
                    label="📐 构图方式（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                scene_camera = gr.Textbox(
                    label="📷 镜头角度与景别（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                scene_lighting = gr.Textbox(
                    label="💡 光影氛围（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                scene_positive = gr.Textbox(
                    label="🎨 场景正向Prompt（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                scene_negative = gr.Textbox(
                    label="🚫 场景负向Prompt（过滤后，可编辑）",
                    placeholder="等待分析结果...",
                    lines=2,
                    interactive=True
                )

            with gr.Row():
                gr.Markdown("""
                **使用说明：**
                上传场景参考图 → 点击「分析场景构图」→ 自动叠加三分法/对角线构图参考线 → llama.cpp视觉模型分析 → 自动填入场景描述、构图方式、镜头角度、光影氛围和正/负向Prompt（均可自由编辑）
                """)

        # ============================================================
        # Tab 3: 图片打光分析（从打光辅助模块合并而来）
        # ============================================================
        with gr.Tab("💡 图片打光分析"):
            gr.Markdown("### 🖼️ 上传图片，AI 自动分析打光方式并生成关键词")
            gr.Markdown("支持 **llama.cpp / LM Studio / Ollama** 三种后端，自动检测可用服务")

            with gr.Row():
                lighting_image = gr.Image(
                    label="📤 上传图片",
                    type="pil",
                    height=300,
                )
                lighting_result_preview = gr.Image(
                    label="预览",
                    interactive=False,
                    height=300,
                )

            with gr.Row():
                lighting_analyze_btn = gr.Button("🔍 分析打光", variant="primary", scale=0, min_width=140)
                lighting_status = gr.Textbox(label="状态", value="等待分析...", interactive=False)

            lighting_analysis_output = gr.Textbox(
                label="📝 打光分析结果",
                lines=12,
                show_copy_button=True,
            )
            lighting_keywords_output = gr.Textbox(
                label="🏷 提取的打光关键词",
                lines=3,
                show_copy_button=True,
            )

            def on_analyze_lighting(img, backend, model):
                if img is None:
                    return "请先上传图片", "", ""
                if not LIGHTING_ANALYSIS_AVAILABLE:
                    return "打光分析模块不可用", "", ""
                analysis, keywords, status = lighting_assistant_module.analyze_image_lighting(
                    image_path=img,
                    model=model,
                    backend=backend,
                )
                return status, analysis, keywords

            lighting_analyze_btn.click(
                fn=on_analyze_lighting,
                inputs=[lighting_image, shared_backend, shared_model],
                outputs=[lighting_status, lighting_analysis_output, lighting_keywords_output],
            )

    # 分析按钮事件（人体比例 + 设计元素）
    analyze_btn.click(
        fn=pg.analyze_reference_image,
        inputs=[analysis_image, shared_backend, shared_model],
        outputs=[analyze_status, analysis_grid_preview, analyzed_raw,
                 analyzed_proportion, analyzed_negative,
                 design_color, design_category, design_typography, design_hierarchy]
    )

    # 场景构图分析按钮事件
    scene_analyze_btn.click(
        fn=pg.analyze_scene_image,
        inputs=[scene_analysis_image, shared_backend, shared_model],
        outputs=[scene_analyze_status, scene_grid_preview, scene_raw_desc,
                 scene_composition, scene_camera, scene_lighting,
                 scene_positive, scene_negative]
    )
