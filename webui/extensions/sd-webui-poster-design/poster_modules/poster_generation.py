"""
海报生成引擎 - 集成 webui 内置的模型
参考 CreatiDesign 的多条件生成管线设计
"""

from __future__ import annotations
import os
from PIL import Image
from typing import Optional, List, Dict

# webui 内部模块
from modules import shared, script_callbacks
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
import modules.images as sd_images

# 同目录模块导入（参考 see-through-sam 的导入方式）
from poster_layout_engine import (
    PosterLayout, TextElement, ImageElement, SubjectElement, render_layout_preview,
    POSTER_TEMPLATES,
)
from poster_image_utils import (
    prepare_condition_image, create_layout_control_image, create_composite_preview,
)


class PosterGenerator:
    """
    海报生成器
    使用 webui 内置的模型，通过 IP-Adapter + ControlNet 实现布局控制
    """

    def __init__(self):
        self.current_layout: Optional[PosterLayout] = None

    def generate(
        self,
        layout: PosterLayout,
        subject_image: Optional[Image.Image] = None,
        element_images: Optional[Dict[str, Image.Image]] = None,
        use_ip_adapter: bool = True,
        use_controlnet: bool = True,
        cfg_scale: float = 3.5,
        denoising_strength: float = 1.0,
        steps: int = 28,
        seed: int = -1,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        """
        生成海报

        参考 CreatiDesign 的管线设计：
        1. 准备条件图像（subject image）
        2. 准备布局控制（layout boxes）
        3. 通过 IP-Adapter 注入主体
        4. 通过 ControlNet 引导布局
        5. 使用当前模型生成
        """
        self.current_layout = layout

        # 构建提示词
        prompt_parts = [layout.global_prompt]
        if layout.background_prompt:
            prompt_parts.append(layout.background_prompt)

        for elem in layout.elements:
            if isinstance(elem, TextElement) and elem.text:
                prompt_parts.append(f"包含文字 '{elem.text}'")
            elif isinstance(elem, SubjectElement) and elem.subject_prompt:
                prompt_parts.append(elem.subject_prompt)

        full_prompt = ", ".join(prompt_parts)
        negative_prompt = layout.negative_prompt

        # 构建 webui 处理参数
        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=int(layout.width),
            height=int(layout.height),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            denoising_strength=float(denoising_strength),
            seed=int(seed) if seed is not None and seed >= 0 else -1,
            batch_size=int(batch_size),
            subseed=-1,
            subseed_strength=0,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            sampler_name="Euler",
            scheduler="Simple",
        )

        # IP-Adapter 注入主体
        if use_ip_adapter and subject_image is not None:
            cond_img = prepare_condition_image(subject_image, (int(layout.width), int(layout.height)))
            try:
                setattr(p, "ip_adapter_image", cond_img)
            except Exception:
                pass

        # ControlNet 布局引导
        if use_controlnet:
            try:
                control_img = create_layout_control_image(
                    layout.elements, (int(layout.width), int(layout.height)),
                )
                setattr(p, "controlnet_images", [control_img])
            except Exception:
                pass

        # 执行生成
        try:
            processed = process_images(p)
        except Exception as e:
            err_img = Image.new("RGB", (int(layout.width), int(layout.height)), "white")
            from PIL import ImageDraw
            d = ImageDraw.Draw(err_img)
            d.text((50, 50), f"生成失败: {e}", fill="red")
            return [err_img]

        images = [r for r in processed.images if isinstance(r, Image.Image)]
        return images

    def generate_with_masks(
        self,
        layout: PosterLayout,
        subject_image: Optional[Image.Image] = None,
        element_images: Optional[Dict[str, Image.Image]] = None,
        use_ip_adapter: bool = True,
        use_controlnet: bool = True,
        cfg_scale: float = 3.5,
        steps: int = 28,
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        带布局描述的海报生成
        参考 CreatiDesign 的 attention_mask 机制，通过提示词精确描述每个元素的位置
        """
        prompt = layout.global_prompt
        if layout.background_prompt:
            prompt += "。" + layout.background_prompt

        layout_desc = []
        for elem in layout.elements:
            x_pct = int(elem.x * 100)
            y_pct = int(elem.y * 100)
            w_pct = int(elem.width * 100)
            h_pct = int(elem.height * 100)

            if isinstance(elem, TextElement) and elem.text:
                layout_desc.append(
                    f"在位置({x_pct}%,{y_pct}%)放置文字'{elem.text}'，大小{w_pct}%×{h_pct}%"
                )
            elif isinstance(elem, SubjectElement) and elem.subject_prompt:
                layout_desc.append(
                    f"在位置({x_pct}%,{y_pct}%)放置{elem.subject_prompt}，占{w_pct}%×{h_pct}%"
                )
            elif isinstance(elem, ImageElement):
                layout_desc.append(
                    f"在位置({x_pct}%,{y_pct}%)放置装饰元素，占{w_pct}%×{h_pct}%"
                )

        if layout_desc:
            prompt += "。布局要求：" + "；".join(layout_desc)

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=layout.negative_prompt,
            width=int(layout.width),
            height=int(layout.height),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            denoising_strength=1.0,
            seed=int(seed) if seed is not None and seed >= 0 else -1,
            batch_size=1,
            subseed=-1,
            subseed_strength=0,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            sampler_name="Euler",
            scheduler="Simple",
        )

        if use_ip_adapter and subject_image is not None:
            cond_img = prepare_condition_image(subject_image, (int(layout.width), int(layout.height)))
            try:
                setattr(p, "ip_adapter_image", cond_img)
            except Exception:
                pass

        if use_controlnet:
            try:
                control_img = create_layout_control_image(
                    layout.elements, (int(layout.width), int(layout.height)),
                )
                setattr(p, "controlnet_images", [control_img])
            except Exception:
                pass

        try:
            processed = process_images(p)
        except Exception as e:
            err_img = Image.new("RGB", (int(layout.width), int(layout.height)), "white")
            from PIL import ImageDraw
            d = ImageDraw.Draw(err_img)
            d.text((50, 50), f"生成失败: {e}", fill="red")
            return [err_img]

        images = [r for r in processed.images if isinstance(r, Image.Image)]
        return images
