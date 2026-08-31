"""
SD Forge Image Stitch API
为 PS 插件提供 REST API 端点，支持多图拼接参考功能
通过 script_callbacks.on_app_started 注册到 FastAPI
"""

import base64
import io

import gradio as gr
import numpy as np
import torch
from fastapi import Body, FastAPI
from PIL import Image

from modules import images, script_callbacks, sd_models, shared
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
    process_images,
)
from modules.sd_samplers_common import images_tensor_to_samples
from modules.shared import device, opts, sd_model


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """将 base64 字符串解码为 PIL Image"""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def closesteight(num: int) -> int:
    """调整为 8 的倍数"""
    rem = num % 8
    if rem <= 4:
        return round(num - rem)
    return round(num + (8 - rem))


def preprocess_reference(img: Image.Image, limit: int) -> Image.Image:
    """预处理参考图：缩放 + 对齐到 64 的倍数"""
    w, h = img.size
    if limit > 0 and max(w, h) > limit:
        ratio = limit / max(w, h)
        _w, _h = int(w * ratio), int(h * ratio)
    else:
        _w, _h = w, h
    if _w % 64 != 0 or _h % 64 != 0:
        _w = round(_w / 64) * 64
        _h = round(_h / 64) * 64
    if w != _w or h != _h:
        return images.resize_image(1, img, _w, _h)
    return img


def encode_reference_images(references: list[Image.Image], max_dim: int, p) -> None:
    """
    将参考图编码为潜空间，注入到模型中
    与 ImageStitch.process 逻辑一致
    """
    from modules.sd_models import FakeInitialModel
    from backend.args import dynamic_args

    if p.sd_model is None or isinstance(p.sd_model, FakeInitialModel):
        raise ValueError("模型未加载，请先在 WebUI 中选择并加载模型后再使用多图拼接参考功能")

    # 清除之前的参考
    p.clear_prompt_cache()
    p.sd_model.clear_references()

    # 设置参考模式标志，让模型（如 Krea2/Klein）的 encode_first_stage 知道要存储参考数据
    dynamic_args.is_referencing = True

    for ref in references:
        ref = preprocess_reference(ref, max_dim)
        image = images.flatten(ref, opts.img2img_background_color)
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image).to(device=device).unsqueeze(0)
        images_tensor_to_samples(image, 0, p.sd_model)

    dynamic_args.is_referencing = False


def image_stitch_api(_: gr.Blocks, app: FastAPI):
    """注册 image_stitch API 端点"""

    @app.post("/sdapi/v1/image-stitch/txt2img")
    async def image_stitch_txt2img(body: dict = Body(None)):
        """
        文生图 + 多图拼接参考
        - reference_images: base64 编码的参考图列表
        - reference_max_dim: 参考图最大边长限制
        """
        prompt = body.get("prompt", "")
        negative_prompt = body.get("negative_prompt", "")
        steps = int(body.get("steps", 20))
        cfg_scale = float(body.get("cfg_scale", 7.0))
        width = int(body.get("width", 512))
        height = int(body.get("height", 512))
        batch_size = int(body.get("batch_size", 1))
        n_iter = int(body.get("n_iter", 1))
        seed = int(body.get("seed", -1))
        sampler_index = body.get("sampler_index", "Euler")
        sd_model_checkpoint = body.get("sd_model_checkpoint", None)
        sd_vae = body.get("sd_vae", None)
        reference_images = body.get("reference_images", [])
        reference_max_dim = int(body.get("reference_max_dim", 1024))

        if not isinstance(reference_images, list):
            reference_images = []

        # 解码参考图（先解码，不依赖模型）
        ref_images = []
        for b64 in reference_images:
            try:
                img = decode_base64_to_image(b64)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                ref_images.append(img)
            except Exception:
                continue

        # 切换模型（在 encode_reference_images 之前确保模型已加载）
        if sd_model_checkpoint:
            try:
                from modules import sd_models as _sd_models
                from modules_forge import main_entry
                # 设置模型选项并刷新加载参数
                main_entry.checkpoint_change(sd_model_checkpoint, preset=None, save=False, refresh=True)
                # 立即加载模型（forge_model_reload 会同步 shared.sd_model）
                _sd_models.forge_model_reload()
            except Exception as e:
                print(f"[Image Stitch] 切换模型失败: {e}")

        # 设置 VAE（必须在模型加载后执行）
        if sd_vae:
            try:
                from modules import sd_vae as _sd_vae
                # 解析 VAE 名称为完整路径
                vae_path = _sd_vae.vae_dict.get(sd_vae, sd_vae)
                _sd_vae.reload_vae_weights(vae_path)
            except Exception as e:
                print(f"[Image Stitch] 设置 VAE 失败: {e}")

        # 创建处理对象（dataclass，参数必须匹配字段名）
        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            batch_size=batch_size,
            n_iter=n_iter,
            seed=seed,
            sampler_name=sampler_index,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        # 编码参考图
        if ref_images:
            encode_reference_images(ref_images, reference_max_dim, p)

        # 执行生成
        processed = process_images(p)
        images_out = []
        for img in processed.images:
            if isinstance(img, Image.Image):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                images_out.append(b64)

        return {
            "images": images_out,
            "parameters": processed.js(),
            "info": processed.infotexts[0] if processed.infotexts else "",
        }

    @app.post("/sdapi/v1/image-stitch/img2img")
    async def image_stitch_img2img(body: dict = Body(None)):
        """
        图生图 + 多图拼接参考
        - init_image: base64 编码的初始图
        - reference_images: base64 编码的额外参考图列表
        - reference_max_dim: 参考图最大边长限制
        """
        init_image = body.get("init_image", "")
        prompt = body.get("prompt", "")
        negative_prompt = body.get("negative_prompt", "")
        steps = int(body.get("steps", 20))
        cfg_scale = float(body.get("cfg_scale", 7.0))
        denoising_strength = float(body.get("denoising_strength", 0.75))
        width = int(body.get("width", 512))
        height = int(body.get("height", 512))
        batch_size = int(body.get("batch_size", 1))
        n_iter = int(body.get("n_iter", 1))
        seed = int(body.get("seed", -1))
        sampler_index = body.get("sampler_index", "Euler")
        sd_model_checkpoint = body.get("sd_model_checkpoint", None)
        sd_vae = body.get("sd_vae", None)
        reference_images = body.get("reference_images", [])
        reference_max_dim = int(body.get("reference_max_dim", 1024))

        if not isinstance(reference_images, list):
            reference_images = []

        # 解码初始图（先解码，不依赖模型）
        init_img = decode_base64_to_image(init_image) if init_image else None
        if init_img and init_img.mode == "RGBA":
            init_img = init_img.convert("RGB")

        # 解码参考图（先解码，不依赖模型）
        ref_images = []
        for b64 in reference_images:
            try:
                img = decode_base64_to_image(b64)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                ref_images.append(img)
            except Exception:
                continue

        # 切换模型（在 encode_reference_images 之前确保模型已加载）
        if sd_model_checkpoint:
            try:
                from modules import sd_models as _sd_models
                from modules_forge import main_entry
                # 设置模型选项并刷新加载参数
                main_entry.checkpoint_change(sd_model_checkpoint, preset=None, save=False, refresh=True)
                # 立即加载模型（forge_model_reload 会同步 shared.sd_model）
                _sd_models.forge_model_reload()
            except Exception as e:
                print(f"[Image Stitch] 切换模型失败: {e}")

        # 设置 VAE（必须在模型加载后执行）
        if sd_vae:
            try:
                from modules import sd_vae as _sd_vae
                # 解析 VAE 名称为完整路径
                vae_path = _sd_vae.vae_dict.get(sd_vae, sd_vae)
                _sd_vae.reload_vae_weights(vae_path)
            except Exception as e:
                print(f"[Image Stitch] 设置 VAE 失败: {e}")

        # 创建处理对象（dataclass，参数必须匹配字段名）
        p = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            denoising_strength=denoising_strength,
            width=width,
            height=height,
            batch_size=batch_size,
            n_iter=n_iter,
            seed=seed,
            sampler_name=sampler_index,
            init_images=[init_img] if init_img else [],
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        # 编码参考图
        if ref_images:
            encode_reference_images(ref_images, reference_max_dim, p)

        # 执行生成
        processed = process_images(p)
        images_out = []
        for img in processed.images:
            if isinstance(img, Image.Image):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                images_out.append(b64)

        return {
            "images": images_out,
            "parameters": processed.js(),
            "info": processed.infotexts[0] if processed.infotexts else "",
        }


script_callbacks.on_app_started(image_stitch_api)