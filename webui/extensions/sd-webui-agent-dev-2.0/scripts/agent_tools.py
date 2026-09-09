# =============================================================================
# Agent Tools — 所有 WebUI 工具函数 + Function Calling 定义
# =============================================================================

import os
import sys
import json
import time
import base64
import traceback
import tempfile
import csv
import zipfile
import subprocess
import re
import html
import urllib.request
import urllib.error
import urllib.parse
from xml.etree import ElementTree
from pathlib import Path
from io import BytesIO

from PIL import Image

from modules import shared, scripts, sd_models, sd_samplers, postprocessing
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    process_images,
)
from modules.shared import device, opts
from modules.images import flatten
from modules.sd_samplers_common import images_tensor_to_samples
from backend.args import dynamic_args
import numpy as np
import torch

# 从配置模块导入共享函数
from scripts.agent_config import (
    load_config, _save_pil_to_tempfile, _get_current_checkpoint, _get_sampler, _scan_model_dir,
    _REGISTRY_AVAILABLE, get_registered_tools, get_tool_function, list_registered_tools
)


def txt2img_tool(prompt, negative_prompt="", steps=None, width=None, height=None,
                 sampler_name=None, cfg_scale=None, seed=-1, batch_size=1, n_iter=1):
    """文生图：根据文字描述生成图片。"""
    cfg = load_config()

    # 如果 forge_preset 已激活，覆盖 LLM 传入的参数为预设值
    try:
        from modules_forge.presets import STEPS, SAMPLERS, SCHEDULERS, CFG, PresetArch
        current_preset = getattr(shared.opts, "forge_preset", "")
        if current_preset and current_preset.lower() != "sd":
            arch = PresetArch[current_preset]
            steps = STEPS.get(arch) or steps
            cfg_scale = CFG.get(arch) if CFG.get(arch) is not None else cfg_scale
            sampler_name = SAMPLERS.get(arch) or sampler_name
    except Exception:
        pass

    p = StableDiffusionProcessingTxt2Img(
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps or cfg["default_steps"],
        cfg_scale=cfg_scale or cfg["default_cfg_scale"],
        width=width or cfg["default_width"],
        height=height or cfg["default_height"],
        batch_size=batch_size,
        n_iter=n_iter,
        sampler_name=sampler_name or _get_sampler(),
        do_not_save_samples=False,
        do_not_save_grid=True,
    )
    p.sd_model = shared.sd_model
    print(f"[Agent] txt2img: '{prompt[:60]}...' {p.width}x{p.height} steps={p.steps}")
    # 通过 Forge main_thread 调度，确保线程安全
    try:
        from modules_forge import main_thread
        processed = main_thread.run_and_wait_result(process_images, p)
    except Exception:
        # 回退到直接调用
        processed = process_images(p)
    if processed is None or not hasattr(processed, 'images'):
        return None, {"status": "error", "error": "模型加载或生图失败，process_images 返回 None",
                      "hint": "可能是主模型与文本编码器/VAE 不匹配，请检查模型组合是否正确"}
    images = [img for img in processed.images]
    info = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "steps": p.steps, "width": p.width, "height": p.height,
        "sampler": p.sampler_name, "seed": p.seed, "model": _get_current_checkpoint(),
        "batch_size": batch_size, "n_iter": n_iter,
    }
    return images, info


def img2img_tool(prompt, image, negative_prompt="", denoising_strength=0.75,
                 steps=None, width=None, height=None, sampler_name=None,
                 cfg_scale=None, seed=-1):
    """图生图：基于参考图片和文字描述生成新图片。"""
    cfg = load_config()

    # 如果 forge_preset 已激活，覆盖 LLM 传入的参数为预设值
    try:
        from modules_forge.presets import STEPS, SAMPLERS, SCHEDULERS, CFG, PresetArch
        current_preset = getattr(shared.opts, "forge_preset", "")
        if current_preset and current_preset.lower() != "sd":
            arch = PresetArch[current_preset]
            steps = STEPS.get(arch) or steps
            cfg_scale = CFG.get(arch) if CFG.get(arch) is not None else cfg_scale
            sampler_name = SAMPLERS.get(arch) or sampler_name
    except Exception:
        pass

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image or file path")

    target_w = width or cfg["default_width"]
    target_h = height or cfg["default_height"]
    if image.width != target_w or image.height != target_h:
        image = image.resize((target_w, target_h), Image.LANCZOS)

    p = StableDiffusionProcessingImg2Img(
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
        prompt=prompt, negative_prompt=negative_prompt, seed=seed,
        steps=steps or cfg["default_steps"],
        cfg_scale=cfg_scale or cfg["default_cfg_scale"],
        width=target_w, height=target_h,
        init_images=[image], denoising_strength=denoising_strength,
        sampler_name=sampler_name or _get_sampler(),
        do_not_save_samples=False, do_not_save_grid=True,
    )
    p.sd_model = shared.sd_model
    print(f"[Agent] img2img: '{prompt[:60]}...' denoise={denoising_strength}")
    # 通过 Forge main_thread 调度，确保线程安全
    try:
        from modules_forge import main_thread
        processed = main_thread.run_and_wait_result(process_images, p)
    except Exception:
        processed = process_images(p)
    if processed is None or not hasattr(processed, 'images'):
        return None, {"status": "error", "error": "模型加载或生图失败，process_images 返回 None",
                      "hint": "可能是主模型与文本编码器/VAE 不匹配，请检查模型组合是否正确"}
    images = [img for img in processed.images]
    info = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": p.steps, "width": p.width, "height": p.height,
        "sampler": p.sampler_name, "seed": p.seed, "model": _get_current_checkpoint(),
    }
    return images, info


def _edit_with_image_stitch(image, instruction, cfg=None):
    """使用 Image Stitch 插件 + txt2img 进行编辑（Klein 等编辑模型专用）。

    无需通过 UI 组件，直接编码参考图到模型潜空间，然后调用 txt2img 生成。
    """
    try:
        from modules.sd_models import FakeInitialModel
        from modules_forge import main_thread
        from modules.images import flatten as _flatten
        from modules.sd_samplers_common import images_tensor_to_samples as _encode
        from backend.args import dynamic_args as _dynargs

        if cfg is None:
            cfg = load_config()

        # 检查当前模型是否支持 image_stitch（Klein 等）
        is_edit_model = any(
            getattr(_dynargs, key, False)
            for key in ("kontext", "edit", "klein", "wan", "krea2")
        )
        if not is_edit_model:
            # 不支持 image_stitch，回退到常规 img2img
            return img2img_tool(image, instruction, cfg_scale=cfg.get("default_cfg_scale"))

        # 检查模型是否已加载（FakeInitialModel 表示未加载）
        if isinstance(shared.sd_model, FakeInitialModel):
            try:
                from modules.sd_models import forge_model_reload
                from modules_forge import main_entry
                checkpoint = getattr(shared.opts, 'sd_model_checkpoint', '')
                if checkpoint:
                    main_entry.checkpoint_change(checkpoint, preset=None, save=False, refresh=True)
                forge_model_reload()
            except Exception as e:
                print(f"[Agent] 尝试加载模型失败: {e}")
                return img2img_tool(image, instruction, cfg_scale=cfg.get("default_cfg_scale"))
            # 如果加载后仍然是 FakeInitialModel，回退
            if isinstance(shared.sd_model, FakeInitialModel):
                return img2img_tool(image, instruction, cfg_scale=cfg.get("default_cfg_scale"))

        # 加载参考图
        if isinstance(image, str):
            ref_img = Image.open(image).convert("RGB")
        else:
            ref_img = image.convert("RGB")

        # 读取当前预设参数
        current_preset = getattr(shared.opts, "forge_preset", "")
        try:
            from modules_forge.presets import STEPS, SAMPLERS, SCHEDULERS, CFG, PresetArch
            if current_preset and current_preset.lower() != "sd":
                arch = PresetArch[current_preset]
                preset_steps = STEPS.get(arch) or cfg.get("default_steps", 20)
                preset_cfg = CFG.get(arch) if CFG.get(arch) is not None else cfg.get("default_cfg_scale", 7)
                preset_sampler = SAMPLERS.get(arch) or _get_sampler()
            else:
                raise ValueError("no preset")
        except Exception:
            preset_steps = cfg.get("default_steps", 20)
            preset_cfg = cfg.get("default_cfg_scale", 7)
            preset_sampler = _get_sampler()

        # 创建 txt2img 处理对象 — 保持原图比例
        def _closesteight(n):
            rem = n % 8
            return n - rem if rem <= 4 else n + (8 - rem)

        out_w = _closesteight(ref_img.width)
        out_h = _closesteight(ref_img.height)
        # 限制最大尺寸，避免显存溢出
        max_dim = 2048
        if max(out_w, out_h) > max_dim:
            ratio = max_dim / max(out_w, out_h)
            out_w = _closesteight(int(out_w * ratio))
            out_h = _closesteight(int(out_h * ratio))

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=instruction,
            negative_prompt="",
            steps=preset_steps,
            cfg_scale=preset_cfg,
            width=out_w,
            height=out_h,
            sampler_name=preset_sampler,
            do_not_save_samples=False,
            do_not_save_grid=True,
            outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "",
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids or "",
        )

        # 编码参考图到模型潜空间（同 Image Stitch 插件 _process_single_task 一致）
        p.clear_prompt_cache()
        p.sd_model.clear_references()
        _dynargs.is_referencing = True

        # 预处理参考图（同 ImageStitch.preprocess 逻辑）
        w, h = ref_img.size
        limit = 1024
        if limit > 0 and max(w, h) > limit:
            ratio = limit / max(w, h)
            _w, _h = int(w * ratio), int(h * ratio)
        else:
            _w, _h = w, h
        if _w % 64 != 0 or _h % 64 != 0:
            _w = round(_w / 64) * 64
            _h = round(_h / 64) * 64
        if w != _w or h != _h:
            from modules.images import resize_image as _resize
            ref_img = _resize(1, ref_img, _w, _h)

        flat = _flatten(ref_img, opts.img2img_background_color)
        arr = np.array(flat, dtype=np.float32) / 255.0
        arr = np.moveaxis(arr, 2, 0)
        t = torch.from_numpy(arr).to(device=device).unsqueeze(0)
        _encode(t, 0, p.sd_model)

        _dynargs.is_referencing = False

        # 执行生成
        try:
            processed = main_thread.run_and_wait_result(process_images, p)
        except Exception:
            processed = process_images(p)

        if processed is None or not hasattr(processed, 'images'):
            return None, {"status": "error", "error": "编辑生成失败"}

        images = [img for img in processed.images if isinstance(img, Image.Image)]
        info = {
            "prompt": instruction,
            "steps": p.steps,
            "width": p.width,
            "height": p.height,
            "sampler": p.sampler_name,
            "seed": p.seed,
            "model": _get_current_checkpoint(),
            "method": "image_stitch_txt2img",
        }
        return images, info
    except Exception as e:
        print(f"[Agent] Image Stitch 编辑失败: {e}")
        traceback.print_exc()
        # 回退到 img2img
        return img2img_tool(image, instruction, cfg_scale=cfg.get("default_cfg_scale") if cfg else None)


def switch_model_tool(model_name):
    """切换 Stable Diffusion 模型 (checkpoint)。

    参数:
        model_name: 模型文件名（支持子目录路径），如
            'krea2_turbo_int8_convrot.safetensors' 或 'klein/Flux2-Klein-9B-True-V3-fp8mixed.safetensors'
    """
    try:
        # 刷新并获取模型列表
        model_filenames = []
        try:
            sd_models.list_models()
            if hasattr(sd_models, "checkpoints_list") and sd_models.checkpoints_list:
                model_filenames = [info.filename for info in sd_models.checkpoints_list.values()]
        except Exception:
            pass
        # 回退：直接扫描文件系统
        if not model_filenames:
            model_filenames = _scan_model_dir("Stable-diffusion")

        # 精确匹配优先，然后模糊匹配
        matched = None
        # 1. 精确匹配（忽略扩展名大小写）
        for fn in model_filenames:
            if fn.lower() == model_name.lower():
                matched = fn
                break
        # 2. 文件名部分匹配
        if matched is None:
            for fn in model_filenames:
                base = os.path.basename(fn)
                if model_name.lower() in fn.lower() or model_name.lower() in base.lower():
                    matched = fn
                    break

        if matched is None:
            return {
                "status": "error",
                "error": f"未找到模型 '{model_name}'",
                "available_models": model_filenames[:30],
                "hint": "请使用 list_models 工具查看准确的模型文件名"
            }

        # 使用 Forge 官方 checkpoint_change API（会自动刷新加载参数）
        try:
            from modules_forge.main_entry import checkpoint_change
            changed = checkpoint_change(matched, preset=None, save=True, refresh=True)
        except Exception:
            # 回退：直接设置 opts
            shared.opts.sd_model_checkpoint = matched
            try:
                from modules_forge.main_entry import refresh_model_loading_parameters
                refresh_model_loading_parameters()
            except Exception:
                pass

        # 获取当前生效的 TE/VAE
        effective_modules = [os.path.basename(m) for m in (getattr(shared.opts, "forge_additional_modules", []) or [])]
        print(f"[Agent] 切换模型: {matched} | 附加模块: {effective_modules}")
        return {
            "status": "success",
            "message": f"已切换模型为: {matched}",
            "model": matched,
            "effective_modules": effective_modules,
            "tip": "模型已切换，下次生图时自动加载（含已设置的 TE/VAE），无需重启"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def update_settings_tool(steps=None, cfg_scale=None, sampler_name=None,
                         width=None, height=None, batch_size=None):
    """修改 WebUI 的生图设置。所有参数都是可选的。

    参数:
        steps: 采样步数
        cfg_scale: CFG 引导强度
        sampler_name: 采样器名称 (如 'Euler a', 'DPM++ 2M Karras')
        width: 图片宽度
        height: 图片高度
        batch_size: 批量大小
    """
    updated = {}
    try:
        if steps is not None:
            shared.opts.steps = int(steps)
            updated["steps"] = shared.opts.steps
        if cfg_scale is not None:
            shared.opts.cfg_scale = float(cfg_scale)
            updated["cfg_scale"] = shared.opts.cfg_scale
        if sampler_name is not None:
            shared.opts.sampler_name = sampler_name
            updated["sampler_name"] = shared.opts.sampler_name
        if width is not None:
            shared.opts.width = int(width)
            updated["width"] = shared.opts.width
        if height is not None:
            shared.opts.height = int(height)
            updated["height"] = shared.opts.height
        if batch_size is not None:
            shared.opts.batch_size = int(batch_size)
            updated["batch_size"] = shared.opts.batch_size

        print(f"[Agent] 更新设置: {updated}")
        return {"status": "success", "updated": updated,
                "current_settings": get_current_settings_tool()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def upscale_tool(image, upscaler_name=None, scale=2, resize_w=None, resize_h=None):
    """图片放大。

    参数:
        image: 要放大的图片 (PIL Image)
        upscaler_name: 放大模型名称，如 '4x-UltraSharp', 'R-ESRGAN 4x+', 'ESRGAN_4x' 等。未指定则优先用 4x-UltraSharp。
        scale: 放大倍数 (默认 2)
        resize_w: 指定宽度 (可选，覆盖 scale)
        resize_h: 指定高度 (可选，覆盖 scale)
    """
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image")

        # 获取可用放大模型
        upscalers = getattr(shared, "sd_upscalers", []) or []
        upscaler_names = [u.name for u in upscalers if hasattr(u, "name")]

        if not upscaler_names:
            # 回退到 Lanczos
            if resize_w and resize_h:
                new_w, new_h = int(resize_w), int(resize_h)
            else:
                new_w, new_h = int(image.width * scale), int(image.height * scale)
            upscaled = image.resize((new_w, new_h), Image.LANCZOS)
            return [upscaled], {"upscaler": "Lanczos (fallback)", "scale": scale,
                                "original_size": f"{image.width}x{image.height}",
                                "new_size": f"{new_w}x{new_h}"}

        # 选择放大模型：优先用户指定，其次 4x-UltraSharp，再退到第一个真实放大器。
        selected = None
        lowered_names = [(un, un.lower()) for un in upscaler_names]
        if upscaler_name and str(upscaler_name).strip().lower() not in ("none", "null"):
            requested = str(upscaler_name).strip().lower()
            for un, lower in lowered_names:
                if requested in lower:
                    selected = un
                    break

        if not selected:
            preferred_terms = ("4x-ultrasharp", "ultrasharp")
            for term in preferred_terms:
                for un, lower in lowered_names:
                    if term in lower and lower != "none":
                        selected = un
                        break
                if selected:
                    break

        if not selected:
            selected = next((un for un in upscaler_names if un and un.lower() != "none"), None)

        if not selected:
            if resize_w and resize_h:
                new_w, new_h = int(resize_w), int(resize_h)
            else:
                new_w, new_h = int(image.width * scale), int(image.height * scale)
            upscaled = image.resize((new_w, new_h), Image.LANCZOS)
            return [upscaled], {"upscaler": "Lanczos (fallback)", "scale": scale,
                                "original_size": f"{image.width}x{image.height}",
                                "new_size": f"{new_w}x{new_h}",
                                "available_upscalers": upscaler_names}

        # 调用 run_extras
        resize_mode = 0  # 0 = scale by factor, 1 = resize to W/H
        if resize_w and resize_h:
            resize_mode = 1

        result = postprocessing.run_extras(
            extras_mode=0,  # 0 = upscale
            resize_mode=resize_mode,
            image=image,
            image_folder="",
            input_dir="",
            output_dir="",
            show_extras_results=False,
            gfpgan_visibility=0,
            codeformer_visibility=0,
            codeformer_weight=0,
            upscaling_resize=scale,
            upscaling_resize_w=resize_w or 512,
            upscaling_resize_h=resize_h or 512,
            upscaling_crop=False,
            extras_upscaler_1=selected,
            extras_upscaler_2="None",
            extras_upscaler_2_visibility=0,
            upscale_first=False,
            save_output=False,
            max_side_length=0,
        )

        images = result.images if hasattr(result, "images") else [image]
        if images and not resize_w and not resize_h and (images[0].width, images[0].height) == (image.width, image.height):
            new_w, new_h = int(image.width * scale), int(image.height * scale)
            images = [images[0].resize((new_w, new_h), Image.LANCZOS)]
            selected = f"{selected} + Lanczos size fallback"
        info = {
            "upscaler": selected,
            "scale": scale,
            "original_size": f"{image.width}x{image.height}",
            "new_size": f"{images[0].width}x{images[0].height}" if images else "unknown",
            "available_upscalers": upscaler_names,
        }
        return images, info
    except Exception as e:
        print(f"[Agent] upscale error: {traceback.format_exc()}")
        # 回退到简单 resize
        if resize_w and resize_h:
            new_w, new_h = int(resize_w), int(resize_h)
        else:
            new_w, new_h = int(image.width * scale), int(image.height * scale)
        upscaled = image.resize((new_w, new_h), Image.LANCZOS)
        return [upscaled], {"upscaler": "Lanczos (fallback)", "scale": scale,
                            "note": f"upscaler failed: {str(e)}, used Lanczos instead"}


def video_keyframe_extract_tool(video_path, num_frames=5, method="even"):
    """从视频中提取关键帧。

    参数:
        video_path: 视频文件路径
        num_frames: 提取帧数 (默认5)
        method: 提取方式 - 'even' 均匀采样, 'first' 首帧, 'last' 尾帧, 'middle' 中间帧
    返回: (images_list, info_dict)
    """
    try:
        import cv2
        import numpy as np

        if not os.path.isfile(video_path):
            return [], {"error": f"视频文件不存在: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {"error": f"无法打开视频: {video_path}"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # 计算要提取的帧索引
        if method == "first":
            indices = [0]
        elif method == "last":
            indices = [max(0, total_frames - 1)]
        elif method == "middle":
            indices = [total_frames // 2]
        else:  # even
            if num_frames <= 1:
                indices = [total_frames // 2]
            else:
                step = total_frames // num_frames
                indices = [i * step for i in range(num_frames)]

        images = []
        extracted_info = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                images.append(img)
                timestamp = idx / fps if fps > 0 else 0
                extracted_info.append({
                    "frame_index": idx,
                    "timestamp": f"{timestamp:.2f}s",
                    "width": width,
                    "height": height,
                })

        cap.release()

        info = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration": f"{duration:.2f}s",
            "resolution": f"{width}x{height}",
            "extracted": len(images),
            "method": method,
            "frames": extracted_info,
        }
        print(f"[Agent] 视频关键帧提取: {video_path} -> {len(images)} 帧")
        return images, info

    except Exception as e:
        print(f"[Agent] video_keyframe_extract error: {traceback.format_exc()}")
        return [], {"error": str(e)}


def trellis2_image_to_3d_tool(image, seed=0, resolution="1024"):
    """调用已安装的 TRELLIS.2 本地扩展，将参考图生成为 GLB。"""
    try:
        if image is None:
            return None, {"status": "error", "error": "请先上传图片"}
        if isinstance(image, dict):
            image = image.get("path")
        input_image = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        import importlib.util
        script_path = os.path.join(scripts.basedir(), "extensions", "sd-webui-trellis2", "scripts", "trellis2_script.py")
        if not os.path.isfile(script_path):
            return None, {"status": "error", "error": "未找到 TRELLIS.2 本地扩展"}
        spec = importlib.util.spec_from_file_location("sd_webui_trellis2_agent", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        status, glb_path, preview = module.generate_3d_from_image(
            input_image, int(seed or 0), False, 7.5, 12, str(resolution),
            0.95, 1024, "无", False, progress=lambda *args, **kwargs: None,
        )
        if not glb_path:
            return None, {"status": "error", "error": status}
        return preview, {"status": "success", "tool": "trellis2_image_to_3d", "glb_path": glb_path, "message": status}
    except Exception as e:
        print(f"[Agent] TRELLIS.2 本地图生3D失败: {traceback.format_exc()}")
        return None, {"status": "error", "error": str(e)}


def video_to_frames_tool(video_path, interval_seconds=1, max_frames=20):
    """从视频中按时间间隔提取帧（每秒/每N秒一帧）。

    参数:
        video_path: 视频文件路径
        interval_seconds: 提取间隔（秒），默认1秒
        max_frames: 最大提取帧数，默认20
    返回: (images_list, info_dict)
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {"error": f"无法打开视频: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps * interval_seconds))

        images = []
        frame_idx = 0
        extracted = 0

        while frame_idx < total_frames and extracted < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb))
                extracted += 1
            frame_idx += frame_interval

        cap.release()
        info = {
            "video_path": video_path,
            "fps": fps,
            "interval_seconds": interval_seconds,
            "extracted": len(images),
            "max_frames": max_frames,
        }
        print(f"[Agent] 视频帧提取: {video_path} -> {len(images)} 帧 (间隔{interval_seconds}s)")
        return images, info
    except Exception as e:
        return [], {"error": str(e)}


def stitch_images_tool(images, columns=2, padding=10, background_color=(255, 255, 255)):
    """图像拼接：把多张图片拼成一张网格图。

    参数:
        images: 图片列表 (PIL Image)
        columns: 列数，默认2
        padding: 图片间距像素，默认10
        background_color: 背景色，默认白色
    返回: (images_list, info_dict)
    """
    try:
        if not images or len(images) < 2:
            return images, {"error": "至少需要2张图片才能拼接"}

        # 处理输入
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))

        cols = min(columns, len(pil_images))
        rows = (len(pil_images) + cols - 1) // cols

        # 计算网格尺寸
        max_w = max(img.width for img in pil_images)
        max_h = max(img.height for img in pil_images)

        total_w = cols * max_w + (cols + 1) * padding
        total_h = rows * max_h + (rows + 1) * padding

        result = Image.new("RGB", (total_w, total_h), background_color)

        for i, img in enumerate(pil_images):
            row = i // cols
            col = i % cols
            x = padding + col * (max_w + padding)
            y = padding + row * (max_h + padding)
            # 居中放置
            offset_x = x + (max_w - img.width) // 2
            offset_y = y + (max_h - img.height) // 2
            result.paste(img, (offset_x, offset_y))

        info = {
            "input_count": len(pil_images),
            "columns": cols,
            "rows": rows,
            "output_size": f"{total_w}x{total_h}",
        }
        print(f"[Agent] 图像拼接: {len(pil_images)}张 -> {total_w}x{total_h}")
        return [result], info
    except Exception as e:
        return [], {"error": str(e)}


def list_preprocessors_tool():
    """列出所有可用的 ControlNet/ControlLLLite 预处理器。"""
    try:
        try:
            from modules_forge.controlnet import get_preprocessor_list
            preprocessors = get_preprocessor_list()
            return {"preprocessors": preprocessors}
        except Exception:
            pass
        try:
            from modules.controlnet import list_preprocessors
            preprocessors = list_preprocessors()
            return {"preprocessors": preprocessors}
        except Exception:
            pass
        return {"preprocessors": [], "note": "ControlNet 模块未找到"}
    except Exception as e:
        return {"error": str(e)}


def apply_adetailer_tool(image, prompt="", model_name=""):
    """ADetailer 脸部修复：检测人脸并对人脸区域做 inpaint 修复。

    参数:
        image: 输入图片 (PIL Image)
        prompt: 修复时使用的提示词（可选，默认使用通用面部修复提示）
        model_name: 检测模型名称（可选，默认 face_yolov8n.pt）
    返回: (images_list, info_dict)
    """
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image")

        # ===== 1. 人脸检测 =====
        try:
            from adetailer.ultralytics import ultralytics_predict
            from adetailer.mediapipe import mediapipe_face_detection
        except ImportError:
            return [image], {"status": "skipped", "note": "ADetailer 模块未加载，返回原图"}

        # 查找检测模型
        adetailer_models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "..", "models", "adetailer"
        )
        model_path = None
        target_model = model_name or "face_yolov8n.pt"
        candidate_paths = [
            os.path.join(adetailer_models_dir, target_model),
            os.path.join(adetailer_models_dir, "face_yolov8n.pt"),
        ]
        for p in candidate_paths:
            if os.path.isfile(p):
                model_path = p
                break

        if model_path is None:
            # 尝试用 mediapipe（无需模型文件）
            try:
                pred = mediapipe_face_detection(image)
            except Exception as e:
                return [image], {"status": "skipped", "note": f"ADetailer 检测模型未找到且 mediapipe 不可用: {e}，返回原图"}
        else:
            try:
                pred = ultralytics_predict(model_path, image, confidence=0.3)
            except Exception as e:
                return [image], {"status": "skipped", "note": f"ADetailer 人脸检测失败: {e}，返回原图"}

        if not pred.masks:
            return [image], {"status": "no_face", "note": "未检测到人脸，返回原图"}

        print(f"[Agent] ADetailer: 检测到 {len(pred.masks)} 张人脸，开始修复")

        # ===== 2. 对每个人脸区域做 inpaint 修复 =====
        result_image = image.copy()
        face_prompt = prompt or "beautiful detailed face, high quality skin, sharp focus, symmetrical face, portrait"
        face_negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

        for i, mask in enumerate(pred.masks):
            try:
                # 确保 mask 尺寸与原图一致
                if mask.size != result_image.size:
                    mask = mask.resize(result_image.size, Image.LANCZOS)

                # 创建 inpaint 任务
                p = StableDiffusionProcessingImg2Img(
                    outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
                    outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
                    prompt=face_prompt,
                    negative_prompt=face_negative,
                    seed=-1,
                    steps=20,
                    cfg_scale=7.0,
                    width=result_image.width,
                    height=result_image.height,
                    init_images=[result_image],
                    mask=mask,
                    denoising_strength=0.45,
                    inpainting_mask_invert=0,  # 修复 mask 白色区域
                    inpainting_fill=1,  # 保留原始内容作为基础
                    inpaint_full_res=True,  # 全分辨率 inpaint
                    inpaint_full_res_padding=32,
                    sampler_name=_get_sampler(),
                    do_not_save_samples=True,
                    do_not_save_grid=True,
                )
                p.sd_model = shared.sd_model

                try:
                    from modules_forge import main_thread
                    processed = main_thread.run_and_wait_result(process_images, p)
                except Exception:
                    processed = process_images(p)

                if processed and hasattr(processed, 'images') and len(processed.images) > 0:
                    # 将修复后的图像合成回原图（仅取 mask 区域）
                    inpainted = processed.images[0]
                    if inpainted.size != result_image.size:
                        inpainted = inpainted.resize(result_image.size, Image.LANCZOS)
                    result_image = Image.composite(inpainted, result_image, mask)
                    print(f"[Agent] ADetailer: 第 {i+1}/{len(pred.masks)} 张人脸修复完成")
                else:
                    print(f"[Agent] ADetailer: 第 {i+1} 张人脸修复返回空，跳过")

            except Exception as e:
                print(f"[Agent] ADetailer: 第 {i+1} 张人脸修复失败: {e}")
                continue

        return [result_image], {"status": "success", "faces_detected": len(pred.masks), "model": os.path.basename(model_path) if model_path else "mediapipe"}

    except Exception as e:
        return [image], {"status": "error", "error": str(e), "note": "返回原图"}


def remove_background_tool(image, mode="auto", points=None, bg_color=None):
    """智能抠图工具：去除图片背景，返回透明背景的图片。

    基于 See-Through-SAM 扩展，支持三种模式：
    - auto: 智能抠图（BiRefNet 自动分割主体，推荐）
    - point_click: 点选分割（用户指定坐标点，需要 points 参数）
    - cleanup: 图像清理（去除指定区域/物体，需要 mask）

    参数:
        image: 输入图片路径
        mode: 'auto' | 'point_click' | 'cleanup'，默认 auto
        points: 点选分割的坐标列表 [[x,y],...]，仅 mode=point_click 时需要
        bg_color: 背景颜色，如 'white'/'black'/'transparent'，默认 transparent
    """
    try:
        import numpy as np

        if image is None:
            return None, {"status": "error", "error": "请提供图片"}
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        results = []
        meta = {"status": "success", "mode": mode}

        if mode == "auto":
            # 智能抠图优先级：InSPyReNet > BiRefNet > rembg
            from modules import shared

            # 辅助函数：将 RGBA 结果按 bg_color 合成
            def _apply_bg(rgba_img):
                target = bg_color or "transparent"
                if target == "transparent":
                    return rgba_img
                bg = Image.new("RGBA", rgba_img.size, target)
                bg.paste(rgba_img, (0, 0), rgba_img.split()[-1])
                return bg.convert("RGB")

            # 优先 1：InSPyReNet (see-through-sam 扩展的默认模型)
            try:
                api_path = os.path.join(scripts.basedir(), "extensions", "sd-webui-ps-plugin-api", "scripts")
                if api_path not in sys.path:
                    sys.path.insert(0, api_path)
                from api_ps_plugin import process_inspyrenet
                print("[Agent] 尝试 InSPyReNet 抠图...")
                rgba = process_inspyrenet(img)
                results.append(_apply_bg(rgba))
                meta["backend"] = "InSPyReNet"
                print("[Agent] InSPyReNet 抠图成功")
            except Exception as e:
                print(f"[Agent] InSPyReNet 抠图失败: {e}，尝试 BiRefNet")

            # 优先 2：BiRefNet
            if not results:
                try:
                    api_path = os.path.join(scripts.basedir(), "extensions", "sd-webui-ps-plugin-api", "scripts")
                    if api_path not in sys.path:
                        sys.path.insert(0, api_path)
                    from api_ps_plugin import process_birefnet
                    print("[Agent] 尝试 BiRefNet 抠图...")
                    rgba = process_birefnet(img, "birefnet-matting")
                    results.append(_apply_bg(rgba))
                    meta["backend"] = "BiRefNet"
                    print("[Agent] BiRefNet 抠图成功")
                except Exception as e:
                    print(f"[Agent] BiRefNet 抠图失败: {e}，回退 rembg")

            # 优先 3：rembg (最后回退)
            if not results:
                try:
                    u2net_path = os.path.join(os.path.expanduser("~"), ".u2net", "u2net.onnx")
                    if not os.path.isfile(u2net_path):
                        meta["note"] = "首次使用 rembg 正在下载 u2net.onnx 模型（约 170MB），请耐心等待..."
                        print("[Agent] rembg 首次使用，正在下载 u2net.onnx 模型...")
                    from rembg import remove
                    out_img = remove(img)
                    results.append(_apply_bg(out_img))
                    meta["fallback"] = "rembg"
                    meta["backend"] = "rembg"
                except ImportError:
                    return None, {"status": "error", "error": "抠图库未安装，请运行: pip install rembg onnxruntime"}
                except Exception as e:
                    return None, {"status": "error", "error": f"rembg 抠图失败: {e}", "hint": "可能是模型下载失败，请检查网络或手动下载 u2net.onnx 到 ~/.u2net/ 目录"}

        elif mode == "point_click":
            # 点选分割：使用 SAM
            try:
                sys.path.insert(0, os.path.join(scripts.basedir(), "extensions", "sd-webui-see-through-sam", "scripts"))
                from segment_anything_ui import point_segmentation
                if not points:
                    return None, {"status": "error", "error": "点选分割需要提供 points 参数，如 [[100,200]]"}
                img_np = np.array(img)
                masks = point_segmentation(img_np, points)
                if masks:
                    # 用第一个 mask 抠图
                    mask = Image.fromarray(masks[0]).convert("L")
                    out_img = img.copy()
                    out_img.putalpha(mask)
                    results.append(out_img)
            except Exception as e:
                return None, {"status": "error", "error": f"SAM 点选分割失败: {e}"}

        elif mode == "cleanup":
            # 图像清理：使用 LiteLaMa 去除物体（需要 mask，此处简化为自动检测+清理）
            try:
                sys.path.insert(0, os.path.join(scripts.basedir(), "extensions", "sd-webui-see-through-sam", "scripts"))
                from cleaner_ui import clean_object
                # 自动生成 mask（简化：用 rembg 生成前景 mask 然后反转做清理）
                from rembg import remove as rembg_remove
                fg = rembg_remove(img)
                fg_mask = fg.split()[-1]
                # 反转 mask 得到背景区域，清理背景中的物体
                bg_mask = Image.eval(fg_mask, lambda x: 255 - x)
                out_img = clean_object(img, bg_mask)
                if isinstance(out_img, list):
                    results.extend(out_img)
                else:
                    results.append(out_img)
            except Exception as e:
                return None, {"status": "error", "error": f"图像清理失败: {e}"}

        else:
            return None, {"status": "error", "error": f"未知模式: {mode}，支持 auto/point_click/cleanup"}

        if not results:
            return None, {"status": "error", "error": "抠图未产生结果"}

        return results[0], meta
    except Exception as e:
        return None, {"status": "error", "error": str(e)}


def layer_separation_tool(image, output_format="psd"):
    """See-Through-SAM 图层分离工具。

    这是独立的图层分离入口，不是抠图、去背景、点选分割或图像清理。
    """
    _t_start = time.time()
    try:
        if image is None:
            return None, {"status": "error", "error": "请提供图片"}
        if isinstance(image, str):
            input_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            input_image = image.convert("RGB")
        else:
            return None, {"status": "error", "error": "image must be a PIL Image or path"}
        print(f"[Agent] layer_separation 开始: 输入图 {input_image.size}, 模式 {input_image.mode}")

        extension_dir = os.path.join(scripts.basedir(), "extensions", "sd-webui-see-through-sam")
        see_through_dir = os.path.join(extension_dir, "see-through")
        script_path = os.path.join(see_through_dir, "inference", "scripts", "inference_psd_optimized.py")
        if not os.path.isfile(script_path):
            return None, {"status": "error", "error": f"See-Through 推理脚本不存在: {script_path}"}

        timestamp = int(time.time() * 1000)
        temp_dir = os.path.join(see_through_dir, "workspace", "agent_inputs")
        output_dir = os.path.join(scripts.basedir(), "output", "See-Through", "layerdiff_output", f"agent_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(temp_dir, f"agent_input_{timestamp}.png")
        input_image.save(input_path)

        # Match the See-Through UI's working NF4 pipeline and choose a safer
        # square inference size from the installed GPU memory.
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            else:
                vram_gb = 0
        except Exception:
            vram_gb = 0
        if vram_gb <= 7:
            resolution = 768
        elif vram_gb <= 10:
            resolution = 832
        elif vram_gb <= 14:
            resolution = 960
        elif vram_gb <= 20:
            resolution = 1024
        else:
            resolution = 1280

        cmd = [
            sys.executable, script_path,
            "--srcp", input_path,
            "--save_dir", output_dir,
            "--resolution", str(resolution),
            "--quant_mode", "nf4",
            "--save_to_psd",
        ]
        print(f"[Agent] layer_separation 启动推理子进程: resolution={resolution}, 命令={' '.join(cmd)}")
        _t_infer = time.time()
        process = subprocess.run(cmd, cwd=see_through_dir, capture_output=True, text=True, timeout=1800)
        _elapsed_infer = time.time() - _t_infer
        print(f"[Agent] layer_separation 推理子进程结束: returncode={process.returncode}, 耗时 {_elapsed_infer:.1f}s")
        if process.returncode != 0:
            detail = (process.stdout or "") + (process.stderr or "")
            return None, {"status": "error", "error": f"See-Through 图层分离失败: {detail[-4000:]}"}

        saved_dir = os.path.join(output_dir, f"agent_input_{timestamp}")
        pngs = []
        if os.path.isdir(saved_dir):
            pngs = [os.path.join(saved_dir, name) for name in os.listdir(saved_dir)
                    if name.lower().endswith(".png") and "depth" not in name.lower()]
        psd_path = os.path.join(output_dir, f"agent_input_{timestamp}.psd")
        if not os.path.isfile(psd_path):
            psd_path = None
        result_images = [Image.open(path).convert("RGBA") for path in pngs if os.path.isfile(path)]
        if not result_images:
            return None, {"status": "error", "error": "See-Through 未生成图层图片", "output_dir": output_dir}
        meta = {"status": "success", "tool": "layer_separation", "backend": "See-Through",
                "source": "sd-webui-see-through-sam", "output_format": output_format,
                "quant_mode": "nf4", "resolution": resolution,
                "output_dir": output_dir, "psd_path": psd_path, "layer_count": len(result_images)}
        _total = time.time() - _t_start
        print(f"[Agent] layer_separation 完成: {len(result_images)} 个图层, PSD={'有' if psd_path else '无'}, 总耗时 {_total:.1f}s")
        return result_images[0], meta
    except Exception as e:
        _total = time.time() - _t_start
        print(f"[Agent] layer_separation 异常（总耗时 {_total:.1f}s）: {e}")
        return None, {"status": "error", "error": f"See-Through 图层分离失败: {e}"}


def edit_image_tool(image, instruction, strength=0.6):
    """通用图像编辑工具：根据文字指令编辑图片（如换背景、改风格、加物体等）。

    内部自动切换到最适合编辑的模型（Flux.2-Klein 多模态编辑模型），
    执行 img2img 编辑，然后切回原模型。用户完全无感。

    参数:
        image: 输入图片路径
        instruction: 编辑指令，如 "把白天换成夜晚"、"加上雨天效果"、"变成水彩画"
        strength: 编辑强度 0.0-1.0，默认 0.6
    """
    try:
        if image is None:
            return None, {"status": "error", "error": "请提供图片"}

        # 1. 记录当前完整状态（模型 + 全部附加模块），用于之后精确恢复
        original_model = getattr(shared.opts, "sd_model_checkpoint", "")
        original_modules = list(getattr(shared.opts, "forge_additional_modules", []) or [])

        # 2. 查找 Klein 编辑模型
        klein_models = []
        try:
            sd_models.list_models()
            if hasattr(sd_models, "checkpoints_list") and sd_models.checkpoints_list:
                klein_models = [info.filename for info in sd_models.checkpoints_list.values() if "klein" in info.filename.lower()]
        except Exception:
            pass
        if not klein_models:
            klein_models = [f for f in _scan_model_dir("Stable-diffusion") if "klein" in f.lower()]

        if not klein_models:
            return None, {"status": "error", "error": "未找到 Klein 编辑模型，已停止，未降级为普通 img2img"}

        klein_model = klein_models[0]

        # 3. 切换到 Klein 编辑模型（含 TE/VAE）
        te_name = None
        vae_name = None
        for key, guide in MODEL_GUIDE.items():
            if key in klein_model.lower():
                te_name = guide["recommended_te"][0] if guide["recommended_te"] else None
                vae_name = guide["recommended_vae"][0] if guide["recommended_vae"] else None
                break

        set_result = set_model_components_tool(model_name=klein_model, te_name=te_name, vae_name=vae_name)
        if set_result.get("status") != "success":
            return None, {"status": "error", "error": f"切换编辑模型失败: {set_result.get('error')}"}

        # 4. 启用 Image Stitch 插件，用 txt2img + 参考图编码进行编辑
        #    Klein 是多模态编辑模型，需要把参考图编码进潜空间并结合上下文，
        #    而不是常规的 img2img
        try:
            dynamic_args.klein = True
        except Exception:
            pass
        edited_img, edit_meta = _edit_with_image_stitch(image, instruction)

        # 5. 精确恢复原模型 + 原附加模块（整个列表恢复，避免 Klein TE 残留）
        try:
            from modules_forge.main_entry import modules_change, checkpoint_change
            # 先恢复附加模块到原始状态（整个列表）
            modules_change(original_modules, preset=None, save=True, refresh=True)
            # 再恢复主模型
            if original_model:
                checkpoint_change(original_model, preset=None, save=True, refresh=True)
            print(f"[Agent] 已恢复原模型: {original_model}")
        except Exception as e:
            print(f"[Agent] 切回原模型异常: {e}")

        if edited_img is None:
            return None, {"status": "error", "error": edit_meta.get("error", "编辑失败")}

        return edited_img, {
            "status": "success",
            "model_used": "Flux.2-Klein (编辑模型)",
            "instruction": instruction,
            "restored_model": original_model,
            **edit_meta
        }
    except Exception as e:
        return None, {"status": "error", "error": str(e), "method": "klein_reference_latent_edit"}


def api_image_edit_tool(image, instruction, model=None, size="auto", response_format="b64_json"):
    """使用外部/API 图像模型编辑图片。用户明确指定 API 图像模型时优先使用。"""
    try:
        if image is None:
            return None, {"status": "error", "error": "请提供图片"}
        if not instruction or not str(instruction).strip():
            return None, {"status": "error", "error": "请提供图像编辑指令"}

        # 比例字符串 → 像素尺寸转换（auto/none/null 保持不变，让 API 自行决定）
        if size and str(size).strip().lower() not in ("auto", "none", "null"):
            size = _resolve_image_size(size)
            print(f"[Agent] api_image_edit: size 参数解析为 {size}")

        cfg = load_config()
        # 当前 UI 选择是唯一模型来源，禁止 LLM 工具参数覆盖它。
        model_id = str(cfg.get("image_model") or "").strip()
        if not model_id:
            return None, {"status": "error", "error": "未指定 API 图像模型"}
        if model_id.lower() == "trellis.2-4b":
            return _modelscope_trellis2_tool(image)
        # 根据模型选择正确的 base URL（Gemini 模型使用专用端点）
        base_url = _get_image_api_base_url(cfg, model_id)
        api_key = _select_image_api_key(cfg)
        provider = str(cfg.get("image_api_provider") or "").strip().lower()
        if not base_url:
            return None, {"status": "error", "error": "未配置图像/视频生成 API Base URL"}
        if not api_key:
            return None, {"status": "error", "error": "未配置图像/视频生成 API Key，请在独立的生成 API 设置中填写，不是 Agent 大脑 API Key"}

        # Gemini 图像编辑模型使用 Google-native generateContent 格式
        if _is_gemini_image_model(model_id):
            # 将输入图片转为 base64
            if not isinstance(image, (list, tuple)):
                image = [image]
            first_img = image[0] if image else None
            img_b64 = None
            img_mime = "image/png"
            if isinstance(first_img, Image.Image):
                buf = BytesIO()
                first_img.convert("RGB").save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            elif isinstance(first_img, str) and os.path.isfile(first_img):
                with open(first_img, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                if first_img.lower().endswith((".jpg", ".jpeg")):
                    img_mime = "image/jpeg"

            if not img_b64:
                return None, {"status": "error", "error": "无法读取输入图片用于 Gemini 编辑"}

            images_b64, err = _call_gemini_generate(base_url, api_key, model_id, instruction, img_b64, img_mime)
            if err:
                return None, err
            images = []
            for b64_data in images_b64:
                try:
                    img_data = base64.b64decode(b64_data)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                except Exception as e:
                    print(f"[Agent] Gemini 编辑图片解码失败: {e}")
            if images:
                info = {"status": "success", "model_used": model_id, "count": len(images)}
                return images, info
            return None, {"status": "error", "error": "Gemini 编辑图片解码失败"}

        if not isinstance(image, (list, tuple)):
            image = [image]

        images = []
        for item in image:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGBA"))
            elif isinstance(item, str) and os.path.isfile(item):
                images.append(Image.open(item).convert("RGBA"))
        if not images:
            return None, {"status": "error", "error": "image must be a PIL Image or file path"}

        image_b64_list = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        # DashScope 图像编辑使用专用 multimodal-generation 协议。
        if provider == "dashscope":
            content = [{"text": str(instruction).strip()}]
            for image_b64 in image_b64_list[:5]:
                content.append({"image": f"data:image/png;base64,{image_b64}"})
            payload = {
                "model": model_id,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": content,
                    }],
                },
                "parameters": {"prompt_extend": True},
            }
            endpoint = base_url
            if endpoint.endswith("/v1"):
                endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                return None, {"status": "error", "error": f"DashScope 图像编辑失败: HTTP {e.code}", "detail": body, "model_used": model_id}

            images = []
            for choice in (result.get("output") or {}).get("choices") or []:
                for item in (choice.get("message") or {}).get("content") or []:
                    output = item.get("image")
                    if output:
                        if output.startswith("http"):
                            with urllib.request.urlopen(output, timeout=120) as image_resp:
                                images.append(Image.open(BytesIO(image_resp.read())).convert("RGB"))
                        else:
                            images.append(Image.open(BytesIO(base64.b64decode(output))).convert("RGB"))
            if not images:
                return None, {"status": "error", "error": "DashScope 未返回图像数据", "raw": result, "model_used": model_id}
            return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": "dashscope_multimodal_generation"}

        # YoboxAI 的 Gemini/banana 接口使用 generateContent 协议，
        # 图片必须放在 parts.inlineData，API Key 放在 query string。
        if _is_gemini_image_model(model_id) and "yoboxai.com" in base_url.lower():
            endpoint = (
                f"{base_url.rstrip('/')}/../gemini/v1beta/models/"
                f"{urllib.parse.quote(model_id, safe='')}:generateContent"
            )
            endpoint = endpoint.replace("/v1/../gemini/", "/gemini/")
            endpoint = f"{endpoint}?{urllib.parse.urlencode({'key': api_key})}"
            parts = [{"text": str(instruction).strip()}]
            for image_b64 in image_b64_list[:5]:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_b64,
                    }
                })
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": parts,
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            inferred_size = _infer_image_size_from_prompt(instruction, size)
            if inferred_size and str(inferred_size).lower() not in ("auto", "none", "null"):
                resolved_size = _resolve_image_size(inferred_size)
                aspect_ratio, image_size = _pixels_to_gemini_config(resolved_size)
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                }
                print(
                    f"[Agent] Gemini 编辑图像参数: aspectRatio={aspect_ratio}, "
                    f"imageSize={image_size}, sourceSize={resolved_size}"
                )
            else:
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": "1:1",
                    "imageSize": "1K",
                }

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                return None, {
                    "status": "error",
                    "error": f"YoboxAI Gemini 图像编辑失败: HTTP {e.code}",
                    "detail": body,
                    "model_used": model_id,
                }

            images = []
            for candidate in result.get("candidates") or []:
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    inline_data = part.get("inlineData") or part.get("inline_data") or {}
                    output_b64 = inline_data.get("data")
                    if output_b64:
                        images.append(
                            Image.open(BytesIO(base64.b64decode(output_b64))).convert("RGB")
                        )
            if not images:
                return None, {
                    "status": "error",
                    "error": "YoboxAI Gemini 未返回图像数据",
                    "raw": result,
                    "model_used": model_id,
                }
            return images, {
                "status": "success",
                "model_used": model_id,
                "instruction": instruction,
                "method": "yoboxai_gemini_generate_content",
            }

        # ModelScope 的兼容层仍使用其专用的 generations 异步协议。
        # OpenAI 兼容的 gpt-image / sanye 编辑必须走真正的 /images/edits，
        # 不能把上传图片塞进 generations JSON，否则会退化成普通图像生成。
        if provider == "modelscope":
            payload = {
                "model": model_id,
                "prompt": str(instruction).strip(),
                "image_url": f"data:image/png;base64,{image_b64_list[0]}",
                "n": 1,
            }
            if size and str(size).lower() not in ("auto", "none", "null"):
                payload["size"] = size
            req = urllib.request.Request(
                f"{base_url}/images/generations",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                },
            )
            if provider == "modelscope":
                req.headers["X-ModelScope-Async-Mode"] = "true"
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                label = "ModelScope" if provider == "modelscope" else provider or "API"
                return None, {"status": "error", "error": f"{label} 图像编辑失败: HTTP {e.code}", "detail": body, "model_used": model_id}

            # 异步模式：轮询任务结果
            if provider == "modelscope" and "task_id" in result:
                print(f"[Agent] ModelScope 编辑异步任务已提交: {result['task_id']}")
                images, err = _modelscope_poll_task(base_url, api_key, result["task_id"], "image_generation")
                if err:
                    return None, err
                return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": "modelscope_async_edit"}

            # 同步模式回退
            data_items = result.get("data") or []
            if not data_items:
                return None, {"status": "error", "error": "ModelScope 未返回图像数据", "raw": result, "model_used": model_id}
            images = []
            for item in data_items:
                b64 = item.get("b64_json") or item.get("image")
                if b64:
                    if isinstance(b64, str) and b64.startswith("data:image"):
                        b64 = b64.split(",", 1)[1]
                    images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
                elif item.get("url"):
                    with urllib.request.urlopen(item["url"], timeout=120) as img_resp:
                        images.append(Image.open(BytesIO(img_resp.read())).convert("RGB"))
            if not images:
                return None, {"status": "error", "error": "ModelScope 编辑未返回可解析图像", "raw": result, "model_used": model_id}
            return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": f"{provider or 'api'}_generations_edit"}

        # OpenAI-compatible图像编辑接口要求 multipart/form-data，不能把 image
        # 作为 data URL 放进 JSON。
        boundary = f"----ForgeAgent{int(time.time() * 1000000)}"
        parts = []

        def add_text(name, value):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
            )

        add_text("model", model_id)
        add_text("prompt", str(instruction).strip())
        add_text("response_format", response_format)
        if size and str(size).lower() not in ("auto", "none", "null"):
            add_text("size", size)
        for idx, image_b64 in enumerate(image_b64_list[:5]):
            png_bytes = base64.b64decode(image_b64)
            # OpenAI Images Edits 使用 image 字段；多张图时重复 image 字段。
            # image0/image1 等字段名会被多数兼容服务忽略，导致“编辑无输入图”。
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input_{idx}.png\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8")
                + png_bytes + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode("ascii"))
        data = b"".join(parts)
        url = f"{base_url}/images/edits"
        edit_headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {api_key}",
        }
        # ModelScope 使用异步模式
        is_modelscope = provider == "modelscope"
        if is_modelscope:
            edit_headers["X-ModelScope-Async-Mode"] = "true"
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=edit_headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return None, _format_api_http_error(e, "图像编辑")

        # ModelScope 异步模式：轮询任务结果
        if is_modelscope and "task_id" in result:
            print(f"[Agent] ModelScope 编辑异步任务已提交: {result['task_id']}")
            images, err = _modelscope_poll_task(base_url, api_key, result["task_id"], "image_generation")
            if err:
                return None, err
            return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": "modelscope_async_edit"}

        data_items = result.get("data") or []
        if not data_items:
            return None, {"status": "error", "error": "API 未返回图像数据", "raw": result, "model_used": model_id}

        images = []
        for item in data_items:
            b64 = item.get("b64_json") or item.get("image")
            if b64:
                if isinstance(b64, str) and b64.startswith("data:image"):
                    b64 = b64.split(",", 1)[1]
                images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
            elif item.get("url"):
                try:
                    with urllib.request.urlopen(item["url"], timeout=120) as image_resp:
                        images.append(Image.open(BytesIO(image_resp.read())).convert("RGB"))
                except Exception as e:
                    return None, {"status": "error", "error": f"API 返回图片 URL 无法读取: {e}", "model_used": model_id}

        if not images:
            return None, {"status": "error", "error": "API 返回中没有可解析的 base64 图像", "raw": result, "model_used": model_id}

        return images, {
            "status": "success",
            "model_used": model_id,
            "instruction": instruction,
            "method": "api_image_edit",
        }
    except Exception as e:
        return None, {"status": "error", "error": str(e), "method": "api_image_edit"}


TRELLIS2_SPACE_URL = "https://studio-xmccln-trellis-2.api-inference.modelscope.net"


def _normalize_modelscope_space_url(url):
    value = str(url or "").strip().rstrip("/")
    if not value:
        return TRELLIS2_SPACE_URL
    parsed = urllib.parse.urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc.lower()
    if host.endswith(".api-inference.modelscope.net") or host.endswith(".ms.show"):
        return f"{parsed.scheme or 'https'}://{parsed.netloc}".rstrip("/")
    print(f"[Agent] TRELLIS.2 Space URL 无效，使用默认地址: {value}")
    return TRELLIS2_SPACE_URL


def _modelscope_trellis2_tool(image):
    """调用 ModelScope TRELLIS.2 API，将输入图片转换为 GLB。"""
    cfg = load_config()
    space_url = _normalize_modelscope_space_url(cfg.get("image_base_url"))
    # ModelScope SDK Token 通常以 ms- 开头；不要把 YoboxAI 的 sk- Key
    # 误发给创空间 API。兼容用户把 Token 填在图像 Key 或 Agent Key 中。
    image_key = str(cfg.get("image_api_key") or "").strip()
    llm_key = str(cfg.get("api_key") or "").strip()
    api_key = next((key for key in (image_key, llm_key) if key.lower().startswith("ms-")), "")
    if not api_key:
        api_key = image_key or llm_key
    if not api_key:
        return None, {
            "status": "error",
            "error": "未配置 ModelScope SDK Token，请在生成 API 设置中填写 image_api_key",
            "method": "modelscope_trellis2",
        }
    if isinstance(image, (list, tuple)):
        image = image[0] if image else None
    if isinstance(image, Image.Image):
        image_path = _save_pil_to_tempfile(image.convert("RGB"))
    elif isinstance(image, str) and os.path.isfile(image):
        image_path = image
    else:
        return None, {"status": "error", "error": "Trellis.2 图生3D需要先上传一张本地图片", "method": "modelscope_trellis2"}
    if not image_path:
        return None, {"status": "error", "error": "无法读取输入图片", "method": "modelscope_trellis2"}

    def _post_json(endpoint, payload, timeout):
        req = urllib.request.Request(
            f"{space_url}/gradio_api/run/{endpoint}",
            data=json.dumps({"data": payload}, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _upload_file(path):
        boundary = f"----ForgeTrellis{int(time.time() * 1000000)}"
        filename = os.path.basename(path)
        with open(path, "rb") as handle:
            file_bytes = handle.read()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("ascii")
        req = urllib.request.Request(
            f"{space_url}/gradio_api/upload",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        if not uploaded or not isinstance(uploaded[0], str):
            raise RuntimeError("Space 文件上传失败")
        return uploaded[0]

    def _file_data(remote_path):
        return {
            "path": remote_path,
            "orig_name": os.path.basename(remote_path),
            "meta": {"_type": "gradio.FileData"},
        }

    try:
        _post_json("start_session", [], 30)
        remote_path = _upload_file(image_path)
        image_data = _file_data(remote_path)
        _post_json("preprocess_image_1", [image_data], 120)
        result = _post_json("image_to_3d", [
            image_data, 0, "1024",
            7.5, 0.7, 12, 5,
            7.5, 0.5, 12, 3,
            1, 0, 12, 3,
        ], 900)
        data = result.get("data") or []
        output = data[0] if data else None
        if isinstance(output, dict):
            output = output.get("path") or output.get("url")
        if not output or not isinstance(output, str):
            return None, {"status": "error", "error": "TRELLIS.2 未返回 GLB 文件", "raw": result, "method": "modelscope_trellis2"}

        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "agent")
        os.makedirs(output_dir, exist_ok=True)
        target = os.path.join(output_dir, f"trellis2_{int(time.time())}.glb")
        if output.startswith(("http://", "https://")):
            download_url = output
        else:
            encoded_path = urllib.parse.quote(output, safe="")
            download_url = f"{space_url}/gradio_api/file={encoded_path}"
        download_request = urllib.request.Request(
            download_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(download_request, timeout=300) as response, open(target, "wb") as handle:
            handle.write(response.read())
        if not os.path.isfile(target) or os.path.getsize(target) == 0:
            raise RuntimeError("返回的 GLB 文件为空")
        return None, {
            "status": "success",
            "model_used": "Trellis.2-4B",
            "method": "modelscope_trellis2",
            "glb_path": target,
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:1200]
        if exc.code == 400 and "Subdomain" in detail and "invalid" in detail:
            detail = (
                detail
                + "\n\n请确认生成 API 供应商为 ModelScope Space，Base URL 为 "
                + TRELLIS2_SPACE_URL
                + "，并重启 WebUI 后重试。"
            )
        return None, {
            "status": "error",
            "error": f"ModelScope TRELLIS.2 请求失败: HTTP {exc.code}",
            "detail": detail,
            "space_url": space_url,
            "method": "modelscope_trellis2",
        }
    except Exception as exc:
        return None, {
            "status": "error",
            "error": f"ModelScope TRELLIS.2 调用失败: {exc}",
            "method": "modelscope_trellis2",
        }


def _modelscope_poll_task(base_url, api_key, task_id, task_type="image_generation", max_polls=60, poll_interval=3):
    """轮询 ModelScope 异步任务，返回 (images_list, error_dict_or_None)。"""
    for i in range(max_polls):
        time.sleep(poll_interval)
        poll_req = urllib.request.Request(
            f"{base_url}/tasks/{task_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-ModelScope-Task-Type": task_type,
            },
        )
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as poll_resp:
                poll_result = json.loads(poll_resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:500]
            return None, {"status": "error", "error": f"ModelScope 轮询失败: HTTP {e.code}", "detail": body}
        except Exception as e:
            # 网络超时等可恢复错误，继续轮询
            print(f"[Agent] ModelScope 轮询 {i+1} 次异常（继续重试）: {e}")
            continue

        status = str(poll_result.get("task_status") or poll_result.get("status", "")).upper()
        print(f"[Agent] ModelScope 任务轮询 {i+1}/{max_polls}: status={status}")

        if status in ("SUCCEEDED", "SUCCEED", "COMPLETED"):
            images = []
            # ModelScope 返回 output_images 数组
            output_images = poll_result.get("output_images") or []
            for img_url in output_images:
                if isinstance(img_url, str) and img_url.startswith("http"):
                    try:
                        with urllib.request.urlopen(img_url, timeout=60) as img_resp:
                            images.append(Image.open(BytesIO(img_resp.read())).convert("RGB"))
                    except Exception as e:
                        print(f"[Agent] ModelScope 下载图片失败: {e}")
            if images:
                return images, None
            return None, {"status": "error", "error": "ModelScope 任务成功但未返回图片", "raw": poll_result}
        elif status in ("FAILED", "FAILURE", "ERROR"):
            err_msg = poll_result.get("errors") or poll_result.get("error") or "未知错误"
            return None, {"status": "error", "error": f"ModelScope 任务失败: {err_msg}", "raw": poll_result}

    return None, {"status": "error", "error": f"ModelScope 任务轮询超时（{max_polls * poll_interval}秒）"}


# 比例 → 像素尺寸映射（gpt-image-2 等 OpenAI 兼容接口只接受像素尺寸）
_ASPECT_RATIO_TO_PIXELS = {
    "1:1": "1024x1024",
    "9:16": "1024x1792",
    "16:9": "1792x1024",
    "3:4": "768x1024",
    "4:3": "1024x768",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "21:9": "1792x768",
}


def _resolve_image_size(size):
    """将 size 参数统一解析为 API 可接受的像素尺寸字符串。

    - 比例格式 (如 '9:16') → 映射为像素尺寸 (如 '1024x1792')
    - 像素格式 (如 '1024x1792') → 原样返回
    - 空值/无效值 → 默认 '1024x1024'
    """
    if not size or not str(size).strip():
        return "1024x1024"
    s = str(size).strip().lower()
    # 比例格式（含冒号）
    if ":" in s:
        mapped = _ASPECT_RATIO_TO_PIXELS.get(s)
        if mapped:
            return mapped
        # 未知比例，回退到默认
        print(f"[Agent] 未知比例 '{s}'，回退到 1024x1024")
        return "1024x1024"
    # 像素格式（含 x）
    if "x" in s:
        return s
    # 其他情况回退默认
    print(f"[Agent] 无法识别的 size '{s}'，回退到 1024x1024")
    return "1024x1024"


def _pixels_to_gemini_config(size):
    """将统一的像素尺寸转换为 Gemini imageConfig 参数。"""
    config = {
        "1024x1024": ("1:1", "1K"),
        "1024x1792": ("9:16", "2K"),
        "1792x1024": ("16:9", "2K"),
        "768x1024": ("3:4", "1K"),
        "1024x768": ("4:3", "1K"),
        "832x1248": ("2:3", "1K"),
        "1248x832": ("3:2", "1K"),
        "1792x768": ("21:9", "2K"),
    }
    return config.get(str(size).lower(), ("1:1", "1K"))


def _infer_image_size_from_prompt(prompt, size):
    """当模型漏传 size 时，从用户提示词中的比例描述兜底推断尺寸。"""
    text = str(prompt or "").lower()
    current = str(size or "").strip().lower()
    if current not in ("", "auto", "none", "null", "1024x1024", "1:1"):
        return size
    if any(token in text for token in ("16:9", "横版", "横屏", "宽幅", "wide", "landscape")):
        return "1792x1024"
    if any(token in text for token in ("9:16", "竖版", "竖屏", "手机壁纸", "vertical", "portrait")):
        return "1024x1792"
    if "4:3" in text:
        return "1024x768"
    if "3:4" in text:
        return "768x1024"
    return size


def api_image_generate_tool(prompt, model=None, size="1024x1024", response_format="b64_json", negative_prompt="", quality=""):
    """使用设置区选择的外部/API 图像模型生成图片。

    参数:
        prompt: 正向提示词
        model: API 模型 ID（为空则用配置中的 image_model）
        size: 输出尺寸，如 1024x1024、16:9 等
        response_format: b64_json 或 url
        negative_prompt: 负向提示词（YoboxAI gpt-image-2 支持）
        quality: 质量档位，如 high/medium/low（YoboxAI gpt-image-2 支持）
    """
    try:
        if not prompt or not str(prompt).strip():
            return None, {"status": "error", "error": "请提供图像生成提示词"}

        # 比例字符串 → 像素尺寸转换（gpt-image-2 等 OpenAI 兼容接口只接受像素尺寸）
        size = _resolve_image_size(_infer_image_size_from_prompt(prompt, size))
        print(f"[Agent] api_image_generate: size 参数解析为 {size}")

        cfg = load_config()
        # 当前 UI 选择是唯一模型来源，禁止 LLM 工具参数覆盖它。
        model_id = str(cfg.get("image_model") or "").strip()
        # 根据模型选择正确的 base URL（Gemini 模型使用专用端点）
        base_url = _get_image_api_base_url(cfg, model_id)
        # 根据供应商选择正确的 key：避免跨供应商混用导致 401
        api_key = _select_image_api_key(cfg)
        provider = str(cfg.get("image_api_provider") or "").strip().lower()
        # 诊断日志
        _kp = cfg.get("image_api_key") or ""
        _kp2 = cfg.get("api_key") or ""
        _k_mask = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else ("(空)" if not api_key else "***")
        _key_source = "LLM_key" if (api_key == _kp2 and _kp2) else ("image_key" if api_key == _kp and _kp else "fallback")
        print(f"[Agent] api_image_generate: model={model_id}, provider={provider}, base_url={base_url}, api_key={_k_mask} (image_key={'有' if _kp else '无'}, LLM_key={'有' if _kp2 else '无'}, key_source={_key_source})")
        if not model_id:
            return None, {"status": "error", "error": "未指定 API 图像模型"}
        if not base_url:
            return None, {"status": "error", "error": "未配置生成 API Base URL"}
        if not api_key:
            return None, {"status": "error", "error": "未配置生成 API Key，请在生成 API 设置中填写"}

        # Gemini 图像模型使用 Google-native generateContent 格式
        if _is_gemini_image_model(model_id):
            aspect_ratio, image_size = _pixels_to_gemini_config(size)
            print(f"[Agent] Gemini 图像参数: aspectRatio={aspect_ratio}, imageSize={image_size}")
            images_b64, err = _call_gemini_generate(
                base_url,
                api_key,
                model_id,
                prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            )
            if err:
                return None, err
            # 将 base64 转为 PIL Image
            images = []
            for b64_data in images_b64:
                try:
                    img_data = base64.b64decode(b64_data)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                except Exception as e:
                    print(f"[Agent] Gemini 图片解码失败: {e}")
            if images:
                info = {"status": "success", "model_used": model_id, "count": len(images)}
                return images, info
            return None, {"status": "error", "error": "Gemini 图片解码失败"}

        if provider == "yoboxai" and _is_gemini_image_model(model_id):
            gemini_model_id = model_id
            endpoint = (
                f"{base_url.rstrip('/')}/../gemini/v1beta/models/"
                f"{urllib.parse.quote(gemini_model_id, safe='')}:generateContent"
            )
            endpoint = endpoint.replace("/v1/../gemini/", "/gemini/")
            endpoint = f"{endpoint}?{urllib.parse.urlencode({'key': api_key})}"
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": str(prompt).strip()}],
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                return None, {"status": "error", "error": f"YoboxAI Gemini 图像生成失败: HTTP {e.code}", "detail": body, "model_used": model_id}

            images = []
            for candidate in result.get("candidates") or []:
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    inline_data = part.get("inlineData") or part.get("inline_data") or {}
                    output_b64 = inline_data.get("data")
                    if output_b64:
                        images.append(Image.open(BytesIO(base64.b64decode(output_b64))).convert("RGB"))
            if not images:
                return None, {"status": "error", "error": "YoboxAI Gemini 未返回图像数据", "raw": result, "model_used": model_id}
            return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "yoboxai_gemini_generate_content"}

        payload = {
            "model": model_id,
            "prompt": str(prompt).strip(),
            "n": 1,
            "size": size,
        }
        # YoboxAI gpt-image-2 支持扩展参数
        if negative_prompt and str(negative_prompt).strip():
            payload["negative_prompt"] = str(negative_prompt).strip()
        if quality and str(quality).strip():
            payload["quality"] = str(quality).strip()
        if response_format:
            payload["response_format"] = response_format
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # ModelScope 使用异步模式
        is_modelscope = provider == "modelscope"
        if is_modelscope:
            request_headers["X-ModelScope-Async-Mode"] = "true"
        req = urllib.request.Request(
            f"{base_url}/images/generations",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=request_headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return None, _format_api_http_error(e, "图像生成")

        # ModelScope 异步模式：轮询任务结果
        if is_modelscope and "task_id" in result:
            print(f"[Agent] ModelScope 异步任务已提交: {result['task_id']}")
            images, err = _modelscope_poll_task(base_url, api_key, result["task_id"], "image_generation")
            if err:
                return None, err
            return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "modelscope_async_generate"}

        images = []
        for item in result.get("data") or []:
            b64 = item.get("b64_json") or item.get("image")
            if b64:
                if isinstance(b64, str) and b64.startswith("data:image"):
                    b64 = b64.split(",", 1)[1]
                images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
            elif item.get("url"):
                with urllib.request.urlopen(item["url"], timeout=120) as image_resp:
                    images.append(Image.open(BytesIO(image_resp.read())).convert("RGB"))
        if not images:
            return None, {"status": "error", "error": "API 未返回图像数据", "raw": result, "model_used": model_id}
        return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "api_image_generate"}
    except Exception as e:
        print(f"[Agent] api_image_generate_tool 异常: {traceback.format_exc()}")
        return None, {"status": "error", "error": str(e), "method": "api_image_generate"}


def change_background_tool(image, atmosphere="night", instruction=""):
    """换背景氛围专用工具：将图片的背景/氛围替换为指定风格（如白天换夜晚）。

    这是 edit_image 的便捷封装，专门用于背景氛围变换。
    会自动使用 Flux.2-Klein 多模态编辑模型，保留主体，只改变背景氛围。

    参数:
        image: 输入图片路径
        atmosphere: 目标氛围，如 'night'(夜晚)/'sunset'(日落)/'rainy'(雨天)/'snowy'(雪天)/'foggy'(雾天)/'cyberpunk'(赛博朋克)/'morning'(早晨)/'studio'(摄影棚)
        instruction: 自定义编辑指令（可选），当指定时将覆盖 atmosphere 预设，直接使用此指令编辑图片。
                     例如: "把背景改为纯白色"、"换成海边背景"、"背景改成红色"
    """
    try:
        if instruction and instruction.strip():
            # 用户提供了自定义指令，直接使用 edit_image 进行编辑
            return edit_image_tool(image, instruction.strip())

        # 氛围关键词映射
        atmosphere_prompts = {
            "night": "change the background to a dark night scene with moonlight, preserve the subject exactly",
            "sunset": "change the background to a golden sunset scene, warm lighting, preserve the subject exactly",
            "rainy": "change the background to a rainy day scene, wet surfaces, preserve the subject exactly",
            "snowy": "change the background to a snowy winter scene, snow falling, preserve the subject exactly",
            "foggy": "change the background to a foggy misty scene, atmospheric perspective, preserve the subject exactly",
            "cyberpunk": "change the background to a cyberpunk neon city at night, preserve the subject exactly",
            "morning": "change the background to a bright morning scene with soft sunlight, preserve the subject exactly",
            "studio": "change the background to a clean studio backdrop with soft lighting, preserve the subject exactly",
        }
        instruction = atmosphere_prompts.get(atmosphere.lower(),
                                             f"change the background atmosphere to {atmosphere}, preserve the subject exactly")
        return edit_image_tool(image, instruction, strength=0.55)
    except Exception as e:
        return None, {"status": "error", "error": str(e)}


def list_extensions_tool():
    """列出所有已安装的扩展插件。"""
    try:
        from modules import extensions
        exts = extensions.list_extensions()
        result = []
        for ext in exts:
            result.append({
                "name": ext.name,
                "enabled": ext.enabled,
                "is_builtin": ext.is_builtin,
            })
        return {"extensions": result}
    except Exception as e:
        return {"error": str(e)}


def research_extension_tool(name):
    """深入研究某个扩展插件的真实功能。

    读取扩展目录下的 README.md、脚本文件注释等，了解扩展的实际用途。
    当用户问到某个你不确定的扩展/功能时，必须先调用此工具调查，绝对不能编造！

    参数:
        name: 扩展名称或关键词（如 'seedvr2', 'controlnet', 'adetailer'）
    """
    try:
        from modules import paths
        # Forge Neo 路径获取：优先用 paths.extensions_dir，多级回退
        extensions_dir = getattr(paths, "extensions_dir", None)
        if not extensions_dir or not os.path.isdir(extensions_dir):
            # 回退1：基于 script_path 推导（webui 根目录）
            webui_root = getattr(paths, "script_path", None)
            if webui_root and os.path.isdir(os.path.join(webui_root, "extensions")):
                extensions_dir = os.path.join(webui_root, "extensions")
            else:
                # 回退2：基于 scripts.basedir() 推导（当前扩展脚本目录的上上级）
                # scripts.basedir() = webui/extensions/sd-webui-agent/scripts
                # 上两级 = webui/extensions
                base = scripts.basedir()
                extensions_dir = os.path.dirname(os.path.dirname(base))
        print(f"[Agent] research_extension: extensions_dir={extensions_dir}")

        name_lower = name.lower()
        found_dir = None

        # 先精确匹配
        if os.path.isdir(os.path.join(extensions_dir, name)):
            found_dir = os.path.join(extensions_dir, name)
        else:
            # 模糊匹配
            for d in os.listdir(extensions_dir):
                d_path = os.path.join(extensions_dir, d)
                if os.path.isdir(d_path) and name_lower in d.lower():
                    found_dir = d_path
                    break

        if not found_dir:
            return {"status": "not_found", "query": name, "message": f"未找到包含 '{name}' 的扩展目录", "hint": "可先调用 list_extensions 查看所有扩展名称"}

        result = {
            "status": "success",
            "extension_dir": os.path.basename(found_dir),
            "path": found_dir,
            "readme": "",
            "scripts_summary": [],
            "key_files": [],
        }

        # 读取 README.md（最多前 1200 字，避免小模型上下文溢出）
        readme_path = os.path.join(found_dir, "README.md")
        if not os.path.isfile(readme_path):
            # 尝试 vendor 目录下的 README
            for root, dirs, files in os.walk(found_dir):
                for f in files:
                    if f.lower() == "readme.md":
                        readme_path = os.path.join(root, f)
                        break
                if os.path.isfile(readme_path):
                    break
        if os.path.isfile(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(1200)
                    # 截断到第一个 "## " 标题之后，避免太长
                    lines = content.split("\n")
                    useful_lines = []
                    for line in lines:
                        useful_lines.append(line)
                        if len(useful_lines) > 40:
                            break
                    result["readme"] = "\n".join(useful_lines)
            except Exception as e:
                result["readme"] = f"读取失败: {e}"

        # 扫描 scripts 目录下的 Python 文件，提取模块级 docstring
        scripts_dir = os.path.join(found_dir, "scripts")
        if os.path.isdir(scripts_dir):
            for f in sorted(os.listdir(scripts_dir))[:10]:
                if f.endswith(".py"):
                    fpath = os.path.join(scripts_dir, f)
                    result["key_files"].append(os.path.join("scripts", f))
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as sf:
                            lines = sf.readlines()[:30]
                            # 提取 docstring（第一个 """ 或 ''' 包裹的内容）
                            doc_lines = []
                            in_doc = False
                            quote = None
                            for line in lines:
                                stripped = line.strip()
                                if not in_doc:
                                    for q in ('"""', "'''"):
                                        if stripped.startswith(q):
                                            in_doc = True
                                            quote = q
                                            content = stripped[3:].strip()
                                            if content and not content.endswith(q):
                                                doc_lines.append(content)
                                            break
                                else:
                                    if quote in stripped:
                                        content = stripped.replace(quote, "").strip()
                                        if content:
                                            doc_lines.append(content)
                                        break
                                    doc_lines.append(stripped)
                            if doc_lines:
                                result["scripts_summary"].append({
                                    "file": f,
                                    "docstring": " ".join(doc_lines)[:500]
                                })
                    except Exception:
                        pass

        # 也扫描扩展根目录下的 py 文件
        for f in sorted(os.listdir(found_dir))[:5]:
            if f.endswith(".py"):
                result["key_files"].append(f)

        return result
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


_TEXT_FILE_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".css", ".html", ".htm",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".log", ".bat", ".cmd", ".sh", ".ps1", ".xml",
}
_BLOCKED_FILE_EXTENSIONS = {
    ".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin", ".dll", ".exe",
    ".msi", ".zip", ".7z", ".rar", ".tar", ".gz",
}


def _get_webui_root():
    """返回实际 WebUI 根目录，兼容 Forge Neo 的不同启动路径。"""
    from modules import paths

    webui_root = getattr(paths, "script_path", None)
    if webui_root and os.path.isdir(webui_root):
        return os.path.realpath(webui_root)

    ext_dir = getattr(paths, "extensions_dir", None)
    if ext_dir and os.path.isdir(os.path.dirname(ext_dir)):
        return os.path.realpath(os.path.dirname(ext_dir))

    return os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(scripts.basedir()))))


def _select_image_api_key(cfg):
    """根据图像供应商选择正确的 API Key，避免跨供应商 key 混用导致 401。

    逻辑：
    - 如果图像供应商 == LLM 供应商（如都是 YoboxAI），使用 LLM 的 api_key
    - 否则使用 image_api_key
    - 如果 image_api_key 为空，回退到 api_key
    """
    image_provider = str(cfg.get("image_api_provider") or "").strip().lower()
    llm_provider = str(cfg.get("api_provider") or "").strip().lower()
    image_key = str(cfg.get("image_api_key") or "").strip()
    llm_key = str(cfg.get("api_key") or "").strip()

    if image_provider and image_provider == llm_provider:
        # 同一供应商，优先用 LLM key
        return llm_key or image_key
    # 不同供应商，用图像专用 key；如果为空则回退到 LLM key
    return image_key or llm_key


def _get_image_api_base_url(cfg, model_id=""):
    """根据模型 ID 选择正确的 API Base URL。

    YoboxAI 的 Gemini/banana 图像模型需要专用端点 https://api.yoboxai.com/gemini，
    不能用 OpenAI 兼容的 /v1 端点。
    """
    model_lower = str(model_id or "").lower()
    if model_lower.startswith("gemini") or model_lower.startswith("banana"):
        # YoboxAI Gemini 专用端点
        return "https://api.yoboxai.com/gemini"
    base_url = str(cfg.get("image_base_url") or "").rstrip("/")
    # 用户可能直接粘贴文档中的完整接口地址；内部统一保留 OpenAI 兼容根地址，
    # 下面的请求再追加 /images/generations。
    for suffix in ("/images/generations", "/chat/completions"):
        if base_url.lower().endswith(suffix):
            base_url = base_url[: -len(suffix)].rstrip("/")
            break
    return base_url


def _is_gemini_image_model(model_id):
    """判断是否为 Gemini 图像模型（需要 Google-native generateContent 格式）。

    banana2/bananapro 是 YoboxAI 对 Gemini 图像模型的别名，也需要走 Gemini 端点。
    """
    model_lower = str(model_id or "").lower()
    return model_lower.startswith("gemini") or model_lower.startswith("banana")


def _call_gemini_generate(base_url, api_key, model_id, prompt_text, image_b64=None, image_mime="image/png",
                          aspect_ratio="1:1", image_size="1K"):
    """调用 Gemini generateContent API 生成/编辑图片。

    返回 (images_list, error_dict)，成功时 images_list 为 base64 图片列表，error_dict 为 None。
    支持 imageConfig: aspectRatio (1:1, 16:9, 9:16, 4:3, 3:4), imageSize (1K, 2K, 4K)。
    """
    parts = []
    if image_b64:
        parts.append({
            "inlineData": {
                "mimeType": image_mime,
                "data": image_b64,
            }
        })
    parts.append({"text": str(prompt_text).strip()})

    payload = {
        "contents": [{
            "role": "user",
            "parts": parts,
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio or "1:1",
                "imageSize": image_size or "1K",
            },
        },
    }

    # 认证方式：?key= 查询参数（YoboxAI Gemini 端点要求）
    endpoint = f"{base_url}/v1beta/models/{model_id}:generateContent?key={api_key}"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return None, _format_api_http_error(e, "Gemini 图像生成")
    except Exception as e:
        return None, {"status": "error", "error": f"Gemini API 调用异常: {e}"}

    # 解析响应，提取 inlineData 图片
    images = []
    try:
        candidates = result.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])
            for part in parts_list:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    images.append(inline["data"])
    except Exception as e:
        return None, {"status": "error", "error": f"解析 Gemini 响应失败: {e}", "detail": str(result)[:500]}

    if not images:
        # 没有返回图片，可能只返回了文本
        text_parts = []
        try:
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("text"):
                        text_parts.append(part["text"])
        except Exception:
            pass
        text_info = " ".join(text_parts)[:200] if text_parts else "无内容"
        return None, {"status": "error", "error": f"Gemini 未返回图片，仅返回文本: {text_info}"}

    return images, None


def _format_api_http_error(e, action="调用"):
    """格式化 API HTTP 错误，特别是 401 鉴权失败时给出明确提示，防止 LLM 静默回退本地。"""
    body = ""
    try:
        body = e.read().decode("utf-8", errors="ignore")[:1200]
    except Exception:
        pass
    if e.code == 401:
        return {
            "status": "error",
            "error": f"API {action}鉴权失败 (HTTP 401)：当前选择的 API Key 与供应商不匹配或已失效。请检查图像生成 API 设置中的供应商和 Key 是否对应。不要切换到本地模型，应直接告知用户此错误。",
            "detail": body,
            "http_code": 401,
            "do_not_fallback": True,
        }
    if e.code == 403:
        if "1010" in body:
            return {
                "status": "error",
                "error": f"API {action}被供应商网络防火墙拦截 (HTTP 403 / Cloudflare 1010)，不是模型选择错误。请检查 sanye 是否允许当前访问来源调用图像接口。",
                "detail": body,
                "http_code": 403,
                "do_not_fallback": True,
            }
        return {
            "status": "error",
            "error": f"API {action}权限不足 (HTTP 403)：当前 Key 无权限访问该模型。不要切换到本地模型，应直接告知用户此错误。",
            "detail": body,
            "http_code": 403,
            "do_not_fallback": True,
        }
    return {
        "status": "error",
        "error": f"API {action}失败: HTTP {e.code}",
        "detail": body,
        "http_code": e.code,
    }


def _resolve_webui_path(path):
    """将用户提供的相对路径安全解析到 WebUI 根目录内。"""
    webui_root = _get_webui_root()
    target = os.path.realpath(os.path.join(webui_root, str(path or ".")))

    try:
        if os.path.commonpath([webui_root, target]) != webui_root:
            raise ValueError("路径越界：只能访问 WebUI 目录内的内容")
    except ValueError:
        raise ValueError("路径越界：只能访问 WebUI 目录内的内容")

    return webui_root, target


def _limit_chars(value, default=12000, maximum=30000):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(500, min(value, maximum))


def _read_document_text(file_path, max_chars):
    """提取受支持文件的文本，并返回文本和格式说明。"""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in _BLOCKED_FILE_EXTENSIONS:
        raise ValueError(f"为避免读取大型二进制资源，不支持读取 {extension} 文件")

    if extension in _TEXT_FILE_EXTENSIONS or not extension:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(max_chars + 1)

        if extension == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except (ValueError, TypeError):
                pass
        elif extension in {".csv", ".tsv"}:
            try:
                delimiter = "\t" if extension == ".tsv" else ","
                rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
                preview = rows[:100]
                text = "\n".join(" | ".join(column.strip() for column in row) for row in preview)
                if len(rows) > len(preview):
                    text += "\n... (仅显示前 100 行)"
            except (csv.Error, TypeError):
                pass
        return text, "text"

    if extension == ".docx":
        try:
            with zipfile.ZipFile(file_path) as archive:
                document_xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(document_xml)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs = []
            for paragraph in root.iter(f"{namespace}p"):
                pieces = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
                if pieces:
                    paragraphs.append("".join(pieces))
            return "\n".join(paragraphs), "docx"
        except (KeyError, OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise ValueError(f"DOCX 文档解析失败: {exc}") from exc

    if extension == ".pdf":
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name)
                reader = module.PdfReader(file_path)
                return "\n".join((page.extract_text() or "") for page in reader.pages), "pdf"
            except ImportError:
                continue
            except Exception as exc:
                raise ValueError(f"PDF 文档解析失败: {exc}") from exc
        raise ValueError("当前环境没有可用的 PDF 解析组件，暂时无法读取 PDF；可先提供 TXT、MD、DOCX 或复制文本内容")

    raise ValueError(f"暂不支持读取 {extension or '无扩展名'} 文件")


def read_workspace_file_tool(path, max_chars=12000):
    """读取 WebUI 目录内的文本、配置或代码文件内容。"""
    try:
        webui_root, target = _resolve_webui_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"文件不存在: {path}"}

        max_chars = _limit_chars(max_chars)
        text, file_type = _read_document_text(target, max_chars)
        return {
            "status": "success",
            "path": os.path.relpath(target, webui_root),
            "file_type": file_type,
            "size_bytes": os.path.getsize(target),
            "truncated": len(text) > max_chars,
            "content": text[:max_chars],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def analyze_document_tool(path, question="", max_chars=16000):
    """提取文档内容，供智能体据此总结、解释、审阅或回答问题。"""
    result = read_workspace_file_tool(path, max_chars=max_chars)
    if result.get("status") != "success":
        return result
    result["question"] = question or "请总结这份文档的关键内容。"
    result["instruction"] = "请仅基于提取出的内容回答问题；内容截断时需说明结论可能不完整。"
    return result


def _workspace_backup_path(webui_root, target):
    """Create a recoverable backup path for an agent edit."""
    backup_dir = os.path.join(webui_root, ".agent_backups")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    relative = os.path.relpath(target, webui_root).replace(os.sep, "__")
    return os.path.join(backup_dir, f"{stamp}_{relative}.bak")


def repair_workspace_file_tool(path, old_text, new_text, reason=""):
    """Safely patch one WebUI text file after inspecting the current contents.

    The exact old_text match prevents the agent from overwriting a file based on
    stale context. A timestamped backup is created before every edit.
    """
    try:
        webui_root, target = _resolve_webui_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"文件不存在: {path}"}
        if os.path.splitext(target)[1].lower() not in _TEXT_FILE_EXTENSIONS:
            return {"status": "error", "error": "只允许修改文本、代码和配置文件"}
        if any(secret in os.path.basename(target).lower() for secret in ("token", "secret", "credential")):
            return {"status": "error", "error": "为保护密钥，不允许直接修改凭据文件"}
        if not isinstance(old_text, str) or not old_text:
            return {"status": "error", "error": "old_text 不能为空；必须基于当前文件内容进行精确修改"}
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            current = handle.read()
        occurrences = current.count(old_text)
        if occurrences != 1:
            return {"status": "error", "error": f"精确匹配失败：找到 {occurrences} 处，要求恰好 1 处；请先重新读取文件"}
        backup = _workspace_backup_path(webui_root, target)
        with open(backup, "w", encoding="utf-8", newline="") as handle:
            handle.write(current)
        updated = current.replace(old_text, new_text, 1)
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        return {"status": "success", "path": os.path.relpath(target, webui_root),
                "backup": os.path.relpath(backup, webui_root), "reason": reason or "未提供"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def diagnose_workspace_tool(path="."):
    """Run lightweight, non-destructive diagnostics for a WebUI path."""
    try:
        webui_root, target = _resolve_webui_path(path)
        if os.path.isfile(target) and target.lower().endswith(".py"):
            command = [sys.executable, "-m", "py_compile", target]
        elif os.path.isdir(target):
            command = [sys.executable, "-m", "compileall", "-q", target]
        else:
            return {"status": "error", "error": "诊断目标必须是 Python 文件或目录"}
        completed = subprocess.run(command, cwd=webui_root, capture_output=True, text=True, timeout=120)
        return {"status": "success" if completed.returncode == 0 else "error",
                "path": os.path.relpath(target, webui_root), "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "诊断超时（超过 120 秒）"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def audit_extensions_tool(include_readme=True):
    """整理所有已安装扩展的状态、关键脚本和 README 摘要。"""
    try:
        from modules import extensions

        webui_root = _get_webui_root()
        registered = {}
        for extension in extensions.list_extensions():
            registered[extension.name] = {
                "enabled": bool(extension.enabled),
                "is_builtin": bool(extension.is_builtin),
                "path": getattr(extension, "path", None),
            }

        candidates = {}
        for directory, is_builtin in (
            (os.path.join(webui_root, "extensions"), False),
            (os.path.join(webui_root, "extensions-builtin"), True),
        ):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                full_path = os.path.join(directory, name)
                if os.path.isdir(full_path):
                    candidates[name] = {"path": full_path, "is_builtin": is_builtin}

        for name, info in registered.items():
            if name not in candidates and info["path"] and os.path.isdir(info["path"]):
                candidates[name] = {"path": info["path"], "is_builtin": info["is_builtin"]}

        results = []
        for name in sorted(candidates)[:50]:
            info = candidates[name]
            extension_path = info["path"]
            registration = registered.get(name, {})
            scripts_dir = os.path.join(extension_path, "scripts")
            key_files = []
            if os.path.isdir(scripts_dir):
                key_files = [
                    os.path.join("scripts", filename)
                    for filename in sorted(os.listdir(scripts_dir))
                    if filename.endswith(".py")
                ][:8]

            readme_preview = ""
            if include_readme:
                for filename in ("README.md", "readme.md", "README.txt"):
                    readme_path = os.path.join(extension_path, filename)
                    if os.path.isfile(readme_path):
                        try:
                            with open(readme_path, "r", encoding="utf-8", errors="replace") as handle:
                                readme_preview = " ".join(handle.read(700).split())[:350]
                        except OSError:
                            pass
                        break

            results.append({
                "name": name,
                "enabled": registration.get("enabled", True),
                "is_builtin": registration.get("is_builtin", info["is_builtin"]),
                "key_scripts": key_files,
                "readme_preview": readme_preview,
            })

        return {
            "status": "success",
            "extensions": results,
            "total": len(candidates),
            "returned": len(results),
            "truncated": len(candidates) > len(results),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def _fetch_web_text(url, timeout=20, max_bytes=2_000_000):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read(max_bytes)
        content_type = response.headers.get("content-type", "")
    charset = "utf-8"
    match = re.search(r"charset=([\w\-.]+)", content_type, re.I)
    if match:
        charset = match.group(1)
    return raw.decode(charset, errors="replace"), content_type


def _strip_html_text(markup, max_chars=6000):
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", markup or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|section|article)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()[:max_chars]


def web_search_tool(query, max_results=6, region="zh-cn"):
    """联网搜索实时新闻和公开网页资料。"""
    try:
        query = str(query or "").strip()
        if not query:
            return {"status": "error", "error": "请提供搜索关键词"}
        max_results = max(1, min(int(max_results or 6), 10))
        params = urllib.parse.urlencode({"q": query, "kl": region or "zh-cn"})
        search_url = f"https://duckduckgo.com/html/?{params}"
        markup, _ = _fetch_web_text(search_url, timeout=25)

        results = []
        blocks = re.findall(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|</body>)', markup)
        for href, title_html, tail in blocks:
            href = html.unescape(href)
            if "uddg=" in href:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                href = qs.get("uddg", [href])[0]
            title = _strip_html_text(title_html, max_chars=300)
            snippet_match = re.search(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', tail)
            snippet = _strip_html_text(snippet_match.group(1), max_chars=700) if snippet_match else ""
            if title and href.startswith(("http://", "https://")):
                results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= max_results:
                break

        if not results:
            return {"status": "error", "error": "没有解析到搜索结果，可能是网络受限或搜索页格式变化", "query": query}
        return {"status": "success", "query": query, "results": results, "source": "DuckDuckGo HTML"}
    except urllib.error.URLError as exc:
        return {"status": "error", "error": f"联网搜索失败，请检查网络或代理设置: {exc}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def web_read_url_tool(url, max_chars=8000):
    """读取指定网页并提取可引用的正文文本。"""
    try:
        url = str(url or "").strip()
        if not url.startswith(("http://", "https://")):
            return {"status": "error", "error": "请提供 http 或 https 网页地址"}
        max_chars = max(1000, min(int(max_chars or 8000), 20000))
        markup, content_type = _fetch_web_text(url, timeout=30)
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", markup)
        title = _strip_html_text(title_match.group(1), max_chars=300) if title_match else ""
        text = _strip_html_text(markup, max_chars=max_chars)
        return {"status": "success", "url": url, "title": title, "content_type": content_type, "text": text}
    except urllib.error.URLError as exc:
        return {"status": "error", "error": f"网页读取失败，请检查网络或代理设置: {exc}", "url": url}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc(), "url": url}


def explore_webui_tool(path="."):
    """探索 WebUI 的目录结构和内置功能。

    当用户问到 WebUI 有什么功能、某个目录是干什么的时，调用此工具调查。

    参数:
        path: 相对路径，默认为 '.' 表示 webui 根目录
    """
    try:
        webui_root, target = _resolve_webui_path(path)
        print(f"[Agent] explore_webui: webui_root={webui_root}")

        if not os.path.isdir(target):
            # 可能是文件
            if os.path.isfile(target):
                size = os.path.getsize(target)
                return {"status": "file", "path": path, "size_bytes": size, "size_kb": round(size/1024, 1)}
            return {"status": "error", "error": f"路径不存在: {path}"}

        entries = []
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            entry = {"name": name, "type": "dir" if os.path.isdir(full) else "file"}
            if entry["type"] == "file":
                entry["size_kb"] = round(os.path.getsize(full) / 1024, 1)
            entries.append(entry)

        return {
            "status": "success",
            "path": path,
            "absolute_path": target,
            "entries": entries[:100],  # 限制数量
            "total": len(entries),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def list_models_tool():
    """列出所有可用的 Stable Diffusion 模型 (checkpoints)。
    直接扫描文件系统，避免 sd_models.list_models() 返回 None 的问题。"""
    try:
        # 优先从 checkpoints_list 读取（如果已加载）
        names = []
        try:
            sd_models.list_models()  # 刷新列表
            if hasattr(sd_models, "checkpoints_list") and sd_models.checkpoints_list:
                names = [info.filename for info in sd_models.checkpoints_list.values()]
        except Exception:
            pass
        # 回退：直接扫描文件系统
        if not names:
            names = _scan_model_dir("Stable-diffusion")
        current = _get_current_checkpoint()
        return {"current_model": current, "available_models": names}
    except Exception as e:
        return {"error": str(e)}


def list_vae_tool():
    """列出所有可用的 VAE 模型。"""
    try:
        # 优先从 sd_vae.vae_dict 读取
        names = []
        try:
            from modules import sd_vae
            sd_vae.refresh_vae_list()
            if hasattr(sd_vae, "vae_dict") and sd_vae.vae_dict:
                names = list(sd_vae.vae_dict.keys())
        except Exception:
            pass
        # 回退：扫描文件系统
        if not names:
            names = _scan_model_dir("vae", extensions=(".safetensors", ".ckpt", ".pt"))
        current = getattr(shared.opts, "sd_vae", "Automatic")
        return {"current_vae": current, "available_vaes": names}
    except Exception as e:
        return {"error": str(e)}


def list_text_encoders_tool():
    """列出所有可用的文本编码器 (Text Encoders)。"""
    try:
        names = _scan_model_dir("text_encoder", extensions=(".safetensors", ".pt", ".pth"))
        # 也检查 text_encoders（复数，有些配置用这个）
        names += _scan_model_dir("text_encoders", extensions=(".safetensors", ".pt", ".pth"))
        return {"text_encoders": sorted(set(names))}
    except Exception as e:
        return {"error": str(e)}


def list_controlnet_tool():
    """列出所有可用的 ControlNet 模型和预处理器。"""
    try:
        cn_models = _scan_model_dir("ControlNet", extensions=(".safetensors", ".pth", ".pt"))
        # 预处理器目录
        preprocessor_dir = None
        try:
            from modules import paths
            preprocessor_dir = os.path.join(paths.models_path, "ControlNet", "preprocessor")
        except Exception:
            preprocessor_dir = None
        preprocessors = []
        if preprocessor_dir and os.path.isdir(preprocessor_dir):
            for root, dirs, files in os.walk(preprocessor_dir):
                for d in dirs:
                    preprocessors.append(d)
        return {
            "controlnet_models": cn_models,
            "preprocessors": sorted(set(preprocessors)),
            "note": "ControlNet 参数在 WebUI 界面中设置，Agent 可通过此工具查看可用资源"
        }
    except Exception as e:
        return {"error": str(e)}


# ===== 模型搭配指南 =====
# 根据主模型名称自动推荐搭配的 TE 和 VAE
# 基于用户提供的模型指南，文件名已与实际文件系统扫描结果核对
MODEL_GUIDE = {
    "krea2": {
        "name": "Krea2 Turbo",
        "description": "新一代审美向多风格文生图模型，支持中文提示词，适合插画/概念艺术",
        "preset_arch": "krea",
        "recommended_te": ["qwen3vl_4b_fp8_scaled.safetensors"],
        "recommended_vae": ["qwen_image_vae.safetensors"],
        "loras": ["krea2-贴图风格", "krea2-原画风格角色", "krea2场景概念艺术"],
        "tips": "Krea2 专用 Qwen3-VL 文本编码器 + Qwen-Image 专用 VAE，不要用 Flux 的 TE/VAE"
    },
    "flux-2-klein": {
        "name": "Flux.2 Klein 9B",
        "description": "Flux.2 Klein 多模态编辑模型，高质量生图，支持图像编辑",
        "preset_arch": "klein",
        "recommended_te": ["qwen_3_8b_fp8mixed.safetensors"],
        "recommended_vae": ["flux2-vae.safetensors"],
        "loras": ["Klein-万物迁移", "klein-9b奇幻装饰艺术场景概念", "klein-9b古风场景概念", "klein-9b二次元动画"],
        "tips": "Flux.2-Klein 专用 Qwen3-8B 文本编码器 + Flux2 VAE"
    },
    "flux2-klein": {
        "name": "Flux.2 Klein 9B",
        "description": "Flux.2 Klein 多模态编辑模型，高质量生图",
        "preset_arch": "klein",
        "recommended_te": ["qwen_3_8b_fp8mixed.safetensors"],
        "recommended_vae": ["flux2-vae.safetensors"],
        "loras": ["Klein-万物迁移", "klein-9b奇幻装饰艺术场景概念", "klein-9b古风场景概念"],
        "tips": "Flux.2-Klein 专用 Qwen3-8B 文本编码器 + Flux2 VAE"
    },
    "qwen_image_edit": {
        "name": "Qwen Image Edit",
        "description": "Qwen 图像编辑模型，支持图生图编辑",
        "preset_arch": "qwen",
        "recommended_te": ["qwen3vl_4b_fp8_scaled.safetensors"],
        "recommended_vae": ["qwen_image_vae.safetensors"],
        "loras": [],
        "tips": "使用 Qwen3-VL 文本编码器 + Qwen Image VAE"
    },
    "z_image": {
        "name": "Z-Image Turbo",
        "description": "Zimage 模型，快速生图",
        "preset_arch": "zit",
        "recommended_te": ["qwen_3_4b.safetensors"],
        "recommended_vae": ["flux-ae.safetensors"],
        "loras": [],
        "tips": "Z-Image 专用 Qwen3-4B 文本编码器 + Flux VAE (flux-ae)"
    },
    "anima": {
        "name": "Anima",
        "description": "Anima 二次元高质量专用模型",
        "preset_arch": "anima",
        "recommended_te": ["qwen_3_06b_base.safetensors"],
        "recommended_vae": ["qwen_image_vae.safetensors"],
        "loras": [],
        "tips": "Anima 需要 Qwen3-0.6B 文本编码器 (通义千问) + Qwen-Image 专用 VAE"
    },
    "illustrious": {
        "name": "Illustrious XL",
        "description": "Illustrious SDXL 插画模型，高质量动漫/插画风格",
        "preset_arch": "xl",
        "recommended_te": [],
        "recommended_vae": [],
        "loras": ["XL漫画人设"],
        "tips": "SDXL 标准架构，使用默认 TE/VAE，可搭配 XL 系列 LoRA"
    },
    "xl": {
        "name": "SDXL 系列",
        "description": "SDXL 架构模型",
        "preset_arch": "xl",
        "recommended_te": [],
        "recommended_vae": [],
        "loras": [],
        "tips": "SDXL 标准架构，使用默认 TE/VAE"
    },
}


def get_model_guide_tool(model_name=None):
    """获取模型搭配指南：主模型 + 文本编码器 + VAE + LoRA 的推荐组合。

    参数:
        model_name: 可选，指定模型名称。如果不提供，返回所有已知模型的指南。
    """
    try:
        if model_name:
            # 匹配模型指南
            matched = None
            for key, guide in MODEL_GUIDE.items():
                if key in model_name.lower():
                    matched = guide
                    break
            if matched is None:
                return {
                    "status": "warning",
                    "message": f"未找到 '{model_name}' 的特定指南，返回通用建议",
                    "general_tips": "如果是自定义模型，请查阅模型卡说明。一般来说：SDXL 用默认 TE/VAE；Flux 系用 Qwen3 TE + Flux VAE；Qwen Image 系列用 Qwen-VL TE + Qwen VAE",
                    "all_guides": {k: v["name"] for k, v in MODEL_GUIDE.items()}
                }
            return {"status": "success", "guide": matched}
        else:
            # 返回所有指南摘要
            all_guides = {}
            for key, guide in MODEL_GUIDE.items():
                all_guides[key] = {
                    "name": guide["name"],
                    "description": guide["description"],
                    "recommended_te": guide["recommended_te"],
                    "recommended_vae": guide["recommended_vae"],
                    "loras": guide["loras"],
                }
            return {"status": "success", "all_guides": all_guides}
    except Exception as e:
        return {"error": str(e)}


def _forge_update_additional_modules(module_type, filepath=None):
    """更新 Forge 的 forge_additional_modules 列表（TE/VAE 热切换核心）。

    Forge 通过 forge_additional_modules 选项在运行时指定额外的 TE/VAE 文件，
    无需重启 WebUI。modules_change() 会自动刷新模型加载参数。

    参数:
        module_type: 'vae' 或 'te'
        filepath: 新文件路径（文件名即可，会自动匹配），None 则移除该类型
    """
    from modules_forge.main_entry import modules_change, module_list, refresh_models

    # 确保 module_list 已填充（首次调用时可能还没刷新）
    try:
        refresh_models()
    except Exception:
        pass

    full_path = None
    if filepath:
        basename = os.path.basename(filepath)
        # 优先从 Forge 的 module_list 匹配
        if basename in module_list:
            full_path = module_list[basename]
        else:
            # 回退：从文件系统构造路径
            from modules import paths
            subdir = "vae" if module_type == "vae" else "text_encoder"
            candidate = os.path.join(paths.models_path, subdir, filepath)
            if os.path.isfile(candidate):
                full_path = candidate

    # 读取当前列表
    current = list(getattr(shared.opts, "forge_additional_modules", []) or [])

    # 移除同类型旧文件（通过路径关键词判断）
    new_values = []
    for m in current:
        m_path = m if isinstance(m, str) else str(m)
        m_lower = m_path.lower().replace("\\", "/")
        # VAE 路径特征：包含 /vae/
        is_vae = "/vae/" in m_lower
        # TE 路径特征：包含 /text_encoder/
        is_te = "/text_encoder/" in m_lower
        if module_type == "vae" and is_vae:
            continue
        if module_type == "te" and is_te:
            continue
        new_values.append(m_path)

    # 添加新文件
    if full_path:
        new_values.append(full_path)

    # 调用 Forge 官方 API 设置并刷新加载参数
    changed = modules_change(new_values, preset=None, save=True, refresh=True)
    return new_values, full_path


def set_vae_tool(vae_name):
    """设置 VAE 模型（运行时热切换，无需重启）。

    通过 Forge 的 forge_additional_modules 机制实现，下次生图时自动加载新 VAE。

    参数:
        vae_name: VAE 文件名，如 'flux2-vae.safetensors' 或 'qwen_image_vae.safetensors'
    """
    try:
        available = _scan_model_dir("vae", extensions=(".safetensors", ".ckpt", ".pt"))
        matched = None
        for fn in available:
            if vae_name.lower() in fn.lower():
                matched = fn
                break
        if matched is None:
            return {"status": "error", "error": f"未找到 VAE '{vae_name}'", "available": available}
        new_values, full_path = _forge_update_additional_modules("vae", matched)
        print(f"[Agent] 设置 VAE: {matched} → {full_path}")
        return {
            "status": "success",
            "message": f"已热切换 VAE 为: {matched}",
            "vae": matched,
            "effective_modules": [os.path.basename(m) for m in new_values],
            "tip": "VAE 已设置，下次生图时自动加载，无需重启"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def set_text_encoder_tool(te_name):
    """设置文本编码器 (Text Encoder)（运行时热切换，无需重启）。

    通过 Forge 的 forge_additional_modules 机制实现，下次生图时自动加载新 TE。

    参数:
        te_name: 文本编码器文件名，如 'qwen3vl_4b_fp8_scaled.safetensors'
    """
    try:
        available = _scan_model_dir("text_encoder", extensions=(".safetensors", ".pt", ".pth"))
        matched = None
        for fn in available:
            if te_name.lower() in fn.lower():
                matched = fn
                break
        if matched is None:
            return {"status": "error", "error": f"未找到文本编码器 '{te_name}'", "available": available}
        new_values, full_path = _forge_update_additional_modules("te", matched)
        print(f"[Agent] 设置文本编码器: {matched} → {full_path}")
        return {
            "status": "success",
            "message": f"已热切换文本编码器为: {matched}",
            "text_encoder": matched,
            "effective_modules": [os.path.basename(m) for m in new_values],
            "tip": "文本编码器已设置，下次生图时自动加载，无需重启"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def set_model_components_tool(model_name=None, te_name=None, vae_name=None):
    """一键设置模型的全部组件：主模型 + 文本编码器 + VAE（运行时热切换，无需重启）。

    这是切换模型最推荐的方式：一次性设置全部组件，只触发一次模型加载，
    避免分步操作导致主模型与附加模块不匹配。
    如果只提供 model_name，会自动从模型指南查找推荐的 TE/VAE。

    参数:
        model_name: 主模型文件名（可选），如 'krea2_turbo_int8_convrot.safetensors'
        te_name: 文本编码器文件名（可选），如 'qwen3vl_4b_fp8_scaled.safetensors'
        vae_name: VAE 文件名（可选），如 'qwen_image_vae.safetensors'
    """
    try:
        from modules_forge.main_entry import module_list, refresh_models, refresh_model_loading_parameters
        from modules import paths

        # 1. 如果只给了 model_name，自动从 MODEL_GUIDE 查找推荐 TE/VAE
        if model_name and not te_name and not vae_name:
            model_lower = model_name.lower()
            for key, guide in MODEL_GUIDE.items():
                if key in model_lower:
                    if guide["recommended_te"]:
                        te_name = guide["recommended_te"][0]
                    if guide["recommended_vae"]:
                        vae_name = guide["recommended_vae"][0]
                    break

        # 2. 确保 module_list 已填充
        try:
            refresh_models()
        except Exception:
            pass

        # 3. 构建新的 additional_modules 列表（一次性）
        new_modules = []
        te_full_path = None
        vae_full_path = None

        # 先保留当前列表中不需要替换的模块（通过路径关键词判断类型）
        current = list(getattr(shared.opts, "forge_additional_modules", []) or [])
        for m in current:
            m_path = m if isinstance(m, str) else str(m)
            m_lower = m_path.lower().replace("\\", "/")
            is_vae = "/vae/" in m_lower
            is_te = "/text_encoder/" in m_lower
            # 如果指定了新的 TE/VAE，移除同类型旧文件
            if te_name and is_te:
                continue
            if vae_name and is_vae:
                continue
            new_modules.append(m_path)

        # 添加新 TE
        if te_name:
            te_basename = os.path.basename(te_name)
            if te_basename in module_list:
                te_full_path = module_list[te_basename]
            else:
                candidate = os.path.join(paths.models_path, "text_encoder", te_name)
                if os.path.isfile(candidate):
                    te_full_path = candidate
                else:
                    return {"status": "error", "step": "resolve_te", "error": f"文本编码器文件未找到: {te_name}"}
            new_modules.append(te_full_path)

        # 添加新 VAE
        if vae_name:
            vae_basename = os.path.basename(vae_name)
            if vae_basename in module_list:
                vae_full_path = module_list[vae_basename]
            else:
                candidate = os.path.join(paths.models_path, "vae", vae_name)
                if os.path.isfile(candidate):
                    vae_full_path = candidate
                else:
                    return {"status": "error", "step": "resolve_vae", "error": f"VAE 文件未找到: {vae_name}"}
            new_modules.append(vae_full_path)

        # 4. 一次性更新 opts（主模型 + 附加模块）
        model_changed = False
        if model_name:
            from modules import sd_models
            new_ckpt_info = sd_models.get_closet_checkpoint_match(model_name)
            if new_ckpt_info is None:
                # 回退：直接匹配文件名
                for info in sd_models.checkpoints_list.values():
                    if model_name.lower() in info.filename.lower():
                        new_ckpt_info = info
                        break
            if new_ckpt_info is None:
                return {"status": "error", "step": "resolve_model", "error": f"主模型文件未找到: {model_name}"}
            shared.opts.set("sd_model_checkpoint", new_ckpt_info.title)
            model_changed = True

        modules_changed = (new_modules != current)
        if modules_changed:
            shared.opts.set("forge_additional_modules", new_modules)

        # 5. 只刷新一次（关键！避免分步加载导致主模型与附加模块不匹配）
        if model_changed or modules_changed:
            try:
                shared.opts.save(shared.config_filename)
            except Exception:
                pass
            refresh_model_loading_parameters(refresh=True)

        # 6. 汇总结果
        effective_modules = [os.path.basename(m) for m in (getattr(shared.opts, "forge_additional_modules", []) or [])]
        return {
            "status": "success",
            "message": "模型组件切换完成，下次生图时自动加载，无需重启",
            "model": os.path.basename(model_name) if model_name else None,
            "text_encoder": os.path.basename(te_full_path) if te_full_path else None,
            "vae": os.path.basename(vae_full_path) if vae_full_path else None,
            "effective_modules": effective_modules,
            "model_changed": model_changed,
            "modules_changed": modules_changed,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


def list_samplers_tool():
    """列出所有可用的采样器。"""
    try:
        samplers = sd_samplers.all_samplers()
        return {"samplers": [s.name for s in samplers]}
    except Exception as e:
        return {"error": str(e)}


def list_upscalers_tool():
    """列出所有可用的放大模型。"""
    try:
        upscalers = getattr(shared, "sd_upscalers", []) or []
        return {"upscalers": [u.name for u in upscalers if hasattr(u, "name")]}
    except Exception as e:
        return {"error": str(e)}


def get_current_settings_tool():
    """获取当前 WebUI 的生图设置信息。"""
    return {
        "checkpoint": _get_current_checkpoint(),
        "sampler": getattr(shared.opts, "sampler_name", "Euler a"),
        "steps": getattr(shared.opts, "steps", 20),
        "cfg_scale": getattr(shared.opts, "cfg_scale", 7.0),
        "width": getattr(shared.opts, "width", 1024),
        "height": getattr(shared.opts, "height", 1024),
        "batch_size": getattr(shared.opts, "batch_size", 1),
    }


def list_loras_tool():
    """列出已安装的 LoRA 模型 (直接扫描目录)。"""
    try:
        lora_dir = os.path.join(shared.opts.models_dir, "Lora") if hasattr(shared.opts, "models_dir") else None
        if not lora_dir or not os.path.isdir(lora_dir):
            # 回退路径
            lora_dir = os.path.join(scripts.basedir(), "models", "Lora")

        loras = []
        if os.path.isdir(lora_dir):
            for root, dirs, files in os.walk(lora_dir):
                for f in files:
                    if f.endswith((".safetensors", ".ckpt", ".pt", ".pth")):
                        loras.append(os.path.splitext(f)[0])
        return {"loras": sorted(loras)}
    except Exception as e:
        return {"error": str(e)}


def generate_with_lora_tool(prompt, lora_name, lora_weight=0.8, **kwargs):
    """使用 LoRA 生成图片。在提示词中自动插入 LoRA 语法。

    参数:
        prompt: 基础提示词
        lora_name: LoRA 名称 (不含 .safetensors 扩展名)
        lora_weight: LoRA 权重 (默认 0.8, 范围 0-2)
        其他参数同 txt2img
    """
    # Forge LoRA 语法: <lora:name:weight>
    full_prompt = f"{prompt} <lora:{lora_name}:{lora_weight}>"
    kwargs.pop("lora_name", None)
    kwargs.pop("lora_weight", None)
    return txt2img_tool(prompt=full_prompt, **kwargs)


# =============================================================================
# MiniMax H3 视频生成工具
# =============================================================================

def _get_webui_base_url():
    """自动检测 WebUI 基础 URL，验证 /h3studio/api/bootstrap 端点可用。

    优先使用 cmd_args.cmd_opts.port，然后扫描 7860-7870 端口。
    返回 base url 字符串或 None。
    """
    import httpx

    def _check(base):
        try:
            resp = httpx.get(f"{base}/h3studio/api/bootstrap", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    # 优先：cmd_args.cmd_opts.port
    try:
        from modules import shared
        port = getattr(shared.cmd_opts, "port", None)
        if port:
            base = f"http://127.0.0.1:{port}"
            if _check(base):
                return base
    except Exception:
        pass

    # 回退：扫描 7860-7870
    for port in range(7860, 7871):
        base = f"http://127.0.0.1:{port}"
        if _check(base):
            return base

    return None


def h3_video_generate_tool(prompt, duration=5, aspect_ratio="16:9",
                           first_frame=None, last_frame=None, reference_image=None):
    """调用 forge-h3-studio 插件生成视频（MiniMax H3）。

    支持两种后端模式：
    - api（云端）：直接使用 CloudClient 调用 MiniMax 云端 API
    - managed/external（本地 ComfyUI）：通过 forge-h3-studio HTTP API 提交

    参数:
        prompt: 视频描述提示词（中英文均可）
        duration: 视频时长（秒），4-15，默认5
        aspect_ratio: 宽高比，16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9，默认16:9
        first_frame: 首帧图片（PIL Image 或文件路径，可选）
        last_frame: 尾帧图片（PIL Image 或文件路径，可选）
        reference_image: 参考图片（PIL Image 或文件路径，可选，多模态参考模式）
    返回: dict，包含 status 和 video_path
    """
    import httpx
    import hashlib

    # 本地/托管工作台使用 forge-h3-studio 的 ComfyUI 工作流；只有工作台
    # 明确设为 api 时，才使用 Agent 的 YoboxAI 视频 API 分支。
    h3_backend_mode = "managed"
    try:
        h3_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "forge-h3-studio", "data", "config.json",
        )
        with open(h3_config_path, "r", encoding="utf-8") as h3_config_file:
            h3_backend_mode = str(json.load(h3_config_file).get("backend_mode") or "managed").strip().lower()
    except Exception:
        pass

    # YoboxAI H3 使用独立的视频 API，不依赖 forge-h3-studio 的 MiniMax Key。
    # 该分支必须在本地插件 bootstrap 检查之前执行，避免 API 模式被误判为未配置。
    try:
        agent_cfg = load_config()
        agent_provider = str(agent_cfg.get("api_provider") or "").strip().lower()
        agent_base_url = str(agent_cfg.get("base_url") or "").rstrip("/")
        agent_api_key = str(agent_cfg.get("api_key") or "").strip()
        if h3_backend_mode == "api" and (agent_provider == "yoboxai" or "api.yoboxai.com" in agent_base_url.lower()):
            if not agent_api_key:
                return {"status": "error", "error": "YoboxAI API Key 未配置，请在 Agent API 设置中填写"}

            yobox_base = agent_base_url or "https://api.yoboxai.com/v1"
            if not yobox_base.endswith("/v1"):
                yobox_base = yobox_base.rstrip("/") + "/v1"
            video_url = f"{yobox_base}/videos"
            timestamp = int(time.time() * 1000)

            def _data_url(asset):
                if asset is None:
                    return None
                if isinstance(asset, str) and os.path.isfile(asset):
                    with open(asset, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("ascii")
                    return f"data:image/png;base64,{encoded}"
                if isinstance(asset, Image.Image):
                    buf = BytesIO()
                    asset.convert("RGB").save(buf, format="PNG")
                    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
                return asset if isinstance(asset, str) and asset.startswith(("http://", "https://", "data:")) else None

            start_url = _data_url(first_frame)
            end_url = _data_url(last_frame)
            ref_url = _data_url(reference_image)
            input_data = {
                "prompt": str(prompt),
                "aspect_ratio": aspect_ratio,
                "resolution": "2K",
                "duration": duration,
                "audio": True,
                "n": 1,
            }
            if start_url:
                input_data["start_frames"] = [{"url": start_url}]
            if end_url:
                if not start_url:
                    return {"status": "error", "error": "首尾帧模式必须同时提供首帧和尾帧"}
                input_data["end_frames"] = [{"url": end_url}]
            elif ref_url:
                input_data["image_references"] = [{"url": ref_url, "strength": "MID"}]

            request = {"model": "MiniMax-H3", "input": input_data}
            headers = {
                "Authorization": f"Bearer {agent_api_key}",
                "Content-Type": "application/json",
                "Idempotency-Key": f"forge-agent-{timestamp}",
            }
            try:
                response = httpx.post(video_url, json=request, headers=headers, timeout=60)
                response.raise_for_status()
                submitted = response.json()
            except Exception as e:
                detail = response.text[:1200] if "response" in locals() else ""
                return {"status": "error", "error": f"YoboxAI H3 提交失败: {e}", "detail": detail}

            task_id = ((submitted.get("data") or {}).get("task_id") or submitted.get("task_id"))
            if not task_id:
                return {"status": "error", "error": "YoboxAI H3 未返回 task_id", "raw": submitted}

            deadline = time.time() + 900
            while time.time() < deadline:
                try:
                    poll = httpx.get(f"{video_url}/{task_id}", headers={"Authorization": f"Bearer {agent_api_key}"}, timeout=30)
                    poll.raise_for_status()
                    result = poll.json()
                except Exception as e:
                    return {"status": "error", "error": f"YoboxAI H3 查询失败: {e}"}
                data = result.get("data") or result
                status = str(data.get("status") or "").upper()
                if status in ("SUCCESS", "SUCCEEDED", "COMPLETED"):
                    output_url = data.get("video_url") or data.get("url") or data.get("download_url")
                    if not output_url:
                        return {"status": "error", "error": "YoboxAI H3 已完成但未返回视频 URL", "raw": result}
                    yobox_out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "h3_video")
                    os.makedirs(yobox_out_dir, exist_ok=True)
                    video_path = os.path.join(yobox_out_dir, f"h3_yobox_{timestamp}.mp4")
                    content = httpx.get(output_url, timeout=180).content
                    with open(video_path, "wb") as f:
                        f.write(content)
                    return {"status": "success", "video_path": video_path, "duration": duration,
                            "prompt": prompt, "backend": "yoboxai", "model": "MiniMax-H3", "task_id": task_id}
                if status in ("FAILED", "ERROR", "CANCELED", "CANCELLED"):
                    return {"status": "error", "error": data.get("error") or "YoboxAI H3 视频生成失败", "raw": result}
                time.sleep(8)
            return {"status": "error", "error": "YoboxAI H3 视频生成超时（超过 15 分钟）", "task_id": task_id}
    except Exception as e:
        return {"status": "error", "error": f"YoboxAI H3 参数处理失败: {e}"}

    max_wait = 600  # 10 分钟
    poll_interval = 8

    # 1. bootstrap 检查
    base = _get_webui_base_url()
    if not base:
        return {"status": "error", "error": "forge-h3-studio 插件未检测到，请确认已安装并启用"}

    try:
        resp = httpx.get(f"{base}/h3studio/api/bootstrap", timeout=10)
        resp.raise_for_status()
        bootstrap = resp.json()
    except Exception as e:
        return {"status": "error", "error": f"forge-h3-studio bootstrap 检查失败: {e}"}

    cfg = bootstrap.get("config", {})
    backend = bootstrap.get("backend", {})
    mode = str(cfg.get("backend_mode", "managed")).lower()

    # 输出目录（webui 标准 outputs/h3_video）
    webui_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.path.join(webui_dir, "outputs", "h3_video")
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, f"h3_{int(time.time())}.mp4")

    # 参数校验
    try:
        duration = max(4, min(15, int(duration)))
    except (TypeError, ValueError):
        duration = 5
    valid_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"}
    if aspect_ratio not in valid_ratios:
        aspect_ratio = "16:9"

    # 保存参考图的辅助函数
    def _save_asset(img):
        if img is None:
            return None
        if isinstance(img, str) and os.path.isfile(img):
            return img
        # PIL Image 或 bytes → 保存到临时文件
        name = f"h3_ref_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}.png"
        tmp_dir = os.path.join(tempfile.gettempdir(), "h3_agent_refs")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, name)
        if isinstance(img, Image.Image):
            img.save(path, "PNG")
        elif isinstance(img, (bytes, bytearray)):
            with open(path, "wb") as f:
                f.write(img)
        else:
            return None
        return path

    first_frame_path = _save_asset(first_frame)
    last_frame_path = _save_asset(last_frame)
    ref_path = _save_asset(reference_image)

    # ============================================================
    # 云端模式（api）：直接使用 CloudClient
    # ============================================================
    if mode == "api":
        try:
            from h3studio.cloud_client import CloudClient, CLOUD_UPLOAD_DIR, H3_FPS
        except ImportError as e:
            return {"status": "error", "error": f"无法导入 forge-h3-studio CloudClient: {e}"}

        client = CloudClient(cfg)
        if not client.enabled():
            return {"status": "error", "error": "MiniMax API Key 未配置，请在 forge-h3-studio 设置中填写"}

        # 将参考图复制到 CLOUD_UPLOAD_DIR（CloudClient 从这里读取并转为 data URL）
        import shutil

        def _copy_to_cloud_uploads(src_path):
            if not src_path:
                return None
            dst = CLOUD_UPLOAD_DIR / os.path.basename(src_path)
            if not dst.is_file():
                try:
                    shutil.copy2(src_path, str(dst))
                except Exception:
                    # 如果同名冲突，用唯一文件名
                    dst = CLOUD_UPLOAD_DIR / f"ref_{int(time.time()*1000)}.png"
                    shutil.copy2(src_path, str(dst))
            return dst.name

        cloud_first = _copy_to_cloud_uploads(first_frame_path)
        cloud_last = _copy_to_cloud_uploads(last_frame_path)
        cloud_ref = _copy_to_cloud_uploads(ref_path)

        frames = duration * H3_FPS
        # 根据宽高比选尺寸
        if aspect_ratio in ("9:16", "3:4"):
            w, h = 768, 1344
        else:
            w, h = 1344, 768

        request = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "frames": frames,
            "duration": duration,
            "first_frame": cloud_first or "",
            "last_frame": cloud_last or "",
            "references": [{"kind": "image", "file": cloud_ref}] if cloud_ref else [],
        }

        print(f"[Agent] H3 云端提交: prompt='{prompt[:50]}...' duration={duration}s ratio={aspect_ratio}")

        try:
            task_id = client.submit(request)
        except Exception as e:
            return {"status": "error", "error": f"MiniMax 云端提交失败: {e}"}

        # 轮询
        start = time.time()
        last_status = ""
        while time.time() - start < max_wait:
            try:
                result = client.query(task_id)
            except Exception as e:
                return {"status": "error", "error": f"MiniMax 云端查询失败: {e}"}

            status = str(result.get("status", "")).lower()
            if status != last_status:
                print(f"[Agent] H3 任务状态: {status}")
                last_status = status

            if status in ("succeeded", "success"):
                video_url = result.get("video_url", "")
                if not video_url:
                    return {"status": "error", "error": "任务成功但未返回视频 URL"}
                # 关键修复：处理相对路径 URL
                if video_url.startswith("/"):
                    video_url = f"{base}{video_url}"
                try:
                    content, _ = client.download(video_url)
                    with open(video_path, "wb") as f:
                        f.write(content)
                except Exception as e:
                    return {"status": "error", "error": f"下载视频失败: {e}"}
                print(f"[Agent] H3 视频已保存: {video_path}")
                return {
                    "status": "success",
                    "video_path": video_path,
                    "duration": duration,
                    "prompt": prompt,
                    "backend": "minimax-cloud",
                }
            elif status in ("failed", "error"):
                return {"status": "error", "error": result.get("error") or "MiniMax 云端生成失败"}

            time.sleep(poll_interval)

        return {"status": "error", "error": f"视频生成超时（超过 {max_wait} 秒）"}

    # ============================================================
    # 本地模式（managed/external）：通过 forge-h3-studio HTTP API
    # ============================================================
    else:
        # 检查后端是否就绪
        if not backend.get("ready"):
            return {"status": "error", "error": f"H3 后端未就绪: state={backend.get('state')}。请先启动 ComfyUI 后端。"}

        # 上传参考图
        def _upload_asset(img_path):
            if not img_path:
                return None
            try:
                with open(img_path, "rb") as f:
                    files = {"file": (os.path.basename(img_path), f, "image/png")}
                    up_resp = httpx.post(f"{base}/h3studio/api/assets/upload", files=files, timeout=60)
                    up_resp.raise_for_status()
                    return up_resp.json()
            except Exception as e:
                print(f"[Agent] 上传参考图失败: {e}")
                return None

        first_asset = _upload_asset(first_frame_path)
        last_asset = _upload_asset(last_frame_path)
        ref_asset = _upload_asset(ref_path)

        # 根据宽高比选尺寸
        if aspect_ratio in ("9:16", "3:4"):
            w, h = 768, 1344
        else:
            w, h = 1344, 768

        request = {
            "mode": "t2v",
            "model": "minimax_h3_fl2va_fp8.safetensors",
            "text_encoder": "qwen3vl_32b_minimax_h3_fp8.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            "prompt": prompt,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "frames": duration * 24,
            "steps": 30,
            "sampler": "euler",
            "scheduler": "simple",
            "shift_video": 12,
            "shift_audio": 3,
            "denoise": 1,
        }
        if first_asset:
            request["first_frame"] = first_asset.get("file", "")
        if last_asset:
            request["last_frame"] = last_asset.get("file", "")
        if ref_asset:
            request["references"] = [{"kind": "image", "file": ref_asset.get("file", "")}]

        print(f"[Agent] H3 本地提交: prompt='{prompt[:50]}...' duration={duration}s")

        try:
            submit_resp = httpx.post(f"{base}/h3studio/api/jobs", json=request, timeout=30)
            submit_resp.raise_for_status()
            job = submit_resp.json()
            job_id = job.get("id")
        except Exception as e:
            return {"status": "error", "error": f"提交 H3 任务失败: {e}"}

        if not job_id:
            return {"status": "error", "error": "提交成功但未返回 job_id"}

        # 轮询任务状态
        start = time.time()
        last_state = ""
        while time.time() - start < max_wait:
            try:
                poll_resp = httpx.get(f"{base}/h3studio/api/jobs/{job_id}", timeout=15)
                poll_resp.raise_for_status()
                job = poll_resp.json()
            except Exception as e:
                return {"status": "error", "error": f"轮询任务失败: {e}"}

            state = str(job.get("state", "")).lower()
            if state != last_state:
                print(f"[Agent] H3 任务状态: {state}")
                last_state = state

            if state == "completed":
                outputs = job.get("outputs", [])
                if not outputs:
                    return {"status": "error", "error": "任务完成但无输出文件"}
                # 找到视频输出
                video_url = ""
                for out in outputs:
                    fname = str(out.get("filename", "")).lower()
                    if fname.endswith((".mp4", ".webm", ".mov")):
                        video_url = out.get("url", "")
                        break
                if not video_url and outputs:
                    video_url = outputs[0].get("url", "")
                if not video_url:
                    return {"status": "error", "error": "未找到视频输出 URL"}

                # 关键修复：处理相对路径 URL
                if video_url.startswith("/"):
                    video_url = f"{base}{video_url}"

                try:
                    vid_resp = httpx.get(video_url, timeout=120, follow_redirects=True)
                    vid_resp.raise_for_status()
                    with open(video_path, "wb") as f:
                        f.write(vid_resp.content)
                except Exception as e:
                    return {"status": "error", "error": f"下载视频失败: {e}"}

                print(f"[Agent] H3 视频已保存: {video_path}")
                return {
                    "status": "success",
                    "video_path": video_path,
                    "duration": duration,
                    "prompt": prompt,
                    "backend": "forge-h3-studio",
                }
            elif state in ("failed", "cancelled"):
                return {"status": "error", "error": job.get("error") or f"任务{state}"}

            time.sleep(poll_interval)

        return {"status": "error", "error": f"视频生成超时（超过 {max_wait} 秒）"}


def dreamina_video_generate_tool(prompt, duration=5, aspect_ratio="16:9", model=None,
                                 first_frame=None, last_frame=None, reference_image=None):
    """使用 YoboxAI Dreamina SeaDance 生成视频，独立于 MiniMax H3 本地/云端设置。"""
    import httpx
    import hashlib

    try:
        cfg = load_config()
        provider = str(cfg.get("video_api_provider") or "").strip().lower()
        base_url = str(cfg.get("video_base_url") or "https://api.yoboxai.com/v1").rstrip("/")
        api_key = str(cfg.get("video_api_key") or "").strip()
        prompt_text = str(prompt)
        model = str(model or cfg.get("video_model") or "dreamina-seedance-2-0-hc").strip()
        if "dreamina-seedance-2-5-hc" in prompt_text.lower():
            model = "dreamina-seedance-2-5-hc"
        elif "dreamina-seedance-2-0-hc" in prompt_text.lower():
            model = "dreamina-seedance-2-0-hc"

        if not api_key:
            return {"status": "error", "error": "视频生成 API Key 未配置，请在 Agent 视频生成设置中填写"}
        if provider and provider != "yoboxai" and "yoboxai.com" not in base_url.lower():
            return {"status": "error", "error": "Dreamina 当前只支持 YoboxAI 视频供应商，请在 Agent 视频生成设置中选择 YoboxAI"}
        if "api.yoboxai.com" in base_url.lower():
            base_url = "https://api.yoboxai.com/api/v3/contents/generations/tasks"
        elif not base_url.endswith("/tasks"):
            base_url = base_url.rstrip("/") + "/api/v3/contents/generations/tasks"

        try:
            duration = max(4, min(15, int(duration)))
        except (TypeError, ValueError):
            duration = 5
        valid_ratios = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "16:9"

        def _asset_url(asset):
            if asset is None:
                return None
            if isinstance(asset, str) and asset.startswith(("http://", "https://", "asset://")):
                return asset
            if isinstance(asset, str) and os.path.isfile(asset):
                with open(asset, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            if isinstance(asset, Image.Image):
                buf = BytesIO()
                asset.convert("RGB").save(buf, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
            return None

        start_url = _asset_url(first_frame)
        end_url = _asset_url(last_frame)
        ref_url = _asset_url(reference_image)
        content = [{"type": "text", "text": prompt_text}]
        if start_url:
            content.append({"type": "image_url", "image_url": {"url": start_url}, "role": "first_frame"})
        if end_url:
            if not start_url:
                return {"status": "error", "error": "首尾帧模式必须同时提供首帧和尾帧"}
            content.append({"type": "image_url", "image_url": {"url": end_url}, "role": "last_frame"})
        elif ref_url:
            content.append({"type": "image_url", "image_url": {"url": ref_url}, "role": "reference_image"})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": f"dreamina-{hashlib.md5(str(time.time()).encode()).hexdigest()}",
        }
        payload = {
            "model": model,
            "content": content,
            "duration": duration,
            "resolution": "720p",
            "ratio": aspect_ratio,
            "generate_audio": False,
            "watermark": False,
        }

        response = httpx.post(base_url, json=payload, headers=headers, timeout=60)
        if response.status_code >= 400:
            return {"status": "error", "error": f"Dreamina 视频提交失败（HTTP {response.status_code}）：{response.text[:800]}"}
        try:
            submitted = response.json()
        except ValueError:
            return {"status": "error", "error": f"Dreamina 视频提交返回的不是 JSON（HTTP {response.status_code}）：{response.text[:800]}"}

        task = submitted.get("task") or submitted.get("data") or {}
        if not isinstance(task, dict):
            task = {}
        task_id = str(
            task.get("id")
            or task.get("task_id")
            or submitted.get("id")
            or submitted.get("task_id")
            or submitted.get("taskId")
            or ""
        ).strip()
        if not task_id:
            return {"status": "error", "error": "Dreamina 未返回 task_id", "raw": submitted}

        deadline = time.time() + 900
        while time.time() < deadline:
            poll = httpx.get(f"{base_url}/{task_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            if poll.status_code >= 400:
                return {"status": "error", "error": f"Dreamina 查询失败（HTTP {poll.status_code}）：{poll.text[:800]}"}
            try:
                result = poll.json()
            except ValueError:
                return {"status": "error", "error": f"Dreamina 查询返回的不是 JSON（HTTP {poll.status_code}）：{poll.text[:800]}"}
            task = result.get("task") or result.get("data") or result
            if not isinstance(task, dict):
                task = result if isinstance(result, dict) else {}
            status = str(task.get("status") or "").lower()
            if status in ("completed", "complete", "succeeded", "success", "done"):
                def _find_video_url(value):
                    if isinstance(value, str):
                        return value if value.startswith(("http://", "https://")) and ".mp4" in value.lower() else None
                    if isinstance(value, list):
                        for item in value:
                            found = _find_video_url(item)
                            if found:
                                return found
                    if isinstance(value, dict):
                        for key in ("video_url", "videoUrl", "url", "download_url", "downloadUrl", "uri", "output"):
                            found = _find_video_url(value.get(key))
                            if found:
                                return found
                        for item in value.values():
                            found = _find_video_url(item)
                            if found:
                                return found
                    return None

                output_url = _find_video_url(result)
                if not output_url:
                    return {"status": "error", "error": "Dreamina 已完成但未返回视频 URL", "raw": result}
                out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "dreamina_video")
                os.makedirs(out_dir, exist_ok=True)
                video_path = os.path.join(out_dir, f"dreamina_{int(time.time())}.mp4")
                vid = httpx.get(output_url, timeout=180, follow_redirects=True)
                vid.raise_for_status()
                with open(video_path, "wb") as f:
                    f.write(vid.content)
                return {"status": "success", "video_path": video_path, "duration": duration,
                        "prompt": prompt, "backend": "dreamina-yoboxai", "model": model, "task_id": task_id}
            if status in ("failed", "error", "cancelled", "canceled"):
                return {"status": "error", "error": task.get("error") or "Dreamina 视频生成失败", "raw": result}
            time.sleep(8)
        return {"status": "error", "error": "Dreamina 视频生成超时（超过 15 分钟）", "task_id": task_id}
    except Exception as e:
        return {"status": "error", "error": f"Dreamina 视频生成失败: {e}"}


# =============================================================================
# Function Calling 工具定义 (OpenAI tools 格式)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "txt2img",
            "description": "根据文字描述生成图片。当用户要求画一张图、生成图片、创建图像时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图片描述提示词，建议用英文，越详细越好"},
                    "negative_prompt": {"type": "string", "description": "负向提示词，不想出现的内容"},
                    "steps": {"type": "integer", "description": "采样步数，默认20"},
                    "width": {"type": "integer", "description": "图片宽度，默认1024"},
                    "height": {"type": "integer", "description": "图片高度，默认1024"},
                    "sampler_name": {"type": "string", "description": "采样器名称"},
                    "cfg_scale": {"type": "number", "description": "CFG引导强度，默认7.0"},
                    "seed": {"type": "integer", "description": "随机种子，-1为随机"},
                    "batch_size": {"type": "integer", "description": "每批生成数量，默认1"},
                    "n_iter": {"type": "integer", "description": "生成批次数量，默认1"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "img2img",
            "description": "基于参考图片和文字描述生成新图片。当用户上传了参考图片并要求修改/变换时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图片描述提示词"},
                    "negative_prompt": {"type": "string", "description": "负向提示词"},
                    "denoising_strength": {"type": "number", "description": "重绘强度0-1，默认0.75"},
                    "steps": {"type": "integer", "description": "采样步数"},
                    "width": {"type": "integer", "description": "图片宽度"},
                    "height": {"type": "integer", "description": "图片高度"},
                    "sampler_name": {"type": "string", "description": "采样器名称"},
                    "cfg_scale": {"type": "number", "description": "CFG引导强度"},
                    "seed": {"type": "integer", "description": "随机种子"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_model",
            "description": "切换 Stable Diffusion 模型 (checkpoint)。当用户要求换模型、切换大模型时使用。可先用 list_models 查看可用模型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "模型文件名或名称关键词"}
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_settings",
            "description": "修改 WebUI 的生图设置 (步数、CFG、采样器、尺寸、批量大小)。所有参数可选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "description": "采样步数"},
                    "cfg_scale": {"type": "number", "description": "CFG引导强度"},
                    "sampler_name": {"type": "string", "description": "采样器名称"},
                    "width": {"type": "integer", "description": "图片宽度"},
                    "height": {"type": "integer", "description": "图片高度"},
                    "batch_size": {"type": "integer", "description": "批量大小"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upscale",
            "description": "图片放大或调整大小。当用户要求放大图片、提高分辨率、调整图片尺寸、改大小、resize 时都使用此工具。需要用户上传图片或刚生成的图片。用户说调整大小/改尺寸时，用 resize_w 和 resize_h 指定目标宽高，不要用 scale。",
            "parameters": {
                "type": "object",
                "properties": {
                    "upscaler_name": {"type": "string", "description": "放大模型名称，如 4x-UltraSharp。未指定时默认优先使用 4x-UltraSharp，不要传 None"},
                    "scale": {"type": "number", "description": "放大倍数，默认2。仅当用户说放大几倍时使用"},
                    "resize_w": {"type": "integer", "description": "指定目标宽度像素。用户说调整大小/改尺寸/resize 到具体宽度时使用，覆盖 scale"},
                    "resize_h": {"type": "integer", "description": "指定目标高度像素。用户说调整大小/改尺寸/resize 到具体高度时使用，覆盖 scale"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_with_lora",
            "description": "使用 LoRA 模型生成图片。当用户要求使用某个 LoRA 风格/角色生成图片时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "基础提示词"},
                    "lora_name": {"type": "string", "description": "LoRA 名称 (不含扩展名)"},
                    "lora_weight": {"type": "number", "description": "LoRA 权重 0-2，默认0.8"},
                    "negative_prompt": {"type": "string", "description": "负向提示词"},
                    "steps": {"type": "integer", "description": "采样步数"},
                    "width": {"type": "integer", "description": "图片宽度"},
                    "height": {"type": "integer", "description": "图片高度"},
                    "seed": {"type": "integer", "description": "随机种子"},
                },
                "required": ["prompt", "lora_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "列出所有可用的 Stable Diffusion 主模型 (checkpoints)，直接扫描文件系统获取准确文件名。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_vae",
            "description": "列出所有可用的 VAE 模型。切换模型前建议先查看有哪些 VAE 可搭配。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_text_encoders",
            "description": "列出所有可用的文本编码器 (Text Encoders)。Krea2/Flux 等模型需要搭配特定 TE。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_controlnet",
            "description": "列出所有可用的 ControlNet 模型和预处理器（openpose, lineart, zoedepth 等）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_guide",
            "description": "获取模型搭配指南：根据主模型推荐搭配的文本编码器(TE)、VAE 和 LoRA。切换模型前务必调用此工具了解正确组合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "可选，指定模型名称如 'krea2' 或 'flux'，不填则返回所有指南"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vae",
            "description": "设置 VAE 模型。通常在切换主模型后根据 get_model_guide 的推荐来设置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "vae_name": {"type": "string", "description": "VAE 文件名，如 'flux2-vae.safetensors'"},
                },
                "required": ["vae_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_text_encoder",
            "description": "设置文本编码器 (TE)（运行时热切换，无需重启）。Krea2 需要 qwen3vl_4b，Flux 需要 qwen_3_8b。",
            "parameters": {
                "type": "object",
                "properties": {
                    "te_name": {"type": "string", "description": "文本编码器文件名，如 'qwen3vl_4b_fp8_scaled.safetensors'"},
                },
                "required": ["te_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_model_components",
            "description": "【推荐】一键设置模型的全部组件：主模型+文本编码器+VAE（运行时热切换，无需重启）。只给 model_name 会自动匹配推荐的 TE/VAE。这是切换模型的首选工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "主模型文件名，如 'krea2_turbo_int8_convrot.safetensors'，只给它会自动匹配 TE/VAE"},
                    "te_name": {"type": "string", "description": "可选，指定文本编码器文件名"},
                    "vae_name": {"type": "string", "description": "可选，指定 VAE 文件名"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_samplers",
            "description": "列出所有可用的采样器。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_upscalers",
            "description": "列出所有可用的放大模型。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_settings",
            "description": "获取当前 WebUI 的生图设置信息 (模型、采样器、步数、CFG、尺寸等)。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_loras",
            "description": "列出已安装的 LoRA 模型。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
        {
            "type": "function",
            "function": {
                "name": "trellis2_image_to_3d",
                "description": "使用已安装的 TRELLIS.2 本地扩展将上传图片生成 GLB 三维模型。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer", "description": "随机种子，默认0"},
                        "resolution": {"type": "string", "description": "网格分辨率，默认1024", "enum": ["512", "1024", "1536"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "video_keyframe_extract",
            "description": "从视频中提取关键帧。当用户上传视频并要求提取关键帧、截取画面、获取视频帧时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_frames": {"type": "integer", "description": "提取帧数，默认5"},
                    "method": {"type": "string", "description": "提取方式: even(均匀采样), first(首帧), last(尾帧), middle(中间帧)", "enum": ["even", "first", "last", "middle"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "video_to_frames",
            "description": "从视频中按时间间隔提取帧（每N秒一帧）。适合从长视频中批量提取画面。",
            "parameters": {
                "type": "object",
                "properties": {
                    "interval_seconds": {"type": "number", "description": "提取间隔（秒），默认1"},
                    "max_frames": {"type": "integer", "description": "最大提取帧数，默认20"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "h3_video_generate",
            "description": "【MiniMax H3 视频生成】生成视频。当用户要求'生成视频'/'制作视频'/'动起来'/'T2V'/'文生视频'/'图生视频'时使用。基于 forge-h3-studio 插件，支持云端 API 模式（已配置）和本地 ComfyUI 模式。生成的视频会保存到 outputs/h3_video 目录并在对话中展示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "视频内容描述提示词（中英文均可，越详细越好）"},
                    "duration": {"type": "integer", "description": "视频时长（秒），范围 4-15，默认 5", "default": 5},
                    "aspect_ratio": {"type": "string", "description": "视频宽高比，默认 16:9", "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"], "default": "16:9"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dreamina_video_generate",
            "description": "【Dreamina SeaDance 视频生成】用户明确使用 @dreamina-seedance-2-5-hc 或 @dreamina-seedance-2-0-hc 时必须使用。走 Agent 独立视频生成 API 设置，不依赖 forge-h3-studio、MiniMax Key 或本地 H3 模型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "视频内容描述提示词"},
                    "duration": {"type": "integer", "description": "视频时长（秒），范围 4-15，默认 5", "default": 5},
                    "aspect_ratio": {"type": "string", "description": "视频比例", "enum": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], "default": "16:9"},
                    "model": {"type": "string", "description": "Dreamina 模型 ID，如 dreamina-seedance-2-0-hc 或 dreamina-seedance-2-5-hc"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stitch_images",
            "description": "把多张图片拼成一张网格图。当用户要求拼图、拼接多张图片时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "integer", "description": "列数，默认2"},
                    "padding": {"type": "integer", "description": "图片间距像素，默认10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_preprocessors",
            "description": "列出所有可用的 ControlNet/ControlLLLite 预处理器。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_adetailer",
            "description": "ADetailer 脸部修复，自动检测并优化人脸细节。需要先上传或生成图片。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "修复时的提示词（可选）"},
                    "model_name": {"type": "string", "description": "检测模型名称（可选）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_background",
            "description": "智能抠图/点选分割/图像清理工具。用户说'去除背景'/'抠图'/'点选分割'/'图像清理'时使用。不要用于图层分离；图层分离必须使用 layer_separation 工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["auto", "point_click", "cleanup"], "description": "auto=智能抠图(推荐) / point_click=点选分割 / cleanup=图像清理", "default": "auto"},
                    "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "点选分割的坐标列表 [[x,y],...]，仅 point_click 模式需要"},
                    "bg_color": {"type": "string", "description": "背景颜色: transparent/white/black，默认 transparent", "default": "transparent"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "layer_separation",
            "description": "图层分离：使用 sd-webui-see-through-sam 的 see-through 图层分离功能，把图像拆成可编辑图层。用户说'图层分离'、'分层'、'分离成 PSD 图层'时必须使用此工具，不要使用 remove_background。",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_format": {"type": "string", "enum": ["psd", "images"], "description": "输出形式。psd=优先合成为 PSD；images=返回分离图层图片", "default": "psd"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_image_generate",
            "description": "外部/API 图像生成工具。当前图像生成模型选择远程 API 模型时，用它进行文生图，不需要用户输入 @模型标签。支持 negative_prompt 和 quality 参数（YoboxAI gpt-image-2 等模型可用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图像生成提示词"},
                    "model": {"type": "string", "description": "API 模型 ID，默认读取设置区当前图像生成模型"},
                    "size": {"type": "string", "description": "输出尺寸（像素），默认 1024x1024。常用值：1024x1024(1:1正方形)、1024x1792(9:16竖版)、1792x1024(16:9横版)、768x1024(3:4)、1024x768(4:3)。用户提到竖版/9:16/竖屏时必须传 1024x1792；横版/16:9/横屏时传 1792x1024", "default": "1024x1024"},
                    "negative_prompt": {"type": "string", "description": "负向提示词，描述不希望出现的元素，如 'blurry, low resolution, ugly, deformed, watermark'（可选）", "default": ""},
                    "quality": {"type": "string", "description": "生成质量档位：high/medium/low（可选，仅部分模型支持）", "default": ""},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_image_edit",
            "description": "外部/API 图像编辑工具。用户使用 @banana2、@bananapro、@gpt-image-2、@gpt-image-2.5、@Qwen-Image-Edit-2511、@FireRed-Image-Edit 等 API 图像编辑模型标签时必须优先使用此工具；选择 Trellis.2-4B 时用于上传图片生成 GLB 三维模型。不要改用 remove_background、change_background 或本地 Klein 编辑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "图像编辑指令，如 'change the image background to pure white, preserve the subject exactly'"},
                    "model": {"type": "string", "description": "API 模型 ID。若用户标签指定了模型，必须填写该模型 ID，如 banana2"},
                    "size": {"type": "string", "description": "输出尺寸（像素），默认 auto(保持原图尺寸)。如需指定比例：1024x1792(9:16竖版)、1792x1024(16:9横版)、1024x1024(1:1)", "default": "auto"},
                },
                "required": ["instruction", "model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "通用图像编辑：根据文字指令编辑图片（换背景/改风格/加物体/去物体等）。会自动切换到 Flux.2-Klein 多模态编辑模型执行编辑，完成后自动切回原模型，用户无感。用户说'把这个改成...'/'加上...'/'去掉...'/'修改...'时使用。【重要】换背景：如果用户要的背景不在预设氛围选项（night/sunset/rainy/snowy/foggy/cyberpunk/morning/studio）中，必须使用此工具，不能使用 change_background。",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "编辑指令，如 'change background to night scene, preserve subject'、'add a cat on the table'"},
                    "strength": {"type": "number", "description": "编辑强度 0.0-1.0，默认 0.6，值越大改动越大", "default": 0.6},
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_background",
            "description": "换背景氛围专用工具：将图片的背景/氛围替换为预设风格（night/sunset/rainy/snowy/foggy/cyberpunk/morning/studio 之一）。自动使用 Klein 编辑模型，保留主体，只改背景。用户说'换成夜晚'/'换个背景'/'改成雨天'时优先使用。【注意】此工具只能使用预设氛围，如果用户想要的背景不在预设列表中（如卧室、海边、森林等），请使用 edit_image 工具，不要使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "atmosphere": {"type": "string", "enum": ["night", "sunset", "rainy", "snowy", "foggy", "cyberpunk", "morning", "studio"], "description": "目标氛围预设", "default": "night"},
                },
                "required": ["atmosphere"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_extensions",
            "description": "列出所有已安装的扩展插件及其状态。当用户询问有哪些插件时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索实时新闻、最新资料和公开网页信息。用户问今天/最新/实时/新闻/网络资料/帮我查一下时优先使用，并在回答中给出来源链接。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，建议包含地点、主题和时间范围"},
                    "max_results": {"type": "integer", "description": "返回结果数量，默认6，最多10", "default": 6},
                    "region": {"type": "string", "description": "搜索区域，如 zh-cn、us-en，默认 zh-cn", "default": "zh-cn"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_read_url",
            "description": "读取指定网页内容并提取正文。适合用户给出链接，或 web_search 后需要核对来源细节时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要读取的 http/https 网页地址"},
                    "max_chars": {"type": "integer", "description": "最多提取字符数，默认8000，最多20000", "default": 8000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_extension",
            "description": "深入研究某个扩展插件的真实功能。读取扩展的 README.md 和脚本文件，了解它实际是干什么的。【重要】当你不确定某个扩展/功能/模型是做什么的时，必须先调用此工具调查，绝对不能编造！",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "扩展名称或关键词，如 'seedvr2', 'controlnet', 'adetailer', 'rembg'",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_webui",
            "description": "探索 WebUI 的目录结构和内置功能。当用户问到 WebUI 有什么功能、某个目录是干什么的时，调用此工具调查。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对路径，默认为 '.' 表示 webui 根目录。如 'extensions', 'models', 'modules'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": "读取 WebUI 根目录内的代码、配置和文本文件内容。用户要求查看文件、解释脚本、核对配置或读取 README 时使用；路径必须相对 WebUI 根目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "WebUI 根目录内的相对文件路径，如 'modules/shared.py'、'extensions/插件名/README.md'"},
                    "max_chars": {"type": "integer", "description": "最多读取的字符数，默认 12000，最大 30000", "default": 12000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
            "description": "提取并分析 WebUI 根目录内的文档内容。支持 TXT、MD、JSON、CSV、DOCX，以及环境具备解析组件时的 PDF。用户要求总结、审阅、解释或从文档中找信息时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "WebUI 根目录内的相对文档路径"},
                    "question": {"type": "string", "description": "用户希望从文档中回答的问题或分析角度", "default": ""},
                    "max_chars": {"type": "integer", "description": "最多提取的字符数，默认 16000，最大 30000", "default": 16000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_extensions",
            "description": "汇总所有已安装插件的启用状态、关键脚本和 README 摘要。用户问“我的插件都能做什么”“帮我梳理所有插件”时优先使用；若需了解一个特定插件，再使用 research_extension。",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_readme": {"type": "boolean", "description": "是否包含每个插件的 README 摘要，默认 true", "default": True},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_workspace_file",
            "description": "修复 WebUI 根目录内的代码、配置或文本文件。必须先读取当前文件，再用 old_text 精确替换为 new_text；修改前自动创建可恢复备份。仅在用户授权修复、安装或调整 WebUI 时使用，不修改密钥文件，不允许路径越界。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "WebUI 根目录内的相对文件路径"},
                    "old_text": {"type": "string", "description": "从当前文件读取到的、需要替换的完整原文，必须恰好匹配一处"},
                    "new_text": {"type": "string", "description": "替换后的完整文本"},
                    "reason": {"type": "string", "description": "本次修复原因"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_workspace",
            "description": "对 WebUI 根目录或指定 Python 文件执行无破坏诊断（Python 语法/编译检查），用于定位并验证报错。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "WebUI 根目录内的 Python 文件或目录，默认为 '.'"},
                },
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "txt2img": txt2img_tool,
    "img2img": img2img_tool,
    "switch_model": switch_model_tool,
    "update_settings": update_settings_tool,
    "upscale": upscale_tool,
    "generate_with_lora": generate_with_lora_tool,
    "list_models": list_models_tool,
    "list_vae": list_vae_tool,
    "list_text_encoders": list_text_encoders_tool,
    "list_controlnet": list_controlnet_tool,
    "get_model_guide": get_model_guide_tool,
    "set_vae": set_vae_tool,
    "set_text_encoder": set_text_encoder_tool,
    "set_model_components": set_model_components_tool,
    "list_samplers": list_samplers_tool,
    "list_upscalers": list_upscalers_tool,
    "get_current_settings": get_current_settings_tool,
    "list_loras": list_loras_tool,
    "video_keyframe_extract": video_keyframe_extract_tool,
    "trellis2_image_to_3d": trellis2_image_to_3d_tool,
    "video_to_frames": video_to_frames_tool,
    "h3_video_generate": h3_video_generate_tool,
    "dreamina_video_generate": dreamina_video_generate_tool,
    "stitch_images": stitch_images_tool,
    "list_preprocessors": list_preprocessors_tool,
    "apply_adetailer": apply_adetailer_tool,
    "remove_background": remove_background_tool,
    "layer_separation": layer_separation_tool,
    "api_image_edit": api_image_edit_tool,
    "api_image_generate": api_image_generate_tool,
    "edit_image": edit_image_tool,
    "change_background": change_background_tool,
    "web_search": web_search_tool,
    "web_read_url": web_read_url_tool,
    "list_extensions": list_extensions_tool,
    "research_extension": research_extension_tool,
    "explore_webui": explore_webui_tool,
    "read_workspace_file": read_workspace_file_tool,
    "analyze_document": analyze_document_tool,
    "audit_extensions": audit_extensions_tool,
    "repair_workspace_file": repair_workspace_file_tool,
    "diagnose_workspace": diagnose_workspace_tool,
}

