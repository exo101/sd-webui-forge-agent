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
from pathlib import Path

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
        upscaler_name: 放大模型名称，如 'R-ESRGAN 4x+', 'ESRGAN_4x', 'Lanczos' 等。None 则用第一个。
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

        # 选择放大模型
        selected = upscaler_names[0]
        if upscaler_name:
            for un in upscaler_names:
                if upscaler_name.lower() in un.lower():
                    selected = un
                    break

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
    """ADetailer 脸部修复（如果安装了 aadetailer 插件）。

    参数:
        image: 输入图片 (PIL Image)
        prompt: 修复时使用的提示词（可选）
        model_name: 检测模型名称（可选）
    返回: (images_list, info_dict)
    """
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image")

        # 尝试导入 aadetailer
        try:
            from adetailer import run
            result = run(
                image=image,
                prompt=prompt,
                model_name=model_name or None,
            )
            if isinstance(result, list):
                return result, {"status": "success", "count": len(result)}
            return [result], {"status": "success"}
        except ImportError:
            # aadetailer 不可用，返回原图
            return [image], {"status": "skipped", "note": "ADetailer 插件未安装，返回原图"}
    except Exception as e:
        return [image], {"status": "error", "error": str(e), "note": "返回原图"}


def remove_background_tool(image, mode="auto", points=None, bg_color=None):
    """智能抠图工具：去除图片背景，返回透明背景的图片。

    基于 See-Through-SAM 扩展，支持四种模式：
    - auto: 智能抠图（BiRefNet 自动分割主体，推荐）
    - point_click: 点选分割（用户指定坐标点，需要 points 参数）
    - cleanup: 图像清理（去除指定区域/物体，需要 mask）
    - layer_separation: 图层分离（将图片分离为多个图层）

    参数:
        image: 输入图片路径
        mode: 'auto' | 'point_click' | 'cleanup' | 'layer_separation'，默认 auto
        points: 点选分割的坐标列表 [[x,y],...]，仅 mode=point_click 时需要
        bg_color: 背景颜色，如 'white'/'black'/'transparent'，默认 transparent
    """
    try:
        from PIL import Image
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

        elif mode == "layer_separation":
            # 图层分离：使用 LayerDiff
            try:
                sys.path.insert(0, os.path.join(scripts.basedir(), "extensions", "sd-webui-see-through-sam", "see-through", "inference", "scripts"))
                from scene_segmenter import SceneSegmenter
                segmenter = SceneSegmenter()
                tmp_in = _save_pil_to_tempfile(img, "layer_sep_in.png")
                layer_paths = segmenter.create_layer_images(tmp_in)
                if layer_paths:
                    for p in layer_paths:
                        if os.path.isfile(p):
                            results.append(Image.open(p))
            except Exception as e:
                return None, {"status": "error", "error": f"图层分离失败: {e}"}

        else:
            return None, {"status": "error", "error": f"未知模式: {mode}，支持 auto/point_click/cleanup/layer_separation"}

        if not results:
            return None, {"status": "error", "error": "抠图未产生结果"}

        return results[0], meta
    except Exception as e:
        return None, {"status": "error", "error": str(e)}


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
            # 没有 Klein，回退用当前模型直接 img2img
            return img2img_tool(image, instruction, strength=strength)

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
        #    Klein 是多模态编辑模型，需要配合 image_stitch 插件使用 txt2img，
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
        return None, {"status": "error", "error": str(e)}


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


def explore_webui_tool(path="."):
    """探索 WebUI 的目录结构和内置功能。

    当用户问到 WebUI 有什么功能、某个目录是干什么的时，调用此工具调查。

    参数:
        path: 相对路径，默认为 '.' 表示 webui 根目录
    """
    try:
        from modules import paths
        # Forge Neo 路径获取：优先用 script_path（= webui 根目录），多级回退
        webui_root = getattr(paths, "script_path", None)
        if not webui_root or not os.path.isdir(webui_root):
            # 回退：从 extensions_dir 推导（extensions_dir 的上一级 = webui 根）
            ext_dir = getattr(paths, "extensions_dir", None)
            if ext_dir and os.path.isdir(os.path.dirname(ext_dir)):
                webui_root = os.path.dirname(ext_dir)
            else:
                # 最后回退：基于 scripts.basedir() 上三级
                webui_root = os.path.dirname(os.path.dirname(os.path.dirname(scripts.basedir())))
        print(f"[Agent] explore_webui: webui_root={webui_root}")

        target = os.path.normpath(os.path.join(webui_root, path))
        # 安全限制：不允许访问 webui 之外的目录
        if not target.startswith(os.path.normpath(webui_root)):
            return {"status": "error", "error": "路径越界：只能访问 WebUI 目录内的内容"}

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
            "description": "图片放大。当用户要求放大图片、提高分辨率时使用。需要用户上传图片或刚生成的图片。",
            "parameters": {
                "type": "object",
                "properties": {
                    "upscaler_name": {"type": "string", "description": "放大模型名称，如 R-ESRGAN 4x+。None 则自动选第一个"},
                    "scale": {"type": "number", "description": "放大倍数，默认2"},
                    "resize_w": {"type": "integer", "description": "指定宽度 (覆盖 scale)"},
                    "resize_h": {"type": "integer", "description": "指定高度 (覆盖 scale)"},
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
            "description": "智能抠图：去除图片背景，返回透明背景图。基于 See-Through-SAM 扩展，支持智能抠图/点选分割/图像清理/图层分离四种模式。用户说'去除背景'/'抠图'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["auto", "point_click", "cleanup", "layer_separation"], "description": "auto=智能抠图(推荐) / point_click=点选分割 / cleanup=图像清理 / layer_separation=图层分离", "default": "auto"},
                    "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "点选分割的坐标列表 [[x,y],...]，仅 point_click 模式需要"},
                    "bg_color": {"type": "string", "description": "背景颜色: transparent/white/black，默认 transparent", "default": "transparent"},
                },
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
    "video_to_frames": video_to_frames_tool,
    "stitch_images": stitch_images_tool,
    "list_preprocessors": list_preprocessors_tool,
    "apply_adetailer": apply_adetailer_tool,
    "remove_background": remove_background_tool,
    "edit_image": edit_image_tool,
    "change_background": change_background_tool,
    "list_extensions": list_extensions_tool,
    "research_extension": research_extension_tool,
    "explore_webui": explore_webui_tool,
}