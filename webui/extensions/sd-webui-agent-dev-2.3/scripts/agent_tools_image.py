# =============================================================================
# Agent Tools — 本地生图与图像处理工具
# =============================================================================

import os
import sys
import subprocess
import time
import traceback

import numpy as np
import torch
from PIL import Image

from modules import shared, scripts, sd_models, postprocessing
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    process_images,
)
from modules.shared import device, opts
from backend.args import dynamic_args

from scripts.agent_config import (
    load_config, _get_current_checkpoint, _get_sampler, _scan_model_dir,
)
from scripts.agent_tools_common import _ui_aspect_size
from scripts.agent_tools_models import MODEL_GUIDE, set_model_components_tool


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

    if width is None and height is None:
        width, height = _ui_aspect_size(cfg)
    elif width is None:
        width = cfg["default_width"]
    elif height is None:
        height = cfg["default_height"]

    p = StableDiffusionProcessingTxt2Img(
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps or cfg["default_steps"],
        cfg_scale=cfg_scale or cfg["default_cfg_scale"],
        width=width,
        height=height,
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


def remove_background_tool(image, mode="auto", bg_color=None):
    """智能抠图工具：去除图片背景，返回透明背景的图片。

    基于 See-Through-SAM 扩展，支持三种模式：
    - auto: 智能抠图（BiRefNet 自动分割主体，推荐）

    参数:
        image: 输入图片路径
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

        else:
            return None, {"status": "error", "error": f"未知模式: {mode}，当前仅支持 auto"}

        if not results:
            return None, {"status": "error", "error": "抠图未产生结果"}

        return results[0], meta
    except Exception as e:
        return None, {"status": "error", "error": str(e)}


def layer_separation_tool(image, output_format="psd"):
    """See-Through-SAM 图层分离工具。

    这是独立的图层分离入口，不是抠图或去背景。
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
