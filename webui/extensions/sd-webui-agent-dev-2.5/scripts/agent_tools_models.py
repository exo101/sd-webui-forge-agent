# =============================================================================
# Agent Tools — 模型管理工具（checkpoint / VAE / TE / 设置查询）
# =============================================================================

import os
import traceback

from modules import shared, scripts, sd_models, sd_samplers

from scripts.agent_config import _get_current_checkpoint, _scan_model_dir


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

        # 根治 VAE/TE 残留：切换模型后，根据新模型架构自动清除不兼容的额外模块
        # 例如从 Krea2(qwen_image_vae) 切到 SDXL 时，必须清除 VAE，否则 SDXL 加载失败
        cleared = []
        try:
            model_lower = matched.lower()
            for key, guide in MODEL_GUIDE.items():
                if key in model_lower:
                    rec_te, rec_vae = pick_guide_components(matched, guide)
                    # 如果该模型不需要额外 VAE/TE（如 SDXL），清除已有同类型额外模块
                    if not rec_vae or not rec_te:
                        from modules_forge.main_entry import modules_change
                        current = list(getattr(shared.opts, "forge_additional_modules", []) or [])
                        new_modules = []
                        for m in current:
                            m_path = m if isinstance(m, str) else str(m)
                            m_lower = m_path.lower().replace("\\", "/")
                            is_vae = "/vae/" in m_lower
                            is_te = "/text_encoder/" in m_lower
                            if not rec_vae and is_vae:
                                cleared.append(os.path.basename(m_path))
                                continue
                            if not rec_te and is_te:
                                cleared.append(os.path.basename(m_path))
                                continue
                            new_modules.append(m_path)
                        if cleared:
                            modules_change(new_modules, preset=None, save=True, refresh=True)
                            print(f"[Agent] 切换模型后自动清除不兼容组件: {cleared}")

                        # 同时清除 sd_vae（传统 WebUI VAE 设置）
                        if not rec_vae:
                            sd_vae = getattr(shared.opts, "sd_vae", None)
                            if sd_vae and str(sd_vae).lower() not in ("none", "auto", "automatic", ""):
                                cleared.append(f"sd_vae={sd_vae}")
                                shared.opts.sd_vae = "None"
                    break
        except Exception as e:
            print(f"[Agent] 自动清除额外模块时出错（非致命）: {e}")

        # 获取当前生效的 TE/VAE
        effective_modules = [os.path.basename(m) for m in (getattr(shared.opts, "forge_additional_modules", []) or [])]
        print(f"[Agent] 切换模型: {matched} | 附加模块: {effective_modules}")
        result = {
            "status": "success",
            "message": f"已切换模型为: {matched}",
            "model": matched,
            "effective_modules": effective_modules,
            "tip": "模型已切换，下次生图时自动加载（含已设置的 TE/VAE），无需重启"
        }
        if cleared:
            result["auto_cleared"] = cleared
            result["tip"] = f"模型已切换，已自动清除不兼容的额外组件: {cleared}，下次生图时使用模型自带组件"
        return result
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
    "qwen_image_2.1": {
        "name": "Qwen-Image-2.1",
        "description": "Qwen-Image-2.1 上下文编辑模型：参考图编码进入潜空间，支持多图参考上下文编辑",
        "preset_arch": "qwen21",
        # 默认（普通版主模型）组件；仅主模型文件名带 _ds 后缀（DiffSynth 版）时才用 ds_recommended_*
        "recommended_te": ["qwen3vl_8b_int8_convrot.safetensors"],
        "recommended_vae": ["qwen_image_2.1_vae_bf16.safetensors"],
        "ds_recommended_te": ["qwen3vl_8b_int8_convrot_ds.safetensors"],
        "ds_recommended_vae": ["qwen_image_2.1_vae_bf16_ds.safetensors"],
        "loras": [],
        "tips": "组件由主模型文件名后缀决定：普通版（qwen_image_2.1_int8_convrot.safetensors）用 Qwen3-VL 8B INT8 文本编码器 + Qwen-Image-2.1 VAE；只有主模型带 _ds 后缀（DiffSynth 版）才用对应 _ds 组件。Simple 调度器 / CFG 1.0 / 默认 30 步；可搭配多图参考插件传入参考图"
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


def pick_guide_components(model_name, guide):
    """按主模型文件名后缀选择指南中的推荐 TE/VAE（与 backend/loader.py 的后端路由规则一致）。

    规则：只有主模型文件名带 _ds 后缀（DiffSynth 版）时才取 ds_recommended_* 组件；
    否则一律取普通 recommended_* 组件。

    返回: (te_name 或 None, vae_name 或 None)
    """
    is_ds = "_ds" in os.path.basename(str(model_name or "")).lower()
    if is_ds:
        te = (guide.get("ds_recommended_te") or guide.get("recommended_te") or [None])[0]
        vae = (guide.get("ds_recommended_vae") or guide.get("recommended_vae") or [None])[0]
    else:
        te = (guide.get("recommended_te") or [None])[0]
        vae = (guide.get("recommended_vae") or [None])[0]
    return te, vae


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
            # 推荐为空时明确告知不需要额外组件，避免 Agent 乱设 VAE/TE 导致加载失败
            result = {"status": "success", "guide": matched}
            if not matched.get("recommended_vae"):
                result["vae_notice"] = "此模型不需要额外 VAE（checkpoint 自带）。必须清除已设置的额外 VAE，否则模型加载失败！"
            if not matched.get("recommended_te"):
                result["te_notice"] = "此模型不需要额外文本编码器（checkpoint 自带）。必须清除已设置的额外 TE，否则模型加载失败！"
            return result
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
        vae_name: VAE 文件名，如 'flux2-vae.safetensors' 或 'qwen_image_vae.safetensors'。
                  传 'none'、'clear'、'default' 或空字符串可清除已设置的 VAE（使用 checkpoint 自带 VAE）。
    """
    try:
        # 支持清除 VAE
        if not vae_name or vae_name.strip().lower() in ("none", "clear", "default", "remove", "取消", "清除", "默认"):
            new_values, _ = _forge_update_additional_modules("vae", None)
            print("[Agent] 已清除 VAE 设置，使用 checkpoint 自带 VAE")
            return {
                "status": "success",
                "message": "已清除 VAE 设置，使用 checkpoint 自带 VAE",
                "vae": None,
                "effective_modules": [os.path.basename(m) for m in new_values],
                "tip": "VAE 已清除，下次生图时使用模型自带 VAE"
            }

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
        te_name: 文本编码器文件名，如 'qwen3vl_4b_fp8_scaled.safetensors'。
                 传 'none'、'clear'、'default' 或空字符串可清除已设置的 TE（使用 checkpoint 自带 TE）。
    """
    try:
        # 支持清除 TE
        if not te_name or te_name.strip().lower() in ("none", "clear", "default", "remove", "取消", "清除", "默认"):
            new_values, _ = _forge_update_additional_modules("te", None)
            print("[Agent] 已清除文本编码器设置，使用 checkpoint 自带 TE")
            return {
                "status": "success",
                "message": "已清除文本编码器设置，使用 checkpoint 自带 TE",
                "text_encoder": None,
                "effective_modules": [os.path.basename(m) for m in new_values],
                "tip": "TE 已清除，下次生图时使用模型自带文本编码器"
            }

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
        #    组件按主模型文件名后缀选择：带 _ds 用 DiffSynth 组件，否则用普通组件
        guide_matched = False
        if model_name and not te_name and not vae_name:
            model_lower = model_name.lower()
            for key, guide in MODEL_GUIDE.items():
                if key in model_lower:
                    te_name, vae_name = pick_guide_components(model_name, guide)
                    guide_matched = True
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
            # 只要没指定新的 TE/VAE，就清除旧的同类型模块（SDXL 等自带组件的模型不需要额外 TE/VAE）
            if not te_name and is_te:
                continue
            if not vae_name and is_vae:
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

        # 同步 sd_vae：如果模型不需要额外 VAE（如 SDXL），清除传统 VAE 设置
        # 只要没有指定新 VAE（guide 匹配且推荐为空，或用户未传 vae_name），就清除 sd_vae
        sd_vae_cleared = False
        if not vae_name:
            sd_vae = getattr(shared.opts, "sd_vae", None)
            if sd_vae and str(sd_vae).lower() not in ("none", "auto", "automatic", ""):
                shared.opts.sd_vae = "None"
                sd_vae_cleared = True

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
