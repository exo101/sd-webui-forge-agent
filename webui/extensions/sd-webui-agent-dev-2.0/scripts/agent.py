# =============================================================================
# SD Webui Agent — AI 全能生图智能体（入口模块）
# 子模块: agent_config.py (配置/工具函数), agent_tools.py (工具/TOOLS),
#         agent_prompts.py (系统提示词)
# =============================================================================

import os
import io
import json
import re
import time
import base64
import traceback

import gradio as gr
from PIL import Image

from modules import shared, scripts, sd_models, script_callbacks
from modules.processing import process_images

# =============================================================================
# 子模块导入
# =============================================================================
from scripts.agent_config import (
    load_config, save_config, _save_pil_to_tempfile, _detect_local_llama,
    API_PROVIDERS, all_api_providers, provider_choices, normalize_base_url, provider_base_url,
    _local_detection_cache, _local_detection_time,
    _REGISTRY_AVAILABLE, get_registered_tools, get_tool_function,
)
from scripts.agent_tools import (
    TOOLS, TOOL_FUNCTIONS, MODEL_GUIDE,
    set_model_components_tool,
)
from scripts.agent_prompts import _get_system_prompt

# =============================================================================
# 智能体大脑 LLM 模型列表
# =============================================================================

LLM_MODELS_BY_PROVIDER = {
    "ModelScope": [
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-Flash-Next",
        "ZhipuAI/GLM-5.3",
        "ZhipuAI/GLM-5.3-Flash",
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "deepseek-ai/DeepSeek-V4-Pro",
        "moonshotai/Kimi-K3",
    ],
    "YoboxAI": [
        "gpt-5.4-mini",
        "gpt-5.6-luna",
        "gpt-5.5",
    ],
}

# =============================================================================
# 生成 API 模型列表
# =============================================================================

IMAGE_GENERATION_MODELS_BY_PROVIDER = {
    "ModelScope": [
        "krea/Krea-2-Turbo",
        "Tongyi-MAI/Z-Image",
        "Qwen/Qwen-Image-2512",
        "FireRedTeam/FireRed-Image-Edit-1.1",
        "Qwen/Qwen-Image-Edit-2511",
    ],
    "YoboxAI": [
        "banana2",
        "bananapro",
        "gpt-image-2",
    ],
    "ModelScope Space": [
        "Trellis.2-4B",
    ],
}

IMAGE_GENERATION_MODELS = [
    "banana2",
    "bananapro",
    "gpt-image-2",
    "krea/Krea-2-Turbo",
    "Tongyi-MAI/Z-Image",
    "Qwen/Qwen-Image-2512",
    "FireRedTeam/FireRed-Image-Edit-1.1",
    "Qwen/Qwen-Image-Edit-2511",
    "Trellis.2-4B",
]

VIDEO_GENERATION_MODELS = [
    "dreamina-seedance-2-0-hc",
    "dreamina-seedance-2-5-hc",
    "MiniMax-H3",
]


def _image_models_for_provider(provider):
    models = IMAGE_GENERATION_MODELS_BY_PROVIDER.get(str(provider or "").strip())
    return models or IMAGE_GENERATION_MODELS


def _models_for_provider(cfg, kind, provider, defaults):
    values = list(defaults)
    custom = (cfg.get("custom_models", {}) if isinstance(cfg, dict) else {}).get(kind, {})
    values.extend(custom.get(provider, []) if isinstance(custom, dict) else [])
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))


def _is_saved_image_model(cfg, model):
    if model in IMAGE_GENERATION_MODELS:
        return True
    custom = cfg.get("custom_models", {}).get("image", {}) if isinstance(cfg, dict) else {}
    return any(model in (values or []) for values in custom.values() if isinstance(values, list))


def _default_image_model_for_provider(provider, current_model=None):
    models = _image_models_for_provider(provider)
    current_model = str(current_model or "").strip()
    return current_model if current_model in models else models[0]


# =============================================================================
# @mention 系统：用户用 @标签 快速指定模型/功能
# =============================================================================

# @标签 → 动作映射
MENTION_MAP = {
    # 模型标签（type=model，值为 MODEL_GUIDE 的 key）
    "krea2": ("model", "krea2"),
    "klein": ("model", "flux-2-klein"),
    "klein9b": ("model", "flux-2-klein"),
    "anima": ("model", "anima"),
    "z_image": ("model", "z_image"),
    "zimage": ("model", "z_image"),
    "qwen": ("model", "qwen_image_edit"),
    "qwen-edit": ("model", "qwen_image_edit"),
    "XL": ("model", "xl"),
    "sdxl": ("model", "xl"),
    "illustrious": ("model", "illustrious"),

    # API 模型标签（切换 Forge 到对应 API 供应商和远程模型）
    "Qwen/Qwen-Image-Edit-2511": ("api_model", "Qwen/Qwen-Image-Edit-2511"),
    "FireRedTeam/FireRed-Image-Edit-1.1": ("api_model", "FireRedTeam/FireRed-Image-Edit-1.1"),
    "krea/Krea-2-Turbo": ("api_model", "krea/Krea-2-Turbo"),
    "Tongyi-MAI/Z-Image": ("api_model", "Tongyi-MAI/Z-Image"),
    "Qwen/Qwen-Image-2512": ("api_model", "Qwen/Qwen-Image-2512"),
    "banana2": ("api_model", "banana2"),
    "bananapro": ("api_model", "bananapro"),
    "gpt-image-2": ("api_model", "gpt-image-2"),

    # 本地工具标签（type=local_tool）：只调用本地扩展或本地处理，不调用 API。
    "智能抠图": ("local_tool", "本地智能抠图 / Local smart background removal：使用 InSPyReNet-Base，调用 remove_background(mode=auto)，禁止调用远程 API。"),
    "点选分割": ("local_tool", "本地点选分割 / Local point-click segmentation：使用 SAM，调用 remove_background(mode=point_click)，禁止调用远程 API。"),
    "图像清理": ("local_tool", "本地图像清理 / Local image cleanup：使用 LiteLaMA 清理流程，调用 remove_background(mode=cleanup)，禁止调用远程 API。"),
    "图层分离": ("local_tool", "本地图层分离 / Local layer separation：使用 See-Through LayerDiff，调用 layer_separation，禁止调用 SAM、remove_background 和远程 API。"),
    "视频关键帧": ("local_tool", "本地视频关键帧提取 / Local video keyframe extraction：使用 video_keyframe_extract，禁止调用远程 API。"),
    "TRELLIS图生3D": ("local_tool", "本地 TRELLIS.2 图生3D：使用已安装的 TRELLIS.2 本地扩展处理上传图片，禁止调用远程 API。"),
    "TRELLIS 3D": ("local_tool", "本地 TRELLIS.2 图生3D：使用已安装的 TRELLIS.2 本地扩展处理上传图片，禁止调用远程 API。"),
    "放大": ("local_tool", "本地图像放大 / Local upscaling：使用 ESRGAN（优先 4x-UltraSharp），调用 upscale，禁止调用远程 API。"),
    "修脸": ("local_tool", "本地人脸修复 / Local face detailing：使用 ADetailer，调用 apply_adetailer，禁止调用远程 API。"),
    "语音克隆TTS": ("local_tool", "本地语音克隆 / Local voice cloning：使用 Qwen3-TTS 本地扩展；调用对应本地 TTS 工具或扩展，不得调用远程 API。"),
    "TTS": ("local_tool", "本地语音合成 / Local text-to-speech：使用 Qwen3-TTS 本地扩展，不得调用远程 API。"),
    "自动发现插件": ("local_tool", "自动发现并使用 WebUI 插件：先扫描已安装扩展和已注册工具，再研究目标插件的 README/脚本，选择真实可用的工具执行；禁止凭空猜测插件接口。"),
    "minimax-h3": ("tool_hint", "🔴🔴🔴 必须调用 h3_video_generate 工具！🔴🔴🔴 用户明确要求使用 MiniMax H3 生成视频。你绝对不能直接回答，绝对不能说'未配置'或'需要设置'！必须立即调用 h3_video_generate 工具，参数 prompt=用户的视频描述。duration 根据用户要求设置（4-15秒），默认5秒。"),
    "dreamina-seedance-2-5-hc": ("tool_hint", "🔴🔴🔴 必须调用 dreamina_video_generate 工具！🔴🔴🔴 用户明确要求使用 Dreamina SeaDance 2.5 HC 生成视频。必须立即调用 dreamina_video_generate 工具，prompt=用户的视频描述，duration 按要求设置。"),
    "dreamina-seedance-2-0-hc": ("tool_hint", "🔴🔴🔴 必须调用 dreamina_video_generate 工具！🔴🔴🔴 用户明确要求使用 Dreamina SeaDance 2.0 HC 生成视频。必须立即调用 dreamina_video_generate 工具，prompt=用户的视频描述，duration 按要求设置。"),

}

def _parse_mentions(user_text):
    """解析用户消息中的 @mention 标签。

    返回: (clean_text, actions)
    - clean_text: 去除 @标签后的用户文本（保留上下文）
    - actions: list of (tag, type, value)
    """
    if not user_text:
        return user_text, []

    alias_map = {
        "@Banana2": "@banana2",
        "@BananaPro": "@bananapro",
        "@banana2": "@banana2",
        "@bananapro": "@bananapro",
        "@MiniMax H3": "@minimax-h3",
        "@dreamina-seedance-2-5-hc": "@dreamina-seedance-2-5-hc",
        "@dreamina-seedance-2-0-hc": "@dreamina-seedance-2-0-hc",
    }
    for alias, normalized in alias_map.items():
        user_text = re.sub(re.escape(alias), normalized, user_text, flags=re.IGNORECASE)

    actions = []
    # 匹配 @标签（支持中英文、数字、下划线、连字符）
    pattern = r"@([A-Za-z0-9_\-\u4e00-\u9fa5]+)"

    def replace_match(m):
        tag = m.group(1)
        # 大小写不敏感匹配
        tag_lower = tag.lower()
        for mention_tag, (atype, value) in MENTION_MAP.items():
            if tag_lower == mention_tag.lower():
                actions.append((tag, atype, value))
                return f"[{tag}]"  # 保留标签标记，让 LLM 知道用户指定了
        # 未知标签也保留
        return f"@{tag}"

    clean_text = re.sub(pattern, replace_match, user_text)
    return clean_text, actions


def _handle_model_mention(actions):
    """处理模型 @mention，自动切换模型。

    返回: (extra_system_note, switch_results)
    - extra_system_note: 注入给 LLM 的隐藏系统提示
    - switch_results: 切换结果列表
    """
    model_actions = [a for a in actions if a[1] == "model"]
    if not model_actions:
        return "", []

    notes = []
    results = []
    for tag, atype, guide_key in model_actions:
        try:
            guide = MODEL_GUIDE.get(guide_key, {})

            # 查找匹配的模型 — 关键词匹配，不写死文件名
            # 例如 @krea2 → 匹配标题含"krea2"的模型，@klein → 含"klein"的
            keyword = guide_key.lower()
            try:
                sd_models.list_models()
            except Exception:
                pass

            # 用关键词直接搜索 checkpoint title（比传文件名更可靠）
            matched_model = None
            if hasattr(sd_models, "get_closet_checkpoint_match"):
                matched_model = sd_models.get_closet_checkpoint_match(keyword)

            if not matched_model and hasattr(sd_models, "checkpoints_list") and sd_models.checkpoints_list:
                # 回退：遍历文件名匹配
                for info in sd_models.checkpoints_list.values():
                    if keyword in info.filename.lower():
                        matched_model = info
                        break

            if not matched_model:
                note = f"⚠️ 用户指定了 @{tag}，但未找到对应模型，请先安装该模型"
                notes.append(note)
                results.append({"tag": tag, "status": "not_found"})
                continue

            # 获取推荐的 TE/VAE
            te = guide.get("recommended_te", [None])[0] if guide.get("recommended_te") else None
            vae = guide.get("recommended_vae", [None])[0] if guide.get("recommended_vae") else None

            # 执行切换 — 传入 title 而非文件名，确保 get_closet_checkpoint_match 能匹配
            model_title = matched_model.title if hasattr(matched_model, "title") else matched_model.filename
            switch_result = set_model_components_tool(model_name=model_title, te_name=te, vae_name=vae)
            if switch_result.get("status") == "success":
                # 本地模型标签明确关闭 Forge API 模式，避免沿用上一次远程模型状态。
                try:
                    shared.opts.set("forge_model_mode", "local")
                    shared.opts.set("forge_api_model", "")
                except Exception:
                    pass
                model_name = guide.get("name", model_title)

                # 同步切换 UI preset + 更新 agent 配置默认值
                preset_arch = guide.get("preset_arch")
                if preset_arch and hasattr(shared.opts, "forge_preset"):
                    shared.opts.set("forge_preset", preset_arch)

                    # 从 WebUI 预设系统中读取实际参数值，写入 agent 配置
                    try:
                        from modules_forge.presets import STEPS, SAMPLERS, SCHEDULERS, CFG, PresetArch
                        arch = PresetArch[preset_arch]
                        preset_steps = STEPS.get(arch)
                        preset_sampler = SAMPLERS.get(arch)
                        preset_scheduler = SCHEDULERS.get(arch)
                        preset_cfg = CFG.get(arch)

                        # 更新 agent 配置中的默认值，使 txt2img_tool 等函数使用正确的预设参数
                        _cfg = load_config()
                        if preset_steps:
                            _cfg["default_steps"] = preset_steps
                        if preset_cfg is not None:
                            _cfg["default_cfg_scale"] = preset_cfg
                        save_config(_cfg)
                    except Exception as e:
                        print(f"[Agent] 读取预设参数失败: {e}")

                    try:
                        shared.opts.save(shared.config_filename)
                    except Exception:
                        pass
                    preset_note = f"（已应用 {preset_arch.upper()} 预设: steps={preset_steps}, CFG={preset_cfg}, sampler={preset_sampler}）"
                else:
                    preset_note = ""

                note = f"✅ 用户指定了 @{tag}，已自动切换到 {model_name} 模型，现在可以直接生图，无需再次切换模型。{preset_note}"
                notes.append(note)
                results.append({"tag": tag, "status": "success", "model": model_name, "file": matched_model.filename if hasattr(matched_model, "filename") else model_title, "preset": preset_arch})
            else:
                note = f"⚠️ 用户指定了 @{tag}，但模型切换失败: {switch_result.get('error', '未知错误')}"
                notes.append(note)
                results.append({"tag": tag, "status": "failed", "error": switch_result.get("error")})
        except Exception as e:
            note = f"⚠️ @{tag} 模型切换异常: {e}"
            notes.append(note)
            results.append({"tag": tag, "status": "error", "error": str(e)})

    return "\n".join(notes), results


def _activate_api_model_mentions(actions):
    """让 API 模型标签只切换模型 ID，供应商由用户设置决定。"""
    api_actions = [a for a in actions if a[1] == "api_model"]
    if not api_actions:
        return "", []

    cfg = load_config()
    notes = []
    results = []
    for tag, _, model_id in api_actions:
        try:
            from modules_forge.api_providers import set_session_api_key

            # 持久化到 agent_config.json
            cfg["image_model"] = model_id
            # 自动补全 base_url
            provider_name = str(cfg.get("image_api_provider") or "").strip()
            correct_url = provider_base_url(provider_name)
            if correct_url and normalize_base_url(cfg.get("image_base_url", "")) != correct_url:
                cfg["image_base_url"] = correct_url
            save_config(cfg)

            shared.opts.set("forge_model_mode", "api")
            provider_lower = provider_name.lower()
            provider_aliases = {
                "modelscope": "modelscope",
                "dashscope": "dashscope",
                "pixapi": "pixapi",
                "yoboxai": "yoboxai",
            }
            if provider_lower in provider_aliases:
                shared.opts.set("forge_api_provider", provider_aliases[provider_lower])
            shared.opts.set("forge_api_model", model_id)
            set_session_api_key(cfg.get("image_api_key", ""))

            notes.append(
                f"✅ 用户指定 @{tag}，已设置 API 模型={model_id}。"
                "供应商、Base URL 和 API Key 使用独立的生成 API 设置。"
            )
            results.append({"tag": tag, "status": "success", "model": model_id})
            print(f"[Agent] API 模型已设置: model={model_id}, base_url={cfg.get('image_base_url')}")
        except Exception as e:
            notes.append(f"⚠️ @{tag} API 模型启用失败: {e}")
            results.append({"tag": tag, "status": "error", "error": str(e)})

    return "\n".join(notes), results


def _handle_tool_mentions(actions):
    """处理工具 @mention，收集隐藏指令。"""
    hints = []
    for tag, atype, value in actions:
        if atype in ("tool_hint", "local_tool"):
            hints.append(f"[用户标签 @{tag}] {value}")
        elif atype == "api_model":
            hints.append(f"[用户标签 @{tag}] 用户指定 API 模型：{value}。Forge 已切换到该 API 模型；生图必须使用远程 API，不要切换本地 checkpoint。")
        elif atype == "stub":
            hints.append(f"[用户标签 @{tag}] {value}")
    return "\n".join(hints)


API_IMAGE_EDIT_MODELS = {
    "Qwen/Qwen-Image-Edit-2511",
    "FireRedTeam/FireRed-Image-Edit-1.1",
    "banana2",
    "bananapro",
    "gpt-image-2",
    "Trellis.2-4B",
}


def _configured_image_model():
    cfg = load_config()
    return str(cfg.get("image_model") or "").strip()


def _configured_video_model():
    cfg = load_config()
    return str(cfg.get("video_model") or "").strip()


def _is_configured_api_image_model(model_id):
    model = str(model_id or "").strip()
    if model in IMAGE_GENERATION_MODELS:
        return True
    cfg = load_config(resolve_local=False)
    custom = cfg.get("custom_models", {}).get("image", {}) if isinstance(cfg.get("custom_models"), dict) else {}
    return any(model in (values or []) for values in custom.values() if isinstance(values, list))


def _is_trellis2_api_model(model_id):
    return str(model_id or "").strip().lower() == "trellis.2-4b"


def _is_dreamina_video_model(model_id):
    return str(model_id or "").strip().lower() in {
        "dreamina-seedance-2-0-hc",
        "dreamina-seedance-2-5-hc",
    }


def _is_h3_video_model(model_id):
    return str(model_id or "").strip().lower() == "minimax-h3"


def _extract_requested_api_model(messages):
    """从本轮系统提示中提取用户明确指定的 API 模型。"""
    pattern = r"用户指定 API 模型：([^。\n]+)"
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content", ""))
        matches = re.findall(pattern, content)
        if matches:
            return matches[-1].strip()
    return ""


def _has_requested_local_model(messages):
    """只检查最新的系统消息是否包含本地模型切换标记。

    不能遍历所有历史消息：一旦用户曾经切换过本地模型（如 @klein），
    旧消息会永久存在，导致后续即使选择了 API 模型也被错误跳过路由。
    """
    latest_system = ""
    for msg in messages:
        if msg.get("role") == "system":
            latest_system = str(msg.get("content", ""))
    if not latest_system:
        return False
    # 最新系统消息中如果明确切换到了 API 模式，则不算本地模型请求
    if "API 模型" in latest_system and ("已切换" in latest_system or "已选择" in latest_system):
        return False
    return "用户指定了 @" in latest_system and "已自动切换到" in latest_system


def _should_protect_api_image_model(requested_api_model):
    return requested_api_model in API_IMAGE_EDIT_MODELS


def _has_requested_layer_separation(user_instruction):
    """图层分离使用专用工具，避免误路由到 SAM/抠图。"""
    text = str(user_instruction or "").lower()
    return any(token in text for token in ("图层分离", "分层", "psd 图层", "psd图层", "@图层分离"))


def _has_requested_trellis_local(user_instruction):
    text = str(user_instruction or "").lower()
    return (
        "@trellis图生3d" in text
        or "[trellis图生3d]" in text
        or "@trellis 3d" in text
        or "[trellis 3d]" in text
    )


def _has_requested_keyframe_local(user_instruction):
    text = str(user_instruction or "")
    return "@视频关键帧" in text or "[视频关键帧]" in text


def _coerce_api_image_edit_tool(tool_name, tool_args, requested_api_model, user_instruction):
    """用户指定 API 图像编辑模型时，禁止误回退到本地抠图/Klein 工具。"""
    if not _should_protect_api_image_model(requested_api_model):
        return tool_name, tool_args
    if tool_name == "api_image_edit":
        tool_args = dict(tool_args or {})
        tool_args.setdefault("model", requested_api_model)
        tool_args.setdefault("instruction", user_instruction)
        return tool_name, tool_args
    if tool_name not in ("remove_background", "edit_image", "change_background", "img2img"):
        return tool_name, tool_args

    coerced_args = dict(tool_args or {})
    instruction = coerced_args.get("instruction") or coerced_args.get("prompt") or user_instruction
    if tool_name == "remove_background":
        bg_color = str(coerced_args.get("bg_color") or "").strip().lower()
        mode = str(coerced_args.get("mode") or "").strip().lower()
        if bg_color:
            instruction = f"Change the image background to {bg_color}, preserve the subject exactly."
        elif mode == "auto":
            instruction = user_instruction
    elif tool_name == "change_background" and coerced_args.get("atmosphere"):
        instruction = user_instruction

    return "api_image_edit", {
        "model": requested_api_model,
        "instruction": instruction,
    }


def _coerce_generation_tool_by_selected_model(tool_name, tool_args, user_instruction):
    """Use the saved generation model selection as the routing source of truth."""
    tool_args = dict(tool_args or {})
    image_model = _configured_image_model()
    video_model = _configured_video_model()

    # 检查是否有与当前图像供应商匹配的可用 key
    _cfg = load_config()
    from scripts.agent_tools import _select_image_api_key
    _selected_key = _select_image_api_key(_cfg)
    _can_use_api = bool(_selected_key.strip()) or _is_trellis2_api_model(image_model)
    _is_api_model = _is_configured_api_image_model(image_model)

    # 诊断日志：路由决策可见
    print(f"[Agent] 路由检查: tool={tool_name}, image_model={image_model!r}, is_api_model={_is_api_model}, has_key={_can_use_api}, provider={_cfg.get('image_api_provider')}")

    # 没有 Key 也不能切换到本地工具或其他 API 模型；保持当前模型进入 API 工具，
    # 由 API 工具直接返回“未配置 Key”错误。
    if not _can_use_api and _is_api_model:
        tool_args["model"] = image_model
        if tool_name == "txt2img":
            return "api_image_generate", {"model": image_model, "prompt": tool_args.get("prompt") or user_instruction}
        if tool_name in ("edit_image", "change_background", "img2img", "api_image_edit"):
            return "api_image_edit", {"model": image_model, "instruction": tool_args.get("instruction") or tool_args.get("prompt") or user_instruction}
        print(f"[Agent] ⚠️ 当前 API 模型未配置 Key，保留工具调用并由工具返回错误: {image_model}")
        return tool_name, tool_args

    # API 图像模型由 UI/持久化配置决定。模型可能自行把 api_image_edit
    # 参数填成 banana2 等旧默认值，不能让该值覆盖用户当前选择。
    if _is_api_model and tool_name in ("api_image_edit", "api_image_generate"):
        tool_args["model"] = image_model
        if _is_trellis2_api_model(image_model) and tool_name == "api_image_generate":
            return "api_image_edit", {
                "model": image_model,
                "instruction": tool_args.get("prompt") or user_instruction,
            }
        print(f"[Agent] 强制使用当前选择的 API 图像模型: {image_model}")
        return tool_name, tool_args

    if _is_api_model and tool_name == "txt2img":
        prompt = tool_args.get("prompt") or user_instruction
        if _is_trellis2_api_model(image_model):
            return "api_image_edit", {"model": image_model, "instruction": prompt}
        print(f"[Agent] 路由: txt2img -> api_image_generate (model={image_model})")
        return "api_image_generate", {"model": image_model, "prompt": prompt}

    if _is_api_model and tool_name in ("edit_image", "change_background", "img2img"):
        instruction = tool_args.get("instruction") or tool_args.get("prompt") or user_instruction
        print(f"[Agent] 路由: {tool_name} -> api_image_edit (model={image_model})")
        return "api_image_edit", {"model": image_model, "instruction": instruction}

    if tool_name in ("h3_video_generate", "dreamina_video_generate"):
        prompt = tool_args.get("prompt") or user_instruction
        duration = tool_args.get("duration", 5)
        aspect_ratio = tool_args.get("aspect_ratio") or tool_args.get("ratio") or "16:9"
        if _is_dreamina_video_model(video_model):
            coerced = dict(tool_args)
            coerced.update({"prompt": prompt, "model": video_model, "duration": duration, "aspect_ratio": aspect_ratio})
            return "dreamina_video_generate", coerced
        if _is_h3_video_model(video_model):
            coerced = dict(tool_args)
            coerced.update({"prompt": prompt, "duration": duration, "aspect_ratio": aspect_ratio})
            return "h3_video_generate", coerced

    return tool_name, tool_args


# =============================================================================
# Agent 核心：流式对话 + Function Calling 循环
# =============================================================================

def _execute_tool(tool_name, tool_args, uploaded_image=None, uploaded_video=None, last_tool_images=None):
    """执行工具调用，返回 (result_string, images_list)。
    自动传入上传的图片/视频 + 上次工具生成的图片到对应参数。
    支持工具链：如 提取关键帧 → 拼接 → 放大。"""
    if last_tool_images is None:
        last_tool_images = []

    # 内置工具和动态注册工具都必须经过下面的参数注入流程。
    func = TOOL_FUNCTIONS.get(tool_name)
    registered_func = None
    if func is None and _REGISTRY_AVAILABLE:
        registered_func = get_tool_function(tool_name)
        func = registered_func
    if func is None:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False), []

    try:
        # ===== 自动注入图片参数（优先级：用户上传 > 上次工具生成） =====
        # 单图工具：需要自动注入上传图片或上一步输出图片。
        # layer_separation 虽然不在旧注释列表中，但同样必须接收 image。
        if tool_name in (
            "img2img", "upscale", "apply_adetailer", "remove_background",
            "layer_separation", "trellis2_image_to_3d", "edit_image", "change_background", "api_image_edit",
        ):
            # LLM 可能会生成空字符串或无效占位路径，也视为未提供图片。
            if not _normalize_image_path(tool_args.get("image")):
                if uploaded_image is not None:
                    tool_args["image"] = uploaded_image
                elif last_tool_images:
                    # 用上次生成的最后一张图
                    tool_args["image"] = last_tool_images[-1]
                else:
                    tool_args["image"] = None

        # 导入的 Gradio API 工具统一使用其示例中的图片参数名。
        if tool_name.startswith("gradio_api_") and not _normalize_image_path(tool_args.get("image")):
            if uploaded_image is not None:
                tool_args["image"] = uploaded_image
            elif last_tool_images:
                tool_args["image"] = last_tool_images[-1]

        # H3 视频生成：自动注入参考图（首帧/参考图）
        if tool_name == "h3_video_generate":
            if ("first_frame" not in tool_args or tool_args["first_frame"] is None) and \
               ("reference_image" not in tool_args or tool_args["reference_image"] is None):
                if uploaded_image is not None:
                    tool_args["first_frame"] = uploaded_image
                elif last_tool_images:
                    tool_args["first_frame"] = last_tool_images[-1]

        if tool_name == "dreamina_video_generate":
            if ("first_frame" not in tool_args or tool_args["first_frame"] is None) and \
               ("reference_image" not in tool_args or tool_args["reference_image"] is None):
                if uploaded_image is not None:
                    tool_args["first_frame"] = uploaded_image
                elif last_tool_images:
                    tool_args["first_frame"] = last_tool_images[-1]

        # 多图工具：stitch_images
        if tool_name == "stitch_images":
            if "images" not in tool_args or not tool_args["images"]:
                if last_tool_images:
                    tool_args["images"] = last_tool_images
                elif uploaded_image is not None:
                    tool_args["images"] = [uploaded_image]

        # 自动传入视频参数
        if tool_name in ("video_keyframe_extract", "video_to_frames") and uploaded_video is not None:
            if "video_path" not in tool_args:
                tool_args["video_path"] = uploaded_video

        result = func(**tool_args)

        if isinstance(result, tuple) and len(result) == 2:
            images, info = result
            # 工具可能返回 None 表示失败
            if images is None:
                return json.dumps({"status": "error", "info": info}, ensure_ascii=False), []
            # 部分图像/视频工具用空列表加 error 信息表示失败，不能误报为成功。
            if isinstance(info, dict) and info.get("error"):
                return json.dumps({"status": "error", "info": info}, ensure_ascii=False), []
            # 统一转为列表
            if not isinstance(images, list):
                images = [images]
            result_str = json.dumps({"status": "success", "info": info, "image_count": len(images)}, ensure_ascii=False)
            return result_str, images

        return json.dumps({"status": "success", "data": result}, ensure_ascii=False, default=str), []

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[Agent] 工具执行失败 [{tool_name}]: {error_msg}")
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), []


def _extract_text_from_history_content(content):
    """从 history 的 content 中提取文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    # 图片 dict {"path": ...} 没有文本
    if isinstance(content, dict) and "path" in content:
        return ""
    return str(content)


def _normalize_image_path(image_value):
    """把上传控件里的图片统一转成可读取的文件路径。"""
    if image_value is None:
        return None
    if isinstance(image_value, str):
        return image_value if os.path.isfile(image_value) else None
    if isinstance(image_value, dict):
        path = image_value.get("path")
        return path if path and os.path.isfile(path) else None
    if isinstance(image_value, Image.Image):
        return _save_pil_to_tempfile(image_value)
    return None


def _normalize_image_paths(image_value):
    """把单图或多图上传值统一转成可读取的文件路径列表。"""
    if image_value is None:
        return []
    if isinstance(image_value, (list, tuple)):
        paths = []
        for item in image_value:
            path = _normalize_image_path(item)
            if path:
                paths.append(path)
        return paths
    path = _normalize_image_path(image_value)
    return [path] if path else []


def _latest_image_paths_from_history(history, limit=5):
    """Find recent image files already shown in the chat."""
    paths = []
    for item in reversed(history or []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        path = _normalize_image_path(content)
        if not path:
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            paths.append(path)
            if len(paths) >= limit:
                break
    return paths


def _image_to_compressed_base64(img_path, max_size=1024, quality=85):
    """将图片压缩后转 base64 data URL，避免大 payload 触发上游限流。

    限制最长边为 max_size，转 JPEG 质量 quality。若压缩后反而更大则回退原图。
    返回 (data_url, orig_size, compressed_size)；失败返回 (None, 0, 0)。
    """
    try:
        file_size = os.path.getsize(img_path)
        img = Image.open(img_path)
        # 等比缩小
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        # JPEG 不支持透明通道，统一转 RGB
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        compressed = buf.getvalue()
        # 如果压缩后更大（小图），回退原始 PNG 编码
        if len(compressed) >= file_size:
            with open(img_path, "rb") as f:
                compressed = f.read()
            mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
        else:
            mime = "image/jpeg"
        b64 = base64.b64encode(compressed).decode("utf-8")
        return f"data:{mime};base64,{b64}", file_size, len(compressed)
    except Exception as e:
        print(f"[Agent] ⚠️ 图片压缩失败，回退原图编码: {e}")
        try:
            file_size = os.path.getsize(img_path)
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
            return f"data:{mime};base64,{b64}", file_size, file_size
        except Exception:
            return None, 0, 0


def chat_stream(history, uploaded_image=None, uploaded_video=None):
    """流式聊天生成器。从 history 中提取最后一条用户消息。

    Yields: (history_update, status_message)
    """
    cfg = load_config()
    _refresh_registered_agent_tools()

    # 从 history 提取最后一条用户文本消息（跳过纯图片消息）
    user_message = ""
    for h in reversed(history):
        if isinstance(h, dict) and h.get("role") == "user":
            text = _extract_text_from_history_content(h.get("content", ""))
            if text:
                user_message = text
                break

    if not user_message:
        yield history, "⚠️ 请输入消息"
        return

    uploaded_image_list = uploaded_image if isinstance(uploaded_image, (list, tuple)) else ([uploaded_image] if uploaded_image is not None else [])
    recent_history_image_paths = _latest_image_paths_from_history(history)
    if not uploaded_image_list and recent_history_image_paths:
        uploaded_image_list = recent_history_image_paths
    primary_uploaded_image = uploaded_image_list[0] if uploaded_image_list else None

    # 下拉框选择的本地工具必须确定性执行，不能交给 LLM 自行猜测是否调用。
    forced_tool = None
    forced_args = {}
    if _has_requested_trellis_local(user_message):
        forced_tool = "trellis2_image_to_3d"
        forced_args = {"image": _normalize_image_path(primary_uploaded_image)}
    elif _has_requested_keyframe_local(user_message):
        forced_tool = "video_keyframe_extract"
        if isinstance(uploaded_video, dict):
            forced_args = {"video_path": uploaded_video.get("path")}
        else:
            forced_args = {"video_path": uploaded_video}
    if forced_tool:
        if forced_tool == "trellis2_image_to_3d" and not _normalize_image_path(primary_uploaded_image):
            final_history = list(history)
            final_history.append({"role": "assistant", "content": "TRELLIS图生3D需要先上传一张图片。"})
            yield final_history, "⚠️ 请先上传图片"
            return
        if forced_tool == "video_keyframe_extract":
            video_path = forced_args.get("video_path")
            if not video_path or not os.path.isfile(video_path):
                final_history = list(history)
                final_history.append({"role": "assistant", "content": "视频关键帧需要先上传一个可读取的视频文件。"})
                yield final_history, "⚠️ 请先上传视频"
                return
        yield history, f"🔧 正在执行本地工具: {forced_tool}..."
        result_str, images = _execute_tool(forced_tool, forced_args, primary_uploaded_image, uploaded_video, [])
        try:
            result_data = json.loads(result_str)
        except Exception:
            result_data = {}
        final_history = list(history)
        if result_data.get("status") == "success":
            info = result_data.get("info") or {}
            detail = info.get("glb_path") or info.get("output_dir") or info.get("extracted") or "完成"
            final_history.append({"role": "assistant", "content": f"本地工具 {forced_tool} 完成：{detail}"})
            glb_path = info.get("glb_path")
            if glb_path and os.path.isfile(glb_path):
                final_history.append({"role": "assistant", "content": {"path": glb_path, "alt_text": "TRELLIS.2 GLB 三维模型"}})
            for i, img in enumerate(images or []):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    final_history.append({"role": "assistant", "content": {"path": img_path, "alt_text": f"本地工具输出 {i + 1}"}})
            yield final_history, f"✅ {forced_tool} 完成"
        else:
            info = result_data.get("info") or result_data
            error = info.get("error") if isinstance(info, dict) else str(info)
            final_history.append({"role": "assistant", "content": f"本地工具 {forced_tool} 失败：{error}"})
            yield final_history, f"❌ {forced_tool} 失败：{error}"
        return

    if _has_requested_layer_separation(user_message) and primary_uploaded_image is not None:
        separation_tool = "layer_separation"
        yield history, f"🔧 正在执行: {separation_tool}..."
        result_str, images = _execute_tool(
            separation_tool,
            {"output_format": "psd"},
            primary_uploaded_image,
            uploaded_video,
            [],
        )
        try:
            result_data = json.loads(result_str)
        except Exception:
            result_data = {}
        if result_data.get("status") == "success":
            info = result_data.get("info") or {}
            final_history = list(history)
            final_history.append({
                "role": "assistant",
                "content": f"图层分离完成。输出目录：{info.get('output_dir', '图层分离输出目录')}",
            })
            for i, img in enumerate(images or []):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    final_history.append({
                        "role": "assistant",
                        "content": {"path": img_path, "alt_text": f"分离图层预览 {i + 1}"},
                    })
            yield final_history, f"✅ {separation_tool} 完成"
        else:
            info = result_data.get("info") or result_data
            error = info.get("error") if isinstance(info, dict) else str(info)
            final_history = list(history)
            final_history.append({"role": "assistant", "content": f"图层分离失败：{error}"})
            yield final_history, f"❌ {separation_tool} 失败：{error}"
        return

    # 构建 OpenAI 消息格式
    # Gradio history 中图片用 {"path": filepath, "alt_text": "..."} 格式
    # 需要转换为 OpenAI 的 image_url (base64) 格式
    messages = [{"role": "system", "content": _get_system_prompt(cfg.get("model", ""))}]
    uploaded_image_paths = _normalize_image_paths(uploaded_image)
    context_image_paths = recent_history_image_paths if not uploaded_image_paths else []
    uploaded_image_path = uploaded_image_paths[0] if uploaded_image_paths else None
    uploaded_video_path = uploaded_video if isinstance(uploaded_video, str) else (uploaded_video.get("path") if isinstance(uploaded_video, dict) else None)
    history_has_image = False
    history_has_video = False
    for item in history:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, dict) and content.get("path"):
            if not history_has_image and content["path"] == uploaded_image_path:
                history_has_image = True
            if not history_has_video and content["path"] == uploaded_video_path:
                history_has_video = True

    # 合并连续的同角色消息（特别是 user 的 text + image）
    pending_user_content = []
    image_count = 0

    def _flush_pending_user():
        nonlocal pending_user_content, image_count
        if pending_user_content:
            if len(pending_user_content) == 1 and isinstance(pending_user_content[0], str):
                # 纯文本：可以直接用字符串
                messages.append({"role": "user", "content": pending_user_content[0]})
            else:
                # 混合内容：必须全部用 typed dict 格式（OpenAI API 要求）
                typed_content = []
                for item in pending_user_content:
                    if isinstance(item, str):
                        typed_content.append({"type": "text", "text": item})
                    else:
                        typed_content.append(item)  # 已经是 image_url dict
                messages.append({"role": "user", "content": typed_content})
            pending_user_content = []

    for h in history:
        if not isinstance(h, dict):
            continue
        role = h.get("role", "user")
        content = h.get("content", "")

        if role == "system":
            # 系统提示消息（如 @mention 注入的自动操作提示）
            # 本地 LLM 的 chat template 只允许开头有一条 system 消息，
            # 有多条会报 "System message must be at the beginning"。
            # 因此合并到第一条 system 消息中，而不是追加新消息。
            _flush_pending_user()
            if isinstance(content, str) and content.strip():
                # 合并到第一条 system 消息中
                if len(messages) > 0 and messages[0]["role"] == "system":
                    messages[0]["content"] += "\n\n" + content
                else:
                    messages.append({"role": "system", "content": content})
        elif role == "user":
            if isinstance(content, str):
                pending_user_content.append(content)
            elif isinstance(content, dict) and "path" in content:
                # 图片消息：读取文件转 base64
                try:
                    img_path = content["path"]
                    if not os.path.isfile(img_path):
                        print(f"[Agent] ⚠️ 图片文件不存在: {img_path}")
                        continue
                    data_url, orig_size, comp_size = _image_to_compressed_base64(img_path)
                    if data_url:
                        pending_user_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                        image_count += 1
                        print(f"[Agent] ✅ 图片已加载: {os.path.basename(img_path)} ({orig_size//1024}KB → {comp_size//1024}KB)")
                except Exception as e:
                    print(f"[Agent] ❌ 读取图片失败: {e}")
        else:
            # assistant 消息
            _flush_pending_user()
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            # 图片文件稍后作为当前用户上下文注入，避免模型把图片当成旧助手文本。

    _flush_pending_user()

    for context_image_path in context_image_paths[-3:]:
        try:
            data_url, orig_size, comp_size = _image_to_compressed_base64(context_image_path)
            if data_url:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是聊天上下文中最近生成/展示的图片，若我要求继续修改图片，请默认使用它作为参考图。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                })
                image_count += 1
                print(f"[Agent] ✅ 上下文图片已注入: {os.path.basename(context_image_path)} ({orig_size//1024}KB → {comp_size//1024}KB)")
        except Exception as e:
            print(f"[Agent] ❌ 注入上下文图片失败: {e}")

    for extra_image_path in uploaded_image_paths:
        if history_has_image and extra_image_path == uploaded_image_path:
            continue
        try:
            data_url, orig_size, comp_size = _image_to_compressed_base64(extra_image_path)
            if data_url:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }]
                })
                image_count += 1
                print(f"[Agent] ✅ 上传图片已注入: {os.path.basename(extra_image_path)} ({orig_size//1024}KB → {comp_size//1024}KB)")
        except Exception as e:
            print(f"[Agent] ❌ 注入上传图片失败: {e}")

    if uploaded_video_path and not history_has_video:
        # 仅保留路径由下游视频工具使用；模型本身不直接消费视频二进制。
        print(f"[Agent] ✅ 参考视频已保留: {os.path.basename(uploaded_video_path)}")

    # 调试日志：确认最终消息结构
    print(f"[Agent] 构建消息完成: {len(messages)} 条消息, 其中图片 {image_count} 张")
    for i, m in enumerate(messages):
        c = m.get("content", "")
        if isinstance(c, str):
            preview = c[:80] + ("..." if len(c) > 80 else "")
            print(f"  msg[{i}] role={m['role']} content='{preview}'")
        elif isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        parts.append(f"[image_url len={len(url)}]")
                    else:
                        parts.append(f"[{item.get('type','?')}]")
                else:
                    parts.append(f"[{type(item).__name__}]")
            print(f"  msg[{i}] role={m['role']} parts={parts}")

    # 添加 assistant 占位消息
    assistant_message = {"role": "assistant", "content": ""}
    new_history = list(history)
    new_history.append(assistant_message)

    reasoning_text = ""
    answer_text = ""
    done_reasoning = False
    pending_images = []  # 最终展示的图片
    pending_videos = []   # 最终展示的视频路径
    pending_files = []    # 最终展示的可下载文件（如 PSD）
    last_tool_images = list(recent_history_image_paths) if not uploaded_image_paths else []  # 上次工具生成的图片，用于工具链传递
    requested_api_model = _extract_requested_api_model(messages)
    requested_local_model = _has_requested_local_model(messages)

    try:
        from openai import OpenAI
        chat_model = cfg["model"]
        chat_base_url = cfg["base_url"]
        # 诊断日志：确认 LLM 配置
        _key_preview = cfg.get("api_key") or ""
        _key_mask = _key_preview[:8] + "..." + _key_preview[-4:] if len(_key_preview) > 12 else ("(空)" if not _key_preview else "***")
        print(f"[Agent] LLM 客户端初始化: model={chat_model}, base_url={chat_base_url}, api_key={_key_mask}")
        client = OpenAI(api_key=cfg["api_key"], base_url=chat_base_url, max_retries=3)

        for iteration in range(cfg["max_tool_iterations"]):
            # 注意: Qwen3 thinking mode 不支持 tool_choice="required"（会报 400），
            # 因此始终使用 "auto"；@minimax-h3 等强制调用场景已通过 system prompt
            # 中的 "🔴🔴🔴 必须调用" 指令约束，效果等价。
            current_tool_choice = "auto"

            # 针对 429 上游限流的显式重试（指数退避：3s → 6s → 12s）
            stream = None
            for retry_attempt in range(3):
                try:
                    stream = client.chat.completions.create(
                        model=chat_model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice=current_tool_choice,
                        max_tokens=int(cfg.get("max_completion_tokens", 32768)),
                        stream=True,
                    )
                    break
                except Exception as api_err:
                    is_rate_limit = (
                        getattr(api_err, "status_code", None) == 429
                        or api_err.__class__.__name__ == "RateLimitError"
                    )
                    if not is_rate_limit or retry_attempt == 2:
                        raise
                    wait_sec = 3 * (2 ** retry_attempt)
                    print(f"[Agent] 对话模型限流(429)，第 {retry_attempt + 1}/3 次重试，等待 {wait_sec}s...")
                    yield new_history, f"⏳ 服务繁忙，{wait_sec}秒后重试...（第 {retry_attempt + 1}/3 次）"
                    time.sleep(wait_sec)

            current_answer = ""
            tool_calls_accumulator = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # reasoning_content
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_text += reasoning
                    new_history[-1] = {
                        "role": "assistant",
                        "content": f"**思考过程：**\n{reasoning_text}\n\n---\n\n{answer_text}" if not done_reasoning else answer_text
                    }
                    yield new_history, "思考中..."

                # content
                content = delta.content
                if content:
                    if not done_reasoning and reasoning_text:
                        done_reasoning = True
                        answer_text += "\n\n"
                    current_answer += content
                    answer_text += content
                    new_history[-1] = {"role": "assistant", "content": answer_text}
                    yield new_history, "生成回复中..."

                # tool_calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if tc_delta.index is not None else 0
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                        tc = tool_calls_accumulator[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += tc_delta.function.arguments

            if tool_calls_accumulator:
                assistant_tool_calls = []
                for idx in sorted(tool_calls_accumulator.keys()):
                    tc = tool_calls_accumulator[idx]
                    assistant_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                    })

                messages.append({
                    "role": "assistant",
                    "content": current_answer or None,
                    "tool_calls": assistant_tool_calls,
                })

                for tc in assistant_tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    original_tool_name = tool_name
                    requested_layer_separation = _has_requested_layer_separation(user_message)
                    # 检查配置中是否有 API 模型（用户在设置区保存的优先于历史本地标记）
                    _cfg_img_model_now = _configured_image_model()
                    _cfg_has_api_model_now = _is_configured_api_image_model(_cfg_img_model_now)
                    if requested_layer_separation and tool_name in ("remove_background", "segment_anything", "sam"):
                        tool_name = "layer_separation"
                        tool_args = {"output_format": tool_args.get("output_format", "psd")}
                        print(f"[Agent] 用户明确要求图层分离，已阻止 {original_tool_name}，强制使用 See-Through layer_separation")
                    elif requested_local_model and not _cfg_has_api_model_now and tool_name == "api_image_edit":
                        tool_name = "edit_image"
                        tool_args = {"instruction": tool_args.get("instruction") or user_message}
                        print("[Agent] 用户指定本地模型，已阻止 api_image_edit，改用本地 edit_image")
                    elif requested_local_model and not _cfg_has_api_model_now and tool_name == "api_image_generate":
                        # 用户选择了本地模型（如 @z_image），不能用 API 生图，强制改回本地 txt2img
                        prompt = tool_args.get("prompt") or user_message
                        tool_name = "txt2img"
                        tool_args = {"prompt": prompt}
                        print("[Agent] 用户指定本地模型，已阻止 api_image_generate，改用本地 txt2img")
                    else:
                        tool_name, tool_args = _coerce_api_image_edit_tool(
                            tool_name, tool_args, requested_api_model, user_message
                        )
                        # 只有当用户明确选择了本地模型 AND 配置中没有 API 模型时，才跳过 API 路由
                        # 如果配置了 API 模型（用户在设置区保存的），优先路由到 API
                        _cfg_img_model = _configured_image_model()
                        _cfg_has_api_model = _is_configured_api_image_model(_cfg_img_model)
                        if not requested_local_model or _cfg_has_api_model:
                            if requested_local_model and _cfg_has_api_model:
                                print(f"[Agent] 配置了 API 模型 {_cfg_img_model}，优先使用 API 路由，忽略历史本地模型标记")
                            tool_name, tool_args = _coerce_generation_tool_by_selected_model(
                                tool_name, tool_args, user_message
                            )
                    if tool_name != original_tool_name:
                        print(f"[Agent] 已按当前生成模型设置修正工具: {original_tool_name} -> {tool_name}")

                    yield new_history, f"🔧 正在执行: {tool_name}..."

                    tool_uploaded_image = uploaded_image_list if tool_name == "api_image_edit" else primary_uploaded_image
                    result_str, images = _execute_tool(
                        tool_name, tool_args, tool_uploaded_image, uploaded_video, last_tool_images
                    )
                    # 更新工具链图片（用于下游工具自动注入）
                    if images:
                        last_tool_images = images

                    # 解析工具结果，提取可下载文件（PSD 等）和视频
                    psd_path = None
                    video_path = None
                    glb_path = None
                    try:
                        result_data = json.loads(result_str)
                        info = result_data.get("info", result_data.get("data", {}))
                        if isinstance(info, dict):
                            psd_path = info.get("psd_path")
                            video_path = info.get("video_path")
                            glb_path = info.get("glb_path")
                    except Exception:
                        pass

                    # 图层分离有 PSD 产物时，只展示 PSD 下载，不展示图层预览图
                    if psd_path and os.path.isfile(psd_path):
                        pending_files.append(psd_path)
                    elif glb_path and os.path.isfile(glb_path):
                        pending_files.append(glb_path)
                    else:
                        pending_images.extend(images)

                    # 提取视频路径（h3_video_generate 等视频工具）
                    if video_path and os.path.isfile(video_path):
                        pending_videos.append(video_path)

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})

                    if psd_path and os.path.isfile(psd_path):
                        layer_count = 0
                        try:
                            result_data = json.loads(result_str)
                            layer_count = result_data.get("info", {}).get("layer_count", 0)
                        except Exception:
                            pass
                        yield new_history, f"✅ {tool_name} 完成，生成 {layer_count} 个图层，PSD 已就绪"
                    elif images:
                        yield new_history, f"✅ {tool_name} 完成，生成 {len(images)} 张图片"
                    elif pending_videos and len(pending_videos) > 0:
                        yield new_history, f"✅ {tool_name} 完成，生成视频"
                    else:
                        yield new_history, f"✅ {tool_name} 完成"

                continue

            break

        # 最终回答 + 图片 + 视频 + 可下载文件（PSD 等）
        # Gradio 5 Chatbot type='messages' 不支持列表混合格式，
        # 所以文本和图片/视频/文件分成多条 assistant 消息
        if pending_images or pending_videos or pending_files:
            # 文本消息
            final_text = answer_text if answer_text else "已为你生成结果："
            new_history[-1] = {"role": "assistant", "content": final_text}
            # 每条图片单独一条消息，用 path dict 格式
            for i, img in enumerate(pending_images):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    new_history.append({
                        "role": "assistant",
                        "content": {"path": img_path, "alt_text": f"生成的图片 {i+1}"}
                    })
            # 每条视频单独一条消息，用 path dict 格式
            for i, vpath in enumerate(pending_videos):
                new_history.append({
                    "role": "assistant",
                    "content": {"path": vpath, "alt_text": f"生成的视频 {i+1}"}
                })
            # 每个可下载文件单独一条消息（PSD 等），Gradio 会自动渲染为下载附件
            for fpath in pending_files:
                fname = os.path.basename(fpath)
                new_history.append({
                    "role": "assistant",
                    "content": {"path": fpath, "alt_text": fname}
                })
            yield new_history, "✅ 完成"
        else:
            new_history[-1] = {"role": "assistant", "content": answer_text}
            yield new_history, "✅ 完成"

    except Exception as e:
        if getattr(e, "status_code", None) == 429 or e.__class__.__name__ == "RateLimitError":
            error_msg = (
                "⏳ 对话模型服务繁忙（HTTP 429），上游负载已饱和。"
                f"当前对话模型: {chat_model}，端点: {chat_base_url}。"
                "已自动重试 3 次仍未成功，请稍后再试，或切换其他对话模型/供应商。"
            )
        elif getattr(e, "status_code", None) == 401 or e.__class__.__name__ in ("AuthenticationError", "APIConnectionError"):
            _ck = cfg.get("api_key") or ""
            _ck_mask = _ck[:8] + "..." + _ck[-4:] if len(_ck) > 12 else ("(空)" if not _ck else "***")
            error_msg = (
                "❌ Agent 对话模型认证失败（HTTP 401）。\n"
                f"当前对话模型: {chat_model}，端点: {chat_base_url}，API Key: {_ck_mask}\n"
                "这通常是因为：\n"
                "1. API Key 为空或无效 — 请在「Agent 大脑设置」中重新填写 API Key 并点保存\n"
                "2. API Key 与供应商不匹配 — 请确认供应商选择正确（如 ModelScope 的 key 不能用于 YoboxAI）\n"
                "3. API Key 已过期或额度用尽\n"
                "注意：这里的 Key 是 Agent 大脑（LLM 对话）用的，不是图像生成 Key。"
            )
        elif getattr(e, "status_code", None) == 403 or e.__class__.__name__ == "PermissionDeniedError":
            error_msg = (
                "❌ Agent 对话模型被上游拒绝（HTTP 403）。"
                f"当前对话模型: {chat_model}，端点: {chat_base_url}。"
                "这与图片上传或图像模型无关。请确认该供应商的 API Key 有权调用此模型的 chat/completions 接口，"
                "并确认 Base URL 是该供应商的对话接口地址。"
            )
        else:
            error_msg = f"❌ 出错了: {str(e)}"
        print(f"[Agent] 错误: {traceback.format_exc()}")
        new_history[-1] = {"role": "assistant", "content": error_msg}
        yield new_history, "❌ 错误"


# =============================================================================
# UI
# =============================================================================

def on_ui_tabs():
    # ===== WebUI 启动时恢复 Forge session API key =====
    # session key 是内存中的，重启后丢失；forge_model_mode 可能被持久化为 "api"
    # 导致重启后 Forge 处于 API 模式但无有效 key，首次生图 401
    try:
        _boot_cfg = load_config(resolve_local=False)
        _restore_key = _boot_cfg.get("image_api_key") or _boot_cfg.get("api_key") or ""
        _image_model = str(_boot_cfg.get("image_model") or "").strip()
        _current_mode = getattr(shared.opts, "forge_model_mode", "") or ""
        try:
            from modules_forge.api_providers import set_session_api_key
            set_session_api_key(_restore_key)
        except Exception as _e:
            print(f"[Agent] 恢复 session API key 失败: {_e}")
        # ===== 安全同步：如果 agent_config 中 key 为空，同步清除其他持久化位置 =====
        if not _restore_key:
            # 清除 config.json 中的 forge_api_key
            try:
                if hasattr(shared.opts, 'forge_api_key') and shared.opts.forge_api_key:
                    shared.opts.set('forge_api_key', '')
                    shared.opts.save(shared.opts.data)
                    print('[Agent] 安全同步：已清除 config.json 中残留的 forge_api_key')
            except Exception as _e:
                print(f'[Agent] 安全同步清除 forge_api_key 失败: {_e}')
            # 清除 forge-h3-studio 的 minimax_api_key
            try:
                h3_cfg_path = os.path.join(scripts.basedir(), 'extensions', 'forge-h3-studio', 'data', 'config.json')
                if os.path.exists(h3_cfg_path):
                    with open(h3_cfg_path, 'r', encoding='utf-8') as f:
                        h3_cfg = json.load(f)
                    if h3_cfg.get('minimax_api_key'):
                        h3_cfg['minimax_api_key'] = ''
                        with open(h3_cfg_path, 'w', encoding='utf-8') as f:
                            json.dump(h3_cfg, f, ensure_ascii=False, indent=2)
                        print('[Agent] 安全同步：已清除 forge-h3-studio 中残留的 minimax_api_key')
            except Exception as _e:
                print(f'[Agent] 安全同步清除 h3-studio key 失败: {_e}')
        # 如果 image_api_key 为空且当前处于 API 模式，回退到 local 避免无 key 401。
        # TRELLIS.2 也需要 ModelScope Token，因此同样回退。
        if not _restore_key and _current_mode.lower() == "api":
            try:
                shared.opts.set("forge_model_mode", "local")
                print("[Agent] image_api_key 为空，已将 forge_model_mode 从 api 回退为 local")
            except Exception:
                pass
    except Exception as _e:
        print(f"[Agent] 启动恢复配置异常: {_e}")

    cfg_init = load_config()
    with gr.Blocks(
        analytics_enabled=False,
        css="""
        #agent-chatbot .image-container,
        #agent-chatbot .message-wrap .image-container,
        #agent-chatbot .message-content .image-container,
        #agent-chatbot .prose .image-container,
        #agent-chatbot [data-testid="chatbot"] .image-container {
            display: inline-flex !important;
            width: min(240px, 72vw) !important;
            min-height: 120px !important;
            max-width: min(240px, 72vw) !important;
            max-height: 240px !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            padding: 8px !important;
            border: 1px solid rgba(148, 163, 184, 0.32) !important;
            border-radius: 8px !important;
            background: rgba(15, 23, 42, 0.38) !important;
        }
        #agent-chatbot .message-content:has(.image-container),
        #agent-chatbot .message-wrap:has(.image-container),
        #agent-chatbot .prose:has(.image-container) {
            width: fit-content !important;
            max-width: min(280px, 78vw) !important;
        }
        #agent-chatbot img,
        #agent-chatbot .image-container img,
        #agent-chatbot .message-wrap img,
        #agent-chatbot .message-content img,
        #agent-chatbot .prose img,
        #agent-chatbot [data-testid="chatbot"] img {
            display: block !important;
            width: auto !important;
            height: auto !important;
            max-width: 224px !important;
            max-height: 224px !important;
            object-fit: contain !important;
        }
        #agent-chatbot .message img,
        #agent-chatbot .bubble-wrap img,
        #agent-chatbot .chatbot img,
        #agent-chatbot [data-testid="bot"] img {
            width: auto !important;
            height: auto !important;
            max-width: 224px !important;
            max-height: 224px !important;
            object-fit: contain !important;
        }
        """,
    ) as agent_interface:
        # 辅助函数：生成 API 供应商切换时自动更新 Base URL（需在 UI 定义前声明）
        def on_provider_change(provider, current_url):
            cfg = load_config(resolve_local=False)
            info = all_api_providers(cfg).get(str(provider or "").strip(), {})
            url = info.get("base_url") or provider_base_url(provider) or normalize_base_url(current_url)
            return url

        # 辅助函数：LLM 大脑供应商切换时更新 Base URL 和模型下拉列表（仅显示该供应商模型）
        def on_llm_provider_change(provider, current_url, current_model):
            cfg = load_config(resolve_local=False)
            info = all_api_providers(cfg).get(str(provider or "").strip(), {})
            url = info.get("base_url") or provider_base_url(provider) or normalize_base_url(current_url)
            llm_models = _models_for_provider(cfg, "llm", provider, LLM_MODELS_BY_PROVIDER.get(provider, []))
            # 如果当前模型不在新供应商列表中，回退到该供应商第一个模型
            new_value = current_model if current_model in llm_models else (llm_models[0] if llm_models else current_model)
            return url, gr.update(choices=llm_models, value=new_value, allow_custom_value=True)

        initial_api_choices = [
            ("不使用 API 模型", ""),
            ("🤖 YoboxAI · banana2", "YoboxAI|banana2"),
            ("🤖 YoboxAI · bananapro", "YoboxAI|bananapro"),
            ("🤖 YoboxAI · gpt-image-2", "YoboxAI|gpt-image-2"),
            ("🧩 ModelScope · Krea-2-Turbo", "ModelScope|krea/Krea-2-Turbo"),
            ("🧩 ModelScope · Z-Image", "ModelScope|Tongyi-MAI/Z-Image"),
            ("🧩 ModelScope · Qwen-Image-2512", "ModelScope|Qwen/Qwen-Image-2512"),
            ("🧩 ModelScope · FireRed-Image-Edit", "ModelScope|FireRedTeam/FireRed-Image-Edit-1.1"),
            ("🧩 ModelScope · Qwen-Image-Edit", "ModelScope|Qwen/Qwen-Image-Edit-2511"),
            ("🧊 ModelScope Space · TRELLIS.2", "ModelScope Space|Trellis.2-4B"),
            ("🎬 YoboxAI · dreamina-seedance-2-0", "video|dreamina-seedance-2-0-hc"),
            ("🎬 YoboxAI · dreamina-seedance-2-5", "video|dreamina-seedance-2-5-hc"),
            ("🎬 YoboxAI · MiniMax-H3", "video|MiniMax-H3"),
        ]
        for _provider, _models in cfg_init.get("custom_models", {}).get("image", {}).items():
            initial_api_choices.extend([(_provider + " · " + _m, _provider + "|" + _m) for _m in _models])
        for _provider, _models in cfg_init.get("custom_models", {}).get("video", {}).items():
            initial_api_choices.extend([(_provider + " · " + _m, "video|" + _m) for _m in _models])

        gr.HTML("""
        <div style="text-align:center; margin-bottom: 10px;">
            <h2 style="color: #c084fc;">绘梦智能体助手 — AI 全能生图智能体</h2>
            <p style="color: #9ca3af; font-size: 14px;">
                基于 Qwen3.8 · 可配置任意 OpenAI 兼容 API · 可指挥我生图/改图/换模型/调参数/放大
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话", height=520, type="messages", show_label=False,
                    elem_id="agent-chatbot",
                )

                # 模型选择下拉列表 — 替换原来的 @快捷指令按钮
                with gr.Row():
                    local_model_select = gr.Dropdown(
                        label="🎯 本地模型/工具（选择后发送消息即可使用）",
                        choices=[
                            ("不使用本地模型", ""),
                            ("🎨 @krea2 多风格美学", "@krea2"),
                            ("✏️ @klein 编辑模型", "@klein"),
                            ("🌸 @anima 二次元动漫", "@anima"),
                            ("👤 @z_image 精致人像", "@z_image"),
                            ("📝 @qwen 文字编辑", "@qwen"),
                            ("🐱 @XL 初代动漫", "@XL"),
                            ("✂️ @智能抠图 InSPyReNet", "@智能抠图"),
                            ("📚 @图层分离 See-Through", "@图层分离"),
                            ("🧊 @TRELLIS图生3D", "@TRELLIS图生3D"),
                            ("🎬 @MiniMax H3 工作台", "@MiniMax H3"),
                            ("🧠 自动发现并使用所有插件", "@自动发现插件"),
                            ("🎞️ @视频关键帧", "@视频关键帧"),
                            ("🔍 @放大 ESRGAN", "@放大"),
                            ("😊 @修脸 ADetailer", "@修脸"),
                        ],
                        value="",
                        interactive=True,
                        scale=1,
                    )
                    api_model_select = gr.Dropdown(
                        label="🌐 API 图像模型（选择后立即生效）",
                        choices=initial_api_choices,
                        value=(
                            f"{cfg_init.get('image_api_provider')}|{cfg_init.get('image_model')}"
                            if _is_saved_image_model(cfg_init, cfg_init.get('image_model')) else ""
                        ),
                        interactive=True,
                        scale=1,
                    )
                # 旧版固定列表已由 initial_api_choices 统一生成
                """choices=[
                            ("不使用 API 模型", ""),
                            ("🤖 YoboxAI · banana2", "YoboxAI|banana2"),
                            ("🤖 YoboxAI · bananapro", "YoboxAI|bananapro"),
                            ("🤖 YoboxAI · gpt-image-2", "YoboxAI|gpt-image-2"),
                            ("🧩 ModelScope · Krea-2-Turbo", "ModelScope|krea/Krea-2-Turbo"),
                            ("🧩 ModelScope · Z-Image", "ModelScope|Tongyi-MAI/Z-Image"),
                            ("🧩 ModelScope · Qwen-Image-2512", "ModelScope|Qwen/Qwen-Image-2512"),
                            ("🧩 ModelScope · FireRed-Image-Edit", "ModelScope|FireRedTeam/FireRed-Image-Edit-1.1"),
                            ("🧩 ModelScope · Qwen-Image-Edit", "ModelScope|Qwen/Qwen-Image-Edit-2511"),
                            ("🧊 ModelScope Space · TRELLIS.2", "ModelScope Space|Trellis.2-4B"),
                            ("🎬 YoboxAI · dreamina-seedance-2-0", "video|dreamina-seedance-2-0-hc"),
                            ("🎬 YoboxAI · dreamina-seedance-2-5", "video|dreamina-seedance-2-5-hc"),
                            ("🎬 YoboxAI · MiniMax-H3", "video|MiniMax-H3"),
                        ],
                        value=(
                            f"{cfg_init.get('image_api_provider')}|{cfg_init.get('image_model')}"
                            if cfg_init.get("image_model") in IMAGE_GENERATION_MODELS else ""
                        ),
                        interactive=True,
                        scale=1,
                    )"""

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入消息",
                        placeholder="例如：画一只可爱的橘猫，坐在窗台上晒太阳...",
                        scale=4, show_label=False,
                        elem_id="agent_chat_input",
                    )
                    send_btn = gr.Button("📤 发送", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ 清空", scale=1)

                status = gr.Textbox(label="状态", value="就绪", interactive=False, show_label=False)

            # 隐藏的 State 保存上传的图片/视频（避免被 UI 清除影响事件链）
            state_image = gr.State(None)
            state_video = gr.State(None)

            with gr.Column(scale=1):
                upload_image = gr.File(
                    label="参考图片 (可选，可多选)",
                    file_count="multiple",
                    file_types=["image"],
                    height=180,
                )
                clear_image_btn = gr.Button("清除图片", size="sm")

                upload_video = gr.Video(
                    label="参考视频 (可选, 用于关键帧提取)",
                    height=180,
                    sources=["upload"],
                )
                clear_video_btn = gr.Button("清除视频", size="sm")

                with gr.Accordion("⚙️ API 设置", open=False):
                    gr.Markdown("### Agent 大脑设置")
                    gr.Markdown("供应商列表显示完整 API URL。需要接入其他平台时，可在下方添加自定义供应商。")
                    with gr.Row():
                        custom_provider_name = gr.Textbox(label="自定义供应商名称", placeholder="例如：我的绘图平台", scale=1)
                        custom_provider_url = gr.Textbox(label="完整 API URL", placeholder="https://api.example.com/v1", scale=2)
                        add_provider_btn = gr.Button("➕ 添加供应商", size="sm", scale=1)
                    provider_manage_status = gr.Textbox(show_label=False, interactive=False)
                    local_mode = gr.Checkbox(
                        label="启用本地 llama-server（仅勾选时检测；关闭则仅用云端）",
                        value=cfg_init.get("local_mode", False)
                    )
                    with gr.Row():
                        detect_btn = gr.Button("🔍 重新检测本地模型", size="sm")
                        detect_status = gr.Textbox(show_label=False, interactive=False, scale=3)
                    api_provider = gr.Dropdown(
                        label="API 供应商",
                        choices=provider_choices(cfg_init),
                        value=cfg_init.get("api_provider", "ModelScope"),
                    )
                    with gr.Row():
                        api_key = gr.Textbox(label="API Key（云端用，本地可留空）", value=cfg_init["api_key"], type="text", scale=4, elem_id="agent_api_key_input")
                        clear_api_key_btn = gr.Button("清空 API Key", size="sm", scale=1)
                    base_url = gr.Textbox(label="Base URL", value=cfg_init["base_url"], visible=False)
                    current_provider = cfg_init.get("api_provider", "ModelScope")
                    # 模型列表仅显示当前供应商的模型，切换供应商时动态更新
                    llm_choices = _models_for_provider(cfg_init, "llm", current_provider, LLM_MODELS_BY_PROVIDER.get(current_provider, []))
                    model_name = gr.Dropdown(
                        label="模型 ID（智能体大脑）",
                        choices=llm_choices if llm_choices else [cfg_init["model"]],
                        value=cfg_init["model"],
                        allow_custom_value=True,
                        info="先选择上方 API 供应商，此处显示该供应商的模型；也可手动输入自定义模型 ID",
                    )
                    save_settings_btn = gr.Button("💾 保存设置", variant="secondary")
                    settings_status = gr.Textbox(show_label=False, interactive=False)

                    gr.Markdown("### 图像/视频生成 API 设置（独立于 Agent 大脑）")
                    image_api_provider = gr.Dropdown(
                        label="生成 API 供应商（选择对应平台的 Key，勿混用）",
                        choices=provider_choices(cfg_init),
                        value=cfg_init.get("image_api_provider", "YoboxAI"),
                    )
                    with gr.Row():
                        image_api_key = gr.Textbox(
                            label="生成 API Key（图像模型选择在首页）",
                            value=cfg_init.get("image_api_key") or cfg_init.get("video_api_key", ""),
                            type="text",
                            scale=4,
                            elem_id="agent_image_api_key_input",
                        )
                        clear_image_api_key_btn = gr.Button("清空生成 Key", size="sm", scale=1)
                    image_base_url = gr.Textbox(
                        label="生成 API Base URL",
                        value=cfg_init.get("image_base_url") or cfg_init.get("video_base_url", "https://api.yoboxai.com/v1"),
                        visible=False,
                    )
                    # 切换供应商时自动更新 Base URL
                    image_api_provider.change(
                        fn=on_provider_change,
                        inputs=[image_api_provider, image_base_url],
                        outputs=[image_base_url],
                    )
                    save_image_settings_btn = gr.Button("💾 保存生成 API 设置", variant="secondary")
                    image_settings_status = gr.Textbox(show_label=False, interactive=False)
                    with gr.Row():
                        custom_model_kind = gr.Dropdown(
                            label="模型类型", choices=[("Agent 大脑", "llm"), ("图像", "image"), ("视频", "video")], value="image", scale=1
                        )
                        custom_model_id = gr.Textbox(label="自定义模型 ID", placeholder="例如：provider/model-name", scale=2)
                        add_model_btn = gr.Button("➕ 添加模型 ID", size="sm", scale=1)
                    custom_model_status = gr.Textbox(show_label=False, interactive=False)


                with gr.Accordion("📖 使用提示", open=False):
                    gr.Markdown("""
                    **🎨 生图：**
                    - "画一只可爱的橘猫坐在窗台上"
                    - "生成赛博朋克风格的城市夜景，4K分辨率"

                    **🖼️ 图生图：** 上传参考图后说
                    - "把这张图变成水彩画风格"
                    - "放大这张图2倍" / "修复人脸"

                    **🎬 视频处理：** 上传视频后说
                    - "提取关键帧" / "截取5帧"
                    - "每秒截取一帧"

                    **🔧 指挥操作：**
                    - "切换到 SDXL 模型"
                    - "把步数改成30，尺寸改成1024x1024"
                    - "把这几张图拼起来"

                    **📋 查询：**
                    - "有哪些模型？" / "当前设置是什么？" / "有哪些插件？"
                    """)

        # ===== 事件绑定 =====

        def on_image_upload(img):
            """图片上传时保存到 State。"""
            return img

        def on_video_upload(vid):
            """视频上传时保存到 State。"""
            return vid

        def prepare_message(message, history, image, video, local_model_value="", api_model_value=""):
            """把用户消息加入 history，清空输入框，返回新 history。
            Gradio 5 Chatbot type='messages' 不支持列表混合格式，
            所以文本和图片分成两条消息。
            同时解析 @mention 标签，自动切换模型/注入工具提示。
            下拉列表选择的本地模型/API 模型会自动注入。"""
            new_history = list(history)

            # ===== 下拉列表自动注入 =====
            prefix = ""
            api_switched = False  # 标记本次是否切换了 API 模型
            selected_api_model = ""

            # 本地模型/工具下拉 — 自动 prepend @标签
            if local_model_value and str(local_model_value).strip():
                prefix += str(local_model_value).strip() + " "

            # API 模型下拉 — 格式 "provider|model"，直接切换
            if api_model_value and "|" in str(api_model_value):
                parts = str(api_model_value).split("|", 1)
                provider = parts[0].strip()
                model = parts[1].strip()
                if provider == "video":
                    # 视频模型 — 只保存配置，不切换 Forge API
                    try:
                        cfg = load_config()
                        cfg["video_model"] = model
                        save_config(cfg)
                        print(f"[Agent] 视频模型已切换: {model}")
                    except Exception as e:
                        print(f"[Agent] 视频模型切换失败: {e}")
                elif provider and model:
                    try:
                        cfg = load_config()
                        cfg["image_api_provider"] = provider
                        cfg["image_model"] = model
                        # 自动补全 base_url（如果为空或与供应商不匹配）
                        provider_info = all_api_providers(cfg).get(provider, {})
                        correct_url = provider_info.get("base_url") or provider_base_url(provider)
                        if correct_url and normalize_base_url(cfg.get("image_base_url", "")) != correct_url:
                            cfg["image_base_url"] = correct_url
                        if correct_url and normalize_base_url(cfg.get("video_base_url", "")) != correct_url:
                            cfg["video_base_url"] = correct_url
                        save_config(cfg)
                        # 切换 Forge API 设置
                        from modules_forge.api_providers import set_session_api_key
                        shared.opts.set("forge_model_mode", "api")
                        provider_aliases = {
                            "modelscope": "modelscope",
                            "dashscope": "dashscope",
                            "pixapi": "pixapi",
                            "yoboxai": "yoboxai",
                        }
                        p_lower = provider.lower()
                        if p_lower in provider_aliases:
                            shared.opts.set("forge_api_provider", provider_aliases[p_lower])
                        shared.opts.set("forge_api_model", model)
                        set_session_api_key(cfg.get("image_api_key", ""))
                        print(f"[Agent] API 图像模型已切换: provider={provider}, model={model}")
                        api_switched = True  # 标记切换成功
                        selected_api_model = f"{provider} · {model}"
                    except Exception as e:
                        print(f"[Agent] API 模型切换失败: {e}")

            if prefix:
                message = prefix + message

            # ===== @mention 处理 =====
            clean_text, actions = _parse_mentions(message)

            # 处理模型 @mention（自动切换）
            model_note, switch_results = _handle_model_mention(actions)
            # 处理 API 模型 @mention（切换 Forge API 模式/供应商/模型）
            api_model_note, api_switch_results = _activate_api_model_mentions(actions)
            # 处理工具 @mention（收集隐藏指令）
            tool_hints = _handle_tool_mentions(actions)

            # 如果有隐藏提示，作为 system 消息注入（LLM 能看到但用户 UI 不显示）
            # 注意：不能 append 到 history 末尾——本地 LLM 的 chat template 要求
            # system 消息必须在开头，放在 assistant 后面会报错 "System message must be at the beginning"。
            # 改为合并到已有的第一条 system 消息中，或插入到 history 开头。
            hidden_notes = []
            if api_switched:
                hidden_notes.append(
                    f"✅ 当前已选择 API 图像模型：{selected_api_model}。"
                    "本轮所有生图/改图请求必须使用 api_image_generate 或 api_image_edit；"
                    "不要要求用户再次说出模型名，也不要使用本地 txt2img/edit_image。"
                )
            if model_note:
                hidden_notes.append(model_note)
            if api_model_note:
                hidden_notes.append(api_model_note)
            if tool_hints:
                hidden_notes.append(tool_hints)
            if hidden_notes:
                hint_text = "[系统提示 - 以下是用户 @标签 触发的自动操作]\n" + "\n".join(hidden_notes) + "\n[请根据以上提示执行任务]"
                # 找到开头连续 system 消息中的最后一条
                merge_idx = None
                for i, msg in enumerate(new_history):
                    if msg["role"] == "system":
                        merge_idx = i
                    else:
                        break
                if merge_idx is not None:
                    new_history[merge_idx]["content"] += "\n\n" + hint_text
                else:
                    new_history.insert(0, {"role": "system", "content": hint_text})

            # 用户消息（保留 [标签] 标记让 LLM 理解上下文）
            new_history.append({"role": "user", "content": clean_text})

            for index, img_path in enumerate(_normalize_image_paths(image)):
                new_history.append({"role": "user", "content": {"path": img_path, "alt_text": f"参考图片 {index + 1}"}})
            if video is not None:
                video_path = video if isinstance(video, str) else (video.get("path") if isinstance(video, dict) else None)
                if video_path and os.path.isfile(video_path):
                    new_history.append({"role": "user", "content": {"path": video_path, "alt_text": "参考视频"}})
            return "", new_history

        def send_chat(history, image, video):
            """从 history 提取最后一条用户消息进行回复。
            image/video 从 State 获取，不会被 UI 清除影响。"""
            video_path = video if isinstance(video, str) else (video.get("path") if isinstance(video, dict) else None)
            yield from chat_stream(history, image, video_path)

        def clear_uploads():
            """发送完成后清除上传组件和 State。"""
            return None, None, None, None

        def _sync_forge_api_key(key):
            """同步写入所有持久化位置的 API Key，确保 Forge 核心也能读到。"""
            # 1. 同步到 config.json 的 forge_api_key
            try:
                shared.opts.set('forge_api_key', key)
                shared.opts.save(shared.opts.data)
            except Exception as e:
                print(f'[Agent] 同步 forge_api_key 失败: {e}')
            # 2. 同步到 forge-h3-studio 的 minimax_api_key
            try:
                h3_cfg_path = os.path.join(scripts.basedir(), 'extensions', 'forge-h3-studio', 'data', 'config.json')
                if os.path.exists(h3_cfg_path):
                    with open(h3_cfg_path, 'r', encoding='utf-8') as f:
                        h3_cfg = json.load(f)
                    h3_cfg['minimax_api_key'] = key
                    with open(h3_cfg_path, 'w', encoding='utf-8') as f:
                        json.dump(h3_cfg, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f'[Agent] 同步 h3-studio key 失败: {e}')
            # 3. 同步到内存中的 session key
            try:
                from modules_forge.api_providers import set_session_api_key
                set_session_api_key(key)
            except Exception:
                pass

        def save_settings(provider, key, url, model, use_local):
            cfg = load_config(resolve_local=False)
            cfg["api_provider"] = provider
            cfg["api_key"] = key
            cfg["base_url"] = normalize_base_url(url)
            cfg["model"] = model
            cfg["local_mode"] = use_local
            ok = save_config(cfg)
            # 同步到 Forge 核心配置（用 LLM key 作为回退）
            if key:
                _sync_forge_api_key(key)
            return "✅ 设置已保存" if ok else "❌ 保存失败"

        def save_image_settings(provider, key, url):
            cfg = load_config(resolve_local=False)
            normalized_url = normalize_base_url(url)
            custom_info = all_api_providers(cfg).get(str(provider or "").strip(), {})
            configured_provider_url = custom_info.get("base_url") or provider_base_url(provider)
            # 如果用户没填 base_url，根据供应商自动补全
            if not normalized_url:
                normalized_url = configured_provider_url
                print(f"[Agent] save_image_settings: base_url 为空，自动补全为 {normalized_url}")
            elif configured_provider_url and any(
                isinstance(item, dict) and str(item.get("name") or "").strip() == str(provider or "").strip()
                for item in cfg.get("custom_providers", [])
            ):
                normalized_url = configured_provider_url
            cfg["image_api_provider"] = provider
            cfg["image_api_key"] = key
            cfg["image_base_url"] = normalized_url
            # TRELLIS.2 是图生3D Space，不提供视频接口，不覆盖现有视频配置。
            if provider != "ModelScope Space":
                cfg["video_api_provider"] = provider
                cfg["video_api_key"] = key
                cfg["video_base_url"] = normalized_url
            ok = save_config(cfg)
            # 同步到 Forge 核心配置（图像生成 key 是主要的）
            if key:
                _sync_forge_api_key(key)
            else:
                # key 为空时清除所有位置
                _clear_all_persisted_keys()
            return "✅ 图像/视频生成 API 设置已保存" if ok else "❌ 生成 API 设置保存失败"

        def add_custom_provider(name, url):
            name = str(name or "").strip()
            url = normalize_base_url(url)
            if not name or not url.startswith(("http://", "https://")):
                return gr.update(), gr.update(), gr.update(), gr.update(), "❌ 请填写供应商名称和完整 URL（http:// 或 https://）"
            cfg = load_config(resolve_local=False)
            items = [x for x in cfg.get("custom_providers", []) if isinstance(x, dict)]
            items = [x for x in items if str(x.get("name") or x.get("provider") or "").strip() != name]
            items.append({"name": name, "base_url": url})
            cfg["custom_providers"] = items
            if not save_config(cfg):
                return gr.update(), gr.update(), gr.update(), gr.update(), "❌ 自定义供应商保存失败"
            choices = provider_choices(cfg)
            return (
                gr.update(choices=choices, value=name),
                gr.update(choices=choices, value=name),
                url,
                url,
                f"✅ 已添加：{url}",
            )

        def add_custom_model(kind, model_id, llm_provider, image_provider):
            model_id = str(model_id or "").strip()
            provider = str((llm_provider if kind == "llm" else image_provider) or "").strip()
            if not model_id or not provider:
                return gr.update(), gr.update(), "❌ 请先选择供应商并填写模型 ID"
            cfg = load_config(resolve_local=False)
            custom = cfg.setdefault("custom_models", {"llm": {}, "image": {}, "video": {}})
            bucket = custom.setdefault(kind, {})
            values = bucket.setdefault(provider, [])
            if model_id not in values:
                values.append(model_id)
            save_config(cfg)
            if kind == "llm":
                choices = _models_for_provider(cfg, "llm", provider, LLM_MODELS_BY_PROVIDER.get(provider, []))
                return gr.update(choices=choices, value=model_id), gr.update(), f"✅ 已添加 Agent 模型：{model_id}"
            current_choices = list(getattr(api_model_select, "choices", []) or [])
            value = ("video|" if kind == "video" else provider + "|") + model_id
            if not any((c[1] if isinstance(c, tuple) else c) == value for c in current_choices):
                current_choices.append((provider + " · " + model_id, value))
            return gr.update(), gr.update(choices=current_choices, value=value), f"✅ 已添加{('视频' if kind == 'video' else '图像')}模型：{model_id}"

        def _clear_all_persisted_keys():
            """同步清除所有持久化位置的 API Key，防止重启后复活。"""
            # 1. 清除 Forge 核心 config.json 中的 forge_api_key
            try:
                if hasattr(shared.opts, 'forge_api_key') and shared.opts.forge_api_key:
                    shared.opts.set('forge_api_key', '')
                    shared.opts.save(shared.opts.data)
                    print('[Agent] 已清除 config.json 中的 forge_api_key')
            except Exception as e:
                print(f'[Agent] 清除 forge_api_key 失败: {e}')
            # 2. 清除 forge-h3-studio 扩展的 minimax_api_key
            try:
                h3_cfg_path = os.path.join(scripts.basedir(), 'extensions', 'forge-h3-studio', 'data', 'config.json')
                if os.path.exists(h3_cfg_path):
                    with open(h3_cfg_path, 'r', encoding='utf-8') as f:
                        h3_cfg = json.load(f)
                    if h3_cfg.get('minimax_api_key'):
                        h3_cfg['minimax_api_key'] = ''
                        with open(h3_cfg_path, 'w', encoding='utf-8') as f:
                            json.dump(h3_cfg, f, ensure_ascii=False, indent=2)
                        print('[Agent] 已清除 forge-h3-studio 的 minimax_api_key')
            except Exception as e:
                print(f'[Agent] 清除 h3-studio key 失败: {e}')
            # 3. 清除内存中的 session key
            try:
                from modules_forge.api_providers import set_session_api_key
                set_session_api_key('')
            except Exception:
                pass

        def clear_api_key():
            cfg = load_config(resolve_local=False)
            cfg["api_key"] = ""
            ok = save_config(cfg)
            _clear_all_persisted_keys()
            return "", "✅ API Key 已清空（所有存储位置）" if ok else "❌ API Key 清空失败"

        def clear_image_api_key():
            cfg = load_config(resolve_local=False)
            cfg["image_api_key"] = ""
            cfg["video_api_key"] = ""
            ok = save_config(cfg)
            _clear_all_persisted_keys()
            return "", "✅ 生成 API Key 已清空（所有存储位置）" if ok else "❌ 生成 API Key 清空失败"

        def detect_local(use_local_mode):
            """手动触发本地模型检测，清除缓存强制重新扫描。仅当 local_mode 开启时才执行。"""
            if not use_local_mode:
                return "⚠️ 请先勾选「启用本地 llama-server」再检测", "", "", "", "ModelScope"
            global _local_detection_cache, _local_detection_time
            _local_detection_cache = None
            _local_detection_time = 0
            detected = _detect_local_llama()
            if detected:
                base_url_val, _, model_val = detected
                return f"✅ 检测到本地模型: {model_val} @ {base_url_val}", model_val, base_url_val, "local", "ModelScope"
            return "⚠️ 未检测到本地 llama-server（端口 8079-8090），请先启动启动器", "", "", "", "ModelScope"

        # 上传事件：同步到 State
        upload_image.change(fn=on_image_upload, inputs=[upload_image], outputs=[state_image])
        upload_video.change(fn=on_video_upload, inputs=[upload_video], outputs=[state_video])

        api_provider.change(fn=on_llm_provider_change, inputs=[api_provider, base_url, model_name], outputs=[base_url, model_name])
        add_provider_btn.click(
            fn=add_custom_provider,
            inputs=[custom_provider_name, custom_provider_url],
            outputs=[api_provider, image_api_provider, base_url, image_base_url, provider_manage_status],
        )
        add_model_btn.click(
            fn=add_custom_model,
            inputs=[custom_model_kind, custom_model_id, api_provider, image_api_provider],
            outputs=[model_name, api_model_select, custom_model_status],
        )

        # 首页 API 模型下拉切换 — 即时保存 provider 和 model
        def on_local_model_select(local_model_value):
            # 选择任何本地模型/工具都会关闭 API 图像模型。
            if not local_model_value or not str(local_model_value).strip():
                return gr.update(), ""
            cfg = load_config(resolve_local=False)
            cfg["image_model"] = ""
            save_config(cfg)
            try:
                shared.opts.set("forge_model_mode", "local")
            except Exception:
                pass
            return gr.update(value=""), "✅ 已切换本地模式，API 图像模型已关闭"

        def on_api_model_select(api_model_value):
            if not api_model_value or "|" not in str(api_model_value):
                cfg = load_config(resolve_local=False)
                cfg["image_model"] = ""
                save_config(cfg)
                try:
                    shared.opts.set("forge_model_mode", "local")
                except Exception:
                    pass
                return "✅ 已关闭 API 图像模型，当前为本地模式", gr.update(value="")
            parts = str(api_model_value).split("|", 1)
            provider, model = parts[0].strip(), parts[1].strip()
            if provider == "video":
                # 视频模型
                cfg = load_config()
                cfg["video_model"] = model
                save_config(cfg)
                return f"✅ 视频模型已切换: {model}", gr.update()
            elif provider and model:
                # 图像模型
                cfg = load_config()
                cfg["image_api_provider"] = provider
                cfg["image_model"] = model
                # 切换模型时同步供应商 Base URL，避免沿用旧供应商（如 yunwu.ai）地址。
                provider_info = all_api_providers(cfg).get(provider, {})
                correct_url = provider_info.get("base_url") or provider_base_url(provider)
                if correct_url:
                    cfg["image_base_url"] = correct_url
                save_config(cfg)
                try:
                    from modules_forge.api_providers import set_session_api_key
                    shared.opts.set("forge_model_mode", "api")
                    set_session_api_key(cfg.get("image_api_key", "") or cfg.get("api_key", ""))
                except Exception:
                    pass
                return f"✅ API 图像模型已切换: {provider} · {model}", gr.update(value="")
            return "", gr.update()
        local_model_select.change(fn=on_local_model_select, inputs=[local_model_select], outputs=[api_model_select, image_settings_status])
        api_model_select.change(fn=on_api_model_select, inputs=[api_model_select], outputs=[image_settings_status, local_model_select])

        save_settings_btn.click(fn=save_settings, inputs=[api_provider, api_key, base_url, model_name, local_mode], outputs=[settings_status])
        save_image_settings_btn.click(fn=save_image_settings, inputs=[image_api_provider, image_api_key, image_base_url], outputs=[image_settings_status])
        clear_api_key_btn.click(fn=clear_api_key, outputs=[api_key, settings_status])
        clear_image_api_key_btn.click(fn=clear_image_api_key, outputs=[image_api_key, image_settings_status])
        detect_btn.click(fn=detect_local, inputs=[local_mode], outputs=[detect_status, model_name, base_url, api_key, api_provider])
        clear_image_btn.click(fn=lambda: (None, None), outputs=[upload_image, state_image])
        clear_video_btn.click(fn=lambda: (None, None), outputs=[upload_video, state_video])

        # 发送按钮：prepare → send_chat → clear_uploads
        send_btn.click(
            fn=prepare_message,
            inputs=[msg_input, chatbot, state_image, state_video, local_model_select, api_model_select],
            outputs=[msg_input, chatbot],
        ).then(
            fn=send_chat,
            inputs=[chatbot, state_image, state_video],
            outputs=[chatbot, status],
        ).then(
            fn=clear_uploads,
            outputs=[upload_image, upload_video, state_image, state_video],
        )

        # 回车发送
        msg_input.submit(
            fn=prepare_message,
            inputs=[msg_input, chatbot, state_image, state_video, local_model_select, api_model_select],
            outputs=[msg_input, chatbot],
        ).then(
            fn=send_chat,
            inputs=[chatbot, state_image, state_video],
            outputs=[chatbot, status],
        ).then(
            fn=clear_uploads,
            outputs=[upload_image, upload_video, state_image, state_video],
        )

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

        # 注入图片放大 lightbox 的 CSS 和 HTML 结构（Gradio 5.x 中 gr.HTML 内的 <script> 不会执行）
        gr.HTML("""
        <style>
        #agent-lightbox {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85);
            z-index: 99999;
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        }
        #agent-lightbox.active { display: flex; }
        #agent-lightbox img {
            max-width: 95vw;
            max-height: 95vh;
            object-fit: contain;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
        }
        #agent-lightbox-close {
            position: fixed;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 36px;
            cursor: pointer;
            z-index: 100000;
            background: rgba(0,0,0,0.5);
            border-radius: 50%;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* 隐藏 Gradio 原生 fullscreen 按钮（button 元素），保留下载链接（a 元素） */
        #agent-chatbot .icon-button-wrapper > button {
            display: none !important;
        }
        /* 自定义放大图标 - 悬停时显示在图片右上角 */
        #agent-chatbot .image-container {
            position: relative;
            display: inline-flex !important;
            width: min(240px, 72vw) !important;
            max-width: min(240px, 72vw) !important;
            max-height: 240px !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            padding: 8px !important;
            border: 1px solid rgba(148, 163, 184, 0.32) !important;
            border-radius: 8px !important;
            background: rgba(15, 23, 42, 0.38) !important;
        }
        #agent-chatbot .image-container::after {
            content: "🔍";
            position: absolute;
            top: 8px;
            right: 8px;
            font-size: 18px;
            background: rgba(0,0,0,0.6);
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
            pointer-events: none;
            z-index: 10;
        }
        #agent-chatbot .image-container:hover::after {
            opacity: 1;
        }
        #agent-chatbot img {
            cursor: zoom-in;
            transition: opacity 0.2s;
            width: auto !important;
            height: auto !important;
            max-width: 224px !important;
            max-height: 224px !important;
            object-fit: contain !important;
        }
        #agent-chatbot img:hover { opacity: 0.9; }
        </style>
        """)

        # 加载时从服务端获取配置回填到 Textbox
        # Gradio 5.x 的 Textbox 不会通过 HTML 属性渲染初始值（尤其是懒加载标签页），
        # 所以需要在 .load() 回调中从 load_config() 读取并设置。
        def _load_config_to_ui():
            cfg = load_config(resolve_local=False)
            return cfg.get("api_key", ""), cfg.get("image_api_key", "") or cfg.get("video_api_key", "")

        agent_interface.load(
            fn=_load_config_to_ui,
            outputs=[api_key, image_api_key],
            js="""() => {
                document.querySelectorAll('textarea').forEach(t => t.setAttribute('autocomplete', 'new-password'));
            }"""
        )

    return [(agent_interface, "绘梦智能体助手", "sd_webui_agent")]


script_callbacks.on_ui_tabs(on_ui_tabs, name="sd_webui_agent_tab")


# =============================================================================
# 合并注册系统中的外部工具
# =============================================================================

if _REGISTRY_AVAILABLE:
    # 合并已注册的工具到 TOOLS 和 TOOL_FUNCTIONS
    for reg_tool in get_registered_tools():
        name = reg_tool["function"]["name"]
        if name not in TOOL_FUNCTIONS:
            TOOLS.append(reg_tool)
            TOOL_FUNCTIONS[name] = get_tool_function(name)


def _refresh_registered_agent_tools():
    """在每次对话前同步其他插件新注册的 Agent 工具。"""
    if not _REGISTRY_AVAILABLE:
        return
    for reg_tool in get_registered_tools():
        name = reg_tool["function"]["name"]
        if name not in TOOL_FUNCTIONS:
            TOOLS.append(reg_tool)
            TOOL_FUNCTIONS[name] = get_tool_function(name)

