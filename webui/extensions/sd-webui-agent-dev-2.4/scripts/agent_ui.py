# =============================================================================
# Agent UI — Gradio 界面 / @mention 系统 / 供应商与自定义模型管理
# =============================================================================

import json
import os
import re

import gradio as gr

from modules import shared, scripts, sd_models

from scripts.agent_config import (
    load_config, save_config,
    API_PROVIDERS, all_api_providers, provider_choices,
    normalize_base_url, provider_base_url,
    list_local_llm_models, detect_local_llm,
)
from scripts.agent_models import (
    LLM_MODELS_BY_PROVIDER, VIDEO_GENERATION_MODELS,
    _models_for_provider, _is_saved_image_model,
)
from scripts.agent_tools import MODEL_GUIDE, set_model_components_tool
from scripts.agent_history import _normalize_image_paths
from scripts.agent_chat import chat_stream, request_stop


# =============================================================================
# Forge API 状态同步（启动恢复 / 设置保存 / Key 清空 等场景复用）
# =============================================================================

FORGE_PROVIDER_ALIASES = {
    "modelscope": "modelscope",
    "dashscope": "dashscope",
    "pixapi": "pixapi",
    "yoboxai": "yoboxai",
}


def _h3_studio_key_path():
    return os.path.join(scripts.basedir(), 'extensions', 'forge-h3-studio', 'data', 'config.json')


def _set_h3_studio_minimax_key(key):
    """把 API Key 写入 forge-h3-studio 的 minimax_api_key（文件不存在则跳过）。"""
    try:
        h3_cfg_path = _h3_studio_key_path()
        if os.path.exists(h3_cfg_path):
            with open(h3_cfg_path, 'r', encoding='utf-8') as f:
                h3_cfg = json.load(f)
            h3_cfg['minimax_api_key'] = key
            with open(h3_cfg_path, 'w', encoding='utf-8') as f:
                json.dump(h3_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[Agent] 同步 h3-studio key 失败: {e}')


def _clear_h3_studio_minimax_key():
    """清除 forge-h3-studio 中残留的 minimax_api_key（为空则跳过）。"""
    try:
        h3_cfg_path = _h3_studio_key_path()
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


def _apply_forge_api_opts(provider, model, api_key=""):
    """把 Forge 核心切换到 API 模式（provider/model/session key）。"""
    from modules_forge.api_providers import set_session_api_key
    shared.opts.set("forge_model_mode", "api")
    provider_lower = str(provider or "").lower()
    if provider_lower in FORGE_PROVIDER_ALIASES:
        shared.opts.set("forge_api_provider", FORGE_PROVIDER_ALIASES[provider_lower])
    shared.opts.set("forge_api_model", model)
    set_session_api_key(api_key)


def _sync_forge_api_key(key):
    """同步写入所有持久化位置的 API Key，确保 Forge 核心也能读到。"""
    # 1. 同步到 config.json 的 forge_api_key
    try:
        shared.opts.set('forge_api_key', key)
        shared.opts.save(shared.opts.data)
    except Exception as e:
        print(f'[Agent] 同步 forge_api_key 失败: {e}')
    # 2. 同步到 forge-h3-studio 的 minimax_api_key
    _set_h3_studio_minimax_key(key)
    # 3. 同步到内存中的 session key
    try:
        from modules_forge.api_providers import set_session_api_key
        set_session_api_key(key)
    except Exception:
        pass


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
    _clear_h3_studio_minimax_key()
    # 3. 清除内存中的 session key
    try:
        from modules_forge.api_providers import set_session_api_key
        set_session_api_key('')
    except Exception:
        pass


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
    "图层分离": ("local_tool", "本地图层分离 / Local layer separation：使用 See-Through LayerDiff，调用 layer_separation，禁止调用 SAM、remove_background 和远程 API。"),
    "视频关键帧": ("local_tool", "本地视频关键帧提取 / Local video keyframe extraction：使用 video_keyframe_extract，禁止调用远程 API。"),
    "TRELLIS图生3D": ("local_tool", "本地 TRELLIS.2 图生3D：使用已安装的 TRELLIS.2 本地扩展处理上传图片，禁止调用远程 API。"),
    "TRELLIS 3D": ("local_tool", "本地 TRELLIS.2 图生3D：使用已安装的 TRELLIS.2 本地扩展处理上传图片，禁止调用远程 API。"),
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

            _apply_forge_api_opts(provider_name, model_id, cfg.get("image_api_key", ""))

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


# =============================================================================
# 上传附件辅助
# =============================================================================

def _upload_path(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("path") or item.get("name")
    return getattr(item, "path", None) or getattr(item, "name", None)


def _classify_uploads(files):
    images, videos, attachments = [], [], []
    items = files if isinstance(files, (list, tuple)) else ([files] if files else [])
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    for item in items:
        path = _upload_path(item)
        if not path or not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in image_exts:
            images.append(path)
        elif ext in video_exts:
            videos.append(path)
        else:
            attachments.append(path)
    return images, (videos[0] if videos else None), attachments


# =============================================================================
# UI
# =============================================================================

# 内置 API 模型选项（图像 + 视频），与 custom_models 合并生成完整下拉列表
_BUILTIN_API_MODEL_CHOICES = [
    ("不使用 API 模型", ""),
    ("🤖 YoboxAI · banana2", "YoboxAI|banana2"),
    ("🤖 YoboxAI · bananapro", "YoboxAI|bananapro"),
    ("🤖 YoboxAI · gpt-image-2", "YoboxAI|gpt-image-2"),
    ("🧩 ModelScope · Krea-2-Turbo", "ModelScope|krea/Krea-2-Turbo"),
    ("🧩 ModelScope · Z-Image", "ModelScope|Tongyi-MAI/Z-Image"),
    ("🧩 ModelScope · Qwen-Image-2512", "ModelScope|Qwen/Qwen-Image-2512"),
    ("🧩 ModelScope · FireRed-Image-Edit", "ModelScope|FireRedTeam/FireRed-Image-Edit-1.1"),
    ("🧩 ModelScope · Qwen-Image-Edit", "ModelScope|Qwen/Qwen-Image-Edit-2511"),
    ("🎬 YoboxAI · dreamina-seedance-2-0", "video|dreamina-seedance-2-0-hc"),
    ("🎬 YoboxAI · dreamina-seedance-2-5", "video|dreamina-seedance-2-5-hc"),
    ("🎬 YoboxAI · MiniMax-H3", "video|MiniMax-H3"),
]

_KIND_LABELS = {"llm": "Agent 大脑", "image": "图像", "video": "视频"}


def _build_api_model_choices(cfg):
    """内置 + 自定义 图像/视频模型 → 完整的 API 模型下拉 choices。"""
    choices = list(_BUILTIN_API_MODEL_CHOICES)
    custom = cfg.get("custom_models", {}) if isinstance(cfg, dict) else {}
    for provider, models in (custom.get("image") or {}).items():
        if isinstance(models, list):
            choices.extend(
                [(str(provider) + " · " + str(m), str(provider) + "|" + str(m)) for m in models if str(m).strip()]
            )
    for provider, models in (custom.get("video") or {}).items():
        if isinstance(models, list):
            choices.extend(
                [(str(provider) + " · " + str(m), "video|" + str(m)) for m in models if str(m).strip()]
            )
    return choices


def _custom_provider_names(cfg):
    """返回自定义供应商名称列表（供删除下拉使用）。"""
    names = []
    for item in (cfg.get("custom_providers", []) if isinstance(cfg, dict) else []):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("provider") or "").strip()
            if name:
                names.append(name)
    return list(dict.fromkeys(names))


def _custom_model_entries(cfg):
    """枚举全部自定义模型。返回 [(label, 'kind|provider|model')]。"""
    entries = []
    custom = cfg.get("custom_models", {}) if isinstance(cfg, dict) else {}
    for kind in ("llm", "image", "video"):
        bucket = custom.get(kind, {})
        if not isinstance(bucket, dict):
            continue
        for provider, models in bucket.items():
            if not isinstance(models, list):
                continue
            for m in models:
                m = str(m).strip()
                if not m:
                    continue
                entries.append((f"{_KIND_LABELS[kind]} · {provider} · {m}", f"{kind}|{provider}|{m}"))
    return entries


def _is_choice_value(choices, value):
    """判断 value 是否存在于 choices 中（choices 项为 (label, value) 或纯值）。"""
    if not value:
        return False
    for c in choices:
        cv = c[1] if isinstance(c, (tuple, list)) else c
        if str(cv) == str(value):
            return True
    return False


def _current_api_model_value(cfg, choices):
    """若当前 image_model 仍是合法选项，返回其下拉值 'provider|model'；否则返回 ''。"""
    cur_img = str(cfg.get("image_model") or "").strip()
    if cur_img and _is_saved_image_model(cfg, cur_img):
        cur_prov = str(cfg.get("image_api_provider") or "").strip()
        if cur_prov:
            candidate = f"{cur_prov}|{cur_img}"
            if _is_choice_value(choices, candidate):
                return candidate
    return ""


def remove_custom_provider(selected):
    """删除自定义供应商（连带删除其下所有自定义模型），并重建所有相关下拉。

    输出顺序: api_provider, image_api_provider, model_name, api_model_select,
              remove_provider_select, remove_model_select, provider_manage_status
    """
    name = str(selected or "").strip()
    if not name:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), "❌ 请先选择要删除的自定义供应商")
    if name in API_PROVIDERS:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), f"❌ {name} 是内置供应商，不能删除")
    cfg = load_config(resolve_local=False)
    items = [x for x in cfg.get("custom_providers", []) if isinstance(x, dict)]
    remaining = [x for x in items if str(x.get("name") or x.get("provider") or "").strip() != name]
    if len(remaining) == len(items):
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), f"❌ 未找到自定义供应商 {name}")
    cfg["custom_providers"] = remaining
    # 连带删除该供应商下所有类型的自定义模型
    custom = cfg.get("custom_models")
    if isinstance(custom, dict):
        for kind in ("llm", "image", "video"):
            bucket = custom.get(kind)
            if isinstance(bucket, dict):
                bucket.pop(name, None)
    # 若当前大脑/生成供应商指向被删供应商，回退到 ModelScope
    changed = []
    if str(cfg.get("api_provider") or "").strip() == name:
        cfg["api_provider"] = "ModelScope"
        cfg["base_url"] = API_PROVIDERS["ModelScope"]["base_url"]
        changed.append("Agent 大脑")
    if str(cfg.get("image_api_provider") or "").strip() == name:
        cfg["image_api_provider"] = "ModelScope"
        changed.append("图像生成")
    if str(cfg.get("video_api_provider") or "").strip() == name:
        cfg["video_api_provider"] = "ModelScope"
        if "视频生成" not in changed:
            changed.append("视频生成")
    # 当前图像模型若已失效，清空
    if cfg.get("image_model") and not _is_saved_image_model(cfg, cfg.get("image_model")):
        cfg["image_model"] = ""
    # 当前视频模型若已失效，清空
    _vm = str(cfg.get("video_model") or "").strip()
    if _vm:
        _video_ok = _vm in VIDEO_GENERATION_MODELS
        if not _video_ok:
            for _vals in (cfg.get("custom_models", {}).get("video", {}) or {}).values():
                if isinstance(_vals, list) and _vm in _vals:
                    _video_ok = True
                    break
        if not _video_ok:
            cfg["video_model"] = ""
    ok = save_config(cfg)
    if not ok:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), "❌ 自定义供应商删除保存失败")
    # 重建所有相关下拉
    choices = provider_choices(cfg)
    llm_choices = choices + [("🖥️ 本地 Llama 服务（llama.cpp / Ollama / LM Studio）", "local-llama")]
    llm_model_choices = _models_for_provider(
        cfg, "llm", cfg.get("api_provider"), LLM_MODELS_BY_PROVIDER.get(cfg.get("api_provider"), [])
    )
    api_choices = _build_api_model_choices(cfg)
    rem_provider_choices = _custom_provider_names(cfg)
    rem_model_choices = _custom_model_entries(cfg)
    api_value = _current_api_model_value(cfg, api_choices)
    note = f"✅ 已删除自定义供应商：{name}"
    if changed:
        note += "（它正在被 " + "、".join(changed) + " 使用，已回退到 ModelScope）"
    return (
        gr.update(choices=llm_choices, value=cfg.get("api_provider")),
        gr.update(choices=choices, value=cfg.get("image_api_provider")),
        gr.update(choices=llm_model_choices, value=cfg.get("model")),
        gr.update(choices=api_choices, value=api_value),
        gr.update(choices=rem_provider_choices, value=""),
        gr.update(choices=rem_model_choices, value=""),
        note,
    )


def remove_custom_model(selected):
    """删除单个自定义模型 ID，并重建相关下拉。

    selected 值格式: 'kind|provider|model'
    输出顺序: model_name, api_model_select, remove_model_select, custom_model_status
    """
    value = str(selected or "").strip()
    if not value or value.count("|") < 2:
        return (gr.update(), gr.update(), gr.update(), "❌ 请先选择要删除的自定义模型")
    kind, provider, model_id = [p.strip() for p in value.split("|", 2)]
    if kind not in ("llm", "image", "video"):
        return (gr.update(), gr.update(), gr.update(), "❌ 无效的模型类型，无法删除")
    cfg = load_config(resolve_local=False)
    custom = cfg.setdefault("custom_models", {"llm": {}, "image": {}, "video": {}})
    bucket = custom.get(kind)
    if not isinstance(bucket, dict):
        return (gr.update(), gr.update(), gr.update(), "❌ 模型数据格式异常")
    values = bucket.get(provider)
    if not isinstance(values, list) or model_id not in values:
        return (gr.update(), gr.update(), gr.update(), f"❌ 未找到自定义模型 {model_id}（供应商 {provider}）")
    # 记录并处理“当前正在使用被删模型”的情况
    was_current = False
    if kind == "llm" and str(cfg.get("model") or "").strip() == model_id:
        was_current = True
        fallback = _models_for_provider(
            cfg, "llm", cfg.get("api_provider"), LLM_MODELS_BY_PROVIDER.get(cfg.get("api_provider"), [])
        )
        cfg["model"] = fallback[0] if fallback else ""
    if kind == "image" and str(cfg.get("image_model") or "").strip() == model_id:
        was_current = True
        cfg["image_model"] = ""
    if kind == "video" and str(cfg.get("video_model") or "").strip() == model_id:
        was_current = True
        cfg["video_model"] = ""
    # 执行删除
    new_values = [v for v in values if str(v).strip() != model_id]
    if new_values:
        bucket[provider] = new_values
    else:
        bucket.pop(provider, None)
    ok = save_config(cfg)
    if not ok:
        return (gr.update(), gr.update(), gr.update(), "❌ 自定义模型删除保存失败")
    rem_model_choices = _custom_model_entries(cfg)
    msg = f"✅ 已删除{_KIND_LABELS[kind]}模型：{model_id}"
    if was_current and kind == "llm":
        msg += "（原为当前大脑模型，已切换到 " + (cfg.get("model") or "默认模型") + "）"
    if kind == "llm":
        llm_model_choices = _models_for_provider(
            cfg, "llm", cfg.get("api_provider"), LLM_MODELS_BY_PROVIDER.get(cfg.get("api_provider"), [])
        )
        return (
            gr.update(choices=llm_model_choices, value=cfg.get("model") or (llm_model_choices[0] if llm_model_choices else None)),
            gr.update(),
            gr.update(choices=rem_model_choices, value=""),
            msg,
        )
    # image / video
    api_choices = _build_api_model_choices(cfg)
    api_value = "" if was_current else _current_api_model_value(cfg, api_choices)
    return (
        gr.update(),
        gr.update(choices=api_choices, value=api_value),
        gr.update(choices=rem_model_choices, value=""),
        msg,
    )


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
            _clear_all_persisted_keys()
        # 如果 image_api_key 为空且当前处于 API 模式，回退到 local 避免无 key 401。
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
            width: min(72vw, 800px) !important;
            min-height: 120px !important;
            max-width: min(72vw, 800px) !important;
            max-height: calc(90vh - 160px) !important;
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
            max-width: min(72vw, 820px) !important;
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
            max-width: 100% !important;
            max-height: calc(90vh - 180px) !important;
            object-fit: contain !important;
        }
        #agent-chatbot .message img,
        #agent-chatbot .bubble-wrap img,
        #agent-chatbot .chatbot img,
        #agent-chatbot [data-testid="bot"] img {
            width: auto !important;
            height: auto !important;
            max-width: 100% !important;
            max-height: calc(90vh - 180px) !important;
            object-fit: contain !important;
        }
        /* ===== 思考过程内嵌小窗口（折叠块 + 限高内部滚动） ===== */
        #agent-chatbot details.agent-thinking {
            margin: 6px 0 10px;
            max-width: min(72vw, 760px);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.38);
            font-size: 12px;
            color: #94a3b8;
        }
        #agent-chatbot details.agent-thinking > summary {
            cursor: pointer;
            padding: 6px 10px;
            font-size: 12px;
            color: #a78bfa;
            user-select: none;
            list-style: none;
        }
        #agent-chatbot details.agent-thinking > summary::-webkit-details-marker {
            display: none;
        }
        #agent-chatbot details.agent-thinking > summary::before {
            content: "▸ ";
        }
        #agent-chatbot details.agent-thinking[open] > summary::before {
            content: "▾ ";
        }
        #agent-chatbot details.agent-thinking .agent-thinking-body {
            max-height: 140px;
            overflow-y: auto;
            padding: 2px 12px 10px;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.55;
            scrollbar-width: thin;
            scrollbar-color: rgba(148, 163, 184, 0.4) transparent;
        }
        #agent-chatbot details.agent-thinking .agent-thinking-body::-webkit-scrollbar {
            width: 6px;
        }
        #agent-chatbot details.agent-thinking .agent-thinking-body::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.4);
            border-radius: 3px;
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
            key = str(provider or "").strip()
            if key == "local-llama":
                # 本地 Llama 大脑：使用之前检测/保存的本地服务 URL 与模型分组
                url = normalize_base_url(cfg.get("local_llm_base_url") or current_url)
                llm_models = _models_for_provider(cfg, "llm", "local-llama", [])
                if not llm_models:
                    _saved = str(cfg.get("local_llm_model") or "").strip()
                    llm_models = [_saved] if _saved else []
                new_value = current_model if current_model in llm_models else (llm_models[0] if llm_models else current_model)
                return url, gr.update(choices=llm_models, value=new_value, allow_custom_value=True)
            info = all_api_providers(cfg).get(key, {})
            url = info.get("base_url") or provider_base_url(provider) or normalize_base_url(current_url)
            llm_models = _models_for_provider(cfg, "llm", provider, LLM_MODELS_BY_PROVIDER.get(provider, []))
            # 如果当前模型不在新供应商列表中，回退到该供应商第一个模型
            new_value = current_model if current_model in llm_models else (llm_models[0] if llm_models else current_model)
            return url, gr.update(choices=llm_models, value=new_value, allow_custom_value=True)

        initial_api_choices = _build_api_model_choices(cfg_init)

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

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入消息",
                        placeholder="例如：画一只可爱的橘猫，坐在窗台上晒太阳...",
                        scale=4, show_label=False,
                        elem_id="agent_chat_input",
                    )
                    send_btn = gr.Button("📤 发送", variant="primary", scale=1, elem_id="agent_send_btn")
                    pause_btn = gr.Button("⏹️ 暂停思考", variant="stop", scale=1, elem_id="agent_pause_btn")
                    clear_btn = gr.Button("🗑️ 清空", scale=1)

                status = gr.Textbox(label="状态", value="就绪", interactive=False, show_label=False)

            # 隐藏的 State 保存全能上传块分类后的文件
            state_image = gr.State(None)
            state_video = gr.State(None)
            state_attachments = gr.State([])

            with gr.Column(scale=1):
                upload_all = gr.File(
                    label="📎 全能附件（图片 / 视频 / 文档 / 代码，可多选）",
                    file_count="multiple",
                    file_types=None,
                    height=110,
                )
                clear_upload_btn = gr.Button("清除附件", size="sm")

                with gr.Accordion("⚙️ API 设置", open=False):
                    # ===== 折叠块 1：自定义供应商（供应商 + 自定义模型 ID 管理：添加 / 删除） =====
                    with gr.Accordion("🏷️ 自定义供应商", open=False):
                        gr.Markdown("供应商列表显示完整 API URL。需要接入其他平台时，可添加自定义供应商；也可以为任意供应商添加自定义模型 ID。不再需要时可用下方「🗑️ 删除」移除（删除供应商会连带删除其名下的自定义模型）。")
                        with gr.Row():
                            custom_provider_name = gr.Textbox(label="自定义供应商名称", placeholder="例如：我的绘图平台", scale=1)
                            custom_provider_url = gr.Textbox(label="完整 API URL", placeholder="https://api.example.com/v1", scale=2)
                            add_provider_btn = gr.Button("➕ 添加供应商", size="sm", scale=1)
                        provider_manage_status = gr.Textbox(show_label=False, interactive=False)
                        with gr.Row():
                            custom_model_kind = gr.Dropdown(
                                label="模型类型", choices=[("Agent 大脑", "llm"), ("图像", "image"), ("视频", "video")], value="image", scale=1
                            )
                            custom_model_id = gr.Textbox(label="自定义模型 ID", placeholder="例如：provider/model-name", scale=2)
                            add_model_btn = gr.Button("➕ 添加模型 ID", size="sm", scale=1)
                        custom_model_status = gr.Textbox(show_label=False, interactive=False)
                        # ---- 删除区：移除供应商 / 移除模型 ID ----
                        with gr.Row():
                            remove_provider_select = gr.Dropdown(
                                label="🗑️ 要删除的供应商（连带删除其自定义模型）",
                                choices=_custom_provider_names(cfg_init),
                                value="",
                                scale=2,
                            )
                            remove_provider_btn = gr.Button("🗑️ 删除供应商", size="sm", scale=1)
                        with gr.Row():
                            remove_model_select = gr.Dropdown(
                                label="🗑️ 要删除的自定义模型 ID",
                                choices=_custom_model_entries(cfg_init),
                                value="",
                                scale=2,
                            )
                            remove_model_btn = gr.Button("🗑️ 删除模型 ID", size="sm", scale=1)

                    # ===== 折叠块 2：Agent 大脑设置（LLM 对话模型） =====
                    with gr.Accordion("🧠 Agent 大脑设置", open=False):
                        gr.Markdown("配置 Agent 对话模型（LLM）：供应商、API Key 与模型 ID。使用本地 Llama 请展开下方「🖥️ 本地 Llama 模型」模块。")
                        api_provider = gr.Dropdown(
                            label="API 供应商",
                            choices=provider_choices(cfg_init) + [("🖥️ 本地 Llama 服务（llama.cpp / Ollama / LM Studio）", "local-llama")],
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

                    # ===== 折叠块 3：图像/视频生成 API 设置（独立于 Agent 大脑） =====
                    with gr.Accordion("🌐 图像/视频生成 API 设置", open=False):
                        gr.Markdown("配置图像/视频生成模型的供应商与 API Key，独立于 Agent 大脑；图像模型在首页下拉框选择。")
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

                    # ===== 折叠块 4：本地 Llama 模型（作为智能体大脑，自动检测本地服务） =====
                    with gr.Accordion("🖥️ 本地 Llama 模型（智能体大脑）", open=False):
                        gr.Markdown("使用本机 llama.cpp / Ollama / LM Studio 等 OpenAI 兼容服务作为 Agent 大脑（无需云端 API Key）。点击「🔍 自动检测」扫描本地常见服务端口，自动填充可用模型。")
                        local_llm_url = gr.Textbox(
                            label="本地服务 Base URL（留空则自动检测）",
                            placeholder="例如：http://127.0.0.1:8080/v1（llama.cpp server）",
                            value=cfg_init.get("local_llm_base_url", ""),
                        )
                        local_llm_key = gr.Textbox(
                            label="API Key（本地服务通常留空）",
                            value=cfg_init.get("local_llm_api_key", ""),
                            type="text",
                            elem_id="agent_local_llm_key_input",
                        )
                        with gr.Row():
                            local_llm_detect_btn = gr.Button("🔍 自动检测本地 Llama 服务", variant="primary", scale=1)
                            local_llm_rescan_btn = gr.Button("↻ 刷新模型列表", scale=1)
                        local_llm_status = gr.Textbox(show_label=False, interactive=False, value="")
                        local_llm_model = gr.Dropdown(
                            label="模型 ID（自动检测后从下拉选择，也可手动输入）",
                            choices=[],
                            value=cfg_init.get("local_llm_model", ""),
                            allow_custom_value=True,
                            info="检测成功后会自动填充；也可以直接输入 gguf 模型名",
                        )
                        local_llm_apply_btn = gr.Button("🧠 设为智能体大脑", variant="secondary")
                        local_llm_apply_status = gr.Textbox(show_label=False, interactive=False, value="")

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

        def on_all_upload(files):
            """全能附件上传：按扩展名分流给图像、视频和普通附件。"""
            return _classify_uploads(files)

        def prepare_message(message, history, image, video, attachments, local_model_value="", api_model_value=""):
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
                        _apply_forge_api_opts(provider, model, cfg.get("image_api_key", ""))
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
            for file_path in attachments or []:
                new_history.append({"role": "user", "content": {"path": file_path, "alt_text": f"附件：{os.path.basename(file_path)}"}})
            return "", new_history

        def send_chat(history, image, video, attachments):
            """从 history 提取最后一条用户消息进行回复。
            image/video 从 State 获取，不会被 UI 清除影响。"""
            video_path = video if isinstance(video, str) else (video.get("path") if isinstance(video, dict) else None)
            yield from chat_stream(history, image, video_path, attachments)

        def clear_uploads():
            """发送完成后清除上传组件和 State。"""
            return None, None, None, []

        def save_settings(provider, key, url, model):
            cfg = load_config(resolve_local=False)
            cfg["api_provider"] = provider
            cfg["api_key"] = key
            cfg["base_url"] = normalize_base_url(url)
            cfg["model"] = model
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
            # 大脑下拉额外保留「本地 Llama 服务」选项；生成下拉不需要
            llm_choices = choices + [("🖥️ 本地 Llama 服务（llama.cpp / Ollama / LM Studio）", "local-llama")]
            return (
                gr.update(choices=llm_choices, value=name),
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

        def detect_local_llm_service(url):
            """自动检测本地 Llama 服务，返回 (url_value, model_choices, status_text)。"""
            url = str(url or "").strip()
            found_url, models, label, err = detect_local_llm(url)
            if found_url:
                return found_url, gr.update(choices=models, value=models[0] if models else None), \
                    f"✅ 检测到 {label}（{found_url}），共 {len(models)} 个模型：{', '.join(models[:5])}{'...' if len(models) > 5 else ''}"
            return url, gr.update(), "❌ " + err

        def rescan_local_llm_models(url):
            """对已确定的 Base URL 重新拉取模型列表（模型热加载后刷新用）。"""
            url = str(url or "").strip()
            models, err = list_local_llm_models(url)
            if models:
                return gr.update(choices=models, value=models[0] if models else None), \
                    f"✅ 已刷新：{len(models)} 个模型：{', '.join(models[:5])}{'...' if len(models) > 5 else ''}"
            return gr.update(), "❌ 获取模型列表失败: " + err

        def apply_local_llm_as_brain(url, key, model):
            """把本地 Llama 模型设为智能体大脑：写入 agent_config 的 api_provider/base_url/api_key/model。

            使用 "local-llama" 作为 provider key 时，load_config 的供应商校验会因
            base_url 是 http URL 而放行；base_url/model 才是真正生效的字段
            （chat_stream 直接读取 cfg['base_url'] 与 cfg['model']）。
            同时把模型注册到 custom_models.llm["local-llama"]，让大脑设置下拉框可同步显示。
            """
            url = str(url or "").strip()
            model = str(model or "").strip()
            if not url:
                # 未填 URL：自动检测一次
                found_url, models, label, err = detect_local_llm("")
                if not found_url:
                    return "", "", "", "❌ " + err, gr.update(), gr.update(), gr.update()
                url = found_url
                model = model or (models[0] if models else "")
            if not url.startswith(("http://", "https://")):
                return url, key, model, "❌ Base URL 必须以 http:// 或 https:// 开头", gr.update(), gr.update(), gr.update()
            if not model:
                return url, key, model, "❌ 请先选择或填写模型 ID（可点「自动检测」填充）", gr.update(), gr.update(), gr.update()
            cfg = load_config(resolve_local=False)
            cfg["api_provider"] = "local-llama"
            cfg["base_url"] = normalize_base_url(url)
            cfg["api_key"] = str(key or "")  # 本地服务通常留空，OpenAI SDK 允许空 key
            cfg["model"] = model
            # 注册到大脑模型列表（local-llama 分组），供大脑设置下拉框使用
            custom = cfg.setdefault("custom_models", {"llm": {}, "image": {}, "video": {}})
            llm_bucket = custom.setdefault("llm", {})
            values = llm_bucket.setdefault("local-llama", [])
            if model not in values:
                values.append(model)
            ok = save_config(cfg)
            if ok:
                # 记住本地服务信息，便于下次 UI 回填
                cfg2 = load_config(resolve_local=False)
                cfg2["local_llm_base_url"] = normalize_base_url(url)
                cfg2["local_llm_api_key"] = str(key or "")
                cfg2["local_llm_model"] = model
                save_config(cfg2)
                llm_choices = _models_for_provider(cfg2, "llm", "local-llama", [])
                return url, key, model, (
                    f"✅ 智能体大脑已切换为本地模型：{model}（{normalize_base_url(url)}）。"
                    "新对话将直接走本地推理，无需云端 API Key。"
                ), gr.update(value="local-llama"), gr.update(choices=llm_choices, value=model), \
                    gr.update(value=normalize_base_url(url))
            return url, key, model, "❌ 保存失败，请重试", gr.update(), gr.update(), gr.update()

        # 上传事件：同步到 State
        upload_all.change(fn=on_all_upload, inputs=[upload_all], outputs=[state_image, state_video, state_attachments])

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
        # 删除自定义供应商（连带删除其名下自定义模型），并重建所有相关下拉
        remove_provider_btn.click(
            fn=remove_custom_provider,
            inputs=[remove_provider_select],
            outputs=[api_provider, image_api_provider, model_name, api_model_select,
                     remove_provider_select, remove_model_select, provider_manage_status],
        )
        # 删除单个自定义模型 ID，并重建相关下拉
        remove_model_btn.click(
            fn=remove_custom_model,
            inputs=[remove_model_select],
            outputs=[model_name, api_model_select, remove_model_select, custom_model_status],
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

        save_settings_btn.click(fn=save_settings, inputs=[api_provider, api_key, base_url, model_name], outputs=[settings_status])
        save_image_settings_btn.click(fn=save_image_settings, inputs=[image_api_provider, image_api_key, image_base_url], outputs=[image_settings_status])
        clear_api_key_btn.click(fn=clear_api_key, outputs=[api_key, settings_status])
        clear_image_api_key_btn.click(fn=clear_image_api_key, outputs=[image_api_key, image_settings_status])
        # 本地 Llama 模块事件
        local_llm_detect_btn.click(
            fn=detect_local_llm_service,
            inputs=[local_llm_url],
            outputs=[local_llm_url, local_llm_model, local_llm_status],
        )
        local_llm_rescan_btn.click(
            fn=rescan_local_llm_models,
            inputs=[local_llm_url],
            outputs=[local_llm_model, local_llm_status],
        )
        local_llm_apply_btn.click(
            fn=apply_local_llm_as_brain,
            inputs=[local_llm_url, local_llm_key, local_llm_model],
            outputs=[local_llm_url, local_llm_key, local_llm_model, local_llm_apply_status, api_provider, model_name, base_url],
        )
        clear_upload_btn.click(fn=lambda: (None, None, None, []), outputs=[upload_all, state_image, state_video, state_attachments])

        # 发送按钮：prepare → send_chat → clear_uploads
        send_event = send_btn.click(
            fn=prepare_message,
            inputs=[msg_input, chatbot, state_image, state_video, state_attachments, local_model_select, api_model_select],
            outputs=[msg_input, chatbot],
        ).then(
            fn=send_chat,
            inputs=[chatbot, state_image, state_video, state_attachments],
            outputs=[chatbot, status],
        ).then(
            fn=clear_uploads,
            outputs=[upload_all, state_image, state_video, state_attachments],
        )

        # 回车发送
        submit_event = msg_input.submit(
            fn=prepare_message,
            inputs=[msg_input, chatbot, state_image, state_video, state_attachments, local_model_select, api_model_select],
            outputs=[msg_input, chatbot],
        ).then(
            fn=send_chat,
            inputs=[chatbot, state_image, state_video, state_attachments],
            outputs=[chatbot, status],
        ).then(
            fn=clear_uploads,
            outputs=[upload_all, state_image, state_video, state_attachments],
        )

        # 「暂停思考」：置位停止标志。chat_stream 在 LLM 流式循环 / 工具迭代循环中检测到后
        # 立即断开 LLM 流并停止（保留已生成的思考/回答内容）。
        # 不使用 Gradio cancels：对 .then() 链会取消错函数（命中末尾的 clear_uploads 而非 send_chat），
        # 且对"正在运行"的函数无效，因此改用显式停止标志。
        pause_btn.click(fn=request_stop, outputs=[status])

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

        # 图像预览容器 CSS：放大显示尺寸（原 240px/224px 过小，"放大也很小"的根源）。
        # 点击放大的 lightbox 由 agent_interface.load(js=...) 注入的客户端 JS 实现
        # （Gradio 5.x 中 gr.HTML 内的 <script> 不会执行，不能用 gr.HTML 注入 JS），
        # 这里只包含纯样式：容器/图片尺寸 + 悬停提示 + lightbox 全屏层样式。
        gr.HTML("""
        <style>
        /* 隐藏 Gradio 原生 fullscreen 按钮（button 元素），保留下载链接（a 元素） */
        #agent-chatbot .icon-button-wrapper > button {
            display: none !important;
        }
        /* 图像预览容器：宽度随聊天列自适应，高度不超过视口 90% 减边距 */
        #agent-chatbot .image-container {
            position: relative;
            display: inline-flex !important;
            width: min(72vw, 800px) !important;
            max-width: min(72vw, 800px) !important;
            max-height: calc(90vh - 160px) !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            padding: 8px !important;
            border: 1px solid rgba(148, 163, 184, 0.32) !important;
            border-radius: 8px !important;
            background: rgba(15, 23, 42, 0.38) !important;
        }
        /* 消息气泡：宽度跟随图片容器，不挤成一列 */
        #agent-chatbot .message-content:has(.image-container),
        #agent-chatbot .message-wrap:has(.image-container),
        #agent-chatbot .prose:has(.image-container) {
            width: fit-content !important;
            max-width: min(72vw, 820px) !important;
        }
        /* 悬停时显示放大提示图标（右上角） */
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
        /* 聊天内图片：填满容器，不再限制 224px */
        #agent-chatbot img,
        #agent-chatbot .image-container img {
            cursor: zoom-in;
            transition: opacity 0.2s;
            width: auto !important;
            height: auto !important;
            max-width: 100% !important;
            max-height: calc(90vh - 180px) !important;
            object-fit: contain !important;
        }
        #agent-chatbot img:hover { opacity: 0.9; }
        /* ===== 点击放大 lightbox 全屏层 ===== */
        #agent-lightbox {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.88);
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
            border-radius: 6px;
            cursor: default;
        }
        #agent-lightbox-close {
            position: fixed;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 30px;
            line-height: 1;
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
        #agent-lightbox-close:hover { background: rgba(0,0,0,0.8); }
        #agent-lightbox-download {
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            color: #fff;
            background: rgba(59, 130, 246, 0.85);
            padding: 8px 18px;
            border-radius: 999px;
            font-size: 14px;
            text-decoration: none;
            z-index: 100000;
        }
        #agent-lightbox-download:hover { background: rgba(59, 130, 246, 1); }
        /* 发送 / 暂停思考 按钮：文本长短不同导致可见宽度不一致，
           强制两者按相同 flex 基准分配，且内部按钮填满容器，保证尺寸完全一致 */
        #agent_send_btn,
        #agent_pause_btn {
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }
        #agent_send_btn,
        #agent_send_btn button,
        #agent_pause_btn,
        #agent_pause_btn button {
            width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }
        </style>
        """)

        # 加载时从服务端获取配置回填到 Textbox
        # Gradio 5.x 的 Textbox 不会通过 HTML 属性渲染初始值（尤其是懒加载标签页），
        # 所以需要在 .load() 回调中从 load_config() 读取并设置。
        def _load_config_to_ui():
            cfg = load_config(resolve_local=False)
            # 本地 Llama 模块：回填已保存的 URL 并尝试拉取模型列表（服务未启动时用已保存模型兜底）
            local_url = cfg.get("local_llm_base_url", "")
            saved_local_model = cfg.get("local_llm_model", "")
            local_models = []
            if local_url:
                local_models, _err = list_local_llm_models(local_url, timeout=2.0)
            if not local_models and saved_local_model:
                local_models = [saved_local_model]
            return (
                cfg.get("api_key", ""),
                cfg.get("image_api_key", "") or cfg.get("video_api_key", ""),
                local_url,
                gr.update(choices=local_models, value=saved_local_model or (local_models[0] if local_models else None)),
            )

        agent_interface.load(
            fn=_load_config_to_ui,
            outputs=[api_key, image_api_key, local_llm_url, local_llm_model],
            js="""() => {
                document.querySelectorAll('textarea').forEach(t => t.setAttribute('autocomplete', 'new-password'));

                // ===== 思考过程块自动滚动（内嵌小窗口，实时跟随最新思考） =====
                // 用 MutationObserver 监听聊天容器，思考块内容增长时把其滚动条拉到底部；
                // 若用户手动上滑查看旧内容（距底部 > 30px）则不打扰。
                (function installThinkAutoScroll() {
                    if (window.__agentThinkScrollInstalled) return;
                    window.__agentThinkScrollInstalled = true;
                    function install(root) {
                        if (!root) return;
                        var observer = new MutationObserver(function() {
                            var bodies = root.querySelectorAll('details.agent-thinking[open] .agent-thinking-body');
                            bodies.forEach(function(b) {
                                var nearBottom = (b.scrollHeight - b.scrollTop - b.clientHeight) < 30;
                                if (nearBottom && b.scrollHeight > b.clientHeight) {
                                    b.scrollTop = b.scrollHeight;
                                }
                            });
                        });
                        observer.observe(root, { childList: true, subtree: true, characterData: true });
                    }
                    var chat = document.getElementById('agent-chatbot');
                    if (chat) { install(chat); }
                    else {
                        setTimeout(function() {
                            install(document.getElementById('agent-chatbot'));
                        }, 500);
                    }
                })();

                // ===== 图像点击放大 lightbox（修复"容器太小、放大也很小"问题） =====
                // 原实现只有 #agent-lightbox 的 CSS，没有任何 JS 创建 DOM / 绑定事件，
                // 点击放大从未生效。这里通过 Gradio 可靠的客户端 js 回调补全完整功能：
                // 点击聊天内任意图片 → 全屏 95vw×95vh 查看；点击遮罩/关闭按钮/Esc → 关闭；底部按钮可下载原图。
                if (window.__agentLightboxInstalled) return;
                window.__agentLightboxInstalled = true;

                var box = document.createElement('div');
                box.id = 'agent-lightbox';
                var img = document.createElement('img');
                img.id = 'agent-lightbox-img';
                var closeBtn = document.createElement('div');
                closeBtn.id = 'agent-lightbox-close';
                closeBtn.textContent = '\\u2715';
                var downloadLink = document.createElement('a');
                downloadLink.id = 'agent-lightbox-download';
                downloadLink.textContent = '\\ud83d\\udcbe 下载原图';
                downloadLink.target = '_blank';
                downloadLink.rel = 'noopener';
                box.appendChild(img);
                box.appendChild(closeBtn);
                box.appendChild(downloadLink);
                document.body.appendChild(box);

                function openBox(src) {
                    if (!src) return;
                    img.src = src;
                    downloadLink.href = src;
                    downloadLink.download = '';
                    box.classList.add('active');
                }
                function closeBox() {
                    box.classList.remove('active');
                    img.src = '';
                    downloadLink.href = '';
                }

                // 文档级事件委托：自动覆盖聊天流式新消息中的图片，无需重新绑定；
                // 且不受 chatbot 元素渲染时序影响（load 回调时 DOM 未就绪也能正常工作）
                document.addEventListener('click', function(e) {
                    var t = e.target;
                    if (!t || !t.tagName) return;
                    var chat = document.getElementById('agent-chatbot');
                    if (!chat || !chat.contains(t)) return;
                    // 只处理图片元素的点击；下载链接/按钮等其他元素保持原生行为
                    var el = (t.tagName === 'IMG') ? t : (t.closest ? t.closest('img') : null);
                    if (!el) return;
                    var src = el.currentSrc || el.src;
                    if (!src) return;
                    e.preventDefault();
                    e.stopPropagation();
                    openBox(src);
                });

                box.addEventListener('click', function(e) {
                    if (e.target === closeBtn || e.target === box) closeBox();
                });
                closeBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    closeBox();
                });
                downloadLink.addEventListener('click', function(e) {
                    e.stopPropagation();
                });

                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape' && box.classList.contains('active')) closeBox();
                });
            }"""
        )

    return [(agent_interface, "绘梦智能体助手", "sd_webui_agent")]
