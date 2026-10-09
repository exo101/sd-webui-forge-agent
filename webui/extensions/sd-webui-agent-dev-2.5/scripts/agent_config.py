# =============================================================================
# Agent Config — 配置加载、本地 LLM 检测、工具函数
# =============================================================================

import os
import sys
import json
import time
import uuid
import tempfile
import traceback
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import shared, scripts, script_callbacks, sd_models, sd_samplers, postprocessing
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    process_images,
)

# 导入工具注册系统（可选，失败不影响主功能）
try:
    from scripts.agent_tools_registry import get_registered_tools, get_tool_function, list_registered_tools
    _REGISTRY_AVAILABLE = True
except Exception:
    _REGISTRY_AVAILABLE = False

# =============================================================================
# 配置
# =============================================================================

EXT_DIR = Path(__file__).parent.parent
CONFIG_PATH = EXT_DIR / "agent_config.json"

DEFAULT_CONFIG = {
    "api_key": "",
    "api_provider": "ModelScope",
    "base_url": "https://api-inference.modelscope.cn/v1",
    "model": "Qwen/Qwen3.8-27B",
    "image_api_key": "",
    "image_api_provider": "ModelScope",
    "image_base_url": "https://api-inference.modelscope.cn/v1",
    "image_model": "",
    "image_aspect_ratio": "1:1",
    "video_api_key": "",
    "video_api_provider": "ModelScope",
    "video_base_url": "https://api-inference.modelscope.cn/v1",
    "video_model": "",
    "custom_providers": [],
    "custom_models": {"llm": {}, "image": {}, "video": {}},
    "max_tool_iterations": 60,
    "max_completion_tokens": 32768,
    "default_steps": 20,
    "default_width": 1024,
    "default_height": 1024,
    "default_cfg_scale": 7.0,
    "memory_enabled": False,
}

API_PROVIDERS = {
    "ModelScope": {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "note": "ModelScope API",
    },
}


def _custom_provider_map(cfg=None):
    """Return custom providers keyed by stable name, accepting old/simple entries."""
    result = {}
    for item in (cfg or {}).get("custom_providers", []) if isinstance(cfg, dict) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("provider") or "").strip()
        url = normalize_base_url(item.get("base_url") or item.get("url"))
        if name and url:
            result[name] = {"base_url": url, "note": "自定义 API 供应商"}
    return result


def all_api_providers(cfg=None):
    providers = dict(API_PROVIDERS)
    providers.update(_custom_provider_map(cfg))
    return providers


def provider_choices(cfg=None):
    """Gradio choices: full URL is shown, internal provider key remains the value."""
    providers = all_api_providers(cfg)
    return [(str(info.get("base_url") or name), name) for name, info in providers.items()]


def normalize_base_url(url):
    """Normalize OpenAI-compatible base URLs."""
    return (url or "").strip().rstrip("/")


def provider_base_url(provider):
    key = str(provider or "").strip()
    info = API_PROVIDERS.get(key)
    if info:
        return info["base_url"]
    # URL 本身也可作为供应商值，便于自定义配置直接迁移。
    if key.startswith(("http://", "https://")):
        return normalize_base_url(key)
    return ""

def _mask_key(key):
    """脱敏显示 API Key，只显示前8位和后4位。"""
    s = str(key or "")
    if len(s) <= 12:
        return "***" if s else "(空)"
    return s[:8] + "..." + s[-4:]


def load_config(resolve_local=True):
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[Agent] 配置加载失败，使用默认配置: {e}")

    cfg["base_url"] = normalize_base_url(cfg.get("base_url"))
    if not isinstance(cfg.get("custom_models"), dict):
        cfg["custom_models"] = {"llm": {}, "image": {}, "video": {}}

    # 供应商兼容：内置和用户自定义供应商都允许使用。
    known_providers = all_api_providers(cfg)
    for _provider_key in ("api_provider", "image_api_provider", "video_api_provider"):
        _p = str(cfg.get(_provider_key) or "").strip()
        # "local-llama" 是本地 LLM 大脑的专用标记（真正生效的是 base_url/model 字段）
        if _p and _p != "local-llama" and _p not in known_providers and not _p.startswith(("http://", "https://")):
            _fallback = "ModelScope"
            print(f"[Agent] {_provider_key}={_p} 已不再支持，回退到 {_fallback}")
            cfg[_provider_key] = _fallback
            _url_key = _provider_key.replace("_api_provider", "_base_url").replace("api_provider", "base_url")
            cfg[_url_key] = provider_base_url(_fallback)

    # 兜底：LLM/image/video 的 base_url 都按各自 api_provider 自动同步，
    # 防止配置文件中残留的旧 URL 与当前 provider 不一致导致请求发到错误上游。
    # 注意：local-llama 走 local_llm_base_url，不能用供应商 URL 覆盖。
    for _provider_key, _url_key in (
        ("api_provider", "base_url"),
        ("image_api_provider", "image_base_url"),
        ("video_api_provider", "video_base_url"),
    ):
        _p = str(cfg.get(_provider_key) or "").strip()
        if _p == "local-llama":
            continue
        _url = normalize_base_url(cfg.get(_url_key))
        _provider_url = known_providers.get(_p, {}).get("base_url") or provider_base_url(_p)
        if _provider_url and _url != _provider_url:
            cfg[_url_key] = _provider_url

    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Agent] 配置保存失败: {e}")
        return False


# =============================================================================
# WebUI 工具函数 (Agent 可调用的 tools)
# =============================================================================

def _get_webui_output_dir():
    """获取 WebUI 的 outputs/agent 目录。"""
    try:
        # 使用 WebUI 标准路径模块获取根目录（webui/）
        from modules import paths
        webui_root = paths.script_path
        output_dir = os.path.join(webui_root, "outputs", "agent")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    except Exception:
        try:
            # 回退：手动计算路径
            # agent_config.py 在 webui/extensions/sd-webui-agent-dev/scripts/
            # 向上 3 级 = webui/
            script_dir = os.path.dirname(os.path.abspath(__file__))
            webui_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
            output_dir = os.path.join(webui_root, "outputs", "agent")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        except Exception:
            return tempfile.gettempdir()


def _save_pil_to_tempfile(img):
    """保存 PIL Image 到 WebUI outputs/agent 目录，返回文件路径。用于 Gradio Chatbot 显示图片。"""
    if not isinstance(img, Image.Image):
        return None
    try:
        # 优先保存到 WebUI output 目录
        output_dir = _get_webui_output_dir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"agent-{timestamp}-{unique_id}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, format="PNG")
        print(f"[Agent] 图像已保存到: {filepath}")
        return filepath
    except Exception as e:
        print(f"[Agent] 保存到 output 目录失败，回退到临时文件: {e}")
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name, format="PNG")
            tmp.close()
            return tmp.name
        except Exception as e2:
            print(f"[Agent] 保存临时图片也失败: {e2}")
            return None


def _get_current_checkpoint():
    try:
        if hasattr(shared.opts, "sd_model_checkpoint") and shared.opts.sd_model_checkpoint:
            return shared.opts.sd_model_checkpoint
    except Exception:
        pass
    return "unknown"


def _get_sampler():
    return getattr(shared.opts, "sampler_name", "Euler a") if hasattr(shared.opts, "sampler_name") else "Euler a"


def _scan_model_dir(subdir, extensions=None):
    """直接扫描模型目录，返回相对路径列表。"""
    if extensions is None:
        extensions = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf")
    models_dir = getattr(shared.opts, "models_dir", None)
    if not models_dir:
        # 回退：使用 modules.paths.models_path（WebUI 的标准模型目录）
        try:
            from modules import paths
            models_dir = paths.models_path
        except Exception:
            # 最后回退到脚本目录下的 models
            models_dir = os.path.join(scripts.basedir(), "models")
    target_dir = os.path.join(models_dir, subdir)
    results = []
    if os.path.isdir(target_dir):
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if f.endswith(extensions):
                    rel = os.path.relpath(os.path.join(root, f), target_dir)
                    results.append(rel.replace("\\", "/"))
    return sorted(results)


# =============================================================================
# 📐 画面比例设置（Agent Aspect Ratio）
# - 设置页新增「📐 画面比例」选项卡：1:1 / 9:16 / 16:9，持久化到 agent_config.json
# - 作为所有图像模型（本地模型与 API 模型）的默认输出比例；对话中明确指定的比例优先。
# =============================================================================

ASPECT_RATIOS = ["1:1", "9:16", "16:9", "参考图比例"]


def _ar_cfg():
    try:
        return load_config() or {}
    except Exception:
        return {}


def _ui_ratio():
    value = str(_ar_cfg().get("image_aspect_ratio") or "1:1").strip().lower()
    return value if value in {x.lower() for x in ASPECT_RATIOS} else "1:1"


def _save_ratio():
    """onchange 回调：把画面比例持久化到 agent_config.json。

    注意：opts.set() 调用 onchange() 时不传参数，新值已写入 shared.opts。
    """
    choice = str(getattr(shared.opts, "agent_image_aspect_ratio", "1:1") or "1:1").strip()
    if choice not in ASPECT_RATIOS:
        choice = "1:1"
    try:
        cfg = _ar_cfg()
        cfg["image_aspect_ratio"] = choice
        save_config(cfg)
        print(f"[Agent] 画面比例已设置为: {choice}")
    except Exception as e:
        print(f"[Agent] 保存画面比例失败: {e}")
        traceback.print_exc()


def _on_ar_ui_settings():
    """注册「📐 画面比例」WebUI 设置项。

    Forge 中 on_ui_settings 回调执行时没有活动的 gr.Blocks 上下文，
    直接创建 gr.Dropdown 并绑定 .change() 会抛
    'Cannot call change outside of a gradio.Blocks context'。
    改用 shared.opts.add_option() 注册标准设置项后，WebUI 会在自己的
    Blocks 上下文内自动创建 Dropdown、处理持久化和保存，功能完全保留。
    """
    from modules.options import OptionInfo

    try:
        current = _ui_ratio()
    except Exception:
        current = "1:1"
    shared.opts.add_option(
        "agent_image_aspect_ratio",
        OptionInfo(
            current,
            "📐 绘梦助手 · 默认画面比例",
            gr.Dropdown,
            {"choices": list(ASPECT_RATIOS)},
            onchange=_save_ratio,
            section=("agent_ar", "📐 画面比例"),
        ).info(
            "所有图像模型（本地与 API）的默认画面比例；对话中明确指定的比例优先。"
            "像素参考：1:1=1024×1024｜9:16=768×1344（竖版）｜16:9=1344×768（横版）。"
            "SDXL 等本地模型请勿使用过大尺寸，否则质量下降。"
            "对话中明确要求其他比例时以对话优先。"
        ),
    )

try:
    script_callbacks.on_ui_settings(_on_ar_ui_settings)
except Exception as e:
    print(f"[Agent] 注册画面比例功能失败: {e}")
    traceback.print_exc()


# =============================================================================
# 🖥️ 本地 LLM 自动检测（llama.cpp / Ollama / LM Studio / vLLM / TGI 等）
# 启动器可直接拉起本地推理服务，这里只负责按常见端口自动探测 OpenAI 兼容接口。
# =============================================================================

# (候选 Base URL, 服务标签) — 按常见本地服务端口排列
LOCAL_LLM_CANDIDATE_ENDPOINTS = [
    ("http://127.0.0.1:1234/v1", "LM Studio"),
    ("http://127.0.0.1:8080/v1", "llama.cpp server"),
    ("http://127.0.0.1:11434/v1", "Ollama"),
    ("http://127.0.0.1:5000/v1", "text-generation-webui"),
    ("http://127.0.0.1:8000/v1", "vLLM"),
    ("http://localhost:1234/v1", "LM Studio (localhost)"),
    ("http://localhost:8080/v1", "llama.cpp server (localhost)"),
]


def list_local_llm_models(base_url, timeout=3.0):
    """从 OpenAI 兼容的本地服务拉取 /models 列表。返回 (models, error)。"""
    import requests

    url = normalize_base_url(base_url)
    if not url:
        return [], "Base URL 为空"
    try:
        resp = requests.get(url.rstrip("/") + "/models", timeout=timeout)
        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code}"
        data = resp.json()
        models = [str(m.get("id")).strip() for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        models = [m for m in dict.fromkeys(models) if m]
        return models, ""
    except Exception as e:
        return [], str(e)


def detect_local_llm(base_url="", timeout=2.0):
    """自动检测本地 Llama 服务：优先用户填写的 URL，再依次扫描常见端口。

    返回 (base_url, models, service_label, error)；未检测到时 base_url 为 None。
    """
    candidates = []
    custom = normalize_base_url(base_url)
    if custom:
        candidates.append((custom, "自定义填写"))
    candidates.extend(LOCAL_LLM_CANDIDATE_ENDPOINTS)

    tried = []
    for url, label in candidates:
        models, err = list_local_llm_models(url, timeout=timeout)
        tried.append(label)
        if models:
            print(f"[Agent] 本地 LLM 检测命中: {label} {url} -> {len(models)} 个模型")
            return url, models, label, ""
    return None, [], "", (
        "未检测到本地 Llama 服务（已尝试: " + "、".join(tried) + "）。"
        "请先用启动器启动 llama.cpp / Ollama / LM Studio 等本地推理服务，"
        "或手动填写 Base URL 后再次检测。"
    )
