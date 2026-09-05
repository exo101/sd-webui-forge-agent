# =============================================================================
# Agent Config — 配置加载、本地 LLM 检测、工具函数
# =============================================================================

import os
import io
import sys
import json
import time
import base64
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

# 端口 → 模型关键词映射（用户的启动器规则）
# 8080=4B, 8079=2B, 其余端口自动选第一个模型
PORT_MODEL_MAP = {
    8080: ("4b", "4B"),
    8079: ("2b", "2B"),
}
LLAMA_PORT_SCAN = list(range(8079, 8091))  # 扫描 8079-8090

DEFAULT_CONFIG = {
    "api_key": "ms-7f81b5d8-78a6-4daa-a33c-052a856f1196",
    "base_url": "https://api-inference.modelscope.cn/v1",
    "model": "Qwen/Qwen3.8-27B",
    "local_mode": True,  # 优先使用本地 llama-server，找不到则回退云端
    "max_tool_iterations": 8,
    "default_steps": 20,
    "default_width": 1024,
    "default_height": 1024,
    "default_cfg_scale": 7.0,
}

# 本地检测缓存（避免每次对话都扫描端口）
_local_detection_cache = None
_local_detection_time = 0
_LOCAL_CACHE_TTL = 60  # 缓存 60 秒


def _detect_local_llama():
    """检测本地 llama-server，自动匹配模型。

    端口规则：
    - 8080 → 4B 模型
    - 8079 → 2B 模型
    - 其余端口 → 自动选第一个可用模型

    返回: (base_url, api_key, model) 或 None（未找到本地服务）
    """
    global _local_detection_cache, _local_detection_time
    import time as _time

    # 缓存命中
    now = _time.time()
    if _local_detection_cache is not None and (now - _local_detection_time) < _LOCAL_CACHE_TTL:
        return _local_detection_cache

    import urllib.request
    import urllib.error

    for port in LLAMA_PORT_SCAN:
        try:
            url = f"http://localhost:{port}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if not models:
                    continue

                # 端口模型匹配规则
                model_id = None
                if port in PORT_MODEL_MAP:
                    keyword, label = PORT_MODEL_MAP[port]
                    for m in models:
                        mid = m.get("id", "")
                        if keyword in mid.lower() or label.lower() in mid.lower():
                            model_id = mid
                            break
                    # 如果指定端口没找到对应模型，回退到第一个
                    if model_id is None:
                        model_id = models[0].get("id")
                else:
                    # 其他端口选第一个
                    model_id = models[0].get("id")

                result = (f"http://localhost:{port}/v1", "local", model_id)
                _local_detection_cache = result
                _local_detection_time = now
                print(f"[Agent] 检测到本地 llama-server: 端口 {port} → 模型 {model_id}")
                return result
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            continue
        except Exception:
            continue

    _local_detection_cache = None
    _local_detection_time = now
    return None


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[Agent] 配置加载失败，使用默认配置: {e}")

    # 本地模式：自动检测 llama-server
    if cfg.get("local_mode", True):
        detected = _detect_local_llama()
        if detected:
            cfg["base_url"], cfg["api_key"], cfg["model"] = detected
            print(f"[Agent] 使用本地模型: {cfg['model']} @ {cfg['base_url']}")
        else:
            print("[Agent] 未检测到本地 llama-server，回退到云端模型")

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

def _save_pil_to_tempfile(img):
    """保存 PIL Image 到临时文件，返回文件路径。用于 Gradio Chatbot 显示图片。"""
    if not isinstance(img, Image.Image):
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, format="PNG")
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[Agent] 保存临时图片失败: {e}")
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