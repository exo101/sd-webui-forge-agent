"""MiniMax H3 本地 WebUI 后端：在 Forge 进程内直接用 DiffSynth pipeline 运行 H3。

不依赖 ComfyUI。模型来源策略：
- DiT / 文本编码器：直接复用本地量化权重（diffsynth 按文件 hash 自动识别型号，
  包括 Comfy-Org int8_convrot 与 DiffSynth-Studio NF4 系列，放入模型目录即可自动启用）
- 视频 VAE / 音频 VAE / processor：优先复用 webui/models/vae 下的 Comfy-Org 命名文件
  （经 state_dict 键严格校验后注册 hash 直接加载）；缺失时首次运行从 ModelScope
  自动下载，缓存在 webui/models/ 下
  VAE 变体可通过 local_vae_variant 选择：original（INT8 组合配套的标准 fp16/fp32
  VAE，MiniMax/MiniMax-H3）或 nf4（DiffSynth-Studio/MiniMax-H3-NF4 的 4bit 量化版）
- 显存/内存策略：通过 vram_strategy 配置自动检测硬件选择最优策略
  （performance / save_memory / save_vram / extreme），详见 resolve_vram_strategy
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .cloud_client import CLOUD_UPLOAD_DIR
from .config import EXTENSION_ROOT, load_config
from .errors import H3StudioError

# 部分云端环境（如共享 GPU 实例）的 CA 证书链不完整，导致 modelscope.cn
# SSL 验证失败（CERTIFICATE_VERIFY_FAILED）。若用户显式设置了
# H3_DISABLE_SSL_VERIFY=1，则全局跳过 HTTPS 证书验证，确保 VAE 等模型可下载。
if os.environ.get("H3_DISABLE_SSL_VERIFY") == "1":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("MODELSCOPE_SDK_DEBUG", "0")
    print("[H3 Studio] 已按 H3_DISABLE_SSL_VERIFY=1 跳过 HTTPS 证书验证", flush=True)

WEBUI_DIR = EXTENSION_ROOT.parent.parent
LOCAL_OUTPUT_DIR = WEBUI_DIR / "outputs" / "h3_video"
MODEL_CACHE_DIR = WEBUI_DIR / "models"
# H3 权重首选目录（webui 自包含，不依赖 ComfyUI；
# 下载器“MiniMax-H3-INT8 / NF4”组合直接写入 models/diffusion_models 与
# models/text_encoder 子目录，无额外嵌套）
WEBUI_H3_MODELS_DIR = MODEL_CACHE_DIR
# 文本编码器子目录：webui 惯用单数 text_encoder（与 webui 其他 TE 同目录），
# ComfyUI 惯用复数 text_encoders（兼容已存在的权重）
TE_SUBDIRS = ("text_encoder", "text_encoders")
H3_REPO_ID = "MiniMax/MiniMax-H3"
H3_NF4_REPO_ID = "DiffSynth-Studio/MiniMax-H3-NF4"
VIDEO_VAE_PATTERN = "FL2VA/video_vae/source/model.safetensors"
AUDIO_VAE_PATTERN = "FL2VA/audio_vae/model.safetensors"
NF4_VIDEO_VAE_PATTERN = "video_vae_nf4.safetensors"
NF4_AUDIO_VAE_PATTERN = "audio_vae_nf4.safetensors"
PROCESSOR_PATTERN = "FL2VA/processor/"
VAE_VARIANTS = {
    "original": {
        "repo": H3_REPO_ID,
        "video_pattern": VIDEO_VAE_PATTERN,
        "audio_pattern": AUDIO_VAE_PATTERN,
        "label": "INT8 组合 fp16/fp32",
    },
    "nf4": {
        "repo": H3_NF4_REPO_ID,
        "video_pattern": NF4_VIDEO_VAE_PATTERN,
        "audio_pattern": NF4_AUDIO_VAE_PATTERN,
        "label": "NF4 量化",
    },
}
REFERENCE_FPS = 24
OUTPUT_AUDIO_SAMPLE_RATE = 32000


# ── 硬件检测与显存策略 ───────────────────────────────────────────────────────

def _detect_total_vram_gb() -> float:
    """检测当前 GPU 总显存（GB），失败返回 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _detect_total_ram_gb() -> float:
    """检测系统总内存（GB），失败返回 0。"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    # Windows 回退：ctypes 调用 GlobalMemoryStatusEx
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 0.0


# 策略预设：(offload_device, use_gc, use_gc_offload, tiled, label)
_STRATEGY_PRESETS: dict[str, tuple[str, bool, bool, bool, str]] = {
    "performance": ("cpu", False, False, True, "高性能模式"),
    "save_memory": ("disk", False, False, True, "省内存模式"),
    "save_vram": ("cpu", True, True, True, "省显存模式"),
    "extreme": ("disk", True, True, True, "极致省资源模式"),
}


def resolve_vram_strategy(config: dict[str, Any]) -> dict[str, Any]:
    """根据配置策略（或自动检测）返回 vram 管理参数。

    返回字段：
        offload_device / onload_device: "cpu" 或 "disk"
        use_gradient_checkpointing / use_gradient_checkpointing_offload: bool
        tiled: bool
        label: 策略名称（用于日志）
        vram_gb / ram_gb: 检测到的硬件值
    """
    strategy = str(config.get("vram_strategy") or "auto").strip().lower()
    vram_gb = _detect_total_vram_gb()
    ram_gb = _detect_total_ram_gb()

    if strategy not in _STRATEGY_PRESETS:
        # auto：根据硬件自动选择
        # 阈值说明：
        #   - 显存充足（≥20GB）：DiT NF4 ~7GB + 激活值，不开 gc 也能跑
        #   - 内存充足（≥24GB）：H3 全模型 NF4 约 12-15GB，cpu offload 不爆内存
        vram_sufficient = vram_gb >= 20.0
        ram_sufficient = ram_gb >= 24.0

        if vram_sufficient and ram_sufficient:
            strategy = "performance"
        elif vram_sufficient and not ram_sufficient:
            strategy = "save_memory"
        elif not vram_sufficient and ram_sufficient:
            strategy = "save_vram"
        else:
            strategy = "extreme"

    offload_device, use_gc, use_gc_offload, tiled, label = _STRATEGY_PRESETS[strategy]
    return {
        "offload_device": offload_device,
        "onload_device": offload_device,
        "use_gradient_checkpointing": use_gc,
        "use_gradient_checkpointing_offload": use_gc_offload,
        "tiled": tiled,
        "label": label,
        "strategy": strategy,
        "vram_gb": round(vram_gb, 1),
        "ram_gb": round(ram_gb, 1),
    }


def build_vram_config(strategy: dict[str, Any]) -> dict[str, Any]:
    """根据策略生成 diffsynth ModelConfig 所需的 vram 配置。

    官方推荐配置：onload_device="cpu", preparing_device="cuda"。
    权重加载后常驻 CPU，运行时 preparing() 按需搬到 GPU，
    vram_limit 控制同时驻留 GPU 的模块数，平衡显存与速度。
    """
    import torch
    device = strategy["offload_device"]
    return {
        "offload_dtype": "disk" if device == "disk" else torch.bfloat16,
        "offload_device": device,
        "onload_dtype": "disk" if device == "disk" else torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


class LocalJobCancelled(Exception):
    """本地任务被用户中断。"""


def iter_models_dirs(config: dict[str, Any]) -> list[Path]:
    """所有候选 H3 权重目录（按优先级去重）。

    顺序：设置中显式指定 → 环境变量 → webui/models（diffusion_models /
    text_encoder(s) 子目录，下载器默认写入）→ ComfyUI models 目录（兼容已存在的权重）。
    """
    candidates: list[Path] = []
    configured = str(config.get("local_models_dir") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    env_dir = os.environ.get("H3_STUDIO_MODELS_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates += [
        WEBUI_H3_MODELS_DIR,
        WEBUI_DIR.parent / "ComfyUI-aki-v2.2" / "ComfyUI" / "models",
        WEBUI_DIR.parent / "ComfyUI" / "models",
    ]
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            te_subdirs = tuple(sub for sub in TE_SUBDIRS if (candidate / sub).is_dir())
            if (candidate / "diffusion_models").is_dir() or te_subdirs:
                result.append(candidate)
        except OSError:
            continue
    return result


def resolve_models_dir(config: dict[str, Any]) -> Path | None:
    """第一个可用的 H3 权重目录（无可用时返回 None）。"""
    dirs = iter_models_dirs(config)
    return dirs[0] if dirs else None


def _known_model_names(hash_value: str) -> set[str]:
    try:
        from diffsynth.configs import MODEL_CONFIGS
    except Exception:
        return set()
    return {
        str(item.get("model_name") or "")
        for item in MODEL_CONFIGS
        if item.get("model_hash") == hash_value
    }


def _hash_safetensors(path: Path) -> str | None:
    try:
        from diffsynth.core.loader.file import hash_model_file
        return hash_model_file(str(path))
    except Exception:
        return None


def _list_h3_files(directory: Path, model_name: str) -> list[str]:
    """列出目录中属于指定 H3 组件的 safetensors 文件名。

    识别策略（双重保险）：
    1. 优先用 diffsynth 的文件哈希匹配 MODEL_CONFIGS（精确识别型号）
    2. 哈希识别失败时（diffsynth 未安装 / MODEL_CONFIGS 无该模型 / 哈希计算失败），
       回退到文件名特征匹配，避免因 diffsynth 版本问题导致已有模型被漏报
    """
    found: list[str] = []
    if not directory.is_dir():
        return found

    # 文件名 fallback 规则：根据要找的组件类型，匹配文件名中的关键词
    def _match_by_filename(filename: str, target: str) -> bool:
        name_lower = filename.lower()
        if target == "minimax_h3_dit":
            # DiT 主模型：含 h3 且不含 vae / text_encoder / qwen（TE 通常含 qwen）
            return "h3" in name_lower and not any(
                k in name_lower for k in ("vae", "text_encoder", "qwen", "audio")
            )
        if target == "minimax_h3_text_encoder":
            # 文本编码器：含 qwen 或 text_encoder
            return "qwen" in name_lower or "text_encoder" in name_lower
        if target == "minimax_h3_video_vae":
            return "vae" in name_lower and "video" in name_lower
        if target == "minimax_h3_audio_vae":
            return "vae" in name_lower and "audio" in name_lower
        # 通用 VAE：含 vae
        if "vae" in target:
            return "vae" in name_lower
        return False

    for path in sorted(directory.glob("*.safetensors")):
        # 策略 1：diffsynth 哈希识别
        hash_value = _hash_safetensors(path)
        if hash_value and model_name in _known_model_names(hash_value):
            found.append(path.name)
            continue
        # 策略 2：文件名 fallback
        if _match_by_filename(path.name, model_name):
            found.append(path.name)
    return found


def find_lora_path(config: dict[str, Any], lora_name: str) -> Path | None:
    """根据 LoRA 文件名在所有候选 loras 目录中查找，返回完整路径。"""
    models_dirs = iter_models_dirs(config)
    # WebUI 惯例用 Lora（单数），ComfyUI 惯例用 loras（复数），都支持
    lora_dirs = [
        MODEL_CACHE_DIR / "Lora",
        MODEL_CACHE_DIR / "loras",
        MODEL_CACHE_DIR / "Lora" / "minimax_h3",
        MODEL_CACHE_DIR / "loras" / "minimax_h3",
    ]
    lora_dirs += [d / "Lora" for d in models_dirs]
    lora_dirs += [d / "loras" for d in models_dirs]
    lora_dirs += [d / "Lora" / "minimax_h3" for d in models_dirs]
    lora_dirs += [d / "loras" / "minimax_h3" for d in models_dirs]
    for directory in lora_dirs:
        candidate = directory / lora_name
        if candidate.is_file():
            return candidate
    return None


def resolve_processor_dir(config: dict[str, Any]) -> Path | None:
    """返回已存在的 H3 processor 目录；不存在时返回 None（由调用方触发下载）。"""
    configured = str(config.get("local_processor_path") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_dir() else None
    # 检查多个可能的 processor 位置（original + nf4）
    candidates = [
        MODEL_CACHE_DIR / H3_REPO_ID.replace("/", os.sep) / "FL2VA" / "processor",
        MODEL_CACHE_DIR / H3_NF4_REPO_ID.replace("/", os.sep) / "FL2VA" / "processor",
        MODEL_CACHE_DIR / H3_NF4_REPO_ID.replace("/", os.sep) / "processor",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def vae_variant(config: dict[str, Any], dit_path: Path | None = None) -> str:
    """VAE 变体选择：优先自动检测 DiT 量化格式，否则回退配置。

    - 若提供 dit_path：根据 DiT 文件名/state_dict 自动判断 nf4 或 original
    - 否则：读取 config["local_vae_variant"]（默认 original）
    """
    if dit_path is not None:
        return detect_vae_variant_from_dit(dit_path)
    variant = str(config.get("local_vae_variant") or "original").strip() or "original"
    return variant if variant in VAE_VARIANTS else "original"


def detect_vae_variant_from_dit(dit_path: Path) -> str:
    """根据 DiT 模型文件自动检测应使用的 VAE 变体。"""
    name = dit_path.name.lower()
    if "nf4" in name:
        return "nf4"
    # 读取 state_dict dtype 二次确认
    try:
        from backend.utils import weight_dtype
        from safetensors.torch import load_file
        sd = load_file(str(dit_path), device="cpu")
        dtype = weight_dtype(sd)
        del sd
        if dtype == "nf4":
            return "nf4"
    except Exception:
        pass
    return "original"


def vae_cache_path(variant: str, which: str) -> Path:
    """某变体下视频/音频 VAE 在本地缓存中的预期路径（which: "video" | "audio"）。"""
    spec = VAE_VARIANTS[variant]
    pattern = spec["video_pattern"] if which == "video" else spec["audio_pattern"]
    return MODEL_CACHE_DIR / spec["repo"].replace("/", os.sep) / pattern


# ── 本地 VAE 复用（Comfy-Org 命名）────────────────────────────────────────────
# 用户可能已把 Comfy-Org 仓库的 VAE（minimax_h3_video_vae_*.safetensors /
# minimax_h3_audio_vae_*.safetensors）放在 webui/models/vae。这类文件与
# diffsynth 期望的文件字节不同，hash 不在 diffsynth 白名单里，直接加载会报
# "Cannot detect the model type"。这里先按 state_dict 键严格校验兼容性，
# 通过后再把文件 hash 注册进 diffsynth 的 MODEL_CONFIGS 白名单，以本地路径
# 直接加载；校验不通过则回退自动下载。
LOCAL_VAE_SPECS = {
    "video": {
        "pattern": "minimax_h3_video_vae*.safetensors",
        "model_name": "minimax_h3_video_vae",
        "model_class": "diffsynth.models.minimax_h3_video_vae.MiniMaxH3VideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.minimax_h3_video_vae.MiniMaxH3VideoVAEStateDictConverter",
    },
    "audio": {
        "pattern": "minimax_h3_audio_vae*.safetensors",
        "model_name": "minimax_h3_audio_vae",
        "model_class": "diffsynth.models.minimax_h3_audio_vae.MiniMaxH3AudioVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.minimax_h3_audio_vae.MiniMaxH3AudioVAEStateDictConverter",
    },
}
# 允许的额外键（非模型参数，随 checkpoint 附带的统计量）
_VAE_TOLERATED_EXTRA_KEYS = {"latents_mean", "latents_std"}
_VAE_VERIFY_CACHE: dict[str, bool] = {}
_REGISTERED_VAE_HASHES: set[str] = set()


def _import_dotted(dotted: str) -> Any:
    import importlib

    module_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _split_weight_norm(state_dict: dict, model_state_keys: set[str]) -> dict:
    """把 state_dict 中的 *.weight 拆成 *.weight_g / *.weight_v。

    diffsynth 的音频 VAE 使用 WeightNormedConv1d/ConvTranspose1d，模型参数
    为 weight_g + weight_v，但部分第三方 checkpoint 只保存合并后的 weight。
    这里按 PyTorch weight_norm 公式还原：
        weight_g = norm(weight, dim=(1,2), keepdim=True)
        weight_v = weight
    """
    import torch

    result = dict(state_dict)
    weight_keys = [k for k in result if k.endswith(".weight")]
    for k in weight_keys:
        prefix = k[: -len(".weight")]
        g_key = f"{prefix}.weight_g"
        v_key = f"{prefix}.weight_v"
        # 只在模型确实期望 weight_g/weight_v 时才拆分
        if g_key not in model_state_keys or v_key not in model_state_keys:
            continue
        w = result.pop(k)
        if w.dim() >= 3:
            g = torch.norm(w.float(), dim=tuple(range(1, w.dim())), keepdim=True).to(w.dtype)
        else:
            g = torch.norm(w.float(), dim=1, keepdim=True).to(w.dtype)
        result[g_key] = g
        result[v_key] = w
    return result


def _verify_local_vae(path: Path, spec: dict) -> bool:
    """严格校验本地 VAE 文件能否被 diffsynth 模型类加载（结果按路径缓存）。"""
    key = str(path)
    if key in _VAE_VERIFY_CACHE:
        return _VAE_VERIFY_CACHE[key]
    ok = False
    try:
        from safetensors.torch import load_file

        model = _import_dotted(spec["model_class"])()
        model_keys = set(model.state_dict().keys())
        converter = _import_dotted(spec["state_dict_converter"])
        converted = converter(load_file(str(path), device="cpu"))
        missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
        ok = not missing and set(unexpected) <= _VAE_TOLERATED_EXTRA_KEYS
        if not ok and missing:
            # fallback：尝试把合并的 weight 拆成 weight_g/weight_v
            converted2 = _split_weight_norm(converted, model_keys)
            missing2, unexpected2 = model.load_state_dict(converted2, strict=False, assign=True)
            ok = not missing2 and set(unexpected2) <= _VAE_TOLERATED_EXTRA_KEYS
    except Exception:
        ok = False
    _VAE_VERIFY_CACHE[key] = ok
    return ok


def find_local_vae(config: dict[str, Any], which: str, variant: str | None = None) -> Path | None:
    """在 webui/models/vae 与候选模型目录的 vae 子目录中查找可用的本地 VAE。

    - original 变体：搜索 minimax_h3_video_vae*.safetensors / minimax_h3_audio_vae*.safetensors
    - nf4 变体：搜索 *nf4*.safetensors 中匹配 video/audio 的文件
    variant 为 None 时从 config 读取。
    """
    v = variant or vae_variant(config)
    # Linux 大小写敏感，同时扫描 vae / VAE / Vae
    dirs: list[Path] = [MODEL_CACHE_DIR / "vae", MODEL_CACHE_DIR / "VAE", MODEL_CACHE_DIR / "Vae"]
    for d in iter_models_dirs(config):
        dirs += [d / "vae", d / "VAE", d / "Vae"]
    # 也扫描 DiT 同级目录（如 webui/models/MiniMax），用户常把 VAE 放在那里
    dirs += [MODEL_CACHE_DIR / "MiniMax"]
    dirs += list(iter_models_dirs(config))
    # NF4 模型缓存目录（DiffSynth-Studio/MiniMax-H3-NF4）
    dirs += [MODEL_CACHE_DIR / "DiffSynth-Studio" / "MiniMax-H3-NF4"]
    seen: set[str] = set()

    if v == "nf4":
        # NF4 VAE：按文件名匹配，无需 diffsynth converter 校验
        keyword = "video" if which == "video" else "audio"
        for directory in dirs:
            try:
                if not directory.is_dir():
                    continue
                key = str(directory.resolve())
                if key in seen:
                    continue
                seen.add(key)
            except OSError:
                continue
            for path in sorted(directory.glob("*.safetensors")):
                name = path.name.lower()
                if "nf4" in name and keyword in name and "vae" in name:
                    return path
        return None

    spec = LOCAL_VAE_SPECS[which]
    fallback_path: Path | None = None
    for directory in dirs:
        try:
            if not directory.is_dir():
                continue
            key = str(directory.resolve())
            if key in seen:
                continue
            seen.add(key)
        except OSError:
            continue
        for path in sorted(directory.glob(spec["pattern"])):
            if _verify_local_vae(path, spec):
                return path
            # 验证未通过但文件存在：优先用本地文件，避免回退到 ModelScope 下载
            # （云端可能因 SSL/网络问题无法下载）。加载失败时 diffsynth 会报错。
            if fallback_path is None:
                fallback_path = path
                print(f"[H3 Studio] 警告：本地 VAE 严格校验未通过，仍尝试使用 {path.name}（若加载失败请重新从模型下载器获取）", flush=True)
    return fallback_path


_PATCHED_VAE_CONVERTERS: set[str] = set()


def _patch_vae_converter(spec: dict) -> None:
    """Monkey-patch 原始 VAE converter，使其支持 weight norm 拆分。

    diffsynth 通过 dotted path 字符串导入 converter，所以不能直接传函数对象。
    这里把原始 converter 替换成包装版本（保留原始 dotted path），diffsynth 后续
    导入时拿到的就是支持 weight norm 拆分的版本。
    """
    dotted = spec["state_dict_converter"]
    if dotted in _PATCHED_VAE_CONVERTERS:
        return
    import importlib

    module_path, _, func_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    original_converter = getattr(module, func_name)
    model_class = _import_dotted(spec["model_class"])
    model_keys: set[str] | None = None

    def patched_converter(state_dict):
        nonlocal model_keys
        converted = original_converter(state_dict)
        # 移除模型不期望的额外统计键（latents_mean/latents_std）
        for extra_key in ("latents_mean", "latents_std"):
            converted.pop(extra_key, None)
        if model_keys is None:
            model_keys = set(model_class().state_dict().keys())
        has_weight = any(k.endswith(".weight") for k in converted)
        has_weight_g = any(k.endswith(".weight_g") for k in converted)
        if has_weight and not has_weight_g:
            converted = _split_weight_norm(converted, model_keys)
        return converted

    setattr(module, func_name, patched_converter)
    _PATCHED_VAE_CONVERTERS.add(dotted)
    print(f"[H3 Studio] 已增强 VAE converter：{dotted}", flush=True)


def _register_local_vae_hash(path: Path, spec: dict) -> None:
    """把本地 VAE 文件 hash 注册进 diffsynth 模型白名单（进程内生效）。"""
    from diffsynth.core.loader.file import hash_model_file
    from diffsynth.models import model_loader

    hash_value = hash_model_file(str(path))
    if hash_value in _REGISTERED_VAE_HASHES:
        return
    if any(item.get("model_hash") == hash_value for item in model_loader.MODEL_CONFIGS):
        _REGISTERED_VAE_HASHES.add(hash_value)
        return
    # 先增强 converter（支持 weight norm 拆分），再用原始 dotted path 注册
    _patch_vae_converter(spec)
    entry = {
        "model_hash": hash_value,
        "model_name": spec["model_name"],
        "model_class": spec["model_class"],
        "state_dict_converter": spec["state_dict_converter"],
    }
    model_loader.MODEL_CONFIGS = tuple(model_loader.MODEL_CONFIGS) + (entry,)
    _REGISTERED_VAE_HASHES.add(hash_value)
    print(f"[H3 Studio] 本地 VAE 已注册进 diffsynth 白名单：{path.name}（{spec['model_name']}）")


def local_catalog(config: dict[str, Any]) -> dict[str, Any]:
    """本地模式模型目录：DiT/文本编码器/VAE 均显示实际文件名。"""
    models_dirs = iter_models_dirs(config)
    errors: list[str] = []
    dit_files: list[str] = []
    te_files: list[str] = []
    for models_dir in models_dirs:
        for name in _list_h3_files(models_dir / "diffusion_models", "minimax_h3_dit"):
            if name not in dit_files:
                dit_files.append(name)
        for sub in TE_SUBDIRS:
            for name in _list_h3_files(models_dir / sub, "minimax_h3_text_encoder"):
                if name not in te_files:
                    te_files.append(name)
    if not models_dirs:
        errors.append("未找到本地模型目录（已探测 webui/models 与 ComfyUI models，均无 H3 权重），可在模型下载器一键下载 H3 组合（写入 webui/models/diffusion_models 与 text_encoder），或在设置中指定“本地模型目录”")
    else:
        if not dit_files:
            errors.append("diffusion_models 中未找到可识别的 MiniMax H3 DiT 权重")
        if not te_files:
            errors.append("text_encoder / text_encoders 中未找到可识别的 MiniMax H3 文本编码器")
    ready = bool(dit_files and te_files)

    # DiT 排序：pruned 模型优先（显存占用更小，适合 16GB 显卡）
    dit_files.sort(key=lambda name: (0 if "pruned" in name.lower() else 1, name))

    # VAE：自动检测所有可用的本地 VAE 文件（original + nf4）
    vae_files: list[str] = []
    # Linux 大小写敏感，同时扫描 vae / VAE / Vae 三种常见命名
    vae_dirs = [MODEL_CACHE_DIR / "vae", MODEL_CACHE_DIR / "VAE", MODEL_CACHE_DIR / "Vae"]
    for d in models_dirs:
        vae_dirs += [d / "vae", d / "VAE", d / "Vae"]
    # 也扫描 DiT 同级目录（如 webui/models/MiniMax），用户常把 VAE 放在那里
    vae_dirs += [MODEL_CACHE_DIR / "MiniMax"]
    vae_dirs += list(models_dirs)
    # 也包含 nf4 缓存目录
    for variant_spec in VAE_VARIANTS.values():
        cache_dir = MODEL_CACHE_DIR / variant_spec["repo"].replace("/", os.sep)
        if cache_dir.is_dir():
            vae_dirs.append(cache_dir)
    seen_vae: set[str] = set()
    for directory in vae_dirs:
        try:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.safetensors")):
                name = path.name.lower()
                if "vae" in name and ("video" in name or "audio" in name):
                    if path.name not in seen_vae:
                        seen_vae.add(path.name)
                        vae_files.append(path.name)
        except OSError:
            continue

    # LoRA：扫描 Lora/loras 目录（WebUI 用 Lora 单数，ComfyUI 用 loras 复数）
    lora_files: list[str] = []
    lora_dirs = [
        MODEL_CACHE_DIR / "Lora",
        MODEL_CACHE_DIR / "loras",
        MODEL_CACHE_DIR / "Lora" / "minimax_h3",
        MODEL_CACHE_DIR / "loras" / "minimax_h3",
    ]
    lora_dirs += [d / "Lora" for d in models_dirs]
    lora_dirs += [d / "loras" for d in models_dirs]
    lora_dirs += [d / "Lora" / "minimax_h3" for d in models_dirs]
    lora_dirs += [d / "loras" / "minimax_h3" for d in models_dirs]
    seen_lora: set[str] = set()
    for directory in lora_dirs:
        try:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.safetensors")):
                # 只收录 H3 相关 LoRA（文件名含 h3 或 minimax），避免混入其他模型的 LoRA
                name_lower = path.name.lower()
                if "h3" not in name_lower and "minimax" not in name_lower:
                    continue
                if path.name not in seen_lora:
                    seen_lora.add(path.name)
                    lora_files.append(path.name)
        except OSError:
            continue

    return {
        "models": dit_files,
        "text_encoders": te_files,
        "vaes": vae_files,
        "loras": lora_files,
        "samplers": ["euler"],
        "schedulers": ["simple"],
        "nodes": [],
        "h3_ready": ready,
        "missing_nodes": errors,
        "supports_lora_model_only": True,
        "supports_lora_model_clip": True,
        "local": True,
        "models_dir": ", ".join(str(d) for d in models_dirs),
    }


def local_status(config: dict[str, Any]) -> dict[str, Any]:
    """供 backend_manager.status() 使用的本地模式状态。"""
    engine = str(config.get("local_engine") or "diffsynth").strip().lower()
    engine_label = "Forge 原生" if engine == "forge" else "DiffSynth"
    provider_text = f"本地 WebUI（{engine_label}）"
    try:
        catalog = local_catalog(config)
        ready = bool(catalog.get("h3_ready"))
        health: dict[str, Any] = {
            "ok": ready,
            "base_url": "in-process",
            "provider": provider_text,
        }
        vae_files = catalog.get("vaes", [])
        if vae_files:
            health["vae"] = "VAE：" + "、".join(vae_files)
        else:
            health["vae"] = "VAE 未下载，首次生成时自动下载"
        if catalog.get("missing_nodes"):
            health["error"] = "；".join(catalog["missing_nodes"])
    except Exception as exc:
        ready = False
        health = {
            "ok": False,
            "base_url": "in-process",
            "provider": provider_text,
            "error": str(exc),
        }
    return {
        "state": "ready" if ready else "stopped",
        "ready": ready,
        "mode": "local",
        "url": f"in-process（{engine_label}）",
        "process_running": False,
        "pid": None,
        "exit_code": None,
        "started_at": None,
        "command": [],
        "health": health,
        "auto_start_on_tab": bool(config.get("auto_start_on_tab", True)),
        "discovered_paths": [],
    }


def _comfy_input_dir() -> Path | None:
    config = load_config()
    for models_dir in iter_models_dirs(config):
        input_dir = models_dir.parent / "input"
        if input_dir.is_dir():
            return input_dir
    return None


def find_asset(filename: str) -> Path | None:
    """按“上传目录 → ComfyUI input 目录”的顺序解析素材文件。"""
    if not filename:
        return None
    name = str(filename).replace("\\", "/")
    candidates: list[Path] = [
        CLOUD_UPLOAD_DIR / Path(name).name,
        CLOUD_UPLOAD_DIR / name,
    ]
    input_dir = _comfy_input_dir()
    if input_dir is not None:
        candidates.append(input_dir / name)
        candidates.append(input_dir / Path(name).name)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _raise_if_cancelled(job_id: str) -> None:
    if _manager.is_cancelled(job_id):
        raise LocalJobCancelled(job_id)


# ── 静默加载日志 ────────────────────────────────────────────────────────────────
# diffsynth 构建 pipeline 时会打印大量过程日志（downloader 配置框、
# Loading models / Loaded model / Using ... from、bitsandbytes 替换信息、
# modelscope "No files to download" 等）。这些日志每次构建只打印一遍、
# 对用户没有参考价值，还容易被误认为报错。这里通过“替换相应模块的 print +
# 抬高 modelscope 日志级别”定向关闭，不重定向全局 stdout，
# 不影响 webui 其他功能的控制台输出。
_SILENT_MODULES = (
    "diffsynth.core.loader.config",   # downloader 配置框
    "diffsynth.core.loader.file",     # safetensors 权重结构明细（n_keys / prefixes）
    "diffsynth.models.model_loader",  # Loading models / Loaded model / Using ... from
    "diffsynth.core.quant.base",      # Quantization backend ...
    "diffsynth.core.quant.config",    # N nn.Linear layers replaced ...
)


@contextlib.contextmanager
def _silent_diffsynth_loading():
    import importlib
    import logging

    saved_prints: list = []
    saved_levels: list = []
    try:
        for dotted in _SILENT_MODULES:
            module = importlib.import_module(dotted)
            if "print" in vars(module):
                continue  # 嵌套进入时已有补丁，交给外层恢复
            saved_prints.append(module)
            module.print = lambda *args, **kwargs: None
        for name in ("modelscope_hub", "modelscope_hub.download"):
            logger = logging.getLogger(name)
            saved_levels.append((logger, logger.level))
            logger.setLevel(logging.CRITICAL)
        yield
    finally:
        for module in reversed(saved_prints):
            del module.print
        for logger, level in reversed(saved_levels):
            logger.setLevel(level)


class _PipelineManager:
    """按 (DiT, 文本编码器, processor) 指纹缓存 pipeline，并管理取消事件。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pipe: Any = None
        self._fingerprint: tuple[str, str, str, str] | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._generation_lock = threading.Lock()

    def register_cancel(self, job_id: str) -> threading.Event:
        with self._lock:
            event = threading.Event()
            self._cancel_events[job_id] = event
            return event

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancel_events.pop(job_id, None)

    def is_cancelled(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        return event is not None and event.is_set()

    def get_pipeline(self, data: dict[str, Any], config: dict[str, Any]) -> Any:
        models_dirs = iter_models_dirs(config)
        if not models_dirs:
            raise H3StudioError("未找到本地模型目录，可在模型下载器一键下载 H3 组合，或在设置中填写“本地模型目录”")

        def find_weight(name: str, subdirs: tuple[str, ...], label: str) -> Path:
            for models_dir in models_dirs:
                for sub in subdirs:
                    path = models_dir / sub / name
                    if path.is_file():
                        return path
            raise H3StudioError(f"{label}文件不存在：{name}")

        dit_path = find_weight(str(data.get("model") or ""), ("diffusion_models",), "H3 DiT 模型")
        te_path = find_weight(str(data.get("text_encoder") or ""), TE_SUBDIRS, "MiniMax 文本编码器")
        processor_dir = resolve_processor_dir(config)
        if processor_dir is None:
            from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig

            processor_download = ModelConfig(
                model_id=H3_REPO_ID,
                origin_file_pattern=PROCESSOR_PATTERN,
                local_model_path=str(MODEL_CACHE_DIR),
            )
            with _silent_diffsynth_loading():
                processor_download.download_if_necessary()
            path = processor_download.path
            if isinstance(path, list):
                if not path:
                    raise H3StudioError("H3 processor 下载后未找到任何文件")
                processor_dir = Path(os.path.commonpath([str(item) for item in path]))
            else:
                processor_dir = Path(str(path))
            if not (processor_dir / "preprocessor_config.json").is_file() and not processor_dir.is_dir():
                raise H3StudioError(f"H3 processor 目录无效：{processor_dir}")
        fingerprint = (
            str(dit_path),
            str(te_path),
            str(processor_dir),
            vae_variant(config, dit_path=dit_path),
            str(find_local_vae(config, "video", variant=vae_variant(config, dit_path=dit_path)) or ""),
            str(find_local_vae(config, "audio", variant=vae_variant(config, dit_path=dit_path)) or ""),
            # vram_limit 和 vram_strategy 变更时需重建 pipeline
            str(config.get("vram_limit") or 0),
            str(config.get("vram_strategy") or "auto"),
        )
        with self._lock:
            if self._pipe is not None and self._fingerprint == fingerprint:
                return self._pipe
        import torch
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
        from diffsynth.core.quant import QuantizeConfig

        def _is_nf4(path: Path) -> bool:
            return "nf4" in path.name.lower()

        def _is_int8(path: Path) -> bool:
            return "int8" in path.name.lower()

        # DiT 量化配置：NF4 用 bitsandbytes_nf4，INT8 用 comfy_kitchen_int8_w8a8
        if _is_nf4(dit_path):
            dit_quantize = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True)
            print(f"[H3 Studio] 检测到 NF4 DiT 模型，启用 bitsandbytes_nf4 量化推理：{dit_path.name}", flush=True)
        elif _is_int8(dit_path):
            dit_quantize = QuantizeConfig(method="comfy_kitchen_int8_w8a8", load_prequantized=True)
            print(f"[H3 Studio] 检测到 INT8 DiT 模型，启用 comfy_kitchen_int8_w8a8 量化推理：{dit_path.name}", flush=True)
        else:
            dit_quantize = None

        # 文本编码器量化配置
        if _is_nf4(te_path):
            te_quantize = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True)
            print(f"[H3 Studio] 检测到 NF4 文本编码器，启用 bitsandbytes_nf4 量化推理：{te_path.name}", flush=True)
        elif _is_int8(te_path):
            te_quantize = QuantizeConfig(method="comfy_kitchen_int8_w8a8", load_prequantized=True)
            print(f"[H3 Studio] 检测到 INT8 文本编码器，启用 comfy_kitchen_int8_w8a8 量化推理：{te_path.name}", flush=True)
        else:
            te_quantize = None

        # 根据策略选择 vram 管理配置
        strategy = resolve_vram_strategy(config)
        vram = build_vram_config(strategy)
        print(
            f"[H3 Studio] 显存策略: {strategy['label']} "
            f"(检测到 VRAM={strategy['vram_gb']}GB, RAM={strategy['ram_gb']}GB, "
            f"offload={strategy['offload_device']}, gc={strategy['use_gradient_checkpointing']})",
            flush=True,
        )
        vae_variant_name = vae_variant(config, dit_path=dit_path)
        vae_spec = VAE_VARIANTS[vae_variant_name]

        def vae_model_config(which: str) -> "ModelConfig":
            local_path = find_local_vae(config, which, variant=vae_variant_name)
            if local_path is not None:
                _register_local_vae_hash(local_path, LOCAL_VAE_SPECS[which])
                vae_quantize = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True) if _is_nf4(local_path) else None
                return ModelConfig(path=str(local_path), quantize=vae_quantize, **vram)
            pattern = vae_spec["video_pattern"] if which == "video" else vae_spec["audio_pattern"]
            vae_quantize = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True) if "nf4" in pattern.lower() else None
            return ModelConfig(
                model_id=vae_spec["repo"],
                origin_file_pattern=pattern,
                local_model_path=str(MODEL_CACHE_DIR),
                quantize=vae_quantize,
                **vram,
            )

        total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
        # 通过 vram_limit 控制 GPU/CPU 权重分布，平衡显存与内存占用
        # 用户可在设置界面配置 vram_limit（0=自动），覆盖默认值
        user_vram_limit = float(config.get("vram_limit") or 0)
        is_int8_model = (
            dit_quantize is not None and "int8" in getattr(dit_quantize, "method", "")
        ) or (
            te_quantize is not None and "int8" in getattr(te_quantize, "method", "")
        )
        is_nf4_model = (
            (dit_quantize is not None and "nf4" in getattr(dit_quantize, "method", ""))
            or (te_quantize is not None and "nf4" in getattr(te_quantize, "method", ""))
        )
        if user_vram_limit > 0:
            vram_limit = user_vram_limit
            print(
                f"[H3 Studio] 用户自定义显存上限: vram_limit={vram_limit}GB",
                flush=True,
            )
        elif is_int8_model:
            # INT8 权重大(约12-15GB)，按显存大小自动决定是否全驻留GPU
            # 24GB+ 显存：权重全驻留，速度最快
            # 16GB 显存：驻留 12GB，其余 offload 到 CPU
            vram_limit = total_vram_gb - 4
            print(
                f"[H3 Studio] INT8 模型显存/内存平衡策略: vram_limit={vram_limit:.1f}GB",
                flush=True,
            )
        elif is_nf4_model:
            # NF4 权重小(约6-8GB)，用宽松限制让权重全驻留GPU，加快加载速度
            # total_vram - 4 留 4GB 给激活值和临时张量，加载快且显存可控
            vram_limit = total_vram_gb - 4
            print(
                f"[H3 Studio] NF4 模型速度优先策略: vram_limit={vram_limit:.1f}GB "
                f"(权重全驻留GPU，加载快)",
                flush=True,
            )
        else:
            vram_limit = total_vram_gb - 2
        with _silent_diffsynth_loading():
            pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=str(dit_path), quantize=dit_quantize, **vram),
                    ModelConfig(path=str(te_path), quantize=te_quantize, **vram),
                    vae_model_config("video"),
                    vae_model_config("audio"),
                ],
                processor_config=ModelConfig(path=str(processor_dir)),
                vram_limit=vram_limit,
                redirect_common_files=False,
            )
        # 阶段日志：平时静默（只展示生成进度），仅当某阶段超过 2 秒时
        # 打印一行耗时，便于区分“慢”和“卡死”，正常快阶段不刷屏。
        base_unit_runner = pipe.unit_runner

        def logged_unit_runner(unit, *args, **kwargs):
            name = type(unit).__name__
            start = time.time()
            try:
                return base_unit_runner(unit, *args, **kwargs)
            finally:
                elapsed = time.time() - start
                if elapsed > 2.0:
                    print(f"[H3 Studio] 阶段 {name} 耗时 {elapsed:.1f}s", flush=True)

        pipe.unit_runner = logged_unit_runner
        base_load_models = pipe.load_models_to_device

        def logged_load_models(names):
            start = time.time()
            try:
                return base_load_models(names)
            finally:
                elapsed = time.time() - start
                if elapsed > 2.0:
                    print(f"[H3 Studio] 换入模型 {names} 耗时 {elapsed:.1f}s", flush=True)

        pipe.load_models_to_device = logged_load_models
        # 保存模型量化类型，供 run_generation 决定是否启用 gradient checkpointing
        pipe._is_nf4_model = is_nf4_model
        pipe._is_int8_model = is_int8_model
        with self._lock:
            self._pipe = pipe
            self._fingerprint = fingerprint
        return pipe


_manager = _PipelineManager()


def request_cancel(job_id: str) -> bool:
    return _manager.request_cancel(job_id)


# ── Forge 原生引擎 ──────────────────────────────────────────────────────────

def _resolve_model_paths(data: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    """解析 Forge 原生引擎所需的模型路径。"""
    models_dirs = iter_models_dirs(config)

    def find(name: str, subdirs: tuple[str, ...]) -> Path:
        for models_dir in models_dirs:
            for sub in subdirs:
                path = models_dir / sub / name
                if path.is_file():
                    return path
        raise H3StudioError(f"模型文件不存在：{name}")

    dit_path = find(str(data.get("model") or ""), ("diffusion_models",))
    te_path = find(str(data.get("text_encoder") or ""), TE_SUBDIRS)

    # VAE 路径：根据 DiT 量化格式自动选择
    variant = vae_variant(config, dit_path=dit_path)
    vae_spec = VAE_VARIANTS[variant]

    video_vae_local = find_local_vae(config, "video", variant=variant)
    audio_vae_local = find_local_vae(config, "audio", variant=variant)

    if video_vae_local is not None:
        video_vae_path = video_vae_local
    else:
        video_vae_path = vae_cache_path(variant, "video")

    if audio_vae_local is not None:
        audio_vae_path = audio_vae_local
    else:
        audio_vae_path = vae_cache_path(variant, "audio")

    return {
        "dit": dit_path,
        "te": te_path,
        "video_vae": video_vae_path,
        "audio_vae": audio_vae_path,
    }


def run_generation_forge(
    job_id: str,
    data: dict[str, Any],
    config: dict[str, Any],
    progress_cb: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """使用 Forge 原生引擎执行 H3 生成。"""
    from .h3_forge_native import get_generator
    from PIL import Image

    _manager.register_cancel(job_id)
    try:
        progress_cb({"phase": "prepare"})
        paths = _resolve_model_paths(data, config)

        gen = get_generator()
        gen.load_models(
            dit_path=str(paths["dit"]),
            te_path=str(paths["te"]),
            video_vae_path=str(paths["video_vae"]),
            audio_vae_path=str(paths["audio_vae"]),
        )
        _raise_if_cancelled(job_id)

        # 构建参数
        mode = data.get("mode")
        first_frame = None
        last_frame = None
        references = None

        # 加载 LoRA（Forge 原生引擎现在支持 model-only LoRA）
        applied_lora = False
        for lora in data.get("loras") or []:
            name = lora.get("name")
            if not name:
                continue
            lora_path = find_lora_path(config, name)
            if lora_path is None:
                print(f"[H3 Studio] 警告：未找到 LoRA 文件 {name}，已跳过", flush=True)
                continue
            strength = float(lora.get("model_strength", lora.get("strength", 1.0)))
            applied_lora = gen.apply_lora(str(lora_path), strength=strength)
            break  # H3 只支持单个 model-only LoRA

        if mode == "i2v":
            first_frame = _load_pil_image(data["first_frame"])
        elif mode == "fl2v":
            first_frame = _load_pil_image(data["first_frame"])
            last_frame = _load_pil_image(data["last_frame"])
        elif mode in {"ref", "swap"}:
            references = _build_references_forge(data)

        def cancel_check():
            return _manager.is_cancelled(job_id)

        try:
            video, audio = gen.generate(
                prompt=data["prompt"],
                width=int(data["width"]),
                height=int(data["height"]),
                num_frames=int(data["frames"]),
                num_steps=int(data["steps"]),
                seed=int(data["seed"]),
                video_shift=float(data.get("shift_video") or 12),
                audio_shift=float(data.get("shift_audio") or 3),
                first_frame=first_frame,
                last_frame=last_frame,
                references=references,
                progress_cb=progress_cb,
                cancel_check=cancel_check,
            )
        finally:
            # 生成后移除 LoRA，确保不影响后续任务
            if applied_lora:
                gen._remove_lora_hooks()
                print("[H3 Studio] 已移除 LoRA", flush=True)
        _raise_if_cancelled(job_id)

        progress_cb({"phase": "write"})
        filename = _write_output_forge(video, audio, data)
        return [{"filename": filename, "subfolder": "", "type": "output"}]
    finally:
        _manager.clear_cancel(job_id)


def _load_pil_image(file_ref) -> "Image.Image":
    """加载 PIL 图像。"""
    from PIL import Image

    path = find_asset(file_ref)
    if path is None:
        raise H3StudioError(f"找不到图像文件：{file_ref}")
    return Image.open(path).convert("RGB")


def _build_references_forge(data: dict[str, Any]) -> list[dict]:
    """构建 Forge 引擎的参考素材列表。"""
    from PIL import Image

    references = []
    for ref in data.get("references", []):
        kind = ref.get("kind")
        path = find_asset(ref.get("file"))
        if path is None:
            continue
        if kind == "image":
            references.append({"type": "image", "image": Image.open(path).convert("RGB")})
        elif kind == "audio":
            waveform, sr = _decode_audio(path)
            references.append({"type": "audio", "audio": waveform, "sample_rate": sr})
        elif kind == "video":
            frames = _decode_video_frames(path, 32)
            if frames:
                references.append({"type": "image", "image": frames[0]})
    return references


def _write_output_forge(video, audio, data: dict[str, Any]) -> str:
    """写入 Forge 引擎的输出视频。"""
    from .h3_forge_native import write_video_audio
    import time

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = data.get("output") or {}
    fps = int(round(float(output.get("fps") or 24)))
    filename = (
        f"Forge_H3_native_{data.get('mode')}_seed_{int(data.get('seed') or 0)}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    write_video_audio(video, audio, str(LOCAL_OUTPUT_DIR / filename), fps=fps)
    return filename


def _make_progress_wrapper(job_id: str, progress_cb: Callable[[dict[str, Any]], None]) -> Callable[[Any], Any]:
    """把 pipeline 的 progress_bar_cmd 换成带取消检查和进度回调的迭代器。"""

    def wrapper(iterable: Any):
        items = list(iterable)
        total = max(1, len(items))
        print(f"[H3 Studio] 开始采样，共 {total} 步", flush=True)
        for index, item in enumerate(items):
            progress_cb({"phase": "sampling", "step": float(index), "maxSteps": total})
            _raise_if_cancelled(job_id)
            yield item

    return wrapper


def _load_frame(filename: str, crop: dict[str, int] | None, width: int, height: int):
    from PIL import Image

    path = find_asset(filename)
    if path is None:
        raise H3StudioError(f"找不到首尾帧文件：{filename}")
    image = Image.open(path).convert("RGB")
    if crop:
        image = image.crop((crop["x"], crop["y"], crop["x"] + crop["width"], crop["y"] + crop["height"]))
    return image.resize((width, height), Image.LANCZOS)


def _decode_video_frames(path: Path, max_frames: int) -> list:
    """PyAV 解码参考视频帧，按 24fps 重采样，最多取 max_frames 帧。"""
    import av
    import numpy as np
    from PIL import Image

    frames: list = []
    container = av.open(str(path))
    try:
        if not container.streams.video:
            raise H3StudioError(f"参考视频没有视频轨：{path.name}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        last_target = -1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time is None:
                continue
            if fps > 0:
                target_index = int(frame.time * REFERENCE_FPS + 1e-6)
                if target_index <= last_target:
                    continue
            else:
                target_index = len(frames)
            last_target = target_index
            if len(frames) >= max_frames:
                break
            array = frame.to_ndarray(format="rgb24")
            frames.append(Image.fromarray(np.ascontiguousarray(array)))
    finally:
        container.close()
    return frames


def _decode_audio(path: Path) -> tuple[Any, int]:
    """PyAV 解码音频为 float32 tensor [C, T]（C<=2）并返回采样率；pipeline 内部会重采样到 32kHz。"""
    import av
    import numpy as np
    import torch

    chunks: list[np.ndarray] = []
    sample_rate = 0
    container = av.open(str(path))
    try:
        if not container.streams.audio:
            raise H3StudioError(f"文件中没有音频轨：{path.name}")
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate or 0)
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if array.ndim == 1:
                array = array[None, :]
            channels = array.shape[0]
            if channels == 1:
                chunks.append(np.repeat(array, 2, axis=0))
            elif channels >= 2:
                chunks.append(array[:2])
            else:
                chunks.append(np.repeat(array, 2, axis=0))
    finally:
        container.close()
    if not chunks:
        raise H3StudioError(f"音频解码结果为空：{path.name}")
    if not sample_rate:
        sample_rate = 44100
    waveform = np.concatenate(chunks, axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        waveform = waveform.astype("float32") / float(max(abs(int(info.min)), abs(int(info.max))))
    else:
        waveform = waveform.astype("float32")
    return torch.from_numpy(np.ascontiguousarray(waveform)), sample_rate


def _build_references(data: dict[str, Any]) -> list[dict[str, Any]]:
    from PIL import Image

    max_frames = max(5, int(data.get("frames") or 124))
    references: list[dict[str, Any]] = []
    for ref in data.get("references") or []:
        kind = ref.get("kind")
        path = find_asset(str(ref.get("file") or ""))
        if path is None:
            raise H3StudioError(f"找不到参考素材：{ref.get('file')}")
        if kind == "image":
            references.append({"type": "image", "image": Image.open(path).convert("RGB")})
        elif kind == "audio":
            waveform, sample_rate = _decode_audio(path)
            references.append({"type": "audio", "audio": waveform, "sample_rate": sample_rate})
        elif kind == "video":
            frames = _decode_video_frames(path, max_frames)
            if not frames:
                raise H3StudioError(f"参考视频没有可解码帧：{path.name}")
            if ref.get("include_audio", True):
                try:
                    waveform, sample_rate = _decode_audio(path)
                    references.append({
                        "type": "video_audio",
                        "video": frames,
                        "audio": waveform,
                        "sample_rate": sample_rate,
                    })
                    continue
                except Exception:
                    pass
            references.append({"type": "video", "video": frames})
    return references


def _configure_lora_loader(pipe: Any) -> None:
    """根据 DiT 类型配置 LoRA loader 的 qkv 排列方式。

    - MiniMaxH3DiTComfy / ComfyPruned：qkv_proj 输出为 [q;k;v] split 排列
      （forward 用 view(total, 3, heads, head_dim)）
    - MiniMaxH3DiT（原生）：qkv_proj 输出为 head-interleaved 排列
      （forward 用 view(total, heads, 3, head_dim)）
    """
    try:
        from diffsynth.models.minimax_h3_dit_comfy import MiniMaxH3DiTComfy
        from diffsynth.utils.lora.minimax_h3 import MiniMaxH3LoRALoader
    except Exception:
        return

    dit = getattr(pipe, "dit", None)
    if dit is None:
        return

    is_comfy = isinstance(dit, MiniMaxH3DiTComfy)
    layout = "split" if is_comfy else "interleaved"

    class _ConfiguredLoader(MiniMaxH3LoRALoader):
        def __init__(self, device="cpu", torch_dtype=torch.float32):
            super().__init__(device=device, torch_dtype=torch_dtype, qkv_layout=layout)

    pipe.lora_loader = _ConfiguredLoader
    print(f"[H3 Studio] LoRA qkv 布局: {layout} ({'ComfyUI split' if is_comfy else 'DiffSynth interleaved'})", flush=True)


def _debug_lora_weights(dit: Any) -> None:
    """调试：检查 LoRA 权重是否正确应用到量化层，并打印统计信息。"""
    try:
        from diffsynth.core.vram.layers import LoRAHotLoadMixin

        total = 0
        with_lora = 0
        apply_enabled = 0
        sample_stats = []
        for name, module in dit.named_modules():
            if isinstance(module, LoRAHotLoadMixin):
                total += 1
                if getattr(module, "apply_lora", False):
                    apply_enabled += 1
                lora_a = getattr(module, "lora_A_weights", None)
                lora_b = getattr(module, "lora_B_weights", None)
                if lora_a and lora_b and lora_a[0] is not None and lora_b[0] is not None:
                    with_lora += 1
                    if len(sample_stats) < 3:
                        a, b = lora_a[0], lora_b[0]
                        sample_stats.append(
                            f"  {name}: A{tuple(a.shape)} dev={a.device} range=[{a.min().item():.4f}, {a.max().item():.4f}] "
                            f"B{tuple(b.shape)} dev={b.device} range=[{b.min().item():.4f}, {b.max().item():.4f}]"
                        )
        print(f"[H3 Studio] LoRA 调试: 量化层总数={total}, apply_lora=True 的层={apply_enabled}, 已加载 LoRA 权重的层={with_lora}", flush=True)
        for line in sample_stats:
            print(line, flush=True)
    except Exception as exc:
        print(f"[H3 Studio] LoRA 调试失败: {exc}", flush=True)


def _build_call_args(job_id: str, data: dict[str, Any], config: dict[str, Any], progress_cb: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    width = int(data["width"])
    height = int(data["height"])
    strategy = resolve_vram_strategy(config)
    args: dict[str, Any] = {
        "prompt": data["prompt"],
        "height": height,
        "width": width,
        "num_frames": int(data["frames"]),
        "num_inference_steps": int(data["steps"]),
        "seed": int(data["seed"]),
        "cfg_scale": 1.0,
        "flow_shift": float(data.get("shift_video") or 12),
        "audio_flow_shift": float(data.get("shift_audio") or 3),
        "tiled": strategy["tiled"],
        "use_gradient_checkpointing": strategy["use_gradient_checkpointing"],
        "use_gradient_checkpointing_offload": strategy["use_gradient_checkpointing_offload"],
        "progress_bar_cmd": _make_progress_wrapper(job_id, progress_cb),
    }
    mode = data.get("mode")
    if mode == "i2v":
        args["keyframes"] = [_load_frame(data["first_frame"], data.get("first_frame_crop"), width, height)]
        args["keyframe_indices"] = [0]
    elif mode == "fl2v":
        args["keyframes"] = [
            _load_frame(data["first_frame"], data.get("first_frame_crop"), width, height),
            _load_frame(data["last_frame"], data.get("last_frame_crop"), width, height),
        ]
        args["keyframe_indices"] = [0, -1]
    elif mode in {"ref", "swap"}:
        args["references"] = _build_references(data)
        short_edge = 2048 if data.get("ref_image_size") == "max" else min(width, height)
        args["ref_image_short_edge"] = max(32, int(short_edge))
        # 调试日志：确认 references 内容
        ref_types = [r.get("type") for r in args["references"]]
        print(f"[H3 Studio] 参考模式 references: {ref_types}", flush=True)
        for i, r in enumerate(args["references"]):
            if r.get("type") == "audio":
                wf = r.get("audio")
                sr = r.get("sample_rate")
                dur = wf.shape[-1] / sr if hasattr(wf, "shape") and sr else "?"
                print(f"[H3 Studio] 音频参考 #{i}: shape={list(wf.shape) if hasattr(wf, 'shape') else '?'}, sample_rate={sr}, duration={dur:.2f}s", flush=True)
            elif r.get("type") == "image":
                img = r.get("image")
                print(f"[H3 Studio] 图片参考 #{i}: size={img.size if hasattr(img, 'size') else '?'}", flush=True)
    return args


def _write_output(video: Any, audio: Any, data: dict[str, Any]) -> str:
    from diffsynth.utils.data.audio_video import write_video_audio

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = data.get("output") or {}
    fps = int(round(float(output.get("fps") or REFERENCE_FPS)))
    filename = (
        f"Forge_H3_local_{data.get('mode')}_seed_{int(data.get('seed') or 0)}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    write_video_audio(
        video=video,
        audio=audio,
        output_path=str(LOCAL_OUTPUT_DIR / filename),
        fps=fps,
        audio_sample_rate=OUTPUT_AUDIO_SAMPLE_RATE,
    )
    return filename


def run_generation(
    job_id: str,
    data: dict[str, Any],
    config: dict[str, Any],
    progress_cb: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """在进程内执行一次 H3 生成，返回 ComfyUI 风格输出条目列表。"""
    engine = str(config.get("local_engine") or "diffsynth").strip().lower()
    engine_name = "Forge 原生引擎" if engine == "forge" else "DiffSynth Pipeline"
    print(f"[H3 Studio] 使用本地引擎: {engine_name}", flush=True)
    if engine == "forge":
        return run_generation_forge(job_id, data, config, progress_cb)

    _manager.register_cancel(job_id)
    try:
        progress_cb({"phase": "prepare"})
        pipe = _manager.get_pipeline(data, config)
        _raise_if_cancelled(job_id)
        with _manager._generation_lock:
            args = _build_call_args(job_id, data, config, progress_cb)
            # NF4 权重小(6-8GB)，不需要 gradient checkpointing，关闭以恢复速度
            # INT8 权重大(12-15GB)，保留策略中的 gc 设置以节省显存
            if getattr(pipe, "_is_nf4_model", False):
                args["use_gradient_checkpointing"] = False
                args["use_gradient_checkpointing_offload"] = False
            # 加载 LoRA（仅 DiT；H3 LoRA 作用于 DiT，文本编码器暂不支持）
            loaded_loras: list[str] = []
            loras = data.get("loras") or []
            if loras:
                # 根据 DiT 类型选择 qkv 排列：ComfyUI 格式用 [q;k;v] split，原生 DiffSynth 用 head-interleaved
                _configure_lora_loader(pipe)
            for lora in loras:
                name = lora.get("name")
                if not name:
                    continue
                lora_path = find_lora_path(config, name)
                if lora_path is None:
                    print(f"[H3 Studio] 警告：未找到 LoRA 文件 {name}，已跳过", flush=True)
                    continue
                alpha = float(lora.get("model_strength", 1.0))
                try:
                    pipe.load_lora(pipe.dit, str(lora_path), alpha=alpha)
                    loaded_loras.append(name)
                    print(f"[H3 Studio] 已加载 LoRA: {name} (alpha={alpha})", flush=True)
                    # 调试：检查 LoRA 权重是否正确应用到量化层
                    _debug_lora_weights(pipe.dit)
                except Exception as exc:
                    print(f"[H3 Studio] 警告：LoRA {name} 加载失败：{exc}", flush=True)
            try:
                video, audio = pipe(**args)
            finally:
                # 生成完成后清除 LoRA，避免影响后续任务
                if loaded_loras:
                    pipe.clear_lora()
                    print(f"[H3 Studio] 已清除 {len(loaded_loras)} 个 LoRA", flush=True)
                # 推理完成后释放 GPU 临时显存（激活、中间张量等），降低显存碎片
                import gc
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        _raise_if_cancelled(job_id)
        progress_cb({"phase": "write"})
        filename = _write_output(video, audio, data)
        return [{"filename": filename, "subfolder": "", "type": "output"}]
    finally:
        _manager.clear_cancel(job_id)
