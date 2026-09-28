"""MiniMax H3 本地 WebUI 后端：在 Forge 进程内直接用 DiffSynth pipeline 运行 H3。

不依赖 ComfyUI。模型来源策略（对齐官方 low_vram 16GB all-disk 示例）：
- DiT / 文本编码器：直接复用本地量化权重（diffsynth 按文件 hash 自动识别型号，
  包括 Comfy-Org int8_convrot 与 DiffSynth-Studio NF4 系列，放入模型目录即可自动启用）
- 视频 VAE / 音频 VAE / processor：优先复用 webui/models/vae 下的 Comfy-Org 命名文件
  （经 state_dict 键严格校验后注册 hash 直接加载）；缺失时首次运行从 ModelScope
  自动下载，缓存在 webui/models/ 下
  VAE 变体可通过 local_vae_variant 选择：original（INT8 组合配套的标准 fp16/fp32
  VAE，MiniMax/MiniMax-H3）或 nf4（DiffSynth-Studio/MiniMax-H3-NF4 的 4bit 量化版）
- 所有权重 disk offload，适配 16GB 显存
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
    found: list[str] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.safetensors")):
        hash_value = _hash_safetensors(path)
        if hash_value and model_name in _known_model_names(hash_value):
            found.append(path.name)
    return found


def resolve_processor_dir(config: dict[str, Any]) -> Path | None:
    """返回已存在的 H3 processor 目录；不存在时返回 None（由调用方触发下载）。"""
    configured = str(config.get("local_processor_path") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_dir() else None
    downloaded = MODEL_CACHE_DIR / H3_REPO_ID.replace("/", os.sep) / "FL2VA" / "processor"
    return downloaded if downloaded.is_dir() else None


def vae_variant(config: dict[str, Any]) -> str:
    """当前选择的 VAE 变体：original（标准 fp16/fp32，INT8 组合配套）或 nf4（4bit 量化）。"""
    variant = str(config.get("local_vae_variant") or "original").strip() or "original"
    return variant if variant in VAE_VARIANTS else "original"


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


def _verify_local_vae(path: Path, spec: dict) -> bool:
    """严格校验本地 VAE 文件能否被 diffsynth 模型类加载（结果按路径缓存）。"""
    key = str(path)
    if key in _VAE_VERIFY_CACHE:
        return _VAE_VERIFY_CACHE[key]
    ok = False
    try:
        from safetensors.torch import load_file

        model = _import_dotted(spec["model_class"])()
        converter = _import_dotted(spec["state_dict_converter"])
        converted = converter(load_file(str(path), device="cpu"))
        missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
        ok = not missing and set(unexpected) <= _VAE_TOLERATED_EXTRA_KEYS
    except Exception:
        ok = False
    _VAE_VERIFY_CACHE[key] = ok
    return ok


def find_local_vae(config: dict[str, Any], which: str) -> Path | None:
    """在 webui/models/vae 与候选模型目录的 vae 子目录中查找可用的本地 VAE。

    仅在实际使用 fp16/fp32 VAE 的变体（original）下生效：NF4 变体是显式选择的量化方案，
    不被 fp16/fp32 的本地文件覆盖。
    """
    if vae_variant(config) != "original":
        return None
    spec = LOCAL_VAE_SPECS[which]
    dirs: list[Path] = [MODEL_CACHE_DIR / "vae"]
    dirs += [d / "vae" for d in iter_models_dirs(config)]
    seen: set[str] = set()
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
    return None


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
    """本地模式模型目录：DiT/文本编码器来自本地权重，VAE 为自动下载占位。"""
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
    variant_label = VAE_VARIANTS[vae_variant(config)]["label"]
    video_local = find_local_vae(config, "video")
    audio_local = find_local_vae(config, "audio")
    video_note = "本地已检测" if video_local is not None else "自动下载"
    audio_note = "本地已检测" if audio_local is not None else "自动下载"
    return {
        "models": dit_files,
        "text_encoders": te_files,
        "vaes": [
            f"MiniMax-H3 视频 VAE（{variant_label}，{video_note}）",
            f"MiniMax-H3 音频 VAE（{variant_label}，{audio_note}）",
        ],
        "loras": [],
        "samplers": ["euler"],
        "schedulers": ["simple"],
        "nodes": [],
        "h3_ready": ready,
        "missing_nodes": errors,
        "supports_lora_model_only": False,
        "supports_lora_model_clip": False,
        "local": True,
        "models_dir": ", ".join(str(d) for d in models_dirs),
    }


def local_status(config: dict[str, Any]) -> dict[str, Any]:
    """供 backend_manager.status() 使用的本地模式状态。"""
    try:
        catalog = local_catalog(config)
        ready = bool(catalog.get("h3_ready"))
        health: dict[str, Any] = {
            "ok": ready,
            "base_url": "in-process (DiffSynth)",
            "provider": "DiffSynth-Studio 本地模式",
        }
        variant = vae_variant(config)
        video_cached = vae_cache_path(variant, "video").is_file()
        audio_cached = vae_cache_path(variant, "audio").is_file()
        video_local = find_local_vae(config, "video")
        audio_local = find_local_vae(config, "audio")

        def vae_status(which: str, cached: bool, local_path: Path | None) -> str:
            label = "视频 VAE" if which == "video" else "音频 VAE"
            if local_path is not None:
                return f"{label}：已检测本地文件（{local_path.name}，{VAE_VARIANTS[variant]['label']}）"
            if cached:
                return f"{label}（{VAE_VARIANTS[variant]['label']}）已缓存"
            return f"{label}（{VAE_VARIANTS[variant]['label']}）未下载，首次生成时自动下载"

        health["video_vae"] = vae_status("video", video_cached, video_local)
        health["audio_vae"] = vae_status("audio", audio_cached, audio_local)
        if video_local is None and audio_local is None and not (video_cached and audio_cached):
            health["vae"] = f"VAE（{VAE_VARIANTS[variant]['label']}）未下载，首次生成时自动下载"
        else:
            health["vae"] = f"VAE（{VAE_VARIANTS[variant]['label']}）就绪"
        if catalog.get("missing_nodes"):
            health["error"] = "；".join(catalog["missing_nodes"])
    except Exception as exc:
        ready = False
        health = {
            "ok": False,
            "base_url": "in-process (DiffSynth)",
            "provider": "DiffSynth-Studio 本地模式",
            "error": str(exc),
        }
    return {
        "state": "ready" if ready else "stopped",
        "ready": ready,
        "mode": "local",
        "url": "in-process (DiffSynth)",
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
            vae_variant(config),
            str(find_local_vae(config, "video") or ""),
            str(find_local_vae(config, "audio") or ""),
        )
        with self._lock:
            if self._pipe is not None and self._fingerprint == fingerprint:
                return self._pipe
        import torch
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

        vram = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        vae_spec = VAE_VARIANTS[vae_variant(config)]

        def vae_model_config(which: str) -> "ModelConfig":
            local_path = find_local_vae(config, which)
            if local_path is not None:
                _register_local_vae_hash(local_path, LOCAL_VAE_SPECS[which])
                return ModelConfig(path=str(local_path), **vram)
            pattern = vae_spec["video_pattern"] if which == "video" else vae_spec["audio_pattern"]
            return ModelConfig(
                model_id=vae_spec["repo"],
                origin_file_pattern=pattern,
                local_model_path=str(MODEL_CACHE_DIR),
                **vram,
            )

        total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
        with _silent_diffsynth_loading():
            pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(path=str(dit_path), **vram),
                    ModelConfig(path=str(te_path), **vram),
                    vae_model_config("video"),
                    vae_model_config("audio"),
                ],
                processor_config=ModelConfig(path=str(processor_dir)),
                vram_limit=total_vram_gb - 2,
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
        with self._lock:
            self._pipe = pipe
            self._fingerprint = fingerprint
        return pipe


_manager = _PipelineManager()


def request_cancel(job_id: str) -> bool:
    return _manager.request_cancel(job_id)


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


def _build_call_args(job_id: str, data: dict[str, Any], progress_cb: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    width = int(data["width"])
    height = int(data["height"])
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
    elif mode == "ref":
        args["references"] = _build_references(data)
        short_edge = 2048 if data.get("ref_image_size") == "max" else min(width, height)
        args["ref_image_short_edge"] = max(32, int(short_edge))
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
    _manager.register_cancel(job_id)
    try:
        progress_cb({"phase": "prepare"})
        pipe = _manager.get_pipeline(data, config)
        _raise_if_cancelled(job_id)
        with _manager._generation_lock:
            args = _build_call_args(job_id, data, progress_cb)
            video, audio = pipe(**args)
        _raise_if_cancelled(job_id)
        progress_cb({"phase": "write"})
        filename = _write_output(video, audio, data)
        return [{"filename": filename, "subfolder": "", "type": "output"}]
    finally:
        _manager.clear_cancel(job_id)
