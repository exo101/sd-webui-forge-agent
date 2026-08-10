"""配置管理 - 读写启动参数配置"""
import json
import os
import socket
from .paths import BASE_DIR, CONFIG_FILE, WEBUI_DIR

DEFAULT_CONFIG = {
    "port": 7869,
    "theme": "dark",
    "listen": True,
    "autolaunch": True,
    "cuda_malloc": True,
    "cuda_stream": True,
    "pin_shared_memory": True,
    "disable_flash": True,
    "disable_xformers": True,
    "xformers": False,
    "lowvram": False,
    "neveroom": False,
    "no_half": False,
    "no_half_vae": False,
    "precision_full": False,
    "api": True,
    "share": False,
    "skip_install": False,
    "skip_version": False,
    "skip_torch": False,
    "show_console": False,  # 是否显示命令行窗口
    "gpu_device": "",  # GPU设备选择，空字符串表示使用所有可用GPU
    "disable_sage": False,  # 禁用 SageAttention
    "reserve_vram": 0,  # 预留显存大小（GB），0 表示不限制
    "extra_args": "",
    "wallpaper": "",  # 自定义壁纸路径，空字符串表示使用默认壁纸
    "llama": {
        "enabled": True,
        "port": 8080,
        "model": "",
        "ngl": 100,
        "host": "0.0.0.0",
    },
    "comfyui": {
        "enabled": False,
        "path": "",
        "port": 8188,
        "python": "",
    },
    "paths": {
        "ckpt_dir": "",
        "diffusion_models_dir": "",
        "text_encoder_dir": "",
        "lora_dir": "",
        "vae_dir": "",
        "controlnet_dir": "",
        "controlnet_preprocessor_dir": "",
    },
    "presets": {}
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(data)
            merged["paths"] = {**DEFAULT_CONFIG["paths"], **data.get("paths", {})}
            merged["paths"] = _resolve_paths(merged["paths"])
            # deep-merge llama config
            merged["llama"] = {**DEFAULT_CONFIG["llama"], **data.get("llama", {})}
            # deep-merge comfyui config
            merged["comfyui"] = {**DEFAULT_CONFIG["comfyui"], **data.get("comfyui", {})}
            return merged
        except Exception:
            # 配置损坏，备份后用默认值
            backup = CONFIG_FILE + ".bak"
            try:
                import shutil
                shutil.copy2(CONFIG_FILE, backup)
            except Exception:
                pass
    cfg = _init_default_paths(DEFAULT_CONFIG.copy())
    return cfg


def save_config(config: dict):
    save_data = config.copy()
    save_data["paths"] = _relativize_paths(config.get("paths", {}))
    tmp_file = CONFIG_FILE + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        import shutil
        shutil.move(tmp_file, CONFIG_FILE)
    except Exception as e:
        try:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
        except Exception:
            pass


def _resolve_paths(paths: dict) -> dict:
    """相对路径 -> 绝对路径"""
    result = {}
    for k, v in paths.items():
        if isinstance(v, list):
            result[k] = [_to_abs(p) for p in v]
        elif v:
            result[k] = _to_abs(v)
        else:
            result[k] = v
    return result


def _relativize_paths(paths: dict) -> dict:
    """绝对路径 -> 相对路径（相对 BASE_DIR）"""
    result = {}
    for k, v in paths.items():
        if isinstance(v, list):
            result[k] = [_to_rel(p) for p in v]
        elif v:
            result[k] = _to_rel(v)
        else:
            result[k] = v
    return result


def _to_abs(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(BASE_DIR, path))


def _to_rel(path: str) -> str:
    if not path:
        return path
    try:
        rel = os.path.relpath(path, BASE_DIR)
        # 如果在不同盘符（Windows），relpath 会返回绝对路径，保持原样
        if rel.startswith("..\\..\\..") or (len(rel) > 2 and rel[1] == ":"):
            return path
        return rel
    except ValueError:
        return path  # 跨盘符时保持绝对路径


def _init_default_paths(config: dict) -> dict:
    webui = os.path.join(WEBUI_DIR, "models")
    config["paths"] = {
        "ckpt_dir":                    os.path.join(webui, "Stable-diffusion"),
        "diffusion_models_dir":        os.path.join(webui, "diffusion_models"),
        "text_encoder_dir":            os.path.join(webui, "text_encoder"),
        "lora_dir":                    os.path.join(webui, "Lora"),
        "vae_dir":                     os.path.join(webui, "VAE"),
        "controlnet_dir":              os.path.join(webui, "ControlNet"),
        "controlnet_preprocessor_dir": os.path.join(webui, "ControlNet", "preprocessor"),
    }
    return config


def is_port_in_use(port: int) -> bool:
    """检测端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True


def find_available_port(start_port: int = 7869, max_attempts: int = 100) -> int:
    """查找可用端口，从 start_port 开始尝试，失败返回 -1"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return -1
