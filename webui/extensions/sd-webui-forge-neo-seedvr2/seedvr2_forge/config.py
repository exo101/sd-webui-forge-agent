from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class SeedVR2Config:
    enabled: bool = True
    only_final_output: bool = True
    dit_model: str = "seedvr2_ema_3b_fp16.safetensors"
    vae_model: str = "ema_vae_fp16.safetensors"

    # Simple output control. The basic UI exposes this instead of internal tile math.
    # scale_mode: "2x", "3x", "4x", "workflow_6k", "custom"
    scale_mode: str = "2x"
    custom_long_edge: int = 4096

    attention_mode: str = "sdpa"
    device: str = "cuda:0"
    dit_offload_device: str = "none"
    vae_offload_device: str = "none"
    tensor_offload_device: str = "cpu"
    keep_model_cached: bool = False
    unload_forge_model_first: bool = True

    # Outer tiling. Hidden in Advanced Settings for normal users.
    tile_size: int = 1024
    tile_padding: int = 64

    # SeedVR2 parameters
    color_correction: str = "lab"
    batch_size: int = 1
    blocks_to_swap: int = 0
    swap_io_components: bool = False
    uniform_batch_size: bool = False
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    seed: int = 0
    max_resolution: int = 0
    encode_tiled: bool = False
    encode_tile_size: int = 1024
    encode_tile_overlap: int = 128
    decode_tiled: bool = False
    decode_tile_size: int = 1024
    decode_tile_overlap: int = 128
    target_free_vram_gb: int = 18
    debug: bool = False

    def resolve_target_long_edge(self, source_size: Tuple[int, int]) -> int:
        """Resolve an intuitive user-facing scale choice to output long edge."""
        src_long = max(int(source_size[0]), int(source_size[1]))
        mode = str(self.scale_mode).lower().strip()
        if mode.endswith("x") and mode[:-1].replace(".", "", 1).isdigit():
            factor = float(mode[:-1])
            return max(64, int(round(src_long * factor / 8.0) * 8))
        if mode == "workflow_6k":
            # Original ComfyUI graph: 6 * 1024 - 64 = 6080 long edge.
            return 6080
        if mode == "custom":
            return max(64, int(round(int(self.custom_long_edge) / 8.0) * 8))
        raise ValueError(f"Unsupported SeedVR2 scale mode: {self.scale_mode}")


def extension_root() -> Path:
    return Path(__file__).resolve().parents[1]


def forge_root() -> Path:
    return extension_root().parents[1]


def vendor_seedvr2_root() -> Path:
    return extension_root() / "vendor" / "seedvr2"


def models_root() -> Path:
    canonical = forge_root() / "models" / "SEEDVR2"
    legacy = forge_root() / "models" / "SeedVR2"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def list_model_files() -> List[str]:
    root = models_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".safetensors", ".gguf"}
    )


def _is_vae(filename: str) -> bool:
    n = filename.lower()
    return "vae" in n


def list_dit_model_files() -> List[str]:
    return [f for f in list_model_files() if not _is_vae(f)]


def list_vae_model_files() -> List[str]:
    return [f for f in list_model_files() if _is_vae(f)]


def describe_dit_model(filename: str) -> str:
    """Human-readable model label; the filename remains the actual Dropdown value."""
    n = filename.lower()
    size = "7B" if "7b" in n else "3B" if "3b" in n else "SeedVR2"
    sharp = " Sharp" if "sharp" in n else ""
    if "fp8" in n:
        precision = "FP8 · 省显存"
    elif "fp16" in n:
        precision = "FP16 · 最高质量"
    elif filename.lower().endswith(".gguf"):
        if "q4" in n:
            precision = "GGUF Q4 · 最省显存"
        elif "q8" in n:
            precision = "GGUF Q8 · 省显存"
        else:
            precision = "GGUF · 量化"
    elif "int8" in n:
        precision = "INT8 · 省显存"
    else:
        precision = "自定义模型"
    return f"{size}{sharp} — {precision} — {filename}"


def dit_dropdown_choices(default_name: str = "seedvr2_ema_3b_fp16.safetensors"):
    files = list_dit_model_files()
    if default_name not in files:
        files.insert(0, default_name)
    # Gradio supports (display_label, value) choices.
    return [(describe_dit_model(f), f) for f in files]


def vae_dropdown_choices(default_name: str = "ema_vae_fp16.safetensors"):
    files = list_vae_model_files()
    if default_name not in files:
        files.insert(0, default_name)
    return [(f, f) for f in files]


def require_model(filename: str) -> Path:
    path = models_root() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"SeedVR2 model not found: {path}\n"
            f"Put SeedVR2 model files in: {models_root()}"
        )
    return path
