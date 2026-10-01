from __future__ import annotations
import os
import sys
import time
import json
import ssl
import shutil
import tempfile
import requests
import gradio as gr
from typing import List, Dict

os.environ['SSL_CERT_FILE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_VERIFICATION'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

ssl._create_default_https_context = ssl._create_unverified_context

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
    from huggingface_hub.hf_api import HfApi
    from huggingface_hub.utils._http import hf_raise_for_status
    from huggingface_hub.utils import build_hf_headers
except ImportError as e:
    print(f"[ModelDownloader] 核心依赖缺失，请运行: pip install huggingface_hub")
    raise e

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    from modelscope.hub.file_download import model_file_download
    from modelscope import HubApi
except ImportError as e:
    ms_snapshot_download = None
    model_file_download = None
    HubApi = None
    print(f"[ModelDownloader] ModelScope SDK 导入失败: {e}")

from modules import script_callbacks
from modules.shared import opts, cmd_opts
from modules.paths import models_path

MODEL_DOWNLOADER_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CACHE_DIR = os.path.join(MODEL_DOWNLOADER_DIR, "cache")
IMAGE_DIR = os.path.join(MODEL_DOWNLOADER_DIR, "image")
# 项目根目录（webui 的上一级），用于把落盘位置显示为相对短路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL_DOWNLOADER_DIR)))
os.makedirs(CACHE_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# 模型组合预设列表
# 所有文件均从魔搭社区（ModelScope）下载，逐文件下载，不使用 snapshot_download，
# 避免下载无关权重、示例图和文档。
# target_dir 是相对于 models_path 的子目录名（不含 models/ 前缀）。
# ════════════════════════════════════════════════════════════════════
MODEL_PRESETS = [
    # ── 1. Flux-Klein：人像美学写实感模型 ──
    {
        "id": "flux-klein",
        "name": "Flux-Klein",
        "description": "Flux2-Klein-9B-True-V3 完整模型组合",
        "role": "上下文编辑模型",
        "vram": "12 GB",
        "cover": "klein.png",
        "files": [
            {
                "repo_id": "wikeeyang/Flux2-Klein-9B-True-V3",
                "file_path": "Flux2-Klein-9B-True-V3-int8mixedrow.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "9.44 GB",
            },
            {
                "repo_id": "Comfy-Org/flux2-klein-9B",
                "file_path": "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "8.66 GB",
            },
            {
                "repo_id": "Comfy-Org/flux2-klein-9B",
                "file_path": "split_files/vae/flux2-vae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "336 MB",
            },
        ],
    },
    # ── 2. Zimage：人像美学写实感模型 ──
    {
        "id": "zimage",
        "name": "Zimage",
        "description": "Z-Image-Turbo 完整模型组合",
        "role": "人像美学写实感模型",
        "vram": "8 GB",
        "cover": "zimage.png",
        "files": [
            {
                "repo_id": "Comfy-Org/z_image_turbo",
                "file_path": "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "6.20 GB",
            },
            {
                "repo_id": "Comfy-Org/z_image_turbo",
                "file_path": "split_files/text_encoders/qwen_3_4b.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "8.04 GB",
            },
            {
                "repo_id": "Comfy-Org/z_image_turbo",
                "file_path": "split_files/vae/ae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "335 MB",
            },
        ],
    },
    # ── 3. Anima：二次元动漫模型 ──
    {
        "id": "anima",
        "name": "Anima",
        "description": "Anima 二次元动漫模型组合",
        "role": "二次元动漫模型",
        "vram": "8 GB",
        "cover": "anima2.9b.png",
        "files": [
            {
                "repo_id": "aa4a4a/Anima-2.9B",
                "file_path": "Anima-2.9B-preview-v1.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "5.84 GB",
            },
            {
                "repo_id": "circlestone-labs/Anima",
                "file_path": "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "1.19 GB",
            },
            {
                "repo_id": "circlestone-labs/Anima",
                "file_path": "split_files/vae/qwen_image_vae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "254 MB",
            },
        ],
    },
    # ── 4. Krea2：新一代审美向多风格文生图模型 ──
    {
        "id": "krea2",
        "name": "Krea2",
        "description": "Krea2-Turbo 完整模型组合",
        "role": "审美向多风格模型",
        "vram": "12 GB",
        "cover": "krea2.png",
        "files": [
            {
                "repo_id": "Comfy-Org/Krea-2",
                "file_path": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "13.14 GB",
            },
            {
                "repo_id": "Comfy-Org/Krea-2",
                "file_path": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "5.24 GB",
            },
            {
                "repo_id": "Comfy-Org/Krea-2",
                "file_path": "vae/qwen_image_vae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "254 MB",
            },
        ],
    },
    # ── 5. Qwen-Image-2.1：通义千问图像生成模型 ──
    {
        "id": "qwen-image-2.1",
        "name": "Qwen-Image-2.1",
        "description": "Qwen-Image-2.1 int8 convrot 完整模型组合",
        "role": "上下文编辑模型",
        "vram": "12 GB",
        "cover": "qwen-image-2.1.png",
        "files": [
            {
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "file_path": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "6.76 GB",
            },
            {
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "file_path": "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "8.71 GB",
            },
            {
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "file_path": "vae/qwen_image_2.1_vae_bf16.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "644 MB",
            },
        ],
    },
    # ── 6. xl-Illustrious：SDXL 二次元动漫单文件模型 ──
    {
        "id": "xl-illustrious",
        "name": "xl-Illustrious",
        "description": "xl-Illustrious 4.0 单文件 checkpoint",
        "role": "初代二次元动漫模型",
        "vram": "6 GB",
        "cover": "xl-Illustrious.png",
        "files": [
            {
                "repo_id": "yangzxo/xl-Illustrious_4.0",
                "file_path": "xl-Illustrious_4.0.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "6.46 GB",
            },
        ],
    },
]

def get_preset(preset_id: str):
    """按 ID 查找预设，找不到时返回 None。"""
    for p in MODEL_PRESETS:
        if p["id"] == preset_id:
            return p
    return None

# ════════════════════════════════════════════════════════════════════
# LoRA 模型下载预设
# 所有文件均从魔搭社区（ModelScope）逐文件下载。
# target_dir 是相对于 models_path 的子目录名（LoRA）。
# ════════════════════════════════════════════════════════════════════
MODEL_LORAS = [
    {
        "id": "klein-9b-anime",
        "name": "Klein-9B 二次元动画",
        "cover": "klein-9b-anime.png",
        "role": "Klein-9B 动漫风格 LoRA",
        "repo_id": "yangzxo/klein-9b-anime",
        "file_path": "klein-9b二次元动画.safetensors",
        "target_dir": "Lora",
        "size": "158 MB",
    },
    {
        "id": "krea2-sticker",
        "name": "Krea2 贴图风格",
        "cover": "krea2-Sticker.png",
        "role": "Krea2 贴图风格 LoRA",
        "repo_id": "yangzxo/krea2-Sticker",
        "file_path": "krea2-贴图风格.safetensors",
        "target_dir": "Lora",
        "size": "218 MB",
    },
    {
        "id": "anima-character-design",
        "name": "Anima 角色设计",
        "cover": "anima_character_design.png",
        "role": "Anima 角色设计 LoRA",
        "repo_id": "yangzxo/anima_character_design",
        "file_path": "anima_lora.safetensors",
        "target_dir": "Lora",
        "size": "66 MB",
    },
    {
        "id": "krea2-scene-concept-art",
        "name": "Krea2 场景概念艺术",
        "cover": "krea2-sceneconceptart.png",
        "role": "Krea2 场景概念艺术 LoRA",
        "repo_id": "yangzxo/krea2-sceneconceptart",
        "file_path": "krea2场景概念艺术.safetensors",
        "target_dir": "Lora",
        "size": "218 MB",
    },
    {
        "id": "xl-comic-illustration",
        "name": "XL 漫画人设",
        "cover": "XL-comic.png",
        "role": "SDXL 漫画人设 LoRA",
        "repo_id": "yangzxo/XL-comic-illustration",
        "file_path": "XL漫画人设.safetensors",
        "target_dir": "Lora",
        "size": "870 MB",
    },
    {
        "id": "zimage-3d",
        "name": "Zimage 3D 风格",
        "cover": "zimage-3D.png",
        "role": "Zimage 3D 风格 LoRA",
        "repo_id": "yangyufeng/3dzimge",
        "file_path": "3dzimge_1.safetensors",
        "target_dir": "Lora",
        "size": "635 MB",
    },
]


def get_lora(lora_id: str):
    """按 ID 查找 LoRA 预设，找不到时返回 None。"""
    for l in MODEL_LORAS:
        if l["id"] == lora_id:
            return l
    return None

# ── 插件模型组合（一键下载，自动放入正确位置）──
# H3 DiT/文本编码器直接放入 models/diffusion_models 与 models/text_encoder
# （forge-h3-studio 本地 WebUI 模式首选查找目录 webui/models 下的子目录，
# text_encoder 为 webui 标准文本编码器目录，无额外嵌套，不依赖 ComfyUI），
# H3 NF4 的 VAE 放入 models/DiffSynth-Studio/MiniMax-H3-NF4
# （与 forge-h3-studio 本地模式的 VAE 缓存路径一致），
# 图层拆分模型整仓放入 models/diffusers/seethroughv0.0.*_nf4（与 sd-webui-see-through-sam 统一）。
# target_dir 支持绝对路径或相对于 models_path 的子目录。
H3_DIT_DIR = os.path.join(models_path, "diffusion_models")
H3_TE_DIR = os.path.join(models_path, "text_encoder")
H3_VAE_DIR = os.path.join(models_path, "vae")
H3_NF4_VAE_DIR = os.path.join(models_path, "DiffSynth-Studio", "MiniMax-H3-NF4")

PLUGIN_BUNDLES = [
    # ── 1. MiniMax-H3 INT8 组合（Comfy-Org int8_convrot）──
    {
        "id": "h3-int8",
        "name": "MiniMax-H3-INT8",
        "cover": "MiniMax-H3.png",
        "role": "视频生成模型组合（forge-h3-studio 插件）",
        "description": "INT8 量化组合：pruned 双 DiT + 文本编码器 + 原版视频/音频 VAE。",
        "source": "Comfy-Org/MiniMax-H3",
        "total_size": "约 72.5 GB",
        "vram": "16G",
        "files": [
            {
                "file_path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "target_dir": H3_DIT_DIR,
                "display_name": "DiT（FL2VA pruned，文生视频）",
                "size": "19.53 GB",
            },
            {
                "file_path": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "target_dir": H3_DIT_DIR,
                "display_name": "DiT（REF2VA pruned，参考生视频）",
                "size": "19.53 GB",
            },
            {
                "file_path": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "target_dir": H3_TE_DIR,
                "display_name": "文本编码器（Qwen3VL 32B INT8）",
                "size": "25.28 GB",
            },
            {
                "file_path": "FL2VA/video_vae/source/model.safetensors",
                "source": "MiniMax/MiniMax-H3-FL2VA",
                "target_dir": H3_VAE_DIR,
                "target_name": "minimax_h3_video_vae_fp16.safetensors",
                "display_name": "视频 VAE（fp16，INT8 组合配套）",
                "size": "约 0.3 GB",
            },
            {
                "file_path": "FL2VA/audio_vae/model.safetensors",
                "source": "MiniMax/MiniMax-H3-FL2VA",
                "target_dir": H3_VAE_DIR,
                "target_name": "minimax_h3_audio_vae_fp32.safetensors",
                "display_name": "音频 VAE（fp32，INT8 组合配套）",
                "size": "约 0.2 GB",
            },
        ],
    },
    # ── 2. MiniMax-H3 NF4 组合（DiffSynth bitsandbytes NF4）──
    {
        "id": "h3-nf4",
        "name": "MiniMax-H3-NF4",
        "cover": "MiniMax-H3.png",
        "role": "视频生成模型组合（forge-h3-studio 插件）",
        "description": "NF4 量化组合：完整 DiT 四件套 + 文本编码器 + NF4 视频/音频 VAE。",
        "source": "DiffSynth-Studio/MiniMax-H3-NF4",
        "total_size": "约 72.5 GB",
        "vram": "12G",
        "files": [
            {
                "file_path": "minimax-h3-fl2va-nf4.safetensors",
                "target_dir": H3_DIT_DIR,
                "display_name": "DiT（FL2VA，文生视频）",
                "size": "16.28 GB",
            },
            {
                "file_path": "minimax-h3-ref2va-nf4.safetensors",
                "target_dir": H3_DIT_DIR,
                "display_name": "DiT（REF2VA，参考生视频）",
                "size": "16.28 GB",
            },
            {
                "file_path": "minimax-h3-fl2va-pruned-nf4.safetensors",
                "target_dir": H3_DIT_DIR,
                "display_name": "DiT（FL2VA pruned）",
                "size": "9.99 GB",
            },
            {
                "file_path": "minimax-h3-ref2va-pruned-nf4.safetensors",
                "target_dir": H3_DIT_DIR,
                "display_name": "DiT（REF2VA pruned）",
                "size": "9.99 GB",
            },
            {
                "file_path": "minimax-h3-text-encoder-nf4.safetensors",
                "target_dir": H3_TE_DIR,
                "display_name": "文本编码器（Qwen3VL 32B NF4）",
                "size": "14.59 GB",
            },
            {
                "file_path": "video_vae_nf4.safetensors",
                "target_dir": H3_NF4_VAE_DIR,
                "display_name": "视频 VAE（NF4）",
                "size": "1.54 GB",
            },
            {
                "file_path": "audio_vae_nf4.safetensors",
                "target_dir": H3_NF4_VAE_DIR,
                "display_name": "音频 VAE（NF4）",
                "size": "284 MB",
            },
        ],
    },
    # ── 3. 图层拆分模型组合（sd-webui-see-through-sam 插件）──
    {
        "id": "seethrough",
        "name": "图层拆分模型",
        "cover": "see-through.png",
        "role": "图层拆分模型组合（sd-webui-see-through-sam 插件）",
        "description": "LayerDiff 3D 图层拆分 + Marigold 深度估计（NF4），下载完成即可直接使用。",
        "total_size": "约 5.7 GB",
        "vram": "8G",
        "repos": [
            {
                "repo_id": "ljsabc/seethroughv0.0.2_layerdiff3d_nf4",
                # 与 sd-webui-see-through-sam 的本地模型发现路径保持一致。
                "target_dir": os.path.join(models_path, "diffusers", "seethroughv0.0.2_layerdiff3d_nf4"),
                "display_name": "LayerDiff 3D 图层拆分模型",
                "size": "3.76 GB",
            },
            {
                "repo_id": "ljsabc/seethroughv0.0.1_marigold_nf4",
                "target_dir": os.path.join(models_path, "diffusers", "seethroughv0.0.1_marigold_nf4"),
                "display_name": "Marigold 深度估计模型",
                "size": "1.93 GB",
            },
        ],
    },
    # ── 4. Breeze-TTS-2 语音合成模型（forge-h3-studio 工作台）──
    # 整仓下载（双分片权重 + 音频分词器 + tokenizer 配置），
    # 放入 models/Breeze-TTS-2 后，多媒体处理插件的「Breeze-TTS-2 语音合成」标签页可直接使用。
    {
        "id": "breeze-tts-2",
        "name": "Breeze-TTS-2",
        "cover": "Breeze-TTS-2.png",
        "role": "语音合成模型（forge-h3-studio 工作台）",
        "description": "开源双语 TTS：声音克隆 / 声音设计 / 声音引导。整仓下载，含音频分词器，下载完成即可直接使用。",
        "total_size": "约 7.7 GB",
        "vram": "12G",
        "repos": [
            {
                "repo_id": "BreezeBlue/Breeze-TTS-2",
                "target_dir": os.path.join(models_path, "Breeze-TTS-2"),
                "display_name": "Breeze-TTS-2 完整模型（双分片权重 + 音频分词器）",
                "size": "约 7.7 GB",
            },
        ],
    },
    # ── 5. TRELLIS.2 图生3D 模型组合（sd-webui-trellis2 插件）──
    # 主模型 + 背景移除（RMBG-2.0）+ 图像编码器（TRELLIS-image-large）+ DINOv3 特征提取器，
    # 全部放入 models/trellis2/ 对应子目录后，TRELLIS.2 图生成3D 标签页可直接使用。
    {
        "id": "trellis2-img23d",
        "name": "TRELLIS.2 图生3D",
        "cover": "trellis2.png",
        "role": "图生3D 模型组合（sd-webui-trellis2 插件）",
        "description": "TRELLIS.2 单图生成 3D 模型（含 PBR 纹理）。含主模型、背景移除、图像编码器与 DINOv3 特征提取器，下载完成即可直接使用。",
        "total_size": "约 19.7 GB",
        "vram": "16G",
        "repos": [
            {
                "repo_id": "microsoft/TRELLIS.2-4B",
                "target_dir": os.path.join(models_path, "trellis2", "TRELLIS.2-4B"),
                "display_name": "TRELLIS.2-4B 主模型",
                "size": "约 15.1 GB",
            },
            {
                "repo_id": "briaai/RMBG-2.0",
                "target_dir": os.path.join(models_path, "trellis2", "BiRefNet", "RMBG-2.0"),
                "display_name": "RMBG-2.0 背景移除模型",
                "size": "约 0.4 GB",
            },
            {
                "repo_id": "microsoft/TRELLIS-image-large",
                "target_dir": os.path.join(models_path, "trellis2", "TRELLIS-image-large"),
                "display_name": "TRELLIS-image-large 图像编码器",
                "size": "约 3.1 GB",
            },
            {
                "repo_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "target_dir": os.path.join(models_path, "trellis2", "facebook", "dinov3-vitl16-pretrain-lvd1689m"),
                "display_name": "DINOv3 图像特征提取器",
                "size": "约 1.1 GB",
            },
        ],
    },
    # ── 6. 高清放大模型组合（Real-ESRGAN + SeedVR2）──
    # 单文件下载到 models/ESRGAN（Forge 自动扫描上采样器）或 models/SEEDVR2。
    # 多个模型来自不同仓库，每个文件单独指定 source。
    {
        "id": "upscalers",
        "name": "高清放大模型组合",
        "cover": "fangda.png",
        "role": "高清放大模型",
        "description": "Real-ESRGAN（通用/动漫 4 倍放大）+ SeedVR2（AI 图像/视频高清放大）。下载后 ESRGAN 模型自动出现在上采样器列表，SeedVR2 放入 models/SEEDVR2。",
        "total_size": "约 3.71 GB",
        "files": [
            {
                "file_path": "4x-UltraSharp.pth",
                "source": "XiangZL0/4x-UltraSharp",
                "target_dir": "ESRGAN",
                "display_name": "4x-UltraSharp（通用写实 4 倍）",
                "size": "63.87 MB",
            },
            {
                "file_path": "RealESRGAN_x4plus_anime_6B.pth",
                "source": "amd/realesrgan-x4plus-anime-6b",
                "target_dir": "ESRGAN",
                "display_name": "Real-ESRGAN anime 6B（动漫 4 倍）",
                "size": "17.11 MB",
            },
            {
                "file_path": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                "source": "numz/SeedVR2_comfyUI",
                "target_dir": "SEEDVR2",
                "display_name": "SeedVR2 主模型（3B fp8）",
                "size": "3.16 GB",
            },
            {
                "file_path": "ema_vae_fp16.safetensors",
                "source": "numz/SeedVR2_comfyUI",
                "target_dir": "SEEDVR2",
                "display_name": "SeedVR2 VAE",
                "size": "478.09 MB",
            },
        ],
    },
]

def get_plugin_bundle(bundle_id: str):
    """按 ID 查找插件模型组合，找不到时返回 None。"""
    for b in PLUGIN_BUNDLES:
        if b["id"] == bundle_id:
            return b
    return None

def _display_target(target_dir: str) -> str:
    """把目标目录显示为相对于项目根目录的短路径，UI 卡片用。"""
    try:
        return os.path.relpath(target_dir, PROJECT_ROOT).replace(os.sep, "\\")
    except ValueError:
        return target_dir

class ModelDownloader:
    def __init__(self):
        self.downloading = False
        self.current_task = None
        self.download_history = []
        self.history_file = os.path.join(CACHE_DIR, "download_history.json")
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.download_history = json.load(f)
            except Exception:
                self.download_history = []

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_history[:100], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def list_model_files(self, model_name: str, source: str) -> List[str]:
        try:
            if source == "huggingface":
                session = requests.Session()
                session.verify = False
                session.headers.update(build_hf_headers())
                url = f"https://huggingface.co/api/models/{model_name}/tree/main?recursive=True&expand=False"
                response = session.get(url)
                hf_raise_for_status(response)
                data = response.json()
                files = []
                self._extract_files(data, files)
                return files
            else:
                if HubApi:
                    api = HubApi()
                    info = api.model_info(model_name)
                    files = []
                    if hasattr(info, 'siblings') and info.siblings:
                        for item in info.siblings:
                            if hasattr(item, 'rfilename'):
                                files.append(item.rfilename)
                    return files
                return []
        except Exception as e:
            print(f"[ModelDownloader] 获取文件列表失败: {e}")
            return []
    
    def _extract_files(self, data, files):
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('type') == 'file':
                        files.append(item.get('path', ''))
                    elif item.get('type') == 'dir':
                        self._extract_files(item.get('children', []), files)

    def download_preset(self, preset_id: str, progress_callback=None) -> str:
        """从魔搭社区（ModelScope）逐文件下载指定预设组合，放入 Forge 对应目录。

        每个文件独立下载，绝不使用 snapshot_download，避免下载无关权重、示例和文档。
        支持来自不同仓库的文件。progress_callback(i, total, desc) 可选用于 UI 进度反馈。
        """
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        preset = get_preset(preset_id)
        if not preset:
            return f"找不到预设 ID: {preset_id}"

        self.downloading = True
        files = preset["files"]
        total = len(files)
        copied = []
        temp_dir = tempfile.mkdtemp(prefix=f"preset-{preset_id}-", dir=CACHE_DIR)

        try:
            for i, fspec in enumerate(files):
                repo_id = fspec["repo_id"]
                file_path = fspec["file_path"]
                target_dir = fspec["target_dir"]
                display_name = fspec["display_name"]
                filename = os.path.basename(file_path)

                if progress_callback:
                    progress_callback(i, total, f"正在下载 {display_name} ({i+1}/{total}) ...")

                print(f"[ModelDownloader] 下载 {display_name}: {repo_id}/{file_path}")

                # 逐文件下载到临时目录
                result = model_file_download(
                    model_id=repo_id,
                    file_path=file_path,
                    cache_dir=temp_dir,
                )

                # 定位下载后的实际文件路径
                source_file = result if isinstance(result, str) and os.path.isfile(result) else None
                if not source_file:
                    candidates = []
                    for root, _, names in os.walk(temp_dir):
                        if filename in names:
                            candidates.append(os.path.join(root, filename))
                    if not candidates:
                        raise FileNotFoundError(f"下载后找不到文件: {filename}")
                    source_file = candidates[0]

                # 复制到 Forge models_path 下的对应子目录
                destination_dir = os.path.join(models_path, target_dir)
                os.makedirs(destination_dir, exist_ok=True)
                destination = os.path.join(destination_dir, filename)
                shutil.copy2(source_file, destination)
                copied.append(destination)
                print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(total, total, "全部下载完成")

            self.download_history.append({
                "model_name": preset["name"],
                "source": "modelscope",
                "save_path": models_path,
                "filename": ", ".join(os.path.basename(x) for x in copied),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": f"{preset['name']} 组合",
            })
            self._save_history()
            return f"✅ {preset['name']} 组合下载完成，共 {len(copied)} 个文件:\n" + "\n".join(f"  • {x}" for x in copied)
        except Exception as e:
            print(f"[ModelDownloader] {preset['name']} 组合下载错误: {e}")
            return f"❌ {preset['name']} 组合下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download_lora(self, lora_id: str, progress_callback=None) -> str:
        """从魔搭社区（ModelScope）下载指定 LoRA 模型，放入 Forge 的 models/Lora 目录。

        单文件下载。下载完成后文件会被 Forge 自动扫描，出现在 LoRA 下拉列表，可直接使用。
        """
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        lora = get_lora(lora_id)
        if not lora:
            return f"找不到 LoRA 模型 ID: {lora_id}"

        repo_id = lora["repo_id"]
        file_path = lora["file_path"]
        target_dir = lora["target_dir"]
        name = lora["name"]
        filename = os.path.basename(file_path)

        self.downloading = True
        temp_dir = tempfile.mkdtemp(prefix=f"lora-{lora_id}-", dir=CACHE_DIR)
        try:
            if progress_callback:
                progress_callback(0, 1, f"正在下载 {name} ...")
            print(f"[ModelDownloader] 下载 LoRA 模型 {name}: {repo_id}/{file_path}")

            result = model_file_download(
                model_id=repo_id,
                file_path=file_path,
                cache_dir=temp_dir,
            )

            source_file = result if isinstance(result, str) and os.path.isfile(result) else None
            if not source_file:
                candidates = []
                for root, _, names in os.walk(temp_dir):
                    if filename in names:
                        candidates.append(os.path.join(root, filename))
                if not candidates:
                    raise FileNotFoundError(f"下载后找不到文件: {filename}")
                source_file = candidates[0]

            destination_dir = os.path.join(models_path, target_dir)
            os.makedirs(destination_dir, exist_ok=True)
            destination = os.path.join(destination_dir, filename)
            shutil.copy2(source_file, destination)
            print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(1, 1, "下载完成")

            self.download_history.append({
                "model_name": name,
                "source": "modelscope",
                "save_path": destination_dir,
                "filename": filename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "LoRA 模型",
            })
            self._save_history()
            return f"✅ LoRA 模型 {name} 下载完成：\n  • {destination}\n重启 WebUI 后会自动出现在 LoRA 列表，可直接使用。"
        except Exception as e:
            print(f"[ModelDownloader] LoRA 模型 {name} 下载错误: {e}")
            return f"❌ LoRA 模型 {name} 下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _resolve_target_dir(self, target_dir: str) -> str:
        """target_dir 支持绝对路径或相对于 models_path 的子目录。"""
        return target_dir if os.path.isabs(target_dir) else os.path.join(models_path, target_dir)

    def download_plugin_bundle(self, bundle_id: str, progress_callback=None) -> str:
        """从魔搭社区一键下载插件模型组合，自动放入各插件期望的正确位置。

        files 类型：逐文件下载后复制到目标目录（支持绝对路径）；
        repos 类型：整仓 snapshot 下载后整体拷贝（保留 diffusers 目录结构）。
        """
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        bundle = get_plugin_bundle(bundle_id)
        if not bundle:
            return f"找不到插件模型组合 ID: {bundle_id}"

        name = bundle["name"]
        temp_dir = tempfile.mkdtemp(prefix=f"bundle-{bundle_id}-", dir=CACHE_DIR)
        saved = []
        self.downloading = True
        try:
            if bundle.get("repos"):
                # 整仓下载（diffusers 目录布局，保留子目录结构）
                items = bundle["repos"]
                total = len(items)
                for i, rspec in enumerate(items):
                    repo_id = rspec["repo_id"]
                    destination = self._resolve_target_dir(rspec["target_dir"])
                    display_name = rspec["display_name"]
                    if progress_callback:
                        progress_callback(i, total, f"正在下载 {display_name} ({i+1}/{total}) ...")
                    print(f"[ModelDownloader] 整仓下载: {repo_id}")
                    result = ms_snapshot_download(model_id=repo_id, cache_dir=temp_dir) if ms_snapshot_download else None
                    repo_root = result if isinstance(result, str) and os.path.isdir(result) else None
                    if not repo_root:
                        # 兜底：在临时目录中定位含 model_index.json 或 config.json 的仓库根
                        # （Breeze-TTS-2 等仓库无 model_index.json，根目录只有 config.json）
                        for dirpath, _, filenames in os.walk(temp_dir):
                            if "model_index.json" in filenames or "config.json" in filenames:
                                repo_root = dirpath
                                break
                    if not repo_root:
                        raise FileNotFoundError(f"整仓下载后找不到仓库根目录: {repo_id}")
                    os.makedirs(destination, exist_ok=True)
                    shutil.copytree(repo_root, destination, dirs_exist_ok=True)
                    saved.append(destination)
                    print(f"[ModelDownloader] 已放入: {destination}")
            else:
                # 逐文件下载（每个文件可单独指定 source 仓库；未指定时回退到 bundle.source）
                files = bundle["files"]
                total = len(files)
                for i, fspec in enumerate(files):
                    file_path = fspec["file_path"]
                    repo_source = fspec.get("source") or bundle.get("source")
                    destination_dir = self._resolve_target_dir(fspec["target_dir"])
                    display_name = fspec["display_name"]
                    filename = fspec.get("target_name") or os.path.basename(file_path)
                    if progress_callback:
                        progress_callback(i, total, f"正在下载 {display_name} ({i+1}/{total}) ...")
                    print(f"[ModelDownloader] 下载 {display_name}: {repo_source}/{file_path}")
                    result = model_file_download(
                        model_id=repo_source,
                        file_path=file_path,
                        cache_dir=temp_dir,
                    )
                    source_file = result if isinstance(result, str) and os.path.isfile(result) else None
                    if not source_file:
                        candidates = []
                        for root, _, names in os.walk(temp_dir):
                            if filename in names:
                                candidates.append(os.path.join(root, filename))
                        if not candidates:
                            raise FileNotFoundError(f"下载后找不到文件: {filename}")
                        source_file = candidates[0]
                    os.makedirs(destination_dir, exist_ok=True)
                    destination = os.path.join(destination_dir, filename)
                    shutil.copy2(source_file, destination)
                    saved.append(destination)
                    print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(total, total, "全部下载完成")

            self.download_history.append({
                "model_name": name,
                "source": "modelscope",
                "save_path": ", ".join(_display_target(x) for x in saved),
                "filename": ", ".join(os.path.basename(x) for x in saved),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": f"{name} 组合",
            })
            self._save_history()
            return f"✅ {name} 组合下载完成，共 {len(saved)} 项:\n" + "\n".join(f"  • {x}" for x in saved) + "\n无需手动移动文件，插件可直接使用。"
        except Exception as e:
            print(f"[ModelDownloader] {name} 组合下载错误: {e}")
            return f"❌ {name} 组合下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download_model(self, model_name: str, source: str, save_path: str, filename: str = "") -> bool:
        try:
            if self.downloading:
                return False
            self.downloading = True
            
            model_name = model_name.strip()
            if not model_name:
                raise ValueError("模型名称不能为空")
                
            if len(model_name.split('/')) < 2:
                raise ValueError(f"模型 ID 格式错误。正确格式: '用户名/模型名'\n例如: Comfy-Org/Krea-2 或 qwen/Qwen2.5-7B-Instruct")
                
            self.current_task = f"Downloading {model_name}"
            os.makedirs(save_path, exist_ok=True)

            if source == "huggingface":
                print(f"[ModelDownloader] 从 HuggingFace 下载: {model_name}")
                session = requests.Session()
                session.verify = False
                if filename:
                    hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        token=None,
                        session=session,
                    )
                else:
                    ignore_8bit = not getattr(cmd_opts, 'load_in_8bit', False)
                    ignore_patterns = ["*.bin", "*.h5"] if ignore_8bit else []
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        ignore_patterns=ignore_patterns,
                        token=None,
                        session=session,
                    )
            else:
                print(f"[ModelDownloader] 从 ModelScope 下载: {model_name}")
                if filename:
                    if model_file_download:
                        model_file_download(
                            model_id=model_name,
                            file_path=filename,
                            cache_dir=save_path,
                        )
                    else:
                        raise RuntimeError("ModelScope SDK 单文件下载功能不可用，请尝试完整模型下载")
                else:
                    if ms_snapshot_download:
                        ms_snapshot_download(
                            model_id=model_name,
                            cache_dir=save_path,
                        )
                    else:
                        raise RuntimeError("ModelScope SDK 完整模型下载功能不可用")

            self.download_history.append({
                "model_name": model_name, 
                "source": source, 
                "save_path": save_path,
                "filename": filename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), 
                "type": "single" if filename else "full",
            })
            self._save_history()
            return True
        except Exception as e:
            print(f"[ModelDownloader] 下载错误: {str(e)}")
            return False
        finally:
            self.downloading = False
            self.current_task = None

model_downloader = ModelDownloader()

def _preset_file_markdown(preset_id: str):
    """从指定预设生成简洁的文件清单 Markdown（每个文件一行）。"""
    preset = get_preset(preset_id)
    if not preset:
        return ""
    lines = []
    for fspec in preset["files"]:
        filename = os.path.basename(fspec["file_path"])
        lines.append(f"- **{fspec['display_name']}**：{filename}（{fspec['size']}）")
    return "\n".join(lines)


def create_ui():
    # 注意：gr.Blocks(css=...) 的样式在嵌套渲染进主 WebUI Blocks 时会被 Gradio 5
    # 的 Blocks.render() 直接丢弃（只合并子块和事件，不合并 css），所以这里改用
    # gr.HTML + <style> 注入封面样式，确保能到达浏览器。
    # .image-container / .image-frame 是 gr.Image 内部预览层类名（Gradio 5.49.1）。
    cover_style = """
    <style>
    .model-cover {
        width: 360px !important;
        max-width: 100% !important;
        margin: 0 auto 8px;
    }
    .model-cover .image-container {
        min-width: 0 !important;
        height: auto !important;
    }
    .model-cover .image-container button,
    .model-cover .image-frame {
        width: 100% !important;
        height: auto !important;
    }
    .model-cover img {
        width: 100% !important;
        height: auto !important;
        border-radius: 8px;
    }
    /* gr.Row/gr.Column 是 flex 布局，列宽由容器分配（默认各占 50%），
       只改内部组件宽度无效，必须固定列本身的宽度 */
    .preset-row {
        justify-content: flex-start !important;
        gap: 16px !important;
    }
    .preset-col {
        flex: 0 1 360px !important;
        min-width: 0 !important;
    }
    </style>
    """
    with gr.Blocks(
        title="开源社区模型下载器",
        analytics_enabled=False,
    ) as model_downloader_tab:
        gr.HTML(cover_style)
        with gr.Tabs():
            # ── 标签页 1：一键下载模型组合 ──
            with gr.TabItem("一键下载模型组合"):
                # ── 生图模型下载区（最上方）──
                gr.Markdown("## 🖼️ 生图模型下载区")
                gr.Markdown(
                    "主流文生图模型组合，从魔搭社区一键下载主模型、文本编码器、VAE，"
                    "自动放入 Forge 对应目录，下载后即可在模型列表中选择使用。"
                )
                # 为每个预设生成独立的下载区块，5 个块并排一行（列宽 240px）
                with gr.Row(elem_classes=["preset-row"]):
                    for preset in MODEL_PRESETS:
                        pid = preset["id"]
                        with gr.Column(elem_classes=["preset-col"]):
                            cover_path = os.path.join(IMAGE_DIR, preset["cover"])
                            gr.Image(
                                value=cover_path,
                                label=f"{preset['name']} · {preset['role']}",
                                show_label=True,
                                show_download_button=False,
                                show_fullscreen_button=True,
                                interactive=False,
                                elem_classes=["model-cover"],
                                min_width=0,
                            )
                            gr.Markdown(f"显存要求：**≥ {preset['vram']}**")
                            with gr.Accordion(f"{preset['name']} — {preset['role']}", open=False):
                                gr.Markdown(f"**{preset['description']}**")
                                gr.Markdown(_preset_file_markdown(pid))

                            # 一键下载按钮放在折叠块下方
                            btn = gr.Button(
                                f"⚡ 一键下载 {preset['name']}",
                                variant="primary",
                                size="lg",
                            )
                            status_box = gr.Textbox(
                                label="下载状态",
                                interactive=False,
                                lines=3,
                            )

                            def make_download_fn(current_pid):
                                def _download():
                                    prog = gr.Progress()
                                    def progress_callback(i, total, desc):
                                        prog(i / total if total > 0 else 0, desc=desc)
                                    return model_downloader.download_preset(current_pid, progress_callback=progress_callback)
                                return _download

                            btn.click(make_download_fn(pid), outputs=status_box)

                # ── LoRA 模型下载区（主模型区下方）──
                gr.Markdown("---")
                gr.Markdown("## 🎨 LoRA 模型下载区")
                gr.Markdown(
                    "LoRA 微调模型。从魔搭社区下载，自动放入 Forge 的 LoRA 目录"
                    "（默认 `models/Lora`）；下载后重启 WebUI，即可在 LoRA 下拉列表中直接使用。"
                )
                with gr.Row(elem_classes=["preset-row"]):
                    for lora in MODEL_LORAS:
                        lid = lora["id"]
                        with gr.Column(elem_classes=["preset-col"]):
                            lora_cover = os.path.join(IMAGE_DIR, lora.get("cover", ""))
                            if os.path.isfile(lora_cover):
                                gr.Image(
                                    value=lora_cover,
                                    label=f"{lora['name']} · {lora['role']}",
                                    show_label=True,
                                    show_download_button=False,
                                    show_fullscreen_button=True,
                                    interactive=False,
                                    elem_classes=["model-cover"],
                                    min_width=0,
                                )
                            gr.Markdown(
                                f"**{lora['name']}**\n"
                                f"- 类型：{lora['role']}\n"
                                f"- 文件：{os.path.basename(lora['file_path'])}\n"
                                f"- 大小：{lora['size']}\n"
                                f"- 来源仓库：{lora['repo_id']}"
                            )
                            lora_btn = gr.Button(
                                f"⚡ 一键下载 {lora['name']}",
                                variant="primary",
                                size="lg",
                            )
                            lora_status = gr.Textbox(
                                label="下载状态",
                                interactive=False,
                                lines=3,
                            )

                            def make_lora_fn(current_lid):
                                def _download():
                                    prog = gr.Progress()
                                    def progress_callback(i, total, desc):
                                        prog(i / total if total > 0 else 0, desc=desc)
                                    return model_downloader.download_lora(current_lid, progress_callback=progress_callback)
                                return _download

                            lora_btn.click(make_lora_fn(lid), outputs=lora_status)

                # ── 插件模型组合下载区（H3 视频生成 / 图层拆分 / 图生3D / TTS）──
                gr.Markdown("---")
                gr.Markdown("## 🎬 插件模型组合下载区")
                gr.Markdown(
                    "视频生成（forge-h3-studio）、图层拆分（sd-webui-see-through-sam）、"
                    "图生3D（sd-webui-trellis2）与语音合成（forge-h3-studio）的模型组合。"
                    "从魔搭社区下载后**自动放入各插件期望的目录**，无需手动移动文件。"
                )
                with gr.Row(elem_classes=["preset-row"]):
                    for b in PLUGIN_BUNDLES:
                        bid = b["id"]
                        listing_lines = []
                        if b.get("repos"):
                            for rspec in b["repos"]:
                                listing_lines.append(
                                    f"- {rspec['display_name']}：{rspec['repo_id']} 整仓（{rspec['size']}）→ `{_display_target(rspec['target_dir'])}`"
                                )
                        else:
                            for fspec in b["files"]:
                                fsrc = fspec.get("source") or b.get("source", "")
                                listing_lines.append(
                                    f"- {fspec['display_name']}：{os.path.basename(fspec['file_path'])}（{fspec['size']}）"
                                    + (f" 来自 `{fsrc}`" if fsrc else "")
                                    + f" → `{_display_target(fspec['target_dir'])}`"
                                )
                        with gr.Column(elem_classes=["preset-col"]):
                            bundle_cover = os.path.join(IMAGE_DIR, b.get("cover", ""))
                            if os.path.isfile(bundle_cover):
                                gr.Image(
                                    value=bundle_cover,
                                    label=f"{b['name']} · {b['role']}",
                                    show_label=True,
                                    show_download_button=False,
                                    show_fullscreen_button=True,
                                    interactive=False,
                                    elem_classes=["model-cover"],
                                    min_width=0,
                                )
                            gr.Markdown(
                                f"**{b['name']}** — {b['description']}\n"
                                f"- 合计大小：**{b['total_size']}**"
                                + (f"\n- 显存要求：**{b['vram']} 显存**" if b.get("vram") else "")
                            )
                            with gr.Accordion("文件清单与落盘位置", open=False):
                                gr.Markdown(
                                    "\n".join(listing_lines)
                                    + (f"\n- 来源仓库：{b['source']}" if b.get("source") else "")
                                )
                            b_btn = gr.Button(
                                f"⚡ 一键下载 {b['name']}",
                                variant="primary",
                                size="lg",
                            )
                            b_status = gr.Textbox(
                                label="下载状态",
                                interactive=False,
                                lines=3,
                            )

                        def make_bundle_fn(current_bid):
                            def _download():
                                prog = gr.Progress()
                                def progress_callback(i, total, desc):
                                    prog(i / total if total > 0 else 0, desc=desc)
                                return model_downloader.download_plugin_bundle(current_bid, progress_callback=progress_callback)
                            return _download

                        b_btn.click(make_bundle_fn(bid), outputs=b_status)

            # ── 标签页 2：自定义模型下载 ──
            with gr.TabItem("自定义模型下载"):
                with gr.Row():
                    with gr.Column(scale=1):
                        source = gr.Radio(
                            choices=["huggingface", "modelscope"],
                            value="modelscope",
                            label="模型源",
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        model_input = gr.Textbox(
                            label="模型名称/仓库ID",
                            placeholder="例如: Comfy-Org/Krea-2 或 qwen/Qwen2.5-7B-Instruct",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        fetch_files_btn = gr.Button("获取文件列表")
                        download_full_btn = gr.Button("下载完整模型", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        file_dropdown = gr.Dropdown(
                            choices=[],
                            label="选择要下载的文件",
                            interactive=True,
                            multiselect=True,
                        )
                    with gr.Column(scale=1):
                        download_single_btn = gr.Button("下载选中文件")

                save_path = gr.Textbox(
                    label="保存路径",
                    value=os.path.join(models_path, "Stable-diffusion"),
                    interactive=True,
                )

                custom_status = gr.Textbox(label="下载状态", interactive=False)

                def fetch_files(model_name, src):
                    if not model_name:
                        return gr.update(choices=[], value=None), "请输入模型名称"
                    if len(model_name.split('/')) < 2:
                        return gr.update(choices=[], value=None), "模型 ID 格式错误，正确格式: 用户名/模型名"
                    files = model_downloader.list_model_files(model_name, src)
                    if not files:
                        return gr.update(choices=[], value=None), f"未能获取 {model_name} 的文件列表"
                    return gr.update(choices=files, value=None), f"获取到 {len(files)} 个文件"

                def download_full(model_name, src, path):
                    if not model_name: return "请输入模型名称"
                    try:
                        prog = gr.Progress()
                        prog(0, desc="开始下载...")
                        success = model_downloader.download_model(model_name, src, path)
                        prog(1, desc="下载完成" if success else "下载失败")
                        return f"模型 {model_name} 下载完成，保存到: {path}" if success else "下载失败，请查看控制台日志"
                    except Exception as e:
                        return f"下载异常: {str(e)}"

                def download_single(model_name, src, path, filenames):
                    if not model_name: return "请输入模型名称"
                    if not filenames or len(filenames) == 0: return "请先选择要下载的文件"
                    try:
                        prog = gr.Progress()
                        prog(0, desc=f"开始下载 {len(filenames)} 个文件...")
                        for i, filename in enumerate(filenames):
                            success = model_downloader.download_model(model_name, src, path, filename)
                            prog((i+1)/len(filenames), desc=f"正在下载 {filename}...")
                            if not success:
                                return f"文件 {filename} 下载失败，请查看控制台日志"
                        return f"已成功下载 {len(filenames)} 个文件到: {path}"
                    except Exception as e:
                        return f"下载异常: {str(e)}"

                fetch_files_btn.click(fetch_files, inputs=[model_input, source], outputs=[file_dropdown, custom_status])
                download_full_btn.click(download_full, inputs=[model_input, source, save_path], outputs=custom_status)
                download_single_btn.click(download_single, inputs=[model_input, source, save_path, file_dropdown], outputs=custom_status)

            # ── 标签页 3：下载历史 ──
            with gr.TabItem("下载历史"):
                def load_history_data():
                    return [[h["time"], h["model_name"], h["source"], h["type"], h["save_path"]] for h in model_downloader.download_history]

                history_table = gr.Dataframe(
                    headers=["时间", "模型名称", "来源", "类型", "路径"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="下载历史",
                    interactive=False,
                    value=load_history_data(),
                )

                def clear_history():
                    model_downloader.download_history.clear()
                    model_downloader._save_history()
                    return []

                clear_history_btn = gr.Button("清空历史")
                clear_history_btn.click(clear_history, outputs=history_table)

    return [(model_downloader_tab, "开源社区模型下载器", "model-downloader")]

def on_ui_tabs():
    return create_ui()

script_callbacks.on_ui_tabs(on_ui_tabs)
