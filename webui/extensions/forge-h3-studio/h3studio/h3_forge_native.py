"""
H3 生成引擎（DiffSynth LoRA 兼容）
==================================

使用 Forge 后端的 MiniMax H3 模型实现（移植自 ComfyUI），LoRA 格式兼容 DiffSynth。
- Forge 的 int8_convrot 有 Triton 优化内核（Blackwell/GPU 更快）
- 支持 disk offload（小显存）和 stream_blocks 流式加载
- LoRA 自动识别 lightx2v 格式并映射到 Forge 层路径
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 确保 Forge backend 可导入
_WEBUI_ROOT = Path(__file__).resolve().parents[3]
if str(_WEBUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_WEBUI_ROOT))

from backend import memory_management
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict
from backend.utils import no_init_weights, calculate_parameters, weight_dtype


# ── 常量 ────────────────────────────────────────────────────────────────────

VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_LATENT_CH = 2  # 立体声
PATCH_SIZE = (1, 2, 2)
VIDEO_SHIFT_DEFAULT = 12.0
AUDIO_SHIFT_DEFAULT = 3.0
NUM_TRAIN_TIMESTEPS = 1000


# ── Sigma 调度 ───────────────────────────────────────────────────────────────

def make_sigmas(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    """FlowMatch sigma 调度（与 DiffSynth set_timesteps_minimax_h3 一致）。"""
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)[:-1]
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    return sigmas.to(device)


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """在不同 shift 之间换算 sigma。"""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def _log_ram(tag: str) -> None:
    """打印当前 CPU 内存（RSS）使用情况。"""
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        print(f"[H3 Forge] 内存 [{tag}]: RSS {rss:.2f} GiB", flush=True)
    except Exception:
        pass


def load_h3_dit(model_path: str, log_name: str = "MiniMaxH3Model"):
    """加载 H3 DiT（自动检测 int8/nf4 量化，支持 DiffSynth LoRA 格式）。

    小显存（<24GB）时启用 disk offload：blocks 权重不加载到 CPU 内存，
    而是通过 safetensors mmap 在 forward 时按需从磁盘读取到 GPU。
    """
    from backend.nn.minimax_h3 import build_minimax_h3_model
    from safetensors.torch import load_file
    from safetensors import safe_open

    print(f"[H3 Forge] 加载 DiT: {Path(model_path).name}", flush=True)
    total_vram = memory_management.get_total_memory(memory_management.get_torch_device())
    use_disk_offload = total_vram < 24 * 1024 ** 3
    _log_ram("DiT 加载前")

    if use_disk_offload:
        # Disk offload：用 mmap 打开，不加载到 CPU 内存
        disk_map = safe_open(model_path, framework="pt", device="cpu")
        state_dict = {k: disk_map.get_tensor(k) for k in disk_map.keys()}
        model = build_minimax_h3_model(state_dict, log_name=log_name, disk_offload=True)
        model.disk_map = disk_map
        model.stream_blocks = True
        del state_dict
        print(f"[H3 Forge] ★ 启用 Disk offload + 流式 block 加载（显存 {total_vram / 1024**3:.0f} GiB < 24 GiB）", flush=True)
    else:
        state_dict = load_file(model_path, device="cpu")
        model = build_minimax_h3_model(state_dict, log_name=log_name)
        del state_dict
        print(f"[H3 Forge] DiT 完整加载到内存（显存 {total_vram / 1024**3:.0f} GiB >= 24 GiB）", flush=True)

    _log_ram("DiT 加载后")
    return model


def _load_vae_with_quant(model_cls, model_path: str, log_name: str):
    """加载 VAE，自动支持 nf4 预量化。"""
    from safetensors.torch import load_file

    state_dict = load_file(model_path, device="cpu")
    state_dict_dtype = weight_dtype(state_dict)

    if state_dict_dtype in ("nf4", "fp4", "gguf"):
        print(f"[H3 Forge] {log_name} 检测到预量化: {state_dict_dtype}", flush=True)
        # bitsandbytes NF4/FP4 必须在 CUDA 设备上初始化，否则 quant_state 为 None
        load_device = memory_management.get_torch_device()
        if memory_management.should_use_bf16(load_device):
            computation_dtype = torch.bfloat16
        elif memory_management.should_use_fp16(load_device, prioritize_performance=True):
            computation_dtype = torch.float16
        else:
            computation_dtype = torch.float32
        with using_forge_operations(device=load_device, dtype=computation_dtype, manual_cast_enabled=True, bnb_dtype=state_dict_dtype):
            model = model_cls()
        load_state_dict(model, state_dict, log_name=log_name)
        # ConvTranspose1d 未被 Forge 替换，weight_norm 参数可能是 float32，需手动转计算 dtype
        for name, param in model.named_parameters():
            if name.endswith((".weight_g", ".weight_v")) and param.dtype != computation_dtype:
                param.data = param.data.to(computation_dtype)
        # 收尾：非量化参数统一为计算 dtype
        for p in model.parameters():
            p.requires_grad = False
            is_bnb = getattr(p, "bnb_quantized", False)
            if not is_bnb and p.dtype != computation_dtype:
                p.data = p.data.to(computation_dtype)
        # 模型移回 CPU，推理时按需加载到 GPU
        model = model.to("cpu")
        memory_management.soft_empty_cache()
    else:
        model = model_cls()
        load_state_dict(model, state_dict, log_name=log_name)
        model = model.to(torch.bfloat16)

    del state_dict
    return model.eval()


def load_h3_video_vae(model_path: str):
    """加载 H3 视频 VAE（支持 nf4）。"""
    from backend.nn.minimax_h3_vae import MiniMaxH3VideoVAE

    print(f"[H3 Forge] 加载视频 VAE: {Path(model_path).name}", flush=True)
    return _load_vae_with_quant(MiniMaxH3VideoVAE, model_path, "MiniMaxH3VideoVAE")


def load_h3_audio_vae(model_path: str):
    """加载 H3 音频 VAE（支持 nf4）。"""
    from backend.nn.minimax_h3_vae import MiniMaxH3AudioVAE

    print(f"[H3 Forge] 加载音频 VAE: {Path(model_path).name}", flush=True)
    return _load_vae_with_quant(MiniMaxH3AudioVAE, model_path, "MiniMaxH3AudioVAE")


def load_h3_text_encoder(model_path: str):
    """加载 Qwen3-VL 文本编码器（截断层 hidden states，无 final norm）。

    支持 nf4/int8 预量化：
    - nf4/fp4：通过 Forge 的 using_forge_operations(bnb_dtype="nf4") 替换 Linear 为 bitsandbytes NF4
    - int8 (comfy_quant)：通过 using_forge_operations(bnb_dtype={"mixed_ops": True}) 替换 Linear 为
      mixed_precision_ops，权重保持 comfy-kitchen QuantizedTensor（int8 格式，不转 bf16）
    """
    from transformers import AutoTokenizer
    from diffsynth.models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
    from diffsynth.utils.state_dict_converters.minimax_h3_text_encoder import MiniMaxH3TextEncoderComfyOrgStateDictConverter
    from safetensors.torch import load_file
    from backend.state_dict import detect_quantization, load_state_dict as forge_load_state_dict

    print(f"[H3 Forge] 加载文本编码器: {Path(model_path).name}", flush=True)

    # 先检测量化类型
    state_dict = load_file(model_path, device="cpu")
    state_dict_dtype = weight_dtype(state_dict)
    quant_config = detect_quantization(state_dict, is_unet=False)

    # Comfy-Org int8 模型用 model.layers.* 前缀，需转换为 model.language_model.layers.*
    if quant_config is not None:
        state_dict = MiniMaxH3TextEncoderComfyOrgStateDictConverter(state_dict)

    if state_dict_dtype in ("nf4", "fp4", "gguf"):
        print(f"[H3 Forge] 文本编码器检测到预量化: {state_dict_dtype}", flush=True)
        # 在 bnb_dtype 上下文中创建模型，使 nn.Linear 替换为 NF4 版本
        with using_forge_operations(device=memory_management.cpu, dtype=torch.bfloat16, manual_cast_enabled=False, bnb_dtype=state_dict_dtype):
            model = MiniMaxH3TextEncoder(num_retained_layers=50)
        forge_load_state_dict(model, state_dict, log_name="MiniMaxH3TextEncoder")
    elif quant_config is not None:
        # int8 (comfy_quant)：使用 mixed_precision_ops，权重保持 QuantizedTensor（int8 格式）
        print(f"[H3 Forge] 文本编码器检测到 int8 量化（comfy_quant），使用 mixed_precision_ops", flush=True)
        with using_forge_operations(device=memory_management.cpu, dtype=torch.bfloat16, manual_cast_enabled=True, bnb_dtype=quant_config):
            model = MiniMaxH3TextEncoder(num_retained_layers=50)
        forge_load_state_dict(model, state_dict, log_name="MiniMaxH3TextEncoder")
    else:
        model = MiniMaxH3TextEncoder(num_retained_layers=50)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[H3 Forge] 文本编码器缺失键: {len(missing)} 个", flush=True)
        if unexpected:
            print(f"[H3 Forge] 文本编码器多余键: {len(unexpected)} 个", flush=True)

    del state_dict

    # 移除 final norm（H3 需要截断层 hidden states）
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "norm"):
        model.model.language_model.norm = torch.nn.Identity()

    # 仅非量化模型才转 bf16；量化模型（nf4/int8）权重保持原量化格式
    if state_dict_dtype not in ("nf4", "fp4", "gguf") and quant_config is None:
        model = model.to(torch.bfloat16)

    model = model.eval()

    # tokenizer：Qwen3-VL 用的是 Qwen2.5 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    return model, tokenizer


# ── 文本编码 ─────────────────────────────────────────────────────────────────

def encode_text(text_encoder, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """编码提示词，返回 [1, L, 5120] 的 hidden states。

    H3 文本编码器是 Qwen3-VL，取截断层 hidden states（无 final norm）。
    """
    system_prompt = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    full_prompt = system_prompt + user_prompt

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # DiffSynth MiniMaxH3TextEncoder 返回 (hidden_states, ...)
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        hidden_states = outputs.hidden_states[-1]
    elif hasattr(outputs, "last_hidden_state"):
        hidden_states = outputs.last_hidden_state
    else:
        hidden_states = outputs

    return hidden_states.to(torch.bfloat16)


# ── 图像/视频 VAE 编码 ───────────────────────────────────────────────────────

def encode_frame_to_latent(vae, image: Image.Image, width: int, height: int, device: torch.device) -> torch.Tensor:
    """将 PIL 图像编码为 H3 视频 latent [1, 24, 1, H/16, W/16]。"""
    img = image.convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [0,1] -> [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    tensor = tensor.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        latent = vae.encode(tensor)
    return latent


def encode_audio_to_latent(audio_vae, waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    """将音频波形编码为 H3 音频 latent [1, 32, 2, T]。"""
    with torch.no_grad():
        latent = audio_vae.encode(waveform.to(device, dtype=torch.bfloat16))
    return latent


# ── 噪声初始化 ───────────────────────────────────────────────────────────────

def init_noise(shape: tuple, seed: int, device: torch.device, dtype=torch.bfloat16) -> torch.Tensor:
    """生成指定形状的噪声。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=gen, dtype=torch.float32)
    return noise.to(device, dtype=dtype)


# ── 采样循环 ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def sample_h3(
    dit,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    payload: dict,
    num_steps: int,
    video_shift: float,
    audio_shift: float,
    seed: int,
    device: torch.device,
    progress_cb: Callable | None = None,
    cancel_check: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """H3 双 sigma 采样循环（Euler）。"""
    sigmas_video = make_sigmas(num_steps, video_shift, device)
    sigmas_audio = make_sigmas(num_steps, audio_shift, device)

    # audio_scale = shift_audio / shift_video，让采样器用统一的 sigma_v 携带音频
    audio_scale = audio_shift / video_shift

    payload = dict(payload)
    payload["audio_scale"] = audio_scale

    for step_idx in range(num_steps):
        if cancel_check and cancel_check():
            raise RuntimeError("生成已取消")

        sigma_v = float(sigmas_video[step_idx])

        # 调用 DiT
        noise_pred = dit(
            [video_latents, audio_latents],
            sigma_v,
            context,
            payload=payload,
            sample_sigmas=sigmas_video,
        )
        noise_pred_video, noise_pred_audio = noise_pred

        # Euler 步
        sigma_v_next = float(sigmas_video[step_idx + 1]) if step_idx + 1 < num_steps else 0.0
        sigma_a_next = float(sigmas_audio[step_idx + 1]) if step_idx + 1 < num_steps else 0.0

        video_latents = video_latents + noise_pred_video * (sigma_v_next - sigma_v)
        audio_latents = audio_latents + noise_pred_audio * (sigma_a_next - float(sigmas_audio[step_idx]))

        if progress_cb:
            progress_cb({"progress": (step_idx + 1) / num_steps, "step": step_idx + 1, "total": num_steps})

    return video_latents, audio_latents


# ── VAE 解码 ─────────────────────────────────────────────────────────────────

def decode_video(vae, latent: torch.Tensor) -> torch.Tensor:
    """解码视频 latent 为 [T, H, W, 3] 像素张量。"""
    with torch.no_grad():
        frames = vae.decode(latent)
    # frames: [1, 3, T, H, W] -> [T, H, W, 3]
    frames = frames[0].permute(1, 2, 3, 0).cpu().clamp(0, 1)
    return frames


def decode_audio(audio_vae, latent: torch.Tensor) -> torch.Tensor:
    """解码音频 latent 为波形。"""
    with torch.no_grad():
        waveform = audio_vae.decode(latent)
    return waveform.cpu()


# ── 视频合成 ─────────────────────────────────────────────────────────────────

def write_video_audio(frames: torch.Tensor, audio: torch.Tensor, output_path: str, fps: int = 24):
    """用 ffmpeg 合成带音频的视频。"""
    import subprocess
    import tempfile

    frames_np = (frames.numpy() * 255).astype(np.uint8)
    num_frames, h, w, _ = frames_np.shape

    # 写临时音频
    audio_np = audio.squeeze().numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]
    audio_np = audio_np.T  # [T, 2]
    audio_np = np.clip(audio_np, -1, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 写音频文件
        audio_path = os.path.join(tmpdir, "audio.wav")
        import soundfile as sf
        sf.write(audio_path, audio_np, 32000)

        # 通过 ffmpeg 管道写入视频
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-i", audio_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for frame in frames_np:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()

    return output_path


# ── 主生成函数 ───────────────────────────────────────────────────────────────

class H3ForgeGenerator:
    """Forge 原生 H3 生成器。"""

    def __init__(self):
        self.dit = None
        self.text_encoder = None
        self.tokenizer = None
        self.video_vae = None
        self.audio_vae = None
        self._dit_path = None
        self._te_path = None
        self._video_vae_path = None
        self._audio_vae_path = None
        # GPU 常驻策略: 'all'=全部常驻, 'dit'=仅DiT常驻, 'none'=全部卸载
        self._gpu_strategy = 'none'
        # LoRA 已应用的层（用于卸载时移除 hook）
        self._lora_hooks: list = []
        self._applied_lora_path: str | None = None

    def load_models(self, dit_path: str, te_path: str, video_vae_path: str, audio_vae_path: str):
        """加载所有模型到 CPU（带缓存），推理时按需移到 GPU。"""
        if (self.dit is not None and self._dit_path == dit_path and
            self._te_path == te_path and
            self._video_vae_path == video_vae_path and
            self._audio_vae_path == audio_vae_path):
            return

        # 释放旧模型
        self._remove_lora_hooks()
        self.dit = None
        self.text_encoder = None
        self.video_vae = None
        self.audio_vae = None
        memory_management.soft_empty_cache()

        self.dit = load_h3_dit(dit_path)
        self._dit_path = dit_path

        self.text_encoder, self.tokenizer = load_h3_text_encoder(te_path)
        self._te_path = te_path
        _log_ram("TE 加载后")

        self.video_vae = load_h3_video_vae(video_vae_path)
        self._video_vae_path = video_vae_path
        _log_ram("Video VAE 加载后")

        self.audio_vae = load_h3_audio_vae(audio_vae_path)
        self._audio_vae_path = audio_vae_path
        _log_ram("Audio VAE 加载后")

        # 根据总显存决定常驻策略
        # DiT int8 ~9-19GB, TE nvfp4 ~3.5GB, video VAE ~1.5GB, audio VAE ~1GB
        total_vram = memory_management.get_total_memory(memory_management.get_torch_device())
        if total_vram >= 32 * 1024 ** 3:
            self._gpu_strategy = 'all'
        elif total_vram >= 16 * 1024 ** 3:
            self._gpu_strategy = 'dit'  # 尝试 DiT 常驻，OOM 时自动降级
        else:
            self._gpu_strategy = 'none'

        strategy_label = {'all': '全部常驻', 'dit': '仅DiT常驻', 'none': '按需加载'}[self._gpu_strategy]
        print(f"[H3 Forge] 所有模型加载完成，显存策略: {strategy_label} ({total_vram / (1024**3):.0f} GiB)", flush=True)

        # 对于 'dit' 策略，尝试把 DiT 移到 GPU，失败则降级
        # 流式模式（stream_blocks）下不需要整体常驻
        if self._gpu_strategy == 'dit' and not getattr(self.dit, 'stream_blocks', False):
            try:
                self._to_gpu(self.dit)
                self._log_vram("DiT 常驻测试")
            except torch.cuda.OutOfMemoryError:
                self._gpu_strategy = 'none'
                self._to_cpu(self.dit)
                print("[H3 Forge] 显存不足，DiT 无法常驻，降级为按需加载", flush=True)

        if self._gpu_strategy == 'all':
            self._to_gpu(self.dit)
            self._to_gpu(self.text_encoder)
            self._to_gpu(self.video_vae)
            self._to_gpu(self.audio_vae)

        self._log_vram("加载完成")

    def _log_vram(self, tag: str) -> None:
        """打印当前显存使用情况。"""
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[H3 Forge] 显存 [{tag}]: 已分配 {allocated:.2f} GiB, 已保留 {reserved:.2f} GiB", flush=True)
        except Exception:
            pass

    def _to_gpu(self, model) -> None:
        """将模型移到 GPU。"""
        if model is not None:
            model.to(memory_management.get_torch_device())

    def _to_cpu(self, model) -> None:
        """将模型移到 CPU 并释放显存。"""
        if model is not None:
            model.to("cpu")
            memory_management.soft_empty_cache()

    # ── LoRA 支持 ────────────────────────────────────────────────────────────

    def _remove_lora_hooks(self) -> None:
        """移除所有已注册的 LoRA forward hook。"""
        for handle in self._lora_hooks:
            handle.remove()
        self._lora_hooks = []
        self._applied_lora_path = None

    @staticmethod
    def _get_module_by_path(root, path: str):
        """按点分路径获取子模块。"""
        module = root
        if not path:
            return module
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
                if module is None:
                    return None
        return module

    @staticmethod
    def _map_diffsynth_to_forge(layer_path: str) -> tuple[str | None, str | None]:
        """将 DiffSynth LoRA 层路径映射到 Forge H3 层路径。

        返回 (forge_path, qkv_slot)，qkv_slot 为 'q'/'k'/'v'/None。
        如果不是 DiffSynth 格式，返回 (None, None)。
        """
        # transformer_blocks.{i}.attn.to_{q,k,v} → blocks.{i}.attn.qkv_proj
        # transformer_blocks.{i}.attn.to_out.0 → blocks.{i}.attn.out_proj
        # transformer_blocks.{i}.ff.net.0.proj → blocks.{i}.mlp.fc1
        # transformer_blocks.{i}.ff.net.2 → blocks.{i}.mlp.fc2
        # token_refiner.refiner_blocks.{i}.* → token_refiner.blocks.{i}.* (Forge 用 blocks)
        parts = layer_path.split('.')
        if not parts:
            return None, None

        # 判断前缀
        if parts[0] == 'transformer_blocks':
            prefix = 'blocks'
            rest = parts[1:]
        elif parts[0] == 'token_refiner' and len(parts) > 1 and parts[1] == 'refiner_blocks':
            prefix = 'token_refiner.blocks'
            rest = parts[2:]
        else:
            return None, None

        if len(rest) < 2:
            return None, None

        block_idx = rest[0]
        section = rest[1]  # 'attn' 或 'ff'

        if section == 'attn':
            # rest = [block_idx, 'attn', 'to_q'] 或 [block_idx, 'attn', 'to_out', '0']
            if len(rest) < 3:
                return None, None
            sub = rest[2]  # 'to_q'/'to_k'/'to_v'/'to_out'
            if sub in ('to_q', 'to_k', 'to_v'):
                slot = sub[-1]  # 'q'/'k'/'v'
                return f"{prefix}.{block_idx}.attn.qkv_proj", slot
            elif sub == 'to_out':
                return f"{prefix}.{block_idx}.attn.out_proj", None
        elif section == 'ff':
            if len(rest) < 4:
                return None, None
            # ff.net.0.proj → mlp.fc1, ff.net.2 → mlp.fc2
            if rest[2] == 'net':
                if len(rest) >= 5 and rest[3] == '0' and rest[4] == 'proj':
                    return f"{prefix}.{block_idx}.mlp.fc1", None
                elif rest[3] == '2':
                    return f"{prefix}.{block_idx}.mlp.fc2", None

        return None, None

    def apply_lora(self, lora_path: str, strength: float = 1.0) -> bool:
        """将 model-only LoRA 应用到 DiT（通过 forward hook，兼容量化权重）。

        支持两种 LoRA 格式：
        1. ComfyUI 格式: diffusion_model.<layer>.lora_A.weight
        2. DiffSynth 格式: transformer_blocks.{i}.attn.to_q.lora_A.default.weight
           （自动映射到 Forge 的 blocks.{i}.attn.qkv_proj，qkv 自动拼接）

        内存优化：不预计算完整 delta（50 blocks 约 40GB），而是存储 A/B 矩阵，
        在 forward hook 中动态计算 delta（仅占 A/B 矩阵的 ~2GB）。

        LoRA 缩放: delta_W = (B @ A) * strength  [无 alpha 时，等价于 alpha=rank]
                   delta_W = (B @ A) * (strength * alpha / rank)  [有 alpha 时]
        """
        if self.dit is None:
            print("[H3 Forge] apply_lora 失败：DiT 未加载", flush=True)
            return False

        if self._applied_lora_path == lora_path:
            print(f"[H3 Forge] LoRA 已应用: {os.path.basename(lora_path)}", flush=True)
            return True

        self._remove_lora_hooks()

        try:
            import safetensors.torch
            lora_weights = safetensors.torch.load_file(lora_path)
        except Exception as e:
            print(f"[H3 Forge] 加载 LoRA 失败: {e}", flush=True)
            return False

        # 解析 LoRA 权重：收集 {layer_path: {suffix: {A, B, alpha}}}
        raw_lora: dict[str, dict] = {}
        for key, value in lora_weights.items():
            parts = key.split('.')
            lora_idx = None
            lora_type = None
            for i, p in enumerate(parts):
                if p in ('lora_A', 'lora_B', 'lora_up', 'lora_down'):
                    lora_idx = i
                    lora_type = p
                    break
            if lora_idx is None:
                if key.endswith('.alpha'):
                    lp = key[:-len('.alpha')]
                    raw_lora.setdefault(lp, {}).setdefault('_alpha', value.item())
                continue

            suffix = ''
            if lora_idx + 2 < len(parts):
                suffix = parts[lora_idx + 1]

            layer_path = '.'.join(parts[:lora_idx])
            entry = raw_lora.setdefault(layer_path, {}).setdefault(suffix, {})
            if lora_type in ('lora_A', 'lora_down'):
                entry['A'] = value
            else:
                entry['B'] = value

        # target: {forge_path: {qkv_slot or None: (A, B, alpha)}}
        target_deltas: dict[str, dict] = {}

        for layer_path, suffixes in raw_lora.items():
            alpha = suffixes.get('_alpha', None)
            for suffix, entry in suffixes.items():
                if suffix == '_alpha':
                    continue
                a = entry.get('A')
                b = entry.get('B')
                if a is None or b is None:
                    continue

                forge_path, qkv_slot = self._map_diffsynth_to_forge(layer_path)
                if forge_path is None:
                    if layer_path.startswith('diffusion_model.'):
                        forge_path = layer_path[len('diffusion_model.'):]
                    else:
                        forge_path = layer_path
                    qkv_slot = None

                target_deltas.setdefault(forge_path, {})[qkv_slot] = (a, b, alpha)

        # 应用到模型：存储 A/B 矩阵（不预计算完整 delta）
        applied = 0
        skipped = 0

        for forge_path, slots in target_deltas.items():
            try:
                module = self._get_module_by_path(self.dit, forge_path)
                if module is None or not hasattr(module, 'weight'):
                    skipped += 1
                    continue

                # 清理旧的 LoRA 属性
                for attr in ('_lora_a', '_lora_b', '_lora_scale', '_lora_is_qkv',
                             '_lora_is_fc1', '_lora_delta_weight'):
                    if hasattr(module, attr):
                        delattr(module, attr)

                if None in slots:
                    # 单层 LoRA（out_proj, fc1, fc2）
                    a, b, alpha = slots[None]
                    rank = a.shape[0]
                    scale = strength * (alpha / rank if alpha is not None else 1.0)
                    module._lora_a = a.to(torch.bfloat16)
                    module._lora_b = b.to(torch.bfloat16)
                    module._lora_scale = scale
                    module._lora_is_qkv = False
                    module._lora_is_fc1 = forge_path.endswith('.mlp.fc1')
                elif all(s in slots for s in ('q', 'k', 'v')):
                    # qkv：存储三个 A/B 对
                    module._lora_qkv = {}
                    for slot in ('q', 'k', 'v'):
                        a, b, alpha = slots[slot]
                        rank = a.shape[0]
                        scale = strength * (alpha / rank if alpha is not None else 1.0)
                        module._lora_qkv[slot] = (a.to(torch.bfloat16), b.to(torch.bfloat16), scale)
                    module._lora_is_qkv = True
                    module._lora_is_fc1 = False
                else:
                    skipped += 1
                    continue

                def _make_hook():
                    def hook(mod, inp, out):
                        x = inp[0]
                        dev = x.device
                        dt = x.dtype
                        if getattr(mod, '_lora_is_qkv', False):
                            # qkv 动态计算并拼接
                            outs = []
                            for slot in ('q', 'k', 'v'):
                                a, b, sc = mod._lora_qkv[slot]
                                d = (b.to(dev) @ a.to(dev)) * sc
                                outs.append(d.to(dt))
                            delta = torch.cat(outs, dim=0)
                        else:
                            a = mod._lora_a.to(dev)
                            b = mod._lora_b.to(dev)
                            delta = (b @ a) * mod._lora_scale
                            delta = delta.to(dt)
                            if getattr(mod, '_lora_is_fc1', False):
                                half = delta.shape[0] // 2
                                up, gate = delta[:half], delta[half:]
                                delta = torch.cat([gate, up], dim=0)
                        return out + F.linear(x, delta)
                    return hook

                handle = module.register_forward_hook(_make_hook())
                self._lora_hooks.append(handle)
                applied += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"[H3 Forge] 跳过 LoRA 层 {forge_path}: {e}", flush=True)

        if applied > 0:
            self._applied_lora_path = lora_path
            print(f"[H3 Forge] LoRA 应用完成: {os.path.basename(lora_path)} "
                  f"(strength={strength}, 应用 {applied} 层, 跳过 {skipped} 层)", flush=True)
        else:
            print(f"[H3 Forge] LoRA 未应用任何层（可能模型结构不匹配）: {os.path.basename(lora_path)}", flush=True)

        del lora_weights, raw_lora, target_deltas
        memory_management.soft_empty_cache()
        return applied > 0

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        seed: int,
        video_shift: float = VIDEO_SHIFT_DEFAULT,
        audio_shift: float = AUDIO_SHIFT_DEFAULT,
        first_frame: Image.Image | None = None,
        last_frame: Image.Image | None = None,
        references: list[dict] | None = None,
        progress_cb: Callable | None = None,
        cancel_check: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一次 H3 生成，返回 (video_frames, audio_waveform)。

        显存管理策略：根据 _gpu_strategy 决定模型是否常驻 GPU。
        """
        device = memory_management.get_torch_device()
        strat = self._gpu_strategy

        # 1. 文本编码（TE 移到 GPU，'all' 策略常驻）
        if progress_cb:
            progress_cb({"phase": "text_encode"})
        self._to_gpu(self.text_encoder)
        context = encode_text(self.text_encoder, self.tokenizer, prompt, device)
        if strat != 'all':
            self._to_cpu(self.text_encoder)
        self._log_vram("文本编码后")

        # 2. 计算 latent 形状
        latent_t = (num_frames + 3) // 4
        latent_h = height // 16
        latent_w = width // 16
        latent_h = math.ceil(latent_h / 2) * 2
        latent_w = math.ceil(latent_w / 2) * 2

        # 3. 初始化噪声
        if progress_cb:
            progress_cb({"phase": "init_noise"})
        video_noise = init_noise((1, VIDEO_LATENT_CHANNELS, latent_t, latent_h, latent_w), seed, device)
        audio_t = latent_t * 8
        audio_noise = init_noise((1, AUDIO_LATENT_CHANNELS, AUDIO_LATENT_CH, audio_t), seed + 1, device)

        # 4. 构建 payload（VAE 编码）
        payload: dict[str, Any] = {"seed": seed}
        cond_video_latents = []
        cond_audio_latents = []

        vae_needed = first_frame is not None or last_frame is not None or references
        if vae_needed:
            self._to_gpu(self.video_vae)
            self._to_gpu(self.audio_vae)

            if first_frame is not None:
                if progress_cb:
                    progress_cb({"phase": "encode_first_frame"})
                first_latent = encode_frame_to_latent(self.video_vae, first_frame, width, height, device)
                cond_video_latents.append(first_latent)
                payload.setdefault("keyframes", []).append({"index": 0})

            if last_frame is not None:
                if progress_cb:
                    progress_cb({"phase": "encode_last_frame"})
                last_latent = encode_frame_to_latent(self.video_vae, last_frame, width, height, device)
                cond_video_latents.append(last_latent)
                payload.setdefault("keyframes", []).append({"index": -1})

            if references:
                for ref in references:
                    kind = ref.get("type")
                    if kind == "image" and "image" in ref:
                        ref_latent = encode_frame_to_latent(self.video_vae, ref["image"], width, height, device)
                        cond_video_latents.append(ref_latent)
                    elif kind == "audio" and "audio" in ref:
                        ref_latent = encode_audio_to_latent(self.audio_vae, ref["audio"], device)
                        cond_audio_latents.append(ref_latent)

            if strat != 'all':
                self._to_cpu(self.video_vae)
                self._to_cpu(self.audio_vae)
            self._log_vram("VAE 编码后")

        if cond_video_latents:
            payload["cond_video_latents"] = cond_video_latents
        if cond_audio_latents:
            payload["cond_audio_latents"] = cond_audio_latents

        # 5. 采样（DiT 移到 GPU，'all'/'dit' 策略常驻）
        if progress_cb:
            progress_cb({"phase": "sampling"})
        if getattr(self.dit, 'stream_blocks', False):
            # 流式模式：只加载顶层模块到 GPU，blocks 在 forward 中按需换入
            self.dit._ensure_top_on_device(device)
        else:
            self._to_gpu(self.dit)
        self._log_vram("DiT 加载后")
        video_latents, audio_latents = sample_h3(
            dit=self.dit,
            video_latents=video_noise,
            audio_latents=audio_noise,
            context=context,
            payload=payload,
            num_steps=num_steps,
            video_shift=video_shift,
            audio_shift=audio_shift,
            seed=seed,
            device=device,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
        )

        if strat == 'none' and not getattr(self.dit, 'stream_blocks', False):
            self._to_cpu(self.dit)
        self._log_vram("采样完成")

        # 6. 解码（VAE 移到 GPU）
        if strat != 'all':
            self._to_gpu(self.video_vae)
        if progress_cb:
            progress_cb({"phase": "decode_video"})
        video_frames = decode_video(self.video_vae, video_latents)
        if strat != 'all':
            self._to_cpu(self.video_vae)

        if strat != 'all':
            self._to_gpu(self.audio_vae)
        if progress_cb:
            progress_cb({"phase": "decode_audio"})
        audio_waveform = decode_audio(self.audio_vae, audio_latents)
        if strat != 'all':
            self._to_cpu(self.audio_vae)

        self._log_vram("生成完成")
        return video_frames, audio_waveform

    def unload_all(self) -> None:
        """将所有模型移回 CPU 并移除 LoRA hook（释放显存）。"""
        self._to_cpu(self.dit)
        self._to_cpu(self.text_encoder)
        self._to_cpu(self.video_vae)
        self._to_cpu(self.audio_vae)
        print("[H3 Forge] 所有模型已卸载到 CPU", flush=True)


# 全局单例
_generator: H3ForgeGenerator | None = None


def get_generator() -> H3ForgeGenerator:
    global _generator
    if _generator is None:
        _generator = H3ForgeGenerator()
    return _generator
