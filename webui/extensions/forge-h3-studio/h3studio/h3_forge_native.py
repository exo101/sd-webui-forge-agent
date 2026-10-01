"""
Forge 原生 H3 生成引擎
======================

直接使用 Forge 后端的 MiniMax H3 实现（移植自 ComfyUI），绕过 DiffSynth pipeline。
优势：
- Forge 的 int8_convrot 有 Triton 优化内核（Blackwell/GPU 更快）
- 模型常驻 GPU，不卸载到磁盘
- 自动量化检测（int8/nf4）
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

def load_h3_dit(model_path: str, log_name: str = "MiniMaxH3Model"):
    """用 Forge 原生方式加载 H3 DiT（自动检测 int8/nf4 量化）。"""
    from backend.nn.minimax_h3 import build_minimax_h3_model
    from safetensors.torch import load_file

    print(f"[H3 Forge] 加载 DiT: {Path(model_path).name}", flush=True)
    state_dict = load_file(model_path, device="cpu")
    model = build_minimax_h3_model(state_dict, log_name=log_name)
    del state_dict
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

    支持 nf4 预量化：通过 Forge 的 using_forge_operations(bnb_dtype="nf4")
    在模型创建时替换 Linear 层为 bitsandbytes NF4 版本。
    """
    from transformers import AutoTokenizer
    from diffsynth.models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
    from safetensors.torch import load_file

    print(f"[H3 Forge] 加载文本编码器: {Path(model_path).name}", flush=True)

    # 先检测量化类型
    state_dict = load_file(model_path, device="cpu")
    state_dict_dtype = weight_dtype(state_dict)

    if state_dict_dtype in ("nf4", "fp4", "gguf"):
        print(f"[H3 Forge] 文本编码器检测到预量化: {state_dict_dtype}", flush=True)
        # 在 bnb_dtype 上下文中创建模型，使 nn.Linear 替换为 NF4 版本
        with using_forge_operations(device=memory_management.cpu, dtype=torch.bfloat16, manual_cast_enabled=False, bnb_dtype=state_dict_dtype):
            model = MiniMaxH3TextEncoder(num_retained_layers=50)
        load_state_dict(model, state_dict, log_name="MiniMaxH3TextEncoder")
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

    if state_dict_dtype not in ("nf4", "fp4", "gguf"):
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

    def load_models(self, dit_path: str, te_path: str, video_vae_path: str, audio_vae_path: str):
        """加载所有模型到 CPU（带缓存），推理时按需移到 GPU。"""
        if (self.dit is not None and self._dit_path == dit_path and
            self._te_path == te_path and
            self._video_vae_path == video_vae_path and
            self._audio_vae_path == audio_vae_path):
            return

        # 释放旧模型
        self.dit = None
        self.text_encoder = None
        self.video_vae = None
        self.audio_vae = None
        memory_management.soft_empty_cache()

        self.dit = load_h3_dit(dit_path)
        self._dit_path = dit_path

        self.text_encoder, self.tokenizer = load_h3_text_encoder(te_path)
        self._te_path = te_path

        self.video_vae = load_h3_video_vae(video_vae_path)
        self._video_vae_path = video_vae_path

        self.audio_vae = load_h3_audio_vae(audio_vae_path)
        self._audio_vae_path = audio_vae_path

        # 模型常驻 CPU，推理时按需移到 GPU
        print("[H3 Forge] 所有模型加载完成（CPU 常驻，推理时按需加载到 GPU）", flush=True)
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

        显存管理策略：模型常驻 CPU，按需移到 GPU，用完立即卸载。
        """
        device = memory_management.get_torch_device()

        # 1. 文本编码（TE 移到 GPU，用完卸载）
        if progress_cb:
            progress_cb({"phase": "text_encode"})
        self._to_gpu(self.text_encoder)
        context = encode_text(self.text_encoder, self.tokenizer, prompt, device)
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

        # 4. 构建 payload（VAE 编码，用完卸载）
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

            self._to_cpu(self.video_vae)
            self._to_cpu(self.audio_vae)
            self._log_vram("VAE 编码后")

        if cond_video_latents:
            payload["cond_video_latents"] = cond_video_latents
        if cond_audio_latents:
            payload["cond_audio_latents"] = cond_audio_latents

        # 5. 采样（DiT 移到 GPU）
        if progress_cb:
            progress_cb({"phase": "sampling"})
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

        # DiT 用完卸载，为 VAE 解码腾出显存
        self._to_cpu(self.dit)
        self._log_vram("采样完成")

        # 6. 解码（VAE 移到 GPU）
        self._to_gpu(self.video_vae)
        if progress_cb:
            progress_cb({"phase": "decode_video"})
        video_frames = decode_video(self.video_vae, video_latents)
        self._to_cpu(self.video_vae)

        self._to_gpu(self.audio_vae)
        if progress_cb:
            progress_cb({"phase": "decode_audio"})
        audio_waveform = decode_audio(self.audio_vae, audio_latents)
        self._to_cpu(self.audio_vae)

        self._log_vram("生成完成")
        return video_frames, audio_waveform


# 全局单例
_generator: H3ForgeGenerator | None = None


def get_generator() -> H3ForgeGenerator:
    global _generator
    if _generator is None:
        _generator = H3ForgeGenerator()
    return _generator
