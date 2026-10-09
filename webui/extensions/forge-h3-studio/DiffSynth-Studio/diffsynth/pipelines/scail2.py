import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List, Union
from einops import rearrange
import random
import sys
import json
import time
import logging

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import BasePipeline
from ..diffusion import FlowMatchScheduler
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.scail2_dit import SCAIL2Model
from ..models.sam3_tracker import get_tracker
from ..utils.scail2_utils import (
    extract_and_compress_mask_to_latent,
    load_image_to_tensor_chw_normalized,
)

logger = logging.getLogger("system")


class SCAIL2Pipeline(BasePipeline):
    """
    SCAIL-2 pipeline for character animation and replacement.
    
    Two modes:
    - Animation mode (replace_flag=False): character performs driving motion
    - Replacement mode (replace_flag=True): swap tracked person with reference character
    """
    
    def __init__(
        self,
        device=get_device_type(),
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: SCAIL2Model = None
        self.vae: WanVideoVAE = None
        self.image_encoder: WanImageEncoder = None  # CLIP vision
        self.sam3_tracker = None
        self.vae_stride = (4, 8, 8)  # temporal, height, width compression
        self.num_train_timesteps = 1000
        self.in_iteration_models = ("dit",)
        # SCAIL-2 LoRA weights
        self.lightx2v_lora = None
        self.dpo_lora = None

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list = [],
        tokenizer_config: ModelConfig = None,
        sam3_model_path: Optional[str] = None,
        sam3_comfy_root: Optional[str] = None,
        sam3_comfy_python: Optional[str] = None,
        sam3_worker_path: Optional[str] = None,
        lightx2v_lora_path: Optional[str] = None,
        dpo_lora_path: Optional[str] = None,
        vram_limit: Optional[float] = None,
    ):
        # Older SCAIL2 UI callers supplied bare ModelConfig(path=...) objects.
        # For ComfyUI pre-quantized exports this bypasses the quantized loader
        # and leaves the pipeline unable to manage the 14B DiT safely.
        model_configs = list(model_configs)
        dit_config = next((c for c in model_configs if cls._looks_like_scail2_dit(c.path)), None)
        if dit_config is not None and dit_config.quantize is None:
            quantize = cls._quant_config_from_checkpoint(dit_config.path)
            if quantize is not None:
                dit_config.quantize = quantize
                print(f"[SCAIL2] detected pre-quantized DiT, enabling managed load: {quantize.method}", flush=True)
        pipe = cls(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("scail2_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # Offload non-DiT models to CPU to save VRAM. Keep managed/quantized
        # DiTs on their configured offload device; never move a 14B packed
        # checkpoint wholesale to CUDA here.
        if pipe.text_encoder is not None:
            pipe.text_encoder.to("cpu")
        if pipe.vae is not None:
            pipe.vae.to("cpu")
        if pipe.image_encoder is not None:
            pipe.image_encoder.to("cpu")
        torch.cuda.empty_cache()

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=512, clean='whitespace'
            )

        # Load SAM3 tracker
        if sam3_model_path is not None:
            try:
                pipe.sam3_tracker = get_tracker(
                    model_path=sam3_model_path,
                    device="cpu",  # offload until needed
                    dtype=torch_dtype,
                    use_sam3=True,
                    comfy_root=sam3_comfy_root,
                    comfy_python=sam3_comfy_python,
                    worker_path=sam3_worker_path,
                )
                if pipe.sam3_tracker.__class__.__name__ == "SimpleMaskGenerator":
                    raise RuntimeError("SAM3 checkpoint fell back to background subtraction")
            except Exception as e:
                print(
                    f"[SCAIL2] SAM3 unavailable: {e}. The UI may still run with SAM3 disabled, "
                    "but SAM3-enabled character replacement will stop with an explicit error.",
                    flush=True,
                )
                pipe.sam3_tracker = None

        pipe.vram_management_enabled = pipe.check_vram_management_state()

        # Load LoRAs
        pipe.lightx2v_lora_path = lightx2v_lora_path
        pipe.dpo_lora_path = dpo_lora_path

        # Quantized checkpoints cannot be merged in-place. Use runtime LoRA
        # adapters like ComfyUI; non-quantized models retain a safe fp merge.
        pipe._apply_loras()

        return pipe

    @staticmethod
    def _looks_like_scail2_dit(path):
        if not path:
            return False
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="pt") as handle:
                keys = set(handle.keys())
            return "patch_embedding_pose.weight" in keys and "patch_embedding_mask.weight" in keys
        except Exception:
            return False

    @staticmethod
    def _quant_config_from_checkpoint(path):
        """Build DiffSynth's mixed quant config from Comfy's checkpoint metadata."""
        try:
            from safetensors import safe_open
            from ..core import QuantizeConfig, MixedQuantizeConfig
            format_methods = {
                "nvfp4": "comfy_kitchen_nvfp4_w4a8",
                "mxfp8": "comfy_kitchen_mxfp8_w8a8",
                "float8_e4m3fn": "comfy_kitchen_fp8_w8a8",
                "int8_tensorwise": "comfy_kitchen_int8_w8a8",
            }
            with safe_open(str(path), framework="pt") as handle:
                keys = set(handle.keys())
                metadata = handle.metadata() or {}
                markers = [key for key in keys if key.endswith(".comfy_quant")]
                grouped = {}
                if markers:
                    for key in markers:
                        try:
                            marker = json.loads(bytes(handle.get_tensor(key).to(torch.uint8).tolist()).decode("utf-8"))
                        except Exception:
                            continue
                        fmt = marker.get("format")
                        if fmt in format_methods:
                            grouped.setdefault(fmt, []).append(key[:-len(".comfy_quant")])
                if not grouped:
                    quant_meta = json.loads(metadata.get("_quantization_metadata", "{}"))
                    for layer, item in quant_meta.get("layers", {}).items():
                        fmt = item.get("format") if isinstance(item, dict) else None
                        if fmt in format_methods:
                            grouped.setdefault(fmt, []).append(layer)
            if not grouped:
                return None
            configs = [
                QuantizeConfig(method=format_methods[fmt], target_modules=sorted(layers))
                for fmt, layers in grouped.items()
            ]
            if len(configs) == 1:
                configs[0].load_prequantized = True
                return configs[0]
            result = MixedQuantizeConfig(configs=configs, load_prequantized=True)
            return result
        except Exception as error:
            print(f"[SCAIL2] quantized checkpoint detection failed: {error}", flush=True)
            return None

    def _apply_loras(self):
        """Apply Lightx2v distillation LoRA and DPO LoRA to the DiT."""
        from safetensors.torch import load_file
        
        quantized = getattr(self.dit, "quantize_config", None) is not None
        if self.lightx2v_lora_path is not None:
            try:
                lora_sd = load_file(self.lightx2v_lora_path)
                count = self._load_lora(self.dit, lora_sd, alpha=0.8, hotload=quantized)
                print(f"[SCAIL2] Lightx2v LoRA matched {count} layers from {self.lightx2v_lora_path}", flush=True)
            except Exception as e:
                raise RuntimeError(f"Official SCAIL-2 workflow requires LightX2v LoRA at strength 0.8: {e}") from e
        
        if self.dpo_lora_path is not None:
            try:
                lora_sd = load_file(self.dpo_lora_path)
                count = self._load_lora(self.dit, lora_sd, alpha=1.0, hotload=quantized)
                print(f"[SCAIL2] DPO LoRA matched {count} layers from {self.dpo_lora_path}", flush=True)
            except Exception as e:
                raise RuntimeError(f"Official SCAIL-2 workflow requires DPO LoRA at strength 1.0: {e}") from e

    def _load_lora(self, model, state_dict, alpha=1.0, hotload=False):
        """Load Wan/Comfy down-up LoRA keys; return actual matched layer count."""
        if hotload:
            from ..diffusion.base_pipeline import LoRAHotLoadMixin
            from ..core.vram.layers import AutoWrappedQuantizedModule
            from ..utils.lora import GeneralLoRALoader
            lora = GeneralLoRALoader(torch_dtype=self.torch_dtype, device="cpu").convert_state_dict(state_dict)
            updated = 0
            for name, module in model.named_modules():
                if not isinstance(module, (LoRAHotLoadMixin, AutoWrappedQuantizedModule)):
                    continue
                # Wrapped modules are registered under paths containing one
                # or more `.module` segments; `.name` preserves the original
                # checkpoint path (the same convention used by Comfy LoRA).
                target_name = getattr(module, "name", "") or name.replace(".module", "")
                a = lora.get(f"{target_name}.lora_A.weight")
                if a is None:
                    a = lora.get(f"{target_name}.lora_down.weight")
                b = lora.get(f"{target_name}.lora_B.weight")
                if b is None:
                    b = lora.get(f"{target_name}.lora_up.weight")
                if a is None or b is None:
                    continue
                module.lora_A_weights.append(a * alpha)
                module.lora_B_weights.append(b)
                updated += 1
            if updated == 0:
                raise ValueError("LoRA 文件读取成功，但没有层名与 SCAIL2 DiT 匹配")
            return updated
        return self._merge_lora(model, state_dict, alpha=alpha)

    @staticmethod
    def _merge_lora(model, lora_state_dict, alpha=1.0):
        """Simple LoRA merge: W = W + alpha * (B @ A)."""
        lora_a = {}
        lora_b = {}
        for key, value in lora_state_dict.items():
            if "lora_A" in key or "lora_down" in key or key.endswith(".down.weight"):
                base_key = key.replace(".lora_A.weight", "").replace(".lora_down.weight", "").replace(".down.weight", "")
                lora_a[base_key] = value
            elif "lora_B" in key or "lora_up" in key or key.endswith(".up.weight"):
                base_key = key.replace(".lora_B.weight", "").replace(".lora_up.weight", "").replace(".up.weight", "")
                lora_b[base_key] = value
        
        updated = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                base_name = name.removeprefix("diffusion_model.")
                if (name in lora_a and name in lora_b) or (base_name in lora_a and base_name in lora_b):
                    key = name if name in lora_a else base_name
                    if param.device.type == "meta":
                        continue
                    a = lora_a[key].to(param.device, param.dtype)
                    b = lora_b[key].to(param.device, param.dtype)
                    # LoRA: delta = B @ A, scaled by alpha
                    if a.dim() == 2 and b.dim() == 2:
                        delta = (b @ a) * alpha
                    else:
                        delta = (b.flatten(1) @ a.flatten(1)).reshape(param.shape) * alpha
                    param.add_(delta)
                    updated += 1
        if updated == 0:
            raise ValueError("LoRA 文件读取成功，但没有参数与 SCAIL2 DiT 匹配")
        return updated

    def _build_segments(self, total_frames, segment_len=81, segment_overlap=5):
        """Build frame segments for long video generation.
        
        Returns:
            (segments, effective_overlap): list of (start, end) tuples and the
            adjusted overlap actually used.
        """
        if total_frames <= segment_len:
            keep = ((total_frames - 1) // self.vae_stride[0]) * self.vae_stride[0] + 1
            return [(0, keep)], 0
        # Ensure stride > 0 and reasonable: adapt overlap to segment_len.
        # The default 5-frame overlap at segment_len=81 gives the official
        # 76-frame continuation stride. For small segments,
        # reduce overlap so we don't degenerate to stride=1.
        if segment_len <= 9:
            segment_overlap = 1  # stride = segment_len - 1
        elif segment_len <= 17:
            segment_overlap = min(segment_overlap, 3)
        else:
            segment_overlap = min(segment_overlap, segment_len - 1)
        segments = []
        start = 0
        stride = segment_len - segment_overlap
        while start < total_frames:
            end = start + segment_len
            if end >= total_frames:
                # Keep the requested window as a hard maximum. Extending the
                # previous segment to cover a short tail can silently turn an
                # 81-frame Comfy-style window into a full-length 121-frame
                # forward, defeating both the VRAM guard and chunk semantics.
                remaining = total_frames - start
                # Every start is stride-aligned and total_frames is VAE-aligned,
                # so the tail is also 1 mod temporal_stride (minimum 5 frames).
                tail_len = ((remaining - 1) // self.vae_stride[0]) * self.vae_stride[0] + 1
                tail_len = max(tail_len, self.vae_stride[0] + 1)
                segments.append((start, min(start + tail_len, total_frames)))
                break
            segments.append((start, end))
            start += stride
        return segments, segment_overlap

    def _calculate_safe_segment_len(self, h, w, requested_len=81):
        """
        根据分辨率和可用显存计算安全的 segment_len。

        DiT 使用 Flash Attention（O(n) 显存），14B fp8 权重占 ~14GB，
        剩余显存需要容纳 QKV、FFN 等线性层激活值。
        使用顺序 CFG（batch=1）进一步减半显存。

        Returns:
            安全的 segment_len（视频帧数，满足 T ≡ 1 mod 4）
        """
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            avail_gb = max(0.5, free_bytes / (1024 ** 3) - 1.0)
        except Exception:
            return requested_len

        # 分块 FFN（chunk=4096）+ 分块注意力（Q/K/V 都分块，cs=2048）后：
        #   FFN 峰值 = chunk × ffn_dim(13824) × 2 ≈ 108 MB（常量）
        #   注意力 scores = num_heads(40) × cs² × 2 ≈ 335 MB（常量）
        #   Q/K/V 分块 = 3 × cs × dim(5120) × 2 ≈ 63 MB（常量）
        # 常量开销合计 ≈ 506 MB
        constant_overhead_mb = 108 + 335 + 63
        # per-token 显存主要是输入 x 和输出 out：2 × seq_len × dim(5120) × 2 = 20480 × seq_len
        avail_for_tokens_gb = max(0.1, avail_gb - constant_overhead_mb / 1024)
        bytes_per_token = 20480
        max_seq_len = int(avail_for_tokens_gb * (1024 ** 3) / bytes_per_token)
        # 安全系数 0.7（留余量给临时张量、40 层叠加、VAE 缓存等）
        max_seq_len = int(max_seq_len * 0.7)

        # Both streams are Conv3d-patched with spatial patch size 2. The
        # previous estimate counted latent pixels instead of actual tokens,
        # overestimating sequence length by 4x and shrinking 65-frame chunks
        # to ~13 frames on 16GB GPUs.
        tokens_per_rv_frame = (h // 16) * (w // 16)
        tokens_per_pose_frame = (h // 32) * (w // 32)

        # 从最大帧往下找满足 seq_len <= max_seq_len 的最大 segment_len
        # segment_len 必须满足 T ≡ 1 (mod 4)，最小为 5（1 个 latent 帧）
        safe_len = requested_len
        for seg_len in range(requested_len, 4, -4):
            f_lat = (seg_len - 1) // self.vae_stride[0] + 1
            seq_len = (f_lat + 1) * tokens_per_rv_frame + f_lat * tokens_per_pose_frame
            if seq_len <= max_seq_len:
                safe_len = seg_len
                break
        else:
            # 即使最小帧数也超了，用 5（1 latent 帧）
            safe_len = 5

        if safe_len < requested_len:
            print(f"[SCAIL2] 显存不足，自动降低 segment_len: {requested_len} -> {safe_len} "
                  f"(max_seq_len={max_seq_len}, avail={avail_gb:.1f}GB)")
        elif safe_len == 5 and requested_len > 5:
            print(f"[SCAIL2] 警告: 分辨率 {h}x{w} 过高，即使 5 帧也可能 OOM，建议降低分辨率")

        return safe_len

    def _vae_encode_image(self, img_tensor):
        """Encode a single image tensor (1, 3, H, W) to VAE latent (16, 1, H/8, W/8)."""
        # img_tensor: (1, 3, H, W) -> (1, 3, 1, H, W)
        video = img_tensor.permute(0, 2, 1, 3, 4) if img_tensor.dim() == 5 else img_tensor.unsqueeze(2)
        # video: (1, 3, 1, H, W)
        video = video.to(dtype=next(self.vae.parameters()).dtype)
        latents = self.vae.encode([video[0]], self.device)  # returns (1, 16, 1, H/8, W/8)
        return latents[0]  # (16, 1, H/8, W/8)

    def _vae_encode_video(self, video_tensor):
        """Encode video tensor (T, 3, H, W) to VAE latent (16, T_lat, H/8, W/8)."""
        # video_tensor: (T, 3, H, W) -> (3, T, H, W) -> (1, 3, T, H, W)
        video = rearrange(video_tensor, 't c h w -> 1 c t h w')
        video = video.to(dtype=next(self.vae.parameters()).dtype)
        latents = self.vae.encode([video[0]], self.device)
        return latents[0]  # (16, T_lat, H/8, W/8)

    @staticmethod
    def _reinhard_lab_match_frame(frame_rgb, reference_rgb, device, strength=1.0):
        """Match one RGB frame to a reference using a strength-controlled Reinhard-Lab rule."""
        import kornia.color

        frame = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(
            device=device, dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)
        reference = torch.from_numpy(np.ascontiguousarray(reference_rgb)).to(
            device=device, dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)
        source_lab = kornia.color.rgb_to_lab(frame)
        reference_lab = kornia.color.rgb_to_lab(reference)
        source_flat = source_lab.flatten(2)
        reference_flat = reference_lab.flatten(2)
        source_mean = source_flat.mean(dim=-1, keepdim=True)
        source_std = source_flat.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(1e-6)
        reference_mean = reference_flat.mean(dim=-1, keepdim=True)
        reference_std = reference_flat.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(1e-6)
        matched_lab = (source_flat - source_mean) * (reference_std / source_std) + reference_mean
        matched = kornia.color.lab_to_rgb(matched_lab.reshape_as(source_lab))
        strength = min(max(float(strength), 0.0), 1.0)
        if strength < 1.0:
            matched = torch.lerp(frame, matched, strength)
        return matched.squeeze(0).permute(1, 2, 0).clamp_(0, 1).cpu().numpy()

    def _swap_model_to_gpu(self, model_name: str):
        """Move one model to GPU, offload ALL others to CPU."""
        # First move everything to CPU
        for name in ("dit", "text_encoder", "vae", "image_encoder"):
            if name == model_name:
                continue
            model = getattr(self, name, None)
            if model is not None:
                model.to("cpu")
        torch.cuda.empty_cache()
        # Then move requested model to GPU
        model = getattr(self, model_name, None)
        if model is not None:
            if model_name == "dit" and getattr(model, "vram_management_enabled", False):
                if getattr(self, "dit_gpu_resident", False):
                    # Keep the packed quantized weights resident for all denoising
                    # steps. CPU staging on every layer call causes huge latency.
                    print("[SCAIL2] managed quantized DiT stays GPU-resident for sampling", flush=True)
                else:
                    print("[SCAIL2] DiT remains CPU-offloaded; quantized layers stage on demand", flush=True)
            else:
                model.to(self.device)
        torch.cuda.empty_cache()

    def _restore_dit_to_gpu(self):
        """Move DiT to GPU, offload all others to CPU."""
        # Move everything to CPU first
        for name in ("text_encoder", "vae", "image_encoder"):
            model = getattr(self, name, None)
            if model is not None:
                model.to("cpu")
        torch.cuda.empty_cache()
        # Then move DiT to GPU
        if self.dit is not None and (
            not getattr(self.dit, "vram_management_enabled", False)
            or getattr(self, "dit_gpu_resident", False)
        ):
            if getattr(self, "dit_gpu_resident", False):
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                free_gb = free_bytes / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
                # Check the live post-transfer headroom: packed checkpoints
                # can expand in memory, so file size is not a useful proxy.
                minimum_activation_gb = 2.5
                if free_gb < minimum_activation_gb:
                    raise RuntimeError(
                        "SCAIL-2 NVFP4 DiT is on GPU, but too little VRAM remains for sampling: "
                        f"free {free_gb:.1f}/{total_gb:.1f} GiB after loading, "
                        f"minimum required is {minimum_activation_gb:.1f} GiB. "
                        "Close GPU-heavy apps/models or lower output resolution. "
                        "Reducing segment length alone cannot fix insufficient activation memory."
                    )
            try:
                residency = next(self.dit.parameters()).device
            except (StopIteration, AttributeError):
                residency = "unknown"
            if str(residency) != str(self.device):
                print("[SCAIL2] transferring quantized DiT to GPU once before sampling", flush=True)
                transfer_started = time.perf_counter()
                self.dit.to(self.device)
                torch.cuda.synchronize(self.device)
                # AutoWrappedQuantizedModule tracks state separately from the
                # underlying tensor device. Mark layers resident so forward()
                # will not repeatedly deepcopy weights on every attention chunk.
                resident_layers = 0
                for module in self.dit.modules():
                    if hasattr(module, "computation_module") and hasattr(module, "state"):
                        module.onload_device = self.device
                        module.preparing_device = self.device
                        module.state = 2
                        resident_layers += 1
                print(
                    f"[SCAIL2] quantized wrappers set to GPU-resident: {resident_layers}",
                    flush=True,
                )
                print(
                    f"[SCAIL2] quantized DiT GPU transfer complete in {time.perf_counter() - transfer_started:.1f}s",
                    flush=True,
                )
            else:
                print(f"[SCAIL2] DiT already resident on {residency}", flush=True)
        elif self.dit is not None:
            try:
                residency = next(self.dit.parameters()).device
            except (StopIteration, AttributeError):
                residency = "unknown"
            cached_layer_count = 0
            for module in self.dit.modules():
                if hasattr(module, "computation_module") and hasattr(module, "state"):
                    # INT8 ConvRot is larger than available VRAM. Leave weights
                    # in host RAM and let each wrapper cache its layer on CUDA
                    # only while the Comfy-style memory budget allows it.
                    module.onload_device = "cpu"
                    module.preparing_device = self.device
                    if str(module._module_device()) == "cpu":
                        module.state = 0
                    cached_layer_count += 1
            if cached_layer_count:
                print(
                    f"[SCAIL2] managed INT8 DiT on {residency}; "
                    f"per-layer CUDA cache enabled for {cached_layer_count} quantized layers",
                    flush=True,
                )
            else:
                print(f"[SCAIL2] managed DiT parameter residency={residency}; entering diffusion", flush=True)
        torch.cuda.empty_cache()

    def _get_clip_features(self, img_tensor):
        """Get CLIP vision features from reference image."""
        # img_tensor: (1, 3, H, W), range [-1, 1]
        # Cast to image encoder's dtype to avoid mismatch
        enc_dtype = next(self.image_encoder.parameters()).dtype
        img_tensor = img_tensor.to(dtype=enc_dtype)
        features = self.image_encoder.encode_image([img_tensor])
        # features: (1, N, 1280) or similar
        if isinstance(features, list):
            features = features[0]
        if features.dim() == 3:
            features = features.squeeze(0)  # (N, 1280)
        return features.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # (1, N, 1280)

    def _encode_text(self, prompt):
        """Encode text prompt to embeddings."""
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        reference_image: Image.Image = None,
        pose_video: List[Image.Image] = None,
        pose_video_frames: Optional[torch.Tensor] = None,
        replace_flag: bool = False,
        reference_mask: Optional[Image.Image] = None,
        replacement_mode: Optional[bool] = None,
        pose_mask_frames: Optional[List[Image.Image]] = None,
        segment_len: int = 81,
        segment_overlap: int = 5,
        shift: float = 5.0,
        sampling_steps: int = 8,
        guide_scale: float = 5.0,
        seed: int = -1,
        pose_strength: float = 1.0,
        pose_start: float = 0.0,
        pose_end: float = 1.0,
        use_sam3: bool = False,
        sam3_points: Optional[List[List[float]]] = None,
        progress_callback=None,
        **kwargs,
    ):
        """
        Generate video using SCAIL-2.
        
        Args:
            reference_image: Reference character image
            pose_video: Driving video frames (list of PIL Images)
            pose_video_frames: Or driving video as tensor (T, 3, H, W), range [-1, 1]
            replace_flag: False=animation, True=replacement
            reference_mask: Optional pre-made reference mask
            pose_mask_frames: Optional pre-made pose masks
            segment_len: Frames per segment (default 81)
            segment_overlap: Overlap frames between segments (default 5)
            shift: Flow matching shift
            sampling_steps: Number of denoising steps
            guide_scale: CFG scale
            seed: Random seed
            pose_strength: Strength of pose conditioning
            pose_start: Start ratio for pose conditioning (0-1)
            pose_end: End ratio for pose conditioning (0-1)
            use_sam3: Whether to use SAM3 for mask generation
            sam3_points: Initial points for SAM3 tracking [[x, y], ...]
        """
        if reference_image is None:
            raise ValueError("reference_image is required")
        if pose_video is None and pose_video_frames is None:
            raise ValueError("pose_video or pose_video_frames is required")

        def report_stage(value, description):
            logger.info("[SCAIL2][阶段] %s", description)
            if progress_callback is not None:
                try:
                    progress_callback(value, desc=description)
                except TypeError:
                    progress_callback(value)

        # --- Prepare inputs ---
        device = self.device
        dtype = self.torch_dtype

        # Reference image -> tensor
        if isinstance(reference_image, Image.Image):
            ref_img = load_image_to_tensor_chw_normalized(reference_image).to(device)  # (1, 3, H, W)
        else:
            ref_img = reference_image.to(device)
        
        h, w = ref_img.shape[-2], ref_img.shape[-1]
        if use_sam3 and self.sam3_tracker is None:
            raise RuntimeError("SAM3 tracking was requested but no compatible tracker is loaded")

        # Pose video -> tensor
        if pose_video_frames is not None:
            pose_frames = pose_video_frames.to(device)
        elif pose_video is not None:
            pose_tensors = [load_image_to_tensor_chw_normalized(f) for f in pose_video]
            pose_frames = torch.cat(pose_tensors, dim=0).to(device)  # (T, 3, H, W)
        num_frames = pose_frames.shape[0]

        # --- Generate masks ---
        # ComfyUI leaves these optional streams absent unless an actual colored
        # identity mask is provided. Synthetic all-black/all-white masks still
        # pass through a biased Conv3d and change the model output.
        ref_mask_cthw = None
        pose_mask_cthw = None
        if replacement_mode is None:
            replacement_mode = bool(replace_flag)
        if reference_mask is not None:
            ref_mask = load_image_to_tensor_chw_normalized(reference_mask).to(device)
            ref_mask_cthw = ref_mask.permute(1, 0, 2, 3)

        if pose_mask_frames is not None:
            pose_mask_tensors = [load_image_to_tensor_chw_normalized(f) for f in pose_mask_frames]
            pose_mask = torch.cat(pose_mask_tensors, dim=0).to(device)
            pose_mask_cthw = pose_mask.permute(1, 0, 2, 3)
        elif use_sam3 and self.sam3_tracker is not None:
            pose_np = [(pose_frames[i].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1, 2, 0)
                       for i in range(num_frames)]
            self.sam3_tracker.to(device)
            try:
                if hasattr(self.sam3_tracker, "track_pair"):
                    ref_np = np.clip(
                        (ref_img[0].detach().float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5,
                        0,
                        255,
                    ).astype(np.uint8)
                    pose_mask, ref_mask_chw = self.sam3_tracker.track_pair(
                        pose_np,
                        ref_np,
                        replacement=replacement_mode,
                        pose_prompt="human",
                        reference_prompt="girl, handkerchief",
                    )
                    pose_mask_cthw = torch.from_numpy(pose_mask).to(device)
                    if reference_mask is None:
                        reference_mask = Image.fromarray(
                            np.clip((ref_mask_chw[:, 0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
                        )
                        ref_mask = load_image_to_tensor_chw_normalized(reference_mask).to(device)
                        ref_mask_cthw = ref_mask.permute(1, 0, 2, 3)
                else:
                    background = "white" if replacement_mode else "black"
                    pose_mask_cthw = torch.from_numpy(
                        self.sam3_tracker.track_video(
                            pose_np, init_points=sam3_points, background=background
                        )
                    ).to(device)
            finally:
                self.sam3_tracker.cpu()

        if replacement_mode and ref_mask_cthw is not None:
            # Comfy WanSCAILToVideo composites the reference subject over black
            # using every supplied reference mask (not only an auto-generated one).
            ref_mask_rgb = ref_mask_cthw[:, 0].unsqueeze(0)
            is_character = (ref_mask_rgb.amax(dim=1, keepdim=True) > 0.1).to(ref_img.dtype)
            ref_img = ref_img * is_character

        report_stage(0.12, "SCAIL2: VAE 编码参考图和姿势视频")
        # --- Encode reference ---
        self._swap_model_to_gpu("vae")
        ref_latent = self._vae_encode_image(ref_img)  # (16, 1, H/8, W/8)
        ref_mask_28ch = None
        if ref_mask_cthw is not None:
            ref_mask_28ch = extract_and_compress_mask_to_latent(
                ref_mask_cthw.cpu(), additional_spatial_downsample=1
            ).to(device, dtype)  # (28, 1, H_lat, W_lat)

        # Match ComfyUI's WanSCAILToVideo: crop each video segment first, then
        # VAE-encode that crop independently. Wan's temporal VAE is causal, so
        # encoding the full clip once and slicing its latents does not reproduce
        # the per-segment temporal context at chunk boundaries.
        segment_len = self._calculate_safe_segment_len(h, w, segment_len)
        segments, segment_overlap = self._build_segments(num_frames, segment_len, segment_overlap)
        if not segments:
            raise ValueError(f"No valid segments for {num_frames} frames")

        pose_latent_segments = []
        pose_mask_segments = []
        for seg_start, seg_end in segments:
            pose_half = F.interpolate(
                pose_frames[seg_start:seg_end],
                # Match WanSCAILToVideo's `common_upscale(..., "area")`.
                # Bilinear changes pose-mask/edge weights at every crop and
                # can make the first frame after an Extend seam drift.
                size=(h // 2, w // 2), mode='area'
            )
            pose_latent_segments.append(self._vae_encode_video(pose_half))
            del pose_half

            if pose_mask_cthw is None:
                pose_mask_segments.append(None)
            else:
                mask_half = F.interpolate(
                    pose_mask_cthw[:, seg_start:seg_end].permute(1, 0, 2, 3),
                    size=(h // 2, w // 2), mode='area'
                ).permute(1, 0, 2, 3)
                pose_mask_segments.append(
                    extract_and_compress_mask_to_latent(
                        mask_half.cpu(), additional_spatial_downsample=1
                    ).to(device, dtype)
                )
                del mask_half
        del pose_frames
        torch.cuda.empty_cache()

        report_stage(0.20, "SCAIL2: CLIP 参考图编码")
        # --- CLIP vision features ---
        self._swap_model_to_gpu("image_encoder")
        clip_features = self._get_clip_features(ref_img)  # (1, N, 1280)

        report_stage(0.24, "SCAIL2: 文本编码")
        # --- Text encoding ---
        self._swap_model_to_gpu("text_encoder")
        context = self._encode_text(prompt)
        context_null = self._encode_text(negative_prompt)
        if isinstance(context, list):
            context = context[0]
            context_null = context_null[0]

        # Restore DiT to GPU for diffusion
        self._restore_dit_to_gpu()

        report_stage(0.28, "SCAIL2: 准备量化 DiT，首个前向可能较慢")
        total_sampling_steps = max(len(segments) * int(sampling_steps), 1)
        batched_cfg = (guide_scale > 1.0)

        def report_progress(done, description, step_elapsed=None):
            """Report normalized progress without making Gradio a pipeline dependency."""
            completed = min(max(done, 0), total_sampling_steps)
            percent = 100.0 * completed / total_sampling_steps
            elapsed_note = f"，本步 {step_elapsed:.1f}s" if step_elapsed is not None else ""
            logger.info(
                "[SCAIL2][进度] 采样 %d/%d (%.0f%%)：%s%s",
                completed,
                total_sampling_steps,
                percent,
                description,
                elapsed_note,
            )
            if progress_callback is None:
                return
            value = 0.28 + 0.72 * completed / total_sampling_steps
            try:
                progress_callback(value, desc=description)
            except TypeError:
                # Support simple callbacks that only accept a numeric value.
                progress_callback(value)

        if segment_len <= 5:
            print(
                f"[SCAIL2] 警告：当前显存只能使用 segment_len={segment_len}，"
                f"将执行 {len(segments)} 个分段、每段 {len(timesteps) if 'timesteps' in locals() else sampling_steps} 步；"
                "低显存下可能需要较长时间。",
                flush=True,
            )

        # --- Sampling ---
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Set up scheduler
        self.scheduler.set_timesteps(sampling_steps, shift=shift)
        timesteps = self.scheduler.timesteps
        logger.info(
            "[SCAIL2][配置] %sx%s，%s 帧，分段=%s，步数=%s，CFG=%.2f，shift=%.2f",
            w,
            h,
            num_frames,
            [end - start for start, end in segments],
            len(timesteps),
            guide_scale,
            shift,
        )

        report_progress(0, f"SCAIL2 开始采样：{len(segments)} 段，共 {total_sampling_steps} 步")

        # Pre-build zero tensors for batched CFG — no need to reallocate every step.
        zero_clip = torch.zeros_like(clip_features)
        zero_ref_lat = torch.zeros_like(ref_latent.unsqueeze(0))
        zero_ref_mask = None if ref_mask_28ch is None else torch.zeros_like(ref_mask_28ch.unsqueeze(0))

        print(
            f"[SCAIL2] 进入采样：{len(segments)} 段 × {len(timesteps)} 步，"
            f"首段长度={segments[0][1] - segments[0][0]} 帧，CFG batch={2 if batched_cfg else 1}",
            flush=True,
        )

        output_frames = []
        prev_history_latent = None
        prev_color_reference = None

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_len = seg_end - seg_start
            
            # Latent dimensions for this segment
            f_lat = (seg_len - 1) // self.vae_stride[0] + 1
            h_lat = h // 8
            w_lat = w // 8

            pose_lat_seg = pose_latent_segments[seg_idx].unsqueeze(0)
            pose_mask_seg = (
                None if pose_mask_segments[seg_idx] is None
                else pose_mask_segments[seg_idx].unsqueeze(0)
            )
            ref_mask_batch = None if ref_mask_28ch is None else ref_mask_28ch.unsqueeze(0)
            
            # Initialize noise
            noise = torch.randn(
                (1, 16, f_lat, h_lat, w_lat), device=device, dtype=dtype, generator=generator
            )
            latent = noise

            # First frame conditioning (I2V): use reference as first frame
            first_frame_latent = ref_latent.unsqueeze(0)  # (1, 16, 1, H/8, W/8)
            frame_mask = torch.zeros(1, 4, f_lat, h_lat, w_lat, device=device, dtype=dtype)
            frame_mask[:, :, 0] = 1.0  # first frame is known
            history_indicator_mask = torch.zeros_like(frame_mask)
            # Apply clean history from previous segment
            if prev_history_latent is not None:
                history_t = min(prev_history_latent.shape[2], f_lat)
                history_latent = prev_history_latent[:, :, :history_t].to(device, dtype)
                # ComfyUI's WAN21_SCAIL2.scale_latent_inpaint() returns the
                # clean latent_image unchanged at every sigma. Keep this
                # SCAIL2-specific behavior instead of generic noisy inpainting.
                frame_mask[:, :, :history_t] = 1.0  # history frames are known
                history_indicator_mask[:, :, :history_t] = 1.0
            else:
                history_t = 0
                history_latent = None
            # Expand first frame across temporal dimension for conditioning
            first_frame_expanded = first_frame_latent.expand(1, 16, f_lat, h_lat, w_lat).clone()
            if prev_history_latent is not None:
                first_frame_expanded[:, :, :history_t] = prev_history_latent[:, :, :history_t].to(device, dtype)
            y = torch.cat([frame_mask, first_frame_expanded], dim=1)  # (1, 20, f_lat, h_lat, w_lat)

            # Denoising loop
            torch.cuda.empty_cache()  # one purge per segment is enough
            for i, t in enumerate(timesteps):
                done_steps = seg_idx * len(timesteps) + i
                # Match WAN21_SCAIL2.scale_latent_inpaint: preserved frames
                # stay clean when supplied to the DiT, at every timestep.
                if history_t:
                    latent[:, :, :history_t] = history_latent
                # Determine if pose conditioning is active at this timestep
                t_ratio = i / max(len(timesteps) - 1, 1)
                pose_active = pose_start <= t_ratio <= pose_end

                t_device = t.unsqueeze(0).to(device)
                pose_cond = pose_lat_seg * pose_strength if pose_active else pose_lat_seg
                first_forward = seg_idx == 0 and i == 0
                step_started = time.perf_counter()
                if first_forward:
                    effective_tokens = (
                        (f_lat + 1) * (h_lat // 2) * (w_lat // 2)
                        + f_lat * (h_lat // 4) * (w_lat // 4)
                    )
                    logger.info(
                        "[SCAIL2][阶段] 首个 DiT 前向开始：约 %s tokens，分辨率=%sx%s，帧数=%s",
                        f"{effective_tokens:,}", h, w, seg_len,
                    )

                # --- Forward pass(s) ---
                if guide_scale <= 1.0:
                    # No CFG: unconditional branch contributes nothing.
                    noise_pred = self.dit(
                        x=latent,
                        timestep=t_device,
                        context=context,
                        clip_feature=clip_features,
                        y=y,
                        ref_latent=first_frame_latent,
                        ref_mask=ref_mask_batch,
                        pose_latent=pose_cond,
                        pose_mask=pose_mask_seg,
                        ref_mask_flag=not replacement_mode,
                        history_indicator_mask=history_indicator_mask,
                    )
                elif batched_cfg:
                    # Batched CFG: cond + uncond in a single DiT forward (batch=2).
                    # This cuts DiT compute roughly in half vs sequential CFG.
                    uncond_pose = torch.zeros_like(pose_lat_seg)
                    noise_pred = self.dit(
                        x=torch.cat([latent, latent], dim=0),
                        timestep=torch.cat([t_device, t_device], dim=0),
                        context=torch.cat([context, context_null], dim=0),
                        clip_feature=torch.cat([clip_features, zero_clip], dim=0),
                        y=torch.cat([y, y], dim=0),
                        ref_latent=torch.cat([first_frame_latent, zero_ref_lat], dim=0),
                        ref_mask=None if ref_mask_batch is None else torch.cat([ref_mask_batch, zero_ref_mask], dim=0),
                        pose_latent=torch.cat([pose_cond, uncond_pose], dim=0),
                        pose_mask=None if pose_mask_seg is None else torch.cat([pose_mask_seg, pose_mask_seg], dim=0),
                        ref_mask_flag=not replacement_mode,
                        history_indicator_mask=torch.cat(
                            [history_indicator_mask, history_indicator_mask], dim=0
                        ),
                    )
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                torch.cuda.synchronize(device)
                step_elapsed = time.perf_counter() - step_started

                # Scheduler step (flow matching is deterministic)
                latent = self.scheduler.step(noise_pred, t, latent)
                if history_t:
                    latent[:, :, :history_t] = history_latent
                report_progress(
                    done_steps + 1,
                    f"SCAIL2 分段 {seg_idx + 1}/{len(segments)}，步骤 {i + 1}/{len(timesteps)} 完成",
                    step_elapsed=step_elapsed,
                )

            # Match ComfyUI's Base / Extend graph: decode each segment on its
            # own and re-encode the prior segment's tail images as the next
            # segment's clean latent anchor.
            del noise, pose_lat_seg, pose_mask_seg
            torch.cuda.empty_cache()
            self._swap_model_to_gpu("vae")
            segment_latent = latent[0].to(
                self.vae.dtype if hasattr(self.vae, "dtype") else dtype
            )
            segment_video = self.vae.decode(
                [segment_latent], device, tiled=segment_latent.shape[1] > 9,
                tile_size=(64, 64), tile_stride=(48, 48),
            )[0].detach().float().cpu()  # (3, T, H, W), normalized to [-1, 1]

            history_frames = (
                max(0, seg_end - segments[seg_idx + 1][0])
                if seg_idx + 1 < len(segments) else 0
            )
            discard_frames = (
                max(0, segments[seg_idx - 1][1] - seg_start)
                if seg_idx > 0 else 0
            )
            segment_display = (
                (segment_video[:, discard_frames:] / 2 + 0.5)
                .clamp(0, 1).permute(1, 2, 3, 0).numpy()
            )
            # Match ComfyUI's Extend output handling: drop the anchored overlap
            # and color-match novel frames to the previous chunk's last frame.
            if prev_color_reference is not None and len(segment_display):
                segment_display = np.stack([
                    self._reinhard_lab_match_frame(
                        frame,
                        prev_color_reference,
                        device,
                        strength=1.0,
                    )
                    for frame in segment_display
                ])
            output_frames.extend(
                Image.fromarray((frame * 255).astype(np.uint8))
                for frame in segment_display
            )
            if len(segment_display):
                prev_color_reference = segment_display[-1].copy()

            if history_frames:
                # Re-encode the same corrected decoded tail that ComfyUI feeds
                # back as previous_frames, rather than the pre-correction VAE
                # tensor. This keeps both the visible seam and latent anchor in
                # sync for the next segment.
                tail = torch.from_numpy(
                    np.ascontiguousarray(segment_display[-history_frames:])
                ).to(device=device, dtype=torch.float32)
                tail = (tail.permute(0, 3, 1, 2) * 2.0 - 1.0).contiguous()
                prev_history_latent = self._vae_encode_video(
                    tail
                ).unsqueeze(0)
                del tail
                self._restore_dit_to_gpu()
                torch.cuda.empty_cache()
            else:
                prev_history_latent = None

        report_progress(total_sampling_steps, "SCAIL2 视频分段解码与拼接完成")
        return output_frames
