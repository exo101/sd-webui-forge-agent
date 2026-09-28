import math
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.qwen_engine import QwenTextProcessingEngine
from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine
from modules import shared
from modules.shared import opts


class QwenImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.QwenImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen25_7b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen25_7b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = QwenTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen25_7b,
            tokenizer=clip.tokenizer.qwen25_7b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        # 新一轮 conditioning：清掉上一次生成的前缀 KV cache（cache key 本身也带内容校验，这里只是及时回收显存）
        m = getattr(self.forge_objects.unet.model, "diffusion_model", None)
        if m is not None and hasattr(m, "reset_prefix_cache"):
            m.reset_prefix_cache()

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if _references:
                return self.get_learned_conditioning_with_image(prompt, _references)
            else:
                dynamic_args.ref_latents.clear()

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        images_vl, ref_latents, image_prompts = [], [], []
        for i, image in enumerate(images):
            v, r, p = self.encode_vision(image, i)
            images_vl.append(v)
            ref_latents.append(r)
            image_prompts.append(p)

        dynamic_args.ref_latents = ref_latents.copy()
        return self.text_processing_engine_qwen(["\n".join([*image_prompts, *prompt])], images=images_vl)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor, i: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        samples = image.movedim(-1, 1)  # b, c, h, w

        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        _vision = s.movedim(1, -1)

        if opts.qwen_vae_resize:
            total = int(1024 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by / 32.0) * 32
            height = round(samples.shape[2] * scale_by / 32.0) * 32

            s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        else:
            s = samples.clone()
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        _prompt = f"Picture {i}: <|vision_start|><|image_pad|><|vision_end|>"

        return (_vision, _latent, _prompt)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if x.size(0) > 1:
            x = x[0].unsqueeze(0)  # enforce batch_size of 1

        start_image = x.movedim(1, -1) * 0.5 + 0.5
        sample = self.forge_objects.vae.encode(start_image)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)

        if dynamic_args.edit:
            if dynamic_args.is_referencing:
                self.ref_latents.append(start_image.cpu())
            else:
                self.ini_latent = start_image.cpu()

        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
        return sample.to(x)


class QwenImage21(ForgeDiffusionEngine):
    matched_guesses = [model_list.QwenImage21]

    def __init__(self, estimated_config, huggingface_components):
        # 检测 DiffSynth 布局权重（_convrot_ds）是否存在，存在则用 DiffSynth 后端
        self.diffsynth_backend = None
        self.use_diffsynth = bool(getattr(dynamic_args, "qwen21_diffsynth", False))
        if self.use_diffsynth:
            # DiffSynth owns the model components.  Forge's base initializer
            # only needs a VAE object to attach the latent format.
            dummy_vae_config = type("DiffSynthVAEConfig", (), {})()
            dummy_vae_config.latent_format = estimated_config.latent_format
            ForgeDiffusionEngine.__init__(self, estimated_config, {"vae": dummy_vae_config})
            try:
                from backend.diffusion_engine.qwen21_diffsynth import Qwen21DiffSynthBackend
                ds_paths = getattr(dynamic_args, "qwen21_ds_paths", None)
                if ds_paths is None:
                    # 兜底：从 models 目录查找
                    from backend.diffusion_engine.qwen21_diffsynth import find_diffsynth_weights
                    main_path = self._guess_main_model_path()
                    ds_paths = find_diffsynth_weights(main_path) if main_path else None
                if ds_paths:
                    self.diffsynth_backend = Qwen21DiffSynthBackend(**ds_paths)
                    print(f"[Qwen21] DiffSynth backend enabled (int8 convrot_ds weights)")
                    self.diffsynth_backend.build_pipeline()  # 预加载，与其他模型行为一致
                else:
                    raise FileNotFoundError("Qwen21 *_ds checkpoint requires the complete DiffSynth companion files")
            except Exception as e:
                print(f"[Qwen21] DiffSynth backend init failed: {e}")
                raise RuntimeError("Qwen21 *_ds checkpoint must use DiffSynth; refusing native fallback") from e

        if self.use_diffsynth:
            # DiffSynth 模式：不构建 webui 的 unet/clip/vae（权重由 DiffSynth 自己加载）
            from backend.diffusion_engine.qwen21_diffsynth import _DummyUnet, _DummyVae
            self.forge_objects = ForgeObjects(unet=_DummyUnet(), clip=None, vae=_DummyVae(), clipvision=None)
            self.forge_objects_original = self.forge_objects.shallow_copy()
            self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()
            self.text_processing_engine_qwen = None
        else:
            super().__init__(estimated_config, huggingface_components)
            clip = CLIP(model_dict={"qwen3vl_8b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_8b": huggingface_components["tokenizer"]})

            vae = VAE(model=huggingface_components["vae"], is_qwen21=True)

            k_predictor = self._get_predictor()

            unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

            self.text_processing_engine_qwen = Qwen3VLTextProcessingEngine(
                text_encoder=clip.cond_stage_model.qwen3vl_8b,
                tokenizer=clip.tokenizer.qwen3vl_8b,
            )

            self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
            self.forge_objects_original = self.forge_objects.shallow_copy()
            self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # Store reference images: pixels for VL encoder, latents for DiT
        self.ref_images: list[torch.Tensor] = []
        self.ref_latents: list[torch.Tensor] = []
        self.ini_latent = None
        self.ini_image = None  # i2i 首图像素（DiffSynth 模式作为第一张参考图传入 edit_image）
        self._ref_session = False  # 参考图编码会话：多图插件逐张编码，只在首张时清空

    @staticmethod
    def _guess_main_model_path():
        """从 webui models 目录猜测主模型文件路径（兜底）。"""
        try:
            from modules import paths
            sd_dir = os.path.join(paths.models_path, "Stable-diffusion")
            for name in os.listdir(sd_dir):
                if "qwen_image_2.1" in name.lower() and name.endswith(".safetensors"):
                    return os.path.join(sd_dir, name)
        except Exception:
            pass
        return None

    def clear_references(self):
        self.ref_images.clear()
        self.ini_latent = None
        self.ini_image = None
        self._ref_session = False
        super().clear_references()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        if self.use_diffsynth:
            # DiffSynth 模式：conditioning 由 pipeline 内部处理，返回 dummy
            return [torch.zeros(1, 1, 4096, dtype=torch.bfloat16)]

        text_patcher = self.forge_objects.clip.patcher
        _t0 = time.perf_counter()
        memory_management.load_model_gpu(text_patcher)
        print(f"[Qwen21] JointTextEncoder load: {time.perf_counter() - _t0:.2f}s", flush=True)

        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            _images = [*self.ref_images]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if _references:
                return self.get_learned_conditioning_with_image(prompt, _references, _images)
            else:
                dynamic_args.ref_latents.clear()
                dynamic_args.qwen21_image_slots.clear()

        _t1 = time.perf_counter()
        result = self.text_processing_engine_qwen(prompt)
        print(f"[Qwen21] Text conditioning: {time.perf_counter() - _t1:.2f}s", flush=True)
        return result

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], latents: list[torch.Tensor], images: list[torch.Tensor]):
        if self.use_diffsynth:
            return [torch.zeros(1, 1, 4096, dtype=torch.bfloat16)]

        device = memory_management.text_encoder_device()

        images_vl = []
        for image in images:
            vl_image = self.encode_vision(image)
            vl_image = vl_image.to(device)
            images_vl.append(vl_image)

        dynamic_args.ref_latents = latents.copy()
        out = self.text_processing_engine_qwen(prompt, images=images_vl)
        dynamic_args.qwen21_image_slots = [*self.text_processing_engine_qwen.last_image_slots]
        return out

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        samples = image.movedim(-1, 1)

        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        _vision = s.movedim(1, -1)

        return _vision

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        if self.use_diffsynth or self.text_processing_engine_qwen is None:
            return 0, 75
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        batch: int = x.size(0)

        if dynamic_args.is_referencing and not self._ref_session:
            # 参考图编码会话开始（首张图）：清空上一轮的残留。
            # 多图参考插件对每张参考图单独调用一次 encode_first_stage，
            # 若每次调用都清空，前面的图会被后续调用冲掉，只剩最后一张。
            self.ref_images.clear()
            self.ref_latents.clear()
            self._ref_session = True

        for b in range(batch):
            y = x[b].unsqueeze(0)
            pixel_image = y.movedim(1, -1) * 0.5 + 0.5

            if self.use_diffsynth:
                # DiffSynth 模式：用 DiffSynth VAE 编码
                pipe = self.diffsynth_backend.build_pipeline()
                # Qwen-Image-2.1 VAE 要求 4 通道输入（RGB + alpha=1）、范围 [-1,1]
                # （与官方 pipeline 的 image.convert("RGBA") + preprocess_image 一致）
                img = pixel_image.movedim(-1, 1)  # (1, 3, H, W) in [0,1]
                alpha = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), device=img.device, dtype=img.dtype)
                img = torch.cat([img, alpha], dim=1) * 2.0 - 1.0  # (1, 4, H, W) in [-1,1]
                sample = pipe.vae.encode(img.to(device="cuda", dtype=torch.bfloat16))
                sample = sample.to(device=x.device, dtype=x.dtype)
            else:
                sample = self.forge_objects.vae.encode(pixel_image)
                sample = self.forge_objects.vae.first_stage_model.process_in(sample)

            if dynamic_args.is_referencing:
                self.ref_images.append(pixel_image.cpu())
                self.ref_latents.append(sample.cpu())
            elif dynamic_args.qwen21:
                # i2i start image: prepended as the first reference when conditioning
                self.ini_latent = sample.cpu()
                self.ini_image = pixel_image.cpu()  # 像素保留：DiffSynth 模式传给 pipeline edit_image

            samples.append(sample)

        return torch.cat(samples).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        samples: list[torch.Tensor] = []
        batch: int = x.size(0)

        for b in range(batch):
            y = x[b].unsqueeze(0)
            if self.use_diffsynth:
                # DiffSynth 模式：用 DiffSynth VAE 解码
                pipe = self.diffsynth_backend.build_pipeline()
                decoded = pipe.vae.decode(y.to(device="cuda", dtype=torch.bfloat16))
                # decoded: (1, H, W, C) in [0,1] -> (1, C, H, W) in [-1,1]
                sample = decoded.movedim(-1, 1) * 2.0 - 1.0
                sample = sample.to(device=x.device, dtype=x.dtype)
            else:
                sample = self.forge_objects.vae.first_stage_model.process_out(y)
                # vae.decode returns (B, H, W, C); the webui expects (B, C, H, W)
                sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
            samples.append(sample)

        return torch.cat(samples).to(x)
