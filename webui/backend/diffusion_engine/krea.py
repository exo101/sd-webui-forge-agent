import math
import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine
from modules.shared import opts


class Krea2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Krea2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen3VLTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.krea2 = True
        # Store reference images: pixels for VL encoder, latents for DiT
        self.ref_images: list[torch.Tensor] = []  # Original pixels for VL encoder
        self.ref_latents: list[torch.Tensor] = []  # VAE latents for DiT forward
        self.ini_latent = None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        
        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            _images = [*self.ref_images]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None
            
            if _references:
                # Don't clear reference lists - keep them for reuse until user clears via UI
                # Only clear dynamic_args.ref_latents which is passed to DiT
                return self.get_learned_conditioning_with_image(prompt, _references, _images)
            else:
                dynamic_args.ref_latents.clear()
        
        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], latents: list[torch.Tensor], images: list[torch.Tensor]):
        """Process reference images through VL encoder and prepare for DiT.
        
        Args:
            prompt: Text prompts
            latents: VAE-encoded reference latents for DiT forward
            images: Original pixel images for VL encoder
        """
        # Get the device from the text encoder
        device = memory_management.text_encoder_device()
        
        images_vl = []
        
        for image in images:
            # Encode image for VL encoder (downsample to ≤384×384)
            vl_image = self.encode_vision(image)
            # Ensure the image is on the correct device
            vl_image = vl_image.to(device)
            images_vl.append(vl_image)
        
        # Store VAE latents for DiT forward pass
        dynamic_args.ref_latents = latents.copy()
        
        # Pass VL images to text encoder
        # Don't include vision markers in the text - let tokenize handle it automatically
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image for VL encoder (downsample to ≤384×384).
        
        Args:
            image: Image tensor in [B, H, W, C] format, normalized to [0, 1]
        
        Returns:
            Downsampled image tensor for VL encoder
        """
        # image is [B, H, W, C] in [0, 1] range
        samples = image.movedim(-1, 1)  # [B, C, H, W]
        
        # Downsample to ≤384×384 for VL encoder
        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        
        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        _vision = s.movedim(1, -1)  # Back to [B, H, W, C]
        
        return _vision

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_qwen.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        batch: int = x.size(0)
        
        # Clear old references before encoding new ones
        # This ensures we only use the current batch of reference images
        if dynamic_args.is_referencing:
            self.ref_images.clear()
            self.ref_latents.clear()

        for b in range(batch):
            y = x[b].unsqueeze(0)
            # y is [1, C, H, W] in [-1, 1] range
            pixel_image = y.movedim(1, -1) * 0.5 + 0.5  # [1, H, W, C] in [0, 1]
            
            sample = self.forge_objects.vae.encode(pixel_image)
            sample = self.forge_objects.vae.first_stage_model.process_in(sample)
            
            # Store reference images when referencing mode is active
            if dynamic_args.is_referencing:
                self.ref_images.append(pixel_image.cpu())  # Pixels for VL encoder
                self.ref_latents.append(sample.cpu())  # Latents for DiT
            
            samples.append(sample)

        return torch.cat(samples).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        batch: int = x.size(0)

        for b in range(batch):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.first_stage_model.process_out(y)
            sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
            samples.append(sample)

        return torch.cat(samples).to(x)
