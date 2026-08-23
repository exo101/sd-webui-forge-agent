"""BiRefNet image processor — standalone version compatible with transformers 5.14.x.

Replaces the upstream ``transformers.image_processing_backends``-based processor
which requires transformers >= 5.16.  All preprocessing is done via direct
``torchvision`` / ``PIL`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


# ---------------------------------------------------------------------------
# Constants matching ImageNet statistics used by BiRefNet
# ---------------------------------------------------------------------------
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Lightweight image processor
# ---------------------------------------------------------------------------
class BiRefNetImageProcessor:
    """Image processor for BiRefNet.

    Performs: RGB conversion, square resize, [0,1] rescale, ImageNet
    normalization.  Also provides ``post_process_alpha_matting`` and
    ``cutout`` helpers.
    """

    resample = Image.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 1024, "width": 1024}
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    rescale_factor = 1.0 / 255.0

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __call__(self, images=None, **kwargs):
        """Alias for ``preprocess``, matching the transformers API convention."""
        return self.preprocess(images, **kwargs)

    # ---- Preprocessing ----------------------------------------------------

    def preprocess(self, images, **kwargs):
        """Preprocess one or more PIL images into a tensor batch.

        Returns a dict-like ``BatchFeature`` with key ``"pixel_values"``.
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        processed = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img) if isinstance(img, (np.ndarray,)) else img
            if self.do_convert_rgb:
                img = img.convert("RGB")
            if self.do_resize:
                h = self.size.get("height", self.size.get("shortest_edge", 1024))
                w = self.size.get("width", h)
                img = img.resize((w, h), self.resample)
            if self.do_rescale:
                arr = TF.to_tensor(img)  # → [0, 1]  float32 C,H,W
            else:
                arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            if self.do_normalize:
                mean = torch.tensor(self.image_mean).view(3, 1, 1)
                std = torch.tensor(self.image_std).view(3, 1, 1)
                arr = (arr - mean) / std
            processed.append(arr)

        pixel_values = torch.stack(processed, dim=0)  # (B, 3, H, W)
        return BatchFeature({"pixel_values": pixel_values})

    # ---- Post-processing --------------------------------------------------

    def post_process_alpha_matting(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Convert raw BiRefNet logits into per-image alpha mattes in [0, 1].

        Args:
            outputs: Model output dict with ``"logits"`` key, or object with
                ``.logits`` attribute, shape ``(B, 1, H, W)``.
            target_sizes: Optional list of ``(height, width)`` tuples, one per
                image; each matte is bilinearly resized to its target size.

        Returns:
            List of ``(H, W)`` tensors with values in [0, 1].
        """
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if target_sizes is not None and len(logits) != len(target_sizes):
            raise ValueError(
                f"Got {len(target_sizes)} target sizes for a batch of "
                f"{len(logits)} images"
            )
        probs = logits.sigmoid()
        mattes = []
        for idx in range(len(probs)):
            alpha = probs[idx].unsqueeze(0)  # (1, 1, H, W)
            if target_sizes is not None:
                alpha = F.interpolate(
                    alpha, size=target_sizes[idx], mode="bilinear",
                    align_corners=False,
                )
            mattes.append(alpha[0, 0])
        return mattes

    @staticmethod
    def cutout(image: Image.Image, alpha: torch.Tensor | Image.Image) -> Image.Image:
        """Composite an alpha matte onto ``image``, returning an RGBA cutout."""
        if isinstance(alpha, torch.Tensor):
            arr = (alpha.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            alpha = Image.fromarray(arr, mode="L")
        if alpha.size != image.size:
            alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
        cutout = image.convert("RGBA")
        cutout.putalpha(alpha)
        return cutout

    # ---- HuggingFace Hub helpers ------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor config from a local directory or HF hub repo.

        Simplified version that reads ``preprocessor_config.json`` from the
        repo root and instantiates the processor with those settings.
        """
        import json
        import os
        from huggingface_hub import hf_hub_download

        # Try local first, then HF hub
        config_path = os.path.join(pretrained_model_name_or_path,
                                   "preprocessor_config.json")
        if not os.path.isfile(config_path):
            try:
                config_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    "preprocessor_config.json",
                    **{k: v for k, v in kwargs.items()
                       if k in ("token", "revision", "cache_dir")},
                )
            except OSError:
                # No config file — use defaults
                return cls(**kwargs)

        with open(config_path) as f:
            config = json.load(f)

        # Map HF-style keys to our class attributes
        mapping = {
            "size": "size",
            "image_mean": "image_mean",
            "image_std": "image_std",
            "do_resize": "do_resize",
            "do_rescale": "do_rescale",
            "do_normalize": "do_normalize",
            "do_convert_rgb": "do_convert_rgb",
            "rescale_factor": "rescale_factor",
        }
        proc_kwargs = {}
        for hf_key, attr in mapping.items():
            val = config.get(hf_key, getattr(cls, attr, None))
            if val is not None:
                proc_kwargs[attr] = val

        # Size from config (may be a dict or int)
        size_val = config.get("size", cls.size)
        if isinstance(size_val, (int, float)):
            proc_kwargs["size"] = {"height": int(size_val), "width": int(size_val)}

        proc_kwargs.update(kwargs)
        return cls(**proc_kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save processor config to a local directory."""
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        config = {
            "image_processor_type": "BiRefNetImageProcessor",
            "size": self.size,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "do_convert_rgb": self.do_convert_rgb,
            "rescale_factor": self.rescale_factor,
        }
        path = os.path.join(save_directory, "preprocessor_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Minimal dict-like wrapper for the preprocess return value
# ---------------------------------------------------------------------------
class BatchFeature(dict):
    """Minimal stand-in for ``transformers.image_processing_base.BatchFeature``."""

    def __init__(self, data: Mapping[str, Any], tensor_type: str | None = None):
        super().__init__(data)
        self._tensor_type = tensor_type

    def to(self, *args, **kwargs):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(*args, **kwargs)
        return self

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None