import json
import os

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, model_info
from transformers.image_processing_base import ImageProcessingMixin

from .birefnet.image_processing_birefnet import BiRefNetImageProcessor
from .birefnet.modeling_birefnet import BiRefNet
from .utils import set_doc


def _is_local_path(path: str) -> bool:
    """Check if a path is a local directory, not a HuggingFace repo ID."""
    # A valid HF repo ID must not contain path separators, drive letters, or
    # be an absolute/relative filesystem path.
    return os.path.sep in path or (os.path.altsep and os.path.altsep in path)


# Inspect repo parameters and return the appropriate model class
class AutoModel:
    @classmethod
    @set_doc(PyTorchModelHubMixin.from_pretrained.__doc__)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if _is_local_path(pretrained_model_name_or_path):
            # Local path — load directly without HF hub validation
            return BiRefNet.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        tags: list[str] = model_info(pretrained_model_name_or_path).tags or []
        if "nobg-birefnet" in tags or "birefnet" in tags:
            return BiRefNet.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        else:
            raise ValueError("this model is not part of nobg")


# Maps the `image_processor_type` recorded in preprocessor_config.json to its
# class. Mirrors transformers' IMAGE_PROCESSOR_MAPPING; append new processors here.
PROCESSOR_TYPES = {
    "BiRefNetImageProcessor": BiRefNetImageProcessor,
}


# Inspect repo parameters and return the appropriate image processor
class AutoProcessor:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if _is_local_path(pretrained_model_name_or_path):
            # Local path — use our custom from_pretrained directly
            return BiRefNetImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        try:
            config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                pretrained_model_name_or_path, **kwargs
            )
        except OSError:
            # fallback for repos that don't have a preprocessor_config.json (e.g. legacy BiRefNet)
            return cls._from_model_config(pretrained_model_name_or_path, **kwargs)

        processor_type = config_dict.get("image_processor_type")
        processor_class = PROCESSOR_TYPES.get(processor_type)
        if processor_class is None:
            raise ValueError("this model is not part of nobg")
        # Re-load through the resolved class (mirrors transformers, which reads the
        # type then calls the class's own from_pretrained to handle all kwargs).
        return processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def _from_model_config(pretrained_model_name_or_path, **kwargs):
        tags: list[str] = model_info(pretrained_model_name_or_path).tags or []
        if "nobg-birefnet" not in tags and "birefnet" not in tags:
            raise ValueError("this model is not part of nobg")
        config_file = hf_hub_download(
            pretrained_model_name_or_path,
            "config.json",
            token=kwargs.get("token"),
        )
        with open(config_file) as f:
            image_size = json.load(f).get("image_size", 1024)
        return BiRefNetImageProcessor(size={"height": image_size, "width": image_size})