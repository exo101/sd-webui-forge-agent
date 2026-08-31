from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        
        # Normalize path separators for Windows compatibility
        path = path.replace('\\', '/')
        
        # Check if it's a local path (absolute or relative)
        is_local = os.path.isabs(path) or os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file_path = f"{path}/{config_file}"
            
            # Verify file exists
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(f"Config file not found: {config_file_path}")
        else:
            from huggingface_hub import hf_hub_download
            config_file_path = hf_hub_download(path, config_file)

        with open(config_file_path, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            try:
                # Check if v is a HuggingFace repo ID (starts with org name like "microsoft/")
                # Local relative paths (like "ckpts/xxx") should NOT be treated as HF IDs
                if '/' in v and not os.path.isabs(v):
                    # Only treat as HF repo ID if it starts with a known organization
                    hf_orgs = ['microsoft', 'stabilityai', 'runwayml', 'black-forest-labs']
                    is_hf_repo = any(v.startswith(org + '/') for org in hf_orgs)
                    
                    if is_hf_repo:
                        # This is a HF repo ID, try loading directly first
                        print(f"[TRELLIS.2] Detected HF repo ID for {k}: {v}")
                        _models[k] = models.from_pretrained(v)
                    else:
                        # This is a local relative path, construct full path
                        model_path = f"{path}/{v}".replace('\\', '/')
                        _models[k] = models.from_pretrained(model_path)
                else:
                    # Absolute path or simple filename, construct full path
                    model_path = f"{path}/{v}".replace('\\', '/')
                    _models[k] = models.from_pretrained(model_path)
            except Exception as e:
                print(f"[WARNING] Failed to load model {k} from {v}: {e}")
                # Try alternative approach
                if '/' in v and not os.path.isabs(v):
                    hf_orgs = ['microsoft', 'stabilityai', 'runwayml', 'black-forest-labs']
                    is_hf_repo = any(v.startswith(org + '/') for org in hf_orgs)
                    
                    if is_hf_repo:
                        # Already tried HF repo, try with path prefix
                        model_path = f"{path}/{v}".replace('\\', '/')
                        _models[k] = models.from_pretrained(model_path)
                    else:
                        # Try as HF repo ID (fallback)
                        _models[k] = models.from_pretrained(v)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))