import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    
    # Normalize path separators
    path = path.replace('\\', '/')
    
    print(f"[TRELLIS.2] Loading model from path: {path}")
    
    # Map HuggingFace repo IDs to local paths
    # This prevents unnecessary downloads when models are already available locally
    hf_to_local_map = {
        'microsoft/TRELLIS-image-large': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..', 'models', 'trellis2', 'TRELLIS-image-large'),
    }
    
    # Check if path is a HF repo ID and map to local path
    original_path = path
    mapped = False
    for hf_repo_id, local_path in hf_to_local_map.items():
        if path.startswith(hf_repo_id):
            # Extract the sub-path after repo ID
            sub_path = path[len(hf_repo_id):].lstrip('/')
            local_full_path = os.path.join(local_path, sub_path).replace('\\', '/')
            
            print(f"[TRELLIS.2] Attempting to map HF repo '{hf_repo_id}'")
            print(f"[TRELLIS.2]   Original path: {path}")
            print(f"[TRELLIS.2]   Local base path: {local_path}")
            print(f"[TRELLIS.2]   Sub-path: {sub_path}")
            print(f"[TRELLIS.2]   Mapped full path: {local_full_path}")
            
            # Check if local files exist
            json_exists = os.path.exists(f"{local_full_path}.json")
            safetensors_exists = os.path.exists(f"{local_full_path}.safetensors")
            print(f"[TRELLIS.2]   .json exists: {json_exists}")
            print(f"[TRELLIS.2]   .safetensors exists: {safetensors_exists}")
            
            if json_exists and safetensors_exists:
                print(f"[TRELLIS.2] ✓ Mapping HF repo '{hf_repo_id}' to local path: {local_full_path}")
                path = local_full_path
                mapped = True
                break
            else:
                print(f"[TRELLIS.2] ✗ Local files not found, will NOT download from HF (offline mode)")
                # Force use local path even if files don't exist - will raise FileNotFoundError later
                path = local_full_path
                mapped = True
                break
    
    if not mapped and '/' in path and not os.path.isabs(path):
        print(f"[TRELLIS.2] ⚠ Path '{path}' looks like HF repo but no mapping configured")
        print(f"[TRELLIS.2] ⚠ In offline mode, this will fail. Please add mapping or ensure files exist locally.")
    
    # Check if it's a local path (absolute or relative)
    is_local = os.path.isabs(path) or (os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors"))

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
        
        # Verify files exist
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
    else:
        # OFFLINE MODE: Do not attempt to download from HuggingFace
        raise FileNotFoundError(
            f"Model files not found locally: {path}\n"
            f"This is an offline deployment. Please ensure all models are downloaded to the local directory.\n"
            f"Expected files:\n"
            f"  - {path}.json\n"
            f"  - {path}.safetensors"
        )

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
