"""
Download TRELLIS.2 model to the models directory
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.paths import models_path

TRELLIS2_MODEL_DIR = os.path.join(models_path, 'trellis2')


def download_model():
    """Download TRELLIS.2 model from Hugging Face"""
    
    print(f"[TRELLIS.2] Model directory: {TRELLIS2_MODEL_DIR}")
    
    if not os.path.exists(TRELLIS2_MODEL_DIR):
        os.makedirs(TRELLIS2_MODEL_DIR, exist_ok=True)
        print(f"[TRELLIS.2] Created directory: {TRELLIS2_MODEL_DIR}")
    
    # Check if model already exists
    config_file = os.path.join(TRELLIS2_MODEL_DIR, 'config.json')
    if os.path.exists(config_file):
        print("[TRELLIS.2] Model already exists, skipping download")
        return
    
    print("[TRELLIS.2] Downloading model from Hugging Face...")
    print("[TRELLIS.2] This may take a while (model size ~16GB)")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id="microsoft/TRELLIS.2-4B",
            local_dir=TRELLIS2_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("[TRELLIS.2] Model downloaded successfully!")
        
    except ImportError:
        print("[TRELLIS.2] Error: huggingface-hub not installed")
        print("[TRELLIS.2] Please install it: pip install huggingface-hub")
        
    except Exception as e:
        print(f"[TRELLIS.2] Error downloading model: {e}")
        print("[TRELLIS.2] You can also manually download from:")
        print("[TRELLIS.2] https://huggingface.co/microsoft/TRELLIS.2-4B")


if __name__ == "__main__":
    download_model()
