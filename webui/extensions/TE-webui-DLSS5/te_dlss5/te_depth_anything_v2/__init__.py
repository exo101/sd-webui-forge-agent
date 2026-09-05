"""Plugin-local Depth Anything V2 model definition.

The network is kept local so TE can run without importing another custom node.
Weights are loaded by ``frame_guidance.py`` from the plugin runtime directory.
"""

from .dpt import DepthAnythingV2


model_configs = {
    "depth_anything_v2_vits.pth": {
        "encoder": "vits", "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "depth_anything_v2_vitl.pth": {
        "encoder": "vitl", "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

__all__ = ["DepthAnythingV2", "model_configs"]
