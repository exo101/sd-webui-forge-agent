from __future__ import annotations

import gc
from typing import Optional


def unload_forge_models(target_free_gb: int = 18) -> str:
    """Ask Forge-Neo to unload resident diffusion models before SeedVR2 loads."""
    messages = []
    try:
        from backend import memory_management

        device = memory_management.get_torch_device()
        bytes_required = int(target_free_gb * (1024 ** 3))
        memory_management.free_memory(bytes_required, device)
        messages.append(f"Forge free_memory requested {target_free_gb}GB on {device}")
    except Exception as e:
        messages.append(f"Forge memory manager unavailable: {e}")

    try:
        # Older/newer Forge revisions may expose a direct full unload helper.
        from modules import sd_models
        if hasattr(sd_models, "unload_model_weights"):
            try:
                sd_models.unload_model_weights()
                messages.append("Forge checkpoint weights unloaded")
            except Exception:
                pass
    except Exception:
        pass

    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    return "; ".join(messages)


# Backward-compatible alias used by the first scaffold.
def free_vram(target_gb: int = 18) -> Optional[str]:
    return unload_forge_models(target_gb)
