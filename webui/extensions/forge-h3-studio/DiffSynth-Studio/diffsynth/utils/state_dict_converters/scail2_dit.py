"""State dict converter for SCAIL-2 fp8_scaled checkpoints.

ComfyUI's `scaled_fp8` format stores per-layer fp8 weights with a scalar
`scale_weight`. DiffSynth-Studio's comfy_kitchen fp8 backend expects the
`comfy_quant` marker + `weight_scale` naming convention instead.
This converter bridges the two so the pre-quantized checkpoint loads via
`comfy_kitchen_fp8_w8a8` (load_prequantized=True).
"""
import json
import torch


def Scail2Fp8StateDictConverter(state_dict):
    """Convert ComfyUI scaled_fp8 format to comfy_kitchen float8_e4m3fn format."""
    converted = {}
    scale_keys = [k for k in state_dict if k.endswith(".scale_weight")]

    # Drop the global scaled_fp8 flag (not a model parameter)
    for k in state_dict:
        if k == "scaled_fp8":
            continue
        converted[k] = state_dict[k]

    marker_bytes = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")
    marker_tensor = torch.frombuffer(bytearray(marker_bytes), dtype=torch.uint8)

    for scale_key in scale_keys:
        layer_name = scale_key[: -len(".scale_weight")]
        # Rename scale_weight -> weight_scale (comfy_kitchen convention)
        scale = converted.pop(scale_key)
        converted[f"{layer_name}.weight_scale"] = scale
        # Add per-layer comfy_quant marker
        converted[f"{layer_name}.comfy_quant"] = marker_tensor.clone()

    return converted
