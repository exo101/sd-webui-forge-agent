"""State dict converter for Wan2.1 UMT5-XXL text encoder (ComfyUI fp8_scaled repackaged).

The ComfyUI-repackaged UMT5 checkpoint uses diffusers-style key naming
(`encoder.block.N.layer.0.SelfAttention.q.weight`, ...) and the ComfyUI
`scaled_fp8` format (per-layer `scale_weight` + global `scaled_fp8` flag).

DiffSynth-Studio's `WanTextEncoder` uses a different key layout
(`blocks[N].attn.q.weight`, ...) and the comfy_kitchen fp8 backend expects
`weight_scale` + `comfy_quant` markers. This converter bridges both gaps.
"""
import json
import torch


def WanTextEncoderFp8StateDictConverter(state_dict):
    """Convert ComfyUI UMT5 fp8_scaled checkpoint to DiffSynth + comfy_kitchen format."""
    converted = {}

    marker_bytes = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")
    marker_tensor = torch.frombuffer(bytearray(marker_bytes), dtype=torch.uint8)

    for k in state_dict:
        v = state_dict[k]
        # Drop non-parameter keys
        if k == "scaled_fp8" or k == "spiece_model":
            continue

        new_key = k

        # Token embedding
        if k == "shared.weight":
            new_key = "token_embedding.weight"

        # Final layer norm
        elif k == "encoder.final_layer_norm.weight":
            new_key = "norm.weight"

        # Encoder blocks
        elif k.startswith("encoder.block."):
            parts = k.split(".")
            # encoder.block.N.layer.0.SelfAttention.q.weight
            # encoder.block.N.layer.0.layer_norm.weight
            # encoder.block.N.layer.1.DenseReluDense.wi_0.weight
            block_idx = parts[2]
            layer_idx = parts[4]  # 0 = self-attn, 1 = ffn

            if layer_idx == "0":
                # Self-attention layer
                sub = parts[5]  # SelfAttention or layer_norm
                if sub == "SelfAttention":
                    param = parts[6]  # q, k, v, o, relative_attention_bias
                    if param == "relative_attention_bias":
                        new_key = f"blocks.{block_idx}.pos_embedding.embedding.weight"
                    else:
                        # q, k, v, o (and their .scale_weight)
                        suffix = parts[7] if len(parts) > 7 else ""
                        if suffix:
                            new_key = f"blocks.{block_idx}.attn.{param}.{suffix}"
                        else:
                            new_key = f"blocks.{block_idx}.attn.{param}.weight"
                elif sub == "layer_norm":
                    new_key = f"blocks.{block_idx}.norm1.weight"

            elif layer_idx == "1":
                # FFN layer
                sub = parts[5]  # DenseReluDense or layer_norm
                if sub == "DenseReluDense":
                    # wi_0 -> gate.0, wi_1 -> fc1, wo -> fc2
                    dense_param = parts[6]  # wi_0, wi_1, wo
                    name_map = {"wi_0": "gate.0", "wi_1": "fc1", "wo": "fc2"}
                    mapped = name_map.get(dense_param, dense_param)
                    suffix = parts[7] if len(parts) > 7 else ""
                    if suffix:
                        new_key = f"blocks.{block_idx}.ffn.{mapped}.{suffix}"
                    else:
                        new_key = f"blocks.{block_idx}.ffn.{mapped}.weight"
                elif sub == "layer_norm":
                    new_key = f"blocks.{block_idx}.norm2.weight"

        converted[new_key] = v

    # --- fp8 format conversion: scale_weight -> weight_scale + comfy_quant ---
    scale_keys = [k for k in converted if k.endswith(".scale_weight")]
    for scale_key in scale_keys:
        layer_name = scale_key[: -len(".scale_weight")]
        scale = converted.pop(scale_key)
        converted[f"{layer_name}.weight_scale"] = scale
        converted[f"{layer_name}.comfy_quant"] = marker_tensor.clone()

    return converted
