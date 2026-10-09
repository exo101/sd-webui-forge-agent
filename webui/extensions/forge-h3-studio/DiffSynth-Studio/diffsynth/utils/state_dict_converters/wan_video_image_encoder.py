import math
import torch


def WanImageEncoderStateDictConverter(state_dict):
    """Convert ComfyUI/HuggingFace CLIP ViT-H checkpoint to DiffSynth (OpenCLIP) naming.

    Key differences:
    - HF: `vision_model.embeddings.*`  ->  OpenCLIP: `model.visual.*`
    - HF: separate q_proj/k_proj/v_proj  ->  OpenCLIP: fused `attn.to_qkv`
    - HF: position_embedding [N, C]  ->  OpenCLIP: pos_embedding [1, N, C]
    - HF: class_embedding [C]  ->  OpenCLIP: cls_embedding [1, 1, C]
    - HF: visual_projection [out, in]  ->  OpenCLIP: head [in, out] (transposed)
    """
    converted = {}

    # First pass: collect per-layer q/k/v projections for fusion
    # Layer index -> {q: tensor, k: tensor, v: tensor}
    qkv_buffer = {}

    for k in state_dict:
        v = state_dict[k]

        # Skip non-vision keys (textual, position_ids, etc.)
        if not k.startswith("vision_model.") and k != "visual_projection.weight":
            continue
        if k.endswith("position_ids"):
            continue

        # --- Embeddings ---
        if k == "vision_model.embeddings.class_embedding":
            # [C] -> [1, 1, C]
            converted["model.visual.cls_embedding"] = v.unsqueeze(0).unsqueeze(0)
            continue

        if k == "vision_model.embeddings.patch_embedding.weight":
            converted["model.visual.patch_embedding.weight"] = v
            continue

        if k == "vision_model.embeddings.position_embedding.weight":
            # [N, C] -> [1, N, C]
            converted["model.visual.pos_embedding"] = v.unsqueeze(0)
            continue

        # --- Pre/Post layer norm ---
        if k.startswith("vision_model.pre_layrnorm."):
            suffix = k.split("vision_model.pre_layrnorm.", 1)[1]
            converted[f"model.visual.pre_norm.{suffix}"] = v
            continue

        if k.startswith("vision_model.post_layernorm."):
            suffix = k.split("vision_model.post_layernorm.", 1)[1]
            converted[f"model.visual.post_norm.{suffix}"] = v
            continue

        # --- Visual projection (head) ---
        if k == "visual_projection.weight":
            # [out_dim, in_dim] -> [in_dim, out_dim]
            converted["model.visual.head"] = v.t()
            continue

        # --- Transformer blocks ---
        # vision_model.encoder.layers.N.{...}
        if k.startswith("vision_model.encoder.layers."):
            parts = k.split(".")
            # parts: ['vision_model', 'encoder', 'layers', 'N', 'sub', ...]
            layer_idx = parts[3]
            sub = parts[4]  # layer_norm1, self_attn, mlp, layer_norm2

            if sub == "layer_norm1":
                suffix = parts[5]  # weight or bias
                converted[f"model.visual.transformer.{layer_idx}.norm1.{suffix}"] = v
                continue

            if sub == "layer_norm2":
                suffix = parts[5]  # weight or bias
                converted[f"model.visual.transformer.{layer_idx}.norm2.{suffix}"] = v
                continue

            if sub == "mlp":
                # mlp.fc1 -> mlp.0, mlp.fc2 -> mlp.2
                dense = parts[5]  # fc1 or fc2
                suffix = parts[6]  # weight or bias
                idx = "0" if dense == "fc1" else "2"
                converted[f"model.visual.transformer.{layer_idx}.mlp.{idx}.{suffix}"] = v
                continue

            if sub == "self_attn":
                # q_proj/k_proj/v_proj -> fuse into to_qkv
                # out_proj -> proj
                proj = parts[5]  # q_proj, k_proj, v_proj, out_proj
                suffix = parts[6]  # weight or bias
                if proj == "out_proj":
                    converted[f"model.visual.transformer.{layer_idx}.attn.proj.{suffix}"] = v
                    continue
                # Buffer q/k/v for fusion
                qkv_buffer.setdefault(layer_idx, {})[f"{proj}.{suffix}"] = v
                continue

    # --- Fuse q/k/v projections into to_qkv ---
    for layer_idx, projs in qkv_buffer.items():
        for suffix in ("weight", "bias"):
            q = projs.get(f"q_proj.{suffix}")
            k = projs.get(f"k_proj.{suffix}")
            v = projs.get(f"v_proj.{suffix}")
            if q is not None and k is not None and v is not None:
                fused = torch.cat([q, k, v], dim=0)
                converted[f"model.visual.transformer.{layer_idx}.attn.to_qkv.{suffix}"] = fused

    # --- log_scale: not present in HF clip vision checkpoint, use default ---
    converted["model.log_scale"] = torch.tensor(math.log(1 / 0.07))

    return converted
