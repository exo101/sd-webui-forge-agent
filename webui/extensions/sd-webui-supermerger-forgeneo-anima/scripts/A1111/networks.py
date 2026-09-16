from __future__ import annotations
import gradio as gr
import logging
import os
import re

import scripts.A1111.lora_patches as lora_patches
import scripts.A1111.network as network
import scripts.A1111.network_lora as network_lora
import scripts.A1111.network_glora as network_glora
import scripts.A1111.network_hada as network_hada
import scripts.A1111.network_ia3 as network_ia3
import scripts.A1111.network_lokr as network_lokr
import scripts.A1111.network_full as network_full
import scripts.A1111.network_norm as network_norm
import scripts.A1111.network_oft as network_oft

import torch
from typing import Union

from modules import shared, devices, sd_models, errors, scripts, launch_utils
import modules.textual_inversion.textual_inversion as textual_inversion

try:
  from modules import sd_hijack
except (ImportError, ModuleNotFoundError):
  sd_hijack = None

class QkvLinear(torch.nn.Linear):
    pass

module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
    network_oft.ModuleTypeOFT(),
]

forge = hasattr(sd_models, "forge_model_reload") or launch_utils.git_tag()[0:2] == "f2"

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

ANIMA_KEY_MARKERS = (
    "lora_unet_blocks_",
    "lora_unet_patch_embed",
    "lora_unet_final_layer",
    "lora_unet_t_embedder",
    "lora_te1_layers_",
    "lora_te_text_encoders_",
    "lora_te_llm_adapter_",
)
ANIMA_QWEN_ALIAS_TO_DOT = {
    "text_encoders_qwen25_7b_": "text_encoders.qwen25_7b.",
    "text_encoders_qwen3_06b_transformer_": "text_encoders.qwen3_06b.transformer.",
    "text_encoders_qwen3_transformer_": "text_encoders.qwen3.transformer.",
}

def is_anima_lora_key(key):
    return any(marker in key for marker in ANIMA_KEY_MARKERS)

def register_network_alias(mapping, alias, module):
    if alias and alias not in mapping:
        mapping[alias] = module

def anima_mlp_alias_variants(value):
    variants = {value}
    replacements = (
        (".mlp.0.", ".mlp.layer1."),
        (".mlp.2.", ".mlp.layer2."),
        ("_mlp_0_", "_mlp_layer1_"),
        ("_mlp_2_", "_mlp_layer2_"),
    )
    changed = True
    while changed:
        changed = False
        current = list(variants)
        for item in current:
            for source, target in replacements:
                if source in item and item.replace(source, target) not in variants:
                    variants.add(item.replace(source, target))
                    changed = True
                if target in item and item.replace(target, source) not in variants:
                    variants.add(item.replace(target, source))
                    changed = True
    return variants

def _anima_join_tokens(tokens):
    parts = []
    i = 0
    while i < len(tokens):
        if i + 3 < len(tokens) and tokens[i] == "adaln" and tokens[i + 1] == "modulation" and tokens[i + 2] in {"cross", "self"} and tokens[i + 3] == "attn":
            parts.append(f"adaln_modulation_{tokens[i + 2]}_attn")
            i += 4
            continue
        if i + 2 < len(tokens) and tokens[i] == "adaln" and tokens[i + 1] == "modulation":
            parts.append(f"adaln_modulation_{tokens[i + 2]}")
            i += 3
            continue
        if i + 1 < len(tokens) and tokens[i] in {"cross", "self"} and tokens[i + 1] == "attn":
            parts.append(f"{tokens[i]}_attn")
            i += 2
            continue
        if i + 1 < len(tokens) and tokens[i] in {"q", "k"} and tokens[i + 1] == "norm":
            parts.append(f"{tokens[i]}_norm")
            i += 2
            continue
        if i + 1 < len(tokens) and tokens[i] in {"q", "k", "v", "output"} and tokens[i + 1] == "proj":
            parts.append(f"{tokens[i]}_proj")
            i += 2
            continue
        parts.append(tokens[i])
        i += 1
    return parts

def anima_dot_name_from_alias(alias):
    prefixes = ("model_diffusion_model_", "diffusion_model_")
    for prefix in prefixes:
        if alias.startswith(prefix):
            suffix = alias[len(prefix):]
            tokens = _anima_join_tokens(suffix.split("_"))
            return prefix.rstrip("_") + "." + ".".join(tokens)
    if alias.startswith("llm_adapter_"):
        suffix = alias[len("llm_adapter_"):]
        tokens = _anima_join_tokens(suffix.split("_"))
        return "llm_adapter." + ".".join(tokens)
    for prefix, dot_prefix in ANIMA_QWEN_ALIAS_TO_DOT.items():
        if alias.startswith(prefix):
            suffix = alias[len(prefix):]
            tokens = _anima_join_tokens(suffix.split("_"))
            return dot_prefix + ".".join(tokens)
    if alias.startswith("text_encoders_"):
        suffix = alias[len("text_encoders_"):]
        tokens = _anima_join_tokens(suffix.split("_"))
        return "text_encoders." + ".".join(tokens)
    return None

def anima_aliases_from_module_name(network_name):
    aliases = {network_name}
    dot_name = anima_dot_name_from_alias(network_name)
    if dot_name:
        aliases.add(dot_name)
    aliases.update(anima_mlp_alias_variants(network_name))
    if dot_name:
        aliases.update(anima_mlp_alias_variants(dot_name))

    if network_name.startswith("model_diffusion_model_"):
        suffix = network_name.split("model_diffusion_model_", 1)[1]
        aliases.add(suffix)
        aliases.add(f"diffusion_model_{suffix}")
        suffix_dot = anima_dot_name_from_alias(f"diffusion_model_{suffix}")
        if suffix_dot:
            aliases.add(suffix_dot)
        model_dot = anima_dot_name_from_alias(network_name)
        if model_dot:
            aliases.add(model_dot.split("model_diffusion_model.", 1)[1])

    if network_name.startswith("text_encoders_"):
        aliases.add(network_name.split("text_encoders_", 1)[1])
        text_dot = anima_dot_name_from_alias(network_name)
        if text_dot:
            aliases.add(text_dot.split("text_encoders.", 1)[1])

    if network_name.startswith("llm_adapter_"):
        aliases.add(network_name.split("llm_adapter_", 1)[1])
        llm_dot = anima_dot_name_from_alias(network_name)
        if llm_dot:
            aliases.add(llm_dot)
            aliases.add(llm_dot.split("llm_adapter.", 1)[1])

    if "_llm_adapter_" in network_name and not network_name.startswith("llm_adapter_"):
        llm_name = "llm_adapter_" + network_name.split("_llm_adapter_", 1)[1]
        aliases.add(llm_name)
        aliases.add(f"diffusion_model_{llm_name}")
        aliases.add(f"model_diffusion_model_{llm_name}")
        llm_dot = anima_dot_name_from_alias(llm_name)
        if llm_dot:
            aliases.add(llm_dot)
            aliases.add(f"diffusion_model.{llm_dot}")
            aliases.add(f"model.diffusion_model.{llm_dot}")
            aliases.add(llm_dot.split("llm_adapter.", 1)[1])

    if network_name.startswith("model_layers_") or "_model_layers_" in network_name:
        if network_name.startswith("model_layers_"):
            model_layers_name = network_name
        else:
            model_layers_name = "model_layers_" + network_name.split("_model_layers_", 1)[1]

        aliases.add(model_layers_name)
        aliases.add(model_layers_name.split("model_", 1)[1])
        aliases.add(f"layers_{model_layers_name.split('model_layers_', 1)[1]}")

        model_dot = anima_dot_name_from_alias(network_name)
        if model_dot:
            aliases.add(model_dot)
            if ".model.layers." in model_dot:
                aliases.add("model.layers." + model_dot.split(".model.layers.", 1)[1])
                aliases.add("layers." + model_dot.split(".model.layers.", 1)[1])
            elif model_dot.startswith("model."):
                aliases.add(model_dot.split("model.", 1)[1])

    if network_name.startswith("layers_"):
        aliases.add(f"model_{network_name}")
        aliases.add(f"text_encoders_{network_name}")
        aliases.add(f"llm_adapter_{network_name}")

    return aliases

def find_anima_module(mapping, key):
    candidates = [key]
    dot_key = anima_dot_name_from_alias(key)
    if dot_key:
        candidates.append(dot_key)

    if "." in key:
        candidates.append(key.replace(".", "_"))
    else:
        underscored_dot = anima_dot_name_from_alias(key.replace(".", "_"))
        if underscored_dot:
            candidates.append(underscored_dot)

    if key.startswith("layers_"):
        candidates.append(f"model_{key}")
        candidates.append(f"text_encoders_{key}")
        candidates.append(f"llm_adapter_{key}")
    elif key.startswith("model_layers_"):
        candidates.append(key.split("model_", 1)[1])
    elif key.startswith("text_encoders_"):
        candidates.append(key.split("text_encoders_", 1)[1])
    elif key.startswith("llm_adapter_"):
        candidates.append(key.split("llm_adapter_", 1)[1])
    elif ".model.layers." in key:
        suffix = key.split(".model.layers.", 1)[1]
        candidates.append(f"model.layers.{suffix}")
        candidates.append(f"layers.{suffix}")
        candidates.append(f"model_layers_{suffix.replace('.', '_')}")
        candidates.append(f"layers_{suffix.replace('.', '_')}")
    elif key.startswith("text_encoders.") and ".transformer.model.layers." in key:
        suffix = key.split(".transformer.model.layers.", 1)[1]
        candidates.append(f"model.layers.{suffix}")
        candidates.append(f"layers.{suffix}")
        candidates.append(f"model_layers_{suffix.replace('.', '_')}")
        candidates.append(f"layers_{suffix.replace('.', '_')}")

    for candidate in candidates:
        for variant in anima_mlp_alias_variants(candidate):
            module = mapping.get(variant)
            if module is not None:
                return module, variant
        module = mapping.get(candidate)
        if module is not None:
            return module, candidate

    return None, key


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    if match(m, r"lora_te1_text_model_encoder_layers_(\d+)_(.+)"):
        return f"clip_l_transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te1_layers_(\d+)_(.+)"):
        return f"layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te3_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"t5xxl_transformer_text_model_encoder_layers_{m[0]}layer{m[1]}SelfAttention_k"
    
    match_pattern = re.match(r"lora_te3_encoder_block_(\d+)_layer_(\d+)_(.+)", key)
    if match_pattern:
        block_num, layer_num, suffix = match_pattern.groups()
        return f"t5xxl_transformer_encoder_block_{block_num}_layer_{layer_num}_{suffix}"

    if key.startswith("lora_unet_") and is_anima_lora_key(key):
        return "model_diffusion_model_" + key.split("lora_unet_", 1)[1]

    if key.startswith("oft_unet_") and (
        "blocks_" in key or "patch_embed" in key or "final_layer" in key or "t_embedder" in key
    ):
        return "model_diffusion_model_" + key.split("oft_unet_", 1)[1]

    if key.startswith("lora_te_") and (
        "text_encoders_" in key or "llm_adapter_" in key
    ):
        return key.split("lora_te_", 1)[1]

    if match(m, r"lora_te_layers_(\d+)_(.+)"):
        return f"layers_{m[0]}_{m[1]}"
    
    return key


def assign_network_names_to_compvis_modules(sd_model):
    network_layer_mapping = {}

    if shared.sd_model.is_sdxl:
        forges = [shared.sd_model.forge_objects.clip.cond_stage_model.clip_l, shared.sd_model.forge_objects.clip.cond_stage_model.clip_g]
        for i, embedder in enumerate(forges if forge else shared.sd_model.conditioner.embedders):
            if not forge and not hasattr(embedder, 'wrapped'):
                continue
                
            cmodules = embedder.named_modules() if forge else embedder.wrapped.named_modules()

            for name, module in cmodules:
                network_name = f'{i}_{name.replace(".", "_")}'
                if forge and i == 1:
                    network_name = network_name.replace("1_transformer_text_model_encoder_layers","1_model_transformer_resblocks")
                    network_name = network_name.replace("self_attn", "attn")
                    network_name = network_name.replace('mlp_fc2', 'mlp_c_proj')
                    network_name = network_name.replace('mlp_fc1', 'mlp_c_fc')
                register_network_alias(network_layer_mapping, network_name, module)
                module.network_layer_name = network_name

    else:
        cond_stage_model = shared.sd_model.forge_objects.clip.cond_stage_model if forge else getattr( shared.sd_model.cond_stage_model, 'wrapped', shared.sd_model.cond_stage_model)

        for name, module in cond_stage_model.named_modules():
            network_name = name.replace(".", "_")
            for alias in anima_aliases_from_module_name(network_name):
                register_network_alias(network_layer_mapping, alias, module)
            module.network_layer_name = network_name

    for name, module in shared.sd_model.forge_objects.unet.model.named_modules() if forge else shared.sd_model.model.named_modules():
        network_name = name.replace(".", "_")
        for alias in anima_aliases_from_module_name(network_name):
            register_network_alias(network_layer_mapping, alias, module)
        module.network_layer_name = network_name

    sd_model.network_layer_mapping = network_layer_mapping


class BundledTIHash(str):
    def __init__(self, hash_str):
        self.hash = hash_str

    def __str__(self):
        return self.hash if shared.opts.lora_bundled_ti_to_infotext else ''

def read_state_dict(filename):
    if forge:
        from backend.utils import load_torch_file
        return load_torch_file(filename)
    try:
        return sd_models.read_state_dict(filename,map_location = torch.device("cuda:0"))
    except:
        return sd_models.read_state_dict(filename)


def load_network(name, network_on_disk, isxl, is_sd2):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    sd = read_state_dict(network_on_disk.filename)

    #this should not be needed but is here as an emergency fix for an unknown error people are experiencing in 1.2.0
    if not hasattr(shared.sd_model, 'network_layer_mapping'):
        assign_network_names_to_compvis_modules(shared.sd_model)

    keys_failed_to_match = {}
    if hasattr(shared.sd_model, 'diffusers_weight_map'):
        diffusers_weight_map = shared.sd_model.diffusers_weight_map
    elif hasattr(shared.sd_model, 'diffusers_weight_mapping'):
        diffusers_weight_map = {}
        for k, v in shared.sd_model.diffusers_weight_mapping():
            diffusers_weight_map[k] = v
        shared.sd_model.diffusers_weight_map = diffusers_weight_map
    else:
        diffusers_weight_map = None

    matched_networks = {}
    bundle_embeddings = {}

    for key_network, weight in sd.items():

        if diffusers_weight_map:
            network_parts = key_network.rsplit(".", 2)
            if len(network_parts) == 3:
                key_network_without_network_parts, network_name, network_weight = network_parts
                network_part = network_name + '.' + network_weight
            elif len(network_parts) == 2:
                key_network_without_network_parts, network_part = network_parts
            else:
                key_network_without_network_parts = key_network
                network_part = ""
        else:
            network_part_suffixes = {
                "alpha", "scale", "bias", "diff", "diff_b",
                "w_norm", "b_norm",
                "lokr_w1", "lokr_w1_a", "lokr_w1_b",
                "lokr_w2", "lokr_w2_a", "lokr_w2_b",
                "lokr_t2",
            }
            if key_network.rsplit(".", 1)[-1] in network_part_suffixes:
                key_network_without_network_parts, network_part = key_network.rsplit(".", 1)
            else:
                network_parts = key_network.rsplit(".", 2)
                if len(network_parts) == 3:
                    key_network_without_network_parts, network_name, network_weight = network_parts
                    network_part = network_name + "." + network_weight
                elif len(network_parts) == 2:
                    key_network_without_network_parts, network_part = network_parts
                else:
                    key_network_without_network_parts = key_network
                    network_part = ""

        if key_network_without_network_parts == "bundle_emb":
            emb_name, vec_name = network_part.split(".", 1)
            emb_dict = bundle_embeddings.get(emb_name, {})
            if vec_name.split('.')[0] == 'string_to_param':
                _, k2 = vec_name.split('.', 1)
                emb_dict['string_to_param'] = {k2: weight}
            else:
                emb_dict[vec_name] = weight
            bundle_embeddings[emb_name] = emb_dict

        if diffusers_weight_map:
            key = diffusers_weight_map.get(key_network_without_network_parts, key_network_without_network_parts)
        else:
            key = convert_diffusers_name_to_compvis(key_network_without_network_parts, is_sd2)

        sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.network_layer_mapping.get(m.group(1), None)

        # SDXL loras seem to already have correct compvis keys, so only need to replace "lora_unet" with "diffusion_model"
        if sd_module is None and "lora_unet" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("lora_unet", "diffusion_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        elif sd_module is None and "lora_te1_text_model" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("lora_te1_text_model", "0_transformer_text_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

            # some SD1 Loras also have correct compvis keys
            if sd_module is None:
                key = key_network_without_network_parts.replace("lora_te1_text_model", "transformer_text_model")
                sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        # kohya_ss OFT module
        elif sd_module is None and "oft_unet" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("oft_unet", "diffusion_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        # KohakuBlueLeaf OFT module
        if sd_module is None and "oft_diag" in key:
            key = key_network_without_network_parts.replace("lora_unet", "diffusion_model")
            key = key_network_without_network_parts.replace("lora_te1_text_model", "0_transformer_text_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        if sd_module is None:
            sd_module, key = find_anima_module(shared.sd_model.network_layer_mapping, key)

        if sd_module is None:
            keys_failed_to_match[key_network] = key
            continue

        if key not in matched_networks:
            matched_networks[key] = network.NetworkWeights(network_key=key_network, sd_key=key, w={}, sd_module=sd_module)

        matched_networks[key].w[network_part] = weight

    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                break

        if net_module is None:
            raise AssertionError(f"Could not find a module type (out of {', '.join([x.__class__.__name__ for x in module_types])}) that would accept those keys: {', '.join(weights.w)}")

        net.modules[key] = net_module

    embeddings = {}
    for emb_name, data in bundle_embeddings.items():
        embedding = textual_inversion.create_embedding_from_data(data, emb_name, filename=network_on_disk.filename + "/" + emb_name)
        embedding.loaded = None
        embedding.shorthash = BundledTIHash(name)
        embeddings[emb_name] = embedding

    net.bundle_embeddings = embeddings

    if keys_failed_to_match:
        print(f"Network {network_on_disk.filename} didn't match keys: {keys_failed_to_match}")

    return net


def purge_networks_from_memory():
    while len(networks_in_memory) > shared.opts.lora_in_memory_limit and len(networks_in_memory) > 0:
        name = next(iter(networks_in_memory))
        networks_in_memory.pop(name, None)

    devices.torch_gc()


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None, isxl = False, isv2 = False):
    #emb_db = sd_hijack.model_hijack.embedding_db
    already_loaded = {}

    for net in loaded_networks:
        if net.name in names:
            already_loaded[net.name] = net
        # for emb_name, embedding in net.bundle_embeddings.items():
        #     if embedding.loaded:
        #         emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()

        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    failed_to_load_networks = []

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = already_loaded.get(name, None)

        if network_on_disk is not None:
            if net is None:
                net = networks_in_memory.get(name)

            if net is None or os.path.getmtime(network_on_disk.filename) > net.mtime:
                try:
                    net = load_network(name, network_on_disk,isxl, isv2)

                    networks_in_memory.pop(name, None)
                    networks_in_memory[name] = net
                except Exception as e:
                    errors.display(e, f"loading network {network_on_disk.filename}")
                    continue

            net.mentioned_name = name

            network_on_disk.read_hash()

        if net is None:
            failed_to_load_networks.append(name)
            logging.info(f"Couldn't find network with name {name}")
            continue

        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        print(f"dyn_dims = {dyn_dims}")
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        loaded_networks.append(net)

        # for emb_name, embedding in net.bundle_embeddings.items():
        #     if embedding.loaded is None and emb_name in emb_db.word_embeddings:
        #         continue

        #     embedding.loaded = False
        #     if emb_db.expected_shape == -1 or emb_db.expected_shape == embedding.shape:
        #         embedding.loaded = True
        #         emb_db.register_embedding(embedding, shared.sd_model)
        #     else:
        #         emb_db.skipped_embeddings[name] = embedding

    if failed_to_load_networks:
        lora_not_found_message = f'Lora not found: {", ".join(failed_to_load_networks)}'
        sd_hijack.model_hijack.comments.append(lora_not_found_message)
        if shared.opts.lora_not_found_warning_console:
            print(f'\n{lora_not_found_message}\n')
        if shared.opts.lora_not_found_gradio_warning:
            gr.Warning(lora_not_found_message)

    purge_networks_from_memory()


def allowed_layer_without_weight(layer):
    if isinstance(layer, torch.nn.LayerNorm) and not layer.elementwise_affine:
        return True

    return False


def store_weights_backup(weight):
    if weight is None:
        return None

    return weight.to(devices.cpu, copy=True)


def restore_weights_backup(obj, field, weight):
    if weight is None:
        setattr(obj, field, None)
        return

    getattr(obj, field).copy_(weight)


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return

    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            restore_weights_backup(self, 'in_proj_weight', weights_backup[0])
            restore_weights_backup(self.out_proj, 'weight', weights_backup[1])
        else:
            restore_weights_backup(self, 'weight', weights_backup)

    if isinstance(self, torch.nn.MultiheadAttention):
        restore_weights_backup(self.out_proj, 'bias', bias_backup)
    else:
        restore_weights_backup(self, 'bias', bias_backup)


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores original weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != ():
        if current_names != () and not allowed_layer_without_weight(self):
            raise RuntimeError(f"{network_layer_name} - no backup weights found and current weights are not unchanged")

        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (store_weights_backup(self.in_proj_weight), store_weights_backup(self.out_proj.weight))
        else:
            weights_backup = store_weights_backup(self.weight)

        self.network_weights_backup = weights_backup

    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None and wanted_names != ():
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_backup = store_weights_backup(self.out_proj.bias)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = store_weights_backup(self.bias)
        else:
            bias_backup = None

        # Unlike weight which always has value, some modules don't have bias.
        # Only report if bias is not None and current bias are not unchanged.
        if bias_backup is not None and current_names != ():
            raise RuntimeError("no backup bias found and current bias are not unchanged")

        self.network_bias_backup = bias_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)

        for net in loaded_networks:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight') and not isinstance(module, QkvLinear):
                try:
                    with torch.no_grad():
                        if getattr(self, 'fp16_weight', None) is None:
                            weight = self.weight
                            bias = self.bias
                        else:
                            weight = self.fp16_weight.clone().to(self.weight.device)
                            bias = getattr(self, 'fp16_bias', None)
                            if bias is not None:
                                bias = bias.clone().to(self.bias.device)
                        updown, ex_bias = module.calc_updown(weight)

                        if len(weight.shape) == 4 and weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))

                        self.weight.copy_((weight.to(dtype=updown.dtype) + updown).to(dtype=self.weight.dtype))
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias).to(self.weight.dtype)
                            else:
                                self.bias.copy_((bias + ex_bias).to(dtype=self.bias.dtype))
                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.in_proj_weight.chunk(3, 0)
                        updown_q, _ = module_q.calc_updown(qw)
                        updown_k, _ = module_k.calc_updown(kw)
                        updown_v, _ = module_v.calc_updown(vw)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight)

                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                    if ex_bias is not None:
                        if self.out_proj.bias is None:
                            self.out_proj.bias = torch.nn.Parameter(ex_bias)
                        else:
                            self.out_proj.bias += ex_bias

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if isinstance(self, QkvLinear) and module_q and module_k and module_v:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.weight.chunk(3, 0)
                        updown_q, _ = module_q.calc_updown(qw)
                        updown_k, _ = module_k.calc_updown(kw)
                        updown_v, _ = module_v.calc_updown(vw)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        self.weight += updown_qkv

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if module is None:
                continue

            logging.debug(f"Network {net.name} layer {network_layer_name}: couldn't find supported operation")
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

        self.network_current_names = wanted_names


def network_forward(org_module, input, original_forward):
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """

    if len(loaded_networks) == 0:
        return original_forward(org_module, input)

    input = devices.cond_cast_unet(input)

    network_restore_weights_from_backup(org_module)
    network_reset_cached_weight(org_module)

    y = original_forward(org_module, input)

    network_layer_name = getattr(org_module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue

        y = module.forward(input, y)

    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None


def network_Linear_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Linear_forward)

    network_apply_weights(self)

    return originals.Linear_forward(self, input)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Linear_load_state_dict(self, *args, **kwargs)


def network_Conv2d_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Conv2d_forward)

    network_apply_weights(self)

    return originals.Conv2d_forward(self, input)


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


def network_GroupNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.GroupNorm_forward)

    network_apply_weights(self)

    return originals.GroupNorm_forward(self, input)


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


def network_LayerNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.LayerNorm_forward)

    network_apply_weights(self)

    return originals.LayerNorm_forward(self, input)


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)

    return originals.MultiheadAttention_forward(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


def process_network_files(names: list[str] | None = None):
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    try:
        candidates += list(shared.walk_files(shared.cmd_opts.lyco_dir_backcompat, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    except:
        pass
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
