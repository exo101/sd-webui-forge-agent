import os
import torch
import safetensors.torch
import threading
from modules import shared, sd_models
import json

try:
  from modules import sd_models_xl
  xl = True
except:
  xl = False

try:
  from modules import sd_hijack
except (ImportError, ModuleNotFoundError):
  sd_hijack = None

def get_checkpoint_output_dir():
    ckpt_dir = getattr(shared.cmd_opts, "ckpt_dir", None)
    if ckpt_dir:
        return ckpt_dir

    ckpt_dirs = getattr(shared.cmd_opts, "ckpt_dirs", None)
    if ckpt_dirs:
        for path in ckpt_dirs:
            if path:
                return path

    return sd_models.model_path
  

def prune_model(model, isxl=False):
    keys = list(model.keys())
    base_prefix = "conditioner." if isxl else "cond_stage_model."
    for k in keys:
        if (
            "diffusion_model." not in k
            and "first_stage_model." not in k
            and base_prefix not in k
            and not k.startswith("text_encoders.")
            and not k.startswith("vae.")
            and not k.startswith("llm_adapter.")
        ):
            model.pop(k, None)
    return model

def to_half(sd):
    for key in sd.keys():
        if 'model' in key and sd[key].dtype in {torch.float32, torch.float64, torch.bfloat16}:
            sd[key] = sd[key].half()
    return sd

def should_save_anima_net_format(sd):
    has_anima_unet = any(
        key.startswith("net.blocks.")
        or key.startswith("model.diffusion_model.blocks.")
        for key in sd.keys()
    )
    has_llm_adapter = any(
        key.startswith("net.llm_adapter.")
        or key.startswith("model.diffusion_model.llm_adapter.")
        or key.startswith("llm_adapter.")
        or ".llm_adapter." in key
        for key in sd.keys()
    )
    return has_anima_unet and has_llm_adapter

def convert_anima_state_dict_to_net(sd):
    converted = {}
    diffusion_prefix = "model.diffusion_model."

    for key, value in sd.items():
        if key.startswith("net."):
            converted[key] = value
        elif key.startswith(diffusion_prefix):
            converted["net." + key[len(diffusion_prefix):]] = value
        elif key.startswith("llm_adapter."):
            converted["net." + key] = value

    return converted

def cast_state_dict_floating(sd, dtype):
    for key, value in list(sd.items()):
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            sd[key] = value.to(dtype=dtype)
    return sd

def resolve_anima_save_dtype(savesets):
    if "fp16" in savesets:
        return torch.float16
    if "float" in savesets or "float32" in savesets or "use float32" in savesets:
        return torch.float32
    return torch.bfloat16

def savemodel(state_dict,currentmodel,fname,savesets,metadata={}):
    other_dict = {}
    if state_dict is None:
        if shared.sd_model and shared.sd_model.sd_checkpoint_info:
            metadata = shared.sd_model.sd_checkpoint_info.metadata.copy()
        else:
            return "Current model is not a valid merged model"

        checkpoint_info = shared.sd_model.sd_checkpoint_info
        # check if current merged model is a fake checkpoint_info
        if checkpoint_info is not None:
            filename = checkpoint_info.filename
            name = os.path.basename(filename)
            info = sd_models.get_closet_checkpoint_match(name)
            if info == checkpoint_info:
                # this is a valid checkpoint_info
                # no need to save
                return "Current model is not a merged model or you've already saved model"

        # prepare metadata
        save_metadata = "save metadata" in savesets
        if save_metadata:
            metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])
        else:
            metadata = {"format": "pt"}

        if shared.sd_model is not None:
            print("load from shared.sd_model..")

            # restore textencoder
            sd_hijack.model_hijack.undo_hijack(shared.sd_model)

            for name,module in shared.sd_model.named_modules():
                if hasattr(module,"network_weights_backup"):
                    module = network_restore_weights_from_backup(module)

            state_dict = shared.sd_model.state_dict()
            for key in list(state_dict.keys()):
                if key in POPKEYS:
                    other_dict[key] = state_dict[key]
                    del state_dict[key]

            sd_hijack.model_hijack.hijack(shared.sd_model)
        else:
            return "No current loaded model found"

        # name_for_extra was set with the currentmodel
        currentmodel = checkpoint_info.name_for_extra

    if "fp16" in savesets:
        pre = ".fp16"
    else:pre = ""
    ext = ".safetensors" if "safetensors" in savesets else ".ckpt"

    # is it a inpainting or instruct-pix2pix2 model?
    if "model.diffusion_model.input_blocks.0.0.weight" in state_dict.keys():
        shape = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape
        if shape[1] == 9:
            pre += "-inpainting"
        if shape[1] == 8:
            pre += "-instruct-pix2pix"

    if not fname or fname == "":
        model_name = os.path.basename(currentmodel)
        model_name = model_name.replace(" ", "").replace(",", "_").replace("(", "_").replace(")", "_")
        fname = model_name + pre + ext
        if fname[0] == "_":
            fname = fname[1:]
    else:
        fname = fname if ext in fname else fname +pre+ext

    fname = os.path.join(get_checkpoint_output_dir(), fname)
    fname = fname.replace("ProgramFiles_x86_","Program Files (x86)")

    if len(fname) > 255:
       fname.replace(ext,"")
       fname=fname[:240]+ext

    # check if output file already exists
    if os.path.isfile(fname) and not "overwrite" in savesets:
        _err_msg = f"Output file ({fname}) existed and was not saved]"
        print(_err_msg)
        return _err_msg

    print("Saving...")
    if should_save_anima_net_format(state_dict):
        state_dict = convert_anima_state_dict_to_net(state_dict)
        target_dtype = resolve_anima_save_dtype(savesets)
        state_dict = cast_state_dict_floating(state_dict, target_dtype)
        print(f"Saving Anima checkpoint in base-compatible net.* format ({target_dtype}).")
    else:
        isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in state_dict
        if isxl:
            # prune share memory tensors, "cond_stage_model." prefixed base tensors are share memory with "conditioner." prefixed tensors
            for key in list(state_dict.keys()):
                if "cond_stage_model." in key:
                    del state_dict[key]

        if "fp16" in savesets:
            state_dict = to_half(state_dict)
        if "prune" in savesets:
            state_dict = prune_model(state_dict, isxl)

    # for safetensors contiguous error
    print("Check contiguous...")
    for key in state_dict.keys():
        v = state_dict[key]
        v = v.contiguous()
        state_dict[key] = v

    try:
      if ext == ".safetensors":
          safetensors.torch.save_file(state_dict, fname, metadata=metadata)
      else:
          torch.save(state_dict, fname)
    except Exception as e:
      print(f"ERROR: Couldn't saved:{fname},ERROR is {e}")
      return f"ERROR: Couldn't saved:{fname},ERROR is {e}"
    print("Done!")
    if other_dict:
        for key in other_dict.keys():
            state_dict[key] = other_dict[key]
        del other_dict
        if hasattr(sd_models, "load_model"):
            sd_models.load_model(checkpoint_info, already_loaded_state_dict=state_dict)
    return "Merged model saved in "+fname

def filenamecutter(name,model_a = False):
    if name =="" or name ==[]: return
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    name= os.path.splitext(checkpoint_info.filename)[0]

    if not model_a:
        name = os.path.basename(name)
    return name

from typing import Union

def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return self

    with torch.no_grad():
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight = torch.nn.Parameter(weights_backup[0].detach().requires_grad_(self.in_proj_weight.requires_grad))
                self.out_proj.weight = torch.nn.Parameter(weights_backup[1].detach().requires_grad_(self.out_proj.weight.requires_grad))
            else:
                self.weight = torch.nn.Parameter(weights_backup.detach().requires_grad_(self.weight.requires_grad))

        if bias_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.out_proj.bias = torch.nn.Parameter(bias_backup.detach().requires_grad_(self.out_proj.bias.requires_grad))
            else:
                self.bias = torch.nn.Parameter(bias_backup.detach().requires_grad_(self.bias.requires_grad))
        else:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.out_proj.bias = None
            else:
                self.bias = None
    return self

def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None

POPKEYS=[
"betas",
"alphas_cumprod",
"alphas_cumprod_prev",
"sqrt_alphas_cumprod",
"sqrt_one_minus_alphas_cumprod",
"log_one_minus_alphas_cumprod",
"sqrt_recip_alphas_cumprod",
"sqrt_recipm1_alphas_cumprod",
"posterior_variance",
"posterior_log_variance_clipped",
"posterior_mean_coef1",
"posterior_mean_coef2",
]
