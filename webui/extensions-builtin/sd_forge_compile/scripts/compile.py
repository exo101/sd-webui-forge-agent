# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py

import logging
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.args import args as cmd_args
from backend.logging import setup_logger
from backend.utils import get_attr, set_attr_raw
from modules import scripts

try:
    import triton
except ImportError:
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True

_COMPILE_CONFIG_KEY = "_torch_compile_config"
_ORIG_APPLY_KEY = "_orig_apply_model"

logger = logging.getLogger("compile")
setup_logger(logger)


def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class TorchCompileForForge(scripts.Script):
    sorting_priority = 99999

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

    def title(self):
        return "Torch 编译集成"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if TRITON_AVAILABLE else None

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            preset = gr.Dropdown(
                label="预设",
                value="自动",
                choices=[
                    "自动",
                    "禁用",
                    "guard_filter_fn",
                    "dynamic",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                    "reduce-overhead",
                ],
                info='"自动"保持当前编译状态',
            )

            _dynamic = "支持任意分辨率/批次大小"
            _indynamic = "更改分辨率/批次大小需要重新编译"
            _no_malloc = "不支持 --cuda-malloc 参数"

            gr.Markdown(rf"""
**torch.compile** 通过提前编译模型来加速推理
- **guard_filter_fn:** 编译速度最快 ; {_indynamic}
- **dynamic:** {_dynamic} ; 编译速度较慢
- **max-autotune:** 运行时速度最佳 ; {_indynamic} ; {_no_malloc}
- **max-autotune-no-cudagraphs:** {_dynamic} ; 比 **dynamic** 更快 ; 编译速度更慢
- **reduce-overhead:** 类似于 **max-autotune** ; {_indynamic} ; {_no_malloc}
            """)

        return [preset]

    def process_batch(self, p, preset: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        prev_config: tuple[str] = getattr(kmodel, _COMPILE_CONFIG_KEY, None)
        
        preset_map = {
            "自动": "Automatic",
            "禁用": "Disable",
            "guard_filter_fn": "guard_filter_fn",
            "dynamic": "dynamic",
            "max-autotune": "max-autotune",
            "max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
            "reduce-overhead": "reduce-overhead",
        }
        preset_key = preset_map.get(preset, preset)
        
        enable: bool = (prev_config is not None) if preset_key == "Automatic" else (preset_key != "Disable")

        if not enable:
            self._remove_compile_wrapper(kmodel)
            return

        if preset_key in ("max-autotune", "reduce-overhead") and cmd_args.cuda_malloc:
            logger.error(f"{preset_key} does not support --cuda-malloc\nModel is not compiled...")
            return

        _config: tuple[str] = (preset_key,)
        if _config == prev_config:
            return

        setattr(kmodel, _COMPILE_CONFIG_KEY, _config)

        if prev_config is not None:
            self._remove_compile_wrapper(kmodel)

        match preset_key:
            case "guard_filter_fn":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "dynamic":
                config = dict(backend="inductor", dynamic=True, fullgraph=False)
            case "max-autotune":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"coordinate_descent_tuning": True, "max_autotune": True, "triton.cudagraphs": True})
            case "max-autotune-no-cudagraphs":
                config = dict(backend="inductor", dynamic=True, fullgraph=False, options={"coordinate_descent_tuning": True, "max_autotune": True})
            case "reduce-overhead":
                config = dict(backend="inductor", mode="reduce-overhead", dynamic=False, fullgraph=False)

        self._wrap_apply_model(kmodel, config)

        logger.info(f"Model Compiled ({preset_key})")

    @staticmethod
    def _wrap_apply_model(kmodel: "KModel", compile_config: dict):
        setattr(kmodel, _ORIG_APPLY_KEY, kmodel.apply_model)

        @wraps(kmodel._orig_apply_model)
        def apply_model_with_compile(*args, **kwargs):
            orig_model = get_attr(kmodel, "diffusion_model")
            compiled = torch.compile(orig_model, **compile_config)
            set_attr_raw(kmodel, "diffusion_model", compiled)
            try:
                return kmodel._orig_apply_model(*args, **kwargs)
            finally:
                set_attr_raw(kmodel, "diffusion_model", orig_model)

        kmodel.apply_model = apply_model_with_compile

    @staticmethod
    def _remove_compile_wrapper(kmodel: "KModel"):
        if (orig := getattr(kmodel, _ORIG_APPLY_KEY, None)) is not None:
            kmodel.apply_model = orig
            delattr(kmodel, _ORIG_APPLY_KEY)
            delattr(kmodel, _COMPILE_CONFIG_KEY)
            logger.info("Model Decompiled")
