from __future__ import annotations

import random
import sys
from pathlib import Path

_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

import gradio as gr
from PIL import Image

from modules import scripts_postprocessing

from seedvr2_forge.config import (
    SeedVR2Config,
    dit_dropdown_choices,
    models_root,
    vae_dropdown_choices,
)
from seedvr2_forge.engine import SeedVR2ForgeEngine
from seedvr2_forge.image_utils import resize_to_long_edge
from seedvr2_forge.memory import unload_forge_models
from seedvr2_forge.tiling import merge_tiles, split_image


class SeedVR2PostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "SeedVR2"
    # Run before Forge's built-in Upscale (order=1000). In exclusive mode we
    # stop later postprocessors after the final SeedVR2 image is produced.
    order = 100

    def ui(self):
        with gr.Accordion("SeedVR2 高清放大", open=False):
            enabled = gr.Checkbox(label="启用 SeedVR2", value=False)

            gr.Markdown(
                "### 基础设置\n"
                "普通使用只需要选择 **模型** 和 **放大倍率**。其余参数可以保持默认。"
            )

            dit_model = gr.Dropdown(
                label="SeedVR2 模型",
                choices=dit_dropdown_choices(),
                value="seedvr2_ema_3b_fp16.safetensors",
                allow_custom_value=True,
                info="自动扫描 models/SEEDVR2。FP16质量最高；FP8更省显存；GGUF最省显存。",
            )

            with gr.Row():
                scale_mode = gr.Dropdown(
                    label="最终输出倍率",
                    choices=[
                        ("2× · 推荐", "2x"),
                        ("3× · 更高清", "3x"),
                        ("4× · 超高清", "4x"),
                        ("6x · 极致高清", "6x"),
                        ("自定义长边", "custom"),
                    ],
                    value="2x",
                    info="例如原图 1024×1536，选择 2× 后约为 2048×3072，并自动保持宽高比。",
                )
                custom_long_edge = gr.Slider(
                    label="自定义最长边（仅选择‘自定义长边’时生效）",
                    minimum=1024,
                    maximum=12288,
                    step=64,
                    value=4096,
                )

            only_final_output = gr.Checkbox(
                label="仅输出最终 SeedVR2 图像",
                value=True,
                info="推荐保持开启。开启后 SeedVR2 会作为独占后处理器，跳过 Forge Upscale 等后续处理，避免重复放大或多余尺寸。",
            )

            gr.Markdown(
                "**模型怎么选：** 24GB 优先 FP16；想少占显存可直接选 FP8；"
                "显存更紧张再考虑 GGUF。VAE 默认保持 FP16 即可。\n\n"
                "**提示：** 开启“仅输出最终 SeedVR2 图像”后，不需要再开启 Forge 自带的 Upscale。"
            )

            # Everything below is intentionally hidden from beginners.
            with gr.Accordion("高级设置（不了解可完全不动）", open=False):
                gr.Markdown(f"模型目录：`{models_root()}`")

                vae_model = gr.Dropdown(
                    label="VAE 模型",
                    choices=vae_dropdown_choices(),
                    value="ema_vae_fp16.safetensors",
                    allow_custom_value=True,
                )

                with gr.Row():
                    attention_mode = gr.Dropdown(
                        label="Attention",
                        choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                        value="sdpa",
                    )
                    device = gr.Textbox(label="推理设备", value="cuda:0")

                with gr.Row():
                    tile_size = gr.Slider(
                        label="外部分块尺寸",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1024,
                        info="主要影响显存和接缝；24G 建议保持 1024。",
                    )
                    tile_padding = gr.Slider(
                        label="拼接融合宽度",
                        minimum=0,
                        maximum=256,
                        step=8,
                        value=64,
                        info="建议保持 64。",
                    )

                with gr.Row():
                    color_correction = gr.Dropdown(
                        label="色彩校正",
                        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                        value="lab",
                    )
                    batch_size = gr.Dropdown(
                        label="Frame batch（单图保持 1）",
                        choices=[1, 5, 9, 13],
                        value=1,
                    )
                    seed = gr.Number(label="Seed（-1 = 随机）", value=-1, precision=0)

                with gr.Row():
                    input_noise_scale = gr.Slider(label="Input noise", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                    latent_noise_scale = gr.Slider(label="Latent noise", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

                with gr.Accordion("显存 / Offload", open=False):
                    with gr.Row():
                        dit_offload_device = gr.Dropdown(label="DiT offload", choices=["none", "cpu"], value="none")
                        vae_offload_device = gr.Dropdown(label="VAE offload", choices=["none", "cpu"], value="none")
                        tensor_offload_device = gr.Dropdown(label="中间张量 offload", choices=["cpu", "none"], value="cpu")

                    with gr.Row():
                        blocks_to_swap = gr.Slider(label="Blocks to swap", minimum=0, maximum=36, step=1, value=0)
                        swap_io_components = gr.Checkbox(label="Swap I/O components", value=False)

                    with gr.Row():
                        unload_forge_model_first = gr.Checkbox(label="运行前释放 Forge 模型显存", value=True)
                        keep_model_cached = gr.Checkbox(label="缓存 SeedVR2 模型", value=False)
                        debug = gr.Checkbox(label="详细日志", value=False)

                    target_free_vram_gb = gr.Slider(
                        label="加载 SeedVR2 前请求空闲显存（GB）",
                        minimum=8,
                        maximum=24,
                        step=1,
                        value=18,
                    )

                with gr.Accordion("VAE 分块 / 低显存", open=False):
                    with gr.Row():
                        encode_tiled = gr.Checkbox(label="VAE tiled encode", value=False)
                        encode_tile_size = gr.Slider(label="Encode tile", minimum=256, maximum=2048, step=64, value=1024)
                        encode_tile_overlap = gr.Slider(label="Encode overlap", minimum=0, maximum=512, step=16, value=128)
                    with gr.Row():
                        decode_tiled = gr.Checkbox(label="VAE tiled decode", value=False)
                        decode_tile_size = gr.Slider(label="Decode tile", minimum=256, maximum=2048, step=64, value=1024)
                        decode_tile_overlap = gr.Slider(label="Decode overlap", minimum=0, maximum=512, step=16, value=128)

        return {
            "enabled": enabled,
            "dit_model": dit_model,
            "vae_model": vae_model,
            "scale_mode": scale_mode,
            "custom_long_edge": custom_long_edge,
            "only_final_output": only_final_output,
            "attention_mode": attention_mode,
            "device": device,
            "dit_offload_device": dit_offload_device,
            "vae_offload_device": vae_offload_device,
            "tensor_offload_device": tensor_offload_device,
            "tile_size": tile_size,
            "tile_padding": tile_padding,
            "color_correction": color_correction,
            "batch_size": batch_size,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "input_noise_scale": input_noise_scale,
            "latent_noise_scale": latent_noise_scale,
            "seed": seed,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "unload_forge_model_first": unload_forge_model_first,
            "keep_model_cached": keep_model_cached,
            "target_free_vram_gb": target_free_vram_gb,
            "debug": debug,
        }

    def process_firstpass(self, pp, **kwargs):
        """Preserve the true input before any other postprocessor can resize it.

        Forge runs process_firstpass() for every script before running any process().
        Keeping a private copy here makes the requested SeedVR2倍率 relative to the
        user's actual input even if the user has manually reordered Forge Upscale
        ahead of this extension.
        """
        if not kwargs.get("enabled", False):
            return
        pp._seedvr2_original_image = pp.image.copy()
        pp._seedvr2_original_size = pp.image.size

    def process(self, pp, **kwargs):
        if not kwargs.get("enabled", False):
            return

        for key in (
            "custom_long_edge", "tile_size", "tile_padding", "batch_size",
            "blocks_to_swap", "target_free_vram_gb", "encode_tile_size",
            "encode_tile_overlap", "decode_tile_size", "decode_tile_overlap", "seed",
        ):
            kwargs[key] = int(kwargs[key])

        if kwargs["batch_size"] not in (1, 5, 9, 13):
            raise ValueError("SeedVR2 batch size must be 1, 5, 9, 13, ...")
        if kwargs["seed"] < 0:
            kwargs["seed"] = random.randint(0, 2**31 - 1)

        cfg = SeedVR2Config(**kwargs)

        if cfg.unload_forge_model_first:
            print("[SeedVR2 Forge] " + unload_forge_models(cfg.target_free_vram_gb))

        original = getattr(pp, "_seedvr2_original_image", pp.image)
        source: Image.Image = original.convert("RGB")
        target_long_edge = cfg.resolve_target_long_edge(source.size)
        resized = resize_to_long_edge(source, target_long_edge, method="lanczos")
        layout = split_image(resized, tile_size=cfg.tile_size, tile_padding=cfg.tile_padding)

        src_long = max(source.size)
        actual_scale = target_long_edge / float(src_long)
        print(
            f"[SeedVR2 Forge] model={cfg.dit_model}; scale={actual_scale:.2f}x; "
            f"{source.size} -> {resized.size}; grid={layout.grid_size}, tiles={len(layout.tiles)}, "
            f"tile={cfg.tile_size}, assembly_padding={cfg.tile_padding}"
        )

        engine = SeedVR2ForgeEngine(cfg)
        try:
            processed_tiles = engine.process_tiles([tile.image for tile in layout.tiles])
            pp.image = merge_tiles(layout, processed_tiles)

            if cfg.only_final_output:
                # SeedVR2 already owns the final target resolution. Do not let Forge's
                # built-in Upscale or later postprocessors run again on this image.
                # Also discard any extra images attached to this pp object.
                if hasattr(pp, "extra_images") and isinstance(pp.extra_images, list):
                    pp.extra_images.clear()
                pp.disable_processing = True

                # Remove stale metadata that may have been written by an Upscale
                # operation placed before SeedVR2 in a custom processing order.
                if hasattr(pp, "info") and isinstance(pp.info, dict):
                    for key in list(pp.info):
                        if key.startswith("Postprocess upscale") or key in {"Max side length", "Postprocess crop to"}:
                            pp.info.pop(key, None)

            if hasattr(pp, "nametags") and isinstance(pp.nametags, list):
                pp.nametags.append("seedvr2")
            if hasattr(pp, "info") and isinstance(pp.info, dict):
                pp.info["SeedVR2"] = self._build_info(
                    cfg, engine.upstream_version, len(layout.tiles), layout.grid_size,
                    source.size, resized.size, actual_scale,
                )
        finally:
            engine.cleanup()

    @staticmethod
    def _build_info(cfg, upstream_version, tile_count, grid_size, original_size, resized_size, actual_scale):
        return (
            f"v{upstream_version}; model={cfg.dit_model}; vae={cfg.vae_model}; "
            f"scale={actual_scale:.3f}x; orig={original_size[0]}x{original_size[1]}; "
            f"output={resized_size[0]}x{resized_size[1]}; grid={grid_size[0]}x{grid_size[1]}; "
            f"tiles={tile_count}; tile={cfg.tile_size}; padding={cfg.tile_padding}; "
            f"attention={cfg.attention_mode}; color={cfg.color_correction}; "
            f"batch={cfg.batch_size}; blocks_to_swap={cfg.blocks_to_swap}; seed={cfg.seed}"
        )
