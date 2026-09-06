"""Minimal See-through model loading and LayerDiff3D inference example.

This script loads the two model directories exported to a ModelScope model
repository. It runs the LayerDiff3D transparent-layer pass and then the
Marigold depth pass on the generated layers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("."))
    parser.add_argument("--layerdiff", type=Path)
    parser.add_argument("--marigold", type=Path)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="See-through/see-through source directory")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./output"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> tuple[Path, Path]:
    layerdiff = args.layerdiff or args.model_root / "seethroughv0.0.2_layerdiff3d"
    marigold = args.marigold or args.model_root / "seethroughv0.0.1_marigold"
    for path in (layerdiff, marigold):
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")
    return layerdiff.resolve(), marigold.resolve()


def main() -> None:
    args = parse_args()
    layerdiff_dir, marigold_dir = resolve_models(args)
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))

    import numpy as np
    import torch
    from PIL import Image
    from common.modules.layerdiffuse.diffusers_kdiffusion_sdxl import (
        KDiffusionStableDiffusionXLPipeline,
    )
    from common.modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from common.modules.layerdiffuse.vae import TransparentVAE
    from common.modules.marigold import MarigoldDepthPipeline
    from common.utils.cv import center_square_pad_resize

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu for a CPU test.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    args.output.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(args.input).convert("RGBA"))
    fullpage = center_square_pad_resize(image, args.resolution)

    print(f"Loading LayerDiff3D from {layerdiff_dir}")
    trans_vae = TransparentVAE.from_pretrained(layerdiff_dir, subfolder="trans_vae")
    layer_unet = UNetFrameConditionModel.from_pretrained(layerdiff_dir, subfolder="unet")
    layer_pipe = KDiffusionStableDiffusionXLPipeline.from_pretrained(
        layerdiff_dir, trans_vae=trans_vae, unet=layer_unet, scheduler=None
    )
    if args.cpu_offload and device.type == "cuda":
        layer_pipe.enable_model_cpu_offload()
    else:
        layer_pipe.to(device=device, dtype=dtype)

    tags = [
        "front hair", "back hair", "head", "neck", "neckwear",
        "topwear", "handwear", "bottomwear", "legwear", "footwear",
        "tail", "wings", "objects",
    ]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = layer_pipe(
        strength=1.0,
        num_inference_steps=args.steps,
        batch_size=1,
        generator=generator,
        guidance_scale=1.0,
        prompt=tags,
        negative_prompt="",
        fullpage=fullpage,
        group_index=0,
    )
    layer_dir = args.output / "layers"
    layer_dir.mkdir(exist_ok=True)
    for layer, tag in zip(result.images, tags):
        Image.fromarray(layer).save(layer_dir / f"{tag}.png")
    Image.fromarray(fullpage).save(layer_dir / "source.png")

    del layer_pipe, layer_unet, trans_vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading Marigold from {marigold_dir}")
    depth_unet = UNetFrameConditionModel.from_pretrained(marigold_dir, subfolder="unet")
    depth_pipe = MarigoldDepthPipeline.from_pretrained(marigold_dir, unet=depth_unet)
    if args.cpu_offload and device.type == "cuda":
        depth_pipe.enable_model_cpu_offload()
    else:
        depth_pipe.to(device=device, dtype=dtype)

    # The custom Marigold pipeline accepts a list of RGBA layer images.
    layer_images = [np.array(Image.open(p).convert("RGBA"))
                    for p in sorted(layer_dir.glob("*.png"))]
    depth_result = depth_pipe(
        color_map=None,
        show_progress_bar=True,
        img_list=layer_images,
    )
    depth = depth_result.depth_tensor.detach().float().cpu().numpy()
    np.save(args.output / "depth.npy", depth)
    print(f"Saved transparent layers to {layer_dir}")
    print(f"Saved depth tensor to {args.output / 'depth.npy'}")


if __name__ == "__main__":
    main()
