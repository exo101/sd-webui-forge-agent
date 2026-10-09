"""Run the exact ComfyUI SAM3.1 multiplex tracker for SCAIL-2 masks.

This worker deliberately runs under ComfyUI's own Python/runtime because its
checkpoint format and SAM3.1 model implementation are Comfy-specific.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PALETTE = (
    (0.0, 0.0, 1.0),  # blue
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 1.0, 0.0),  # yellow
)


def _render(track_data, background: str) -> np.ndarray:
    from comfy.ldm.sam3.tracker import unpack_masks

    frames = int(track_data.get("n_frames", 1))
    height, width = track_data["orig_size"]
    packed = track_data.get("packed_masks")
    bg = 1.0 if background == "white" else 0.0
    if packed is None or packed.shape[1] == 0:
        return np.full((frames, height, width, 3), bg, dtype=np.float32)

    masks = unpack_masks(packed)
    if tuple(masks.shape[-2:]) != (height, width):
        masks = torch.nn.functional.interpolate(
            masks.reshape(-1, 1, *masks.shape[-2:]).float(),
            size=(height, width),
            mode="nearest",
        ).reshape(masks.shape[0], masks.shape[1], height, width) > 0.5
    masks = masks.detach().to("cpu").numpy().astype(bool)
    # Match SCAIL2ColoredMask's left_to_right identity sorting.
    order_data = []
    for i, mask in enumerate(masks.transpose(1, 0, 2, 3)):
        present = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
        first = int(present[0]) if len(present) else len(masks)
        if len(present):
            ys, xs = np.where(mask[first])
            cx = float(xs.mean()) if len(xs) else 0.0
        else:
            cx = 0.0
        order_data.append((first, cx, i))
    order = [item[2] for item in sorted(order_data, key=lambda item: (item[0], item[1]))]
    masks = masks[:, order]

    out = np.full((masks.shape[0], height, width, 3), bg, dtype=np.float32)
    any_mask = masks.any(axis=1)
    indices = masks.argmax(axis=1)
    for obj_idx in range(masks.shape[1]):
        color = PALETTE[obj_idx % len(PALETTE)]
        selected = any_mask & (indices == obj_idx)
        out[selected] = color
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pose-prompt", default="human")
    parser.add_argument("--reference-prompt", default="girl, handkerchief")
    parser.add_argument("--replacement", action="store_true")
    args = parser.parse_args()

    comfy_root = Path(args.comfy_root).resolve()
    sys.path.insert(0, str(comfy_root))
    import comfy.model_management as model_management
    import comfy.sd
    from comfy.ldm.modules import attention as comfy_attention
    # This helper is launched outside ComfyUI's normal CLI bootstrap, so its
    # global backend selector may pick xFormers even when the installed build
    # cannot dispatch masked attention on Blackwell. Comfy's PyTorch SDPA path
    # is the supported fallback for these SAM3 text-attention tensors.
    comfy_attention.optimized_attention = comfy_attention.attention_pytorch
    comfy_attention.optimized_attention_masked = comfy_attention.attention_pytorch
    import comfy.ldm.sam3.detector as sam3_detector
    import comfy.ldm.sam3.sam as sam3_image
    import comfy.ldm.sam3.tracker as sam3_tracker
    sam3_detector.optimized_attention = comfy_attention.attention_pytorch
    sam3_image.optimized_attention = comfy_attention.attention_pytorch
    sam3_tracker.optimized_attention = comfy_attention.attention_pytorch

    source = np.load(args.input)
    pose = np.asarray(source["pose"], dtype=np.float32)
    reference = np.asarray(source["reference"], dtype=np.float32)
    if pose.ndim != 4 or pose.shape[-1] != 3 or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError(f"Expected pose [T,H,W,3] and reference [H,W,3], got {pose.shape}, {reference.shape}")

    device = model_management.get_torch_device()
    model_patcher, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
        args.checkpoint,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        output_model=True,
        model_options={"load_device": device, "offload_device": model_management.unet_offload_device()},
        te_model_options={"load_device": device, "offload_device": model_management.text_encoder_offload_device()},
    )
    if model_patcher is None or clip is None:
        raise RuntimeError("ComfyUI did not return the SAM3.1 model and its text encoder")

    model_management.load_models_gpu([model_patcher], force_full_load=True)
    tracker_model = model_patcher.model.diffusion_model
    dtype = model_patcher.model.get_dtype()

    def encode(text: str):
        encoded = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
        multi = encoded.get("sam3_multi_cond")
        if multi is not None:
            prompts = []
            for entry in multi:
                condition = entry["cond"].to(device=device, dtype=dtype)
                attention_mask = entry.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                prompts.append((condition, attention_mask, entry.get("max_detections", 1)))
            return prompts
        condition = encoded["cond"].to(device=device, dtype=dtype)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return [(condition, attention_mask, 1)]

    def track(images: np.ndarray, text: str):
        image_tensor = torch.from_numpy(images).to(device=device, dtype=dtype)
        image_tensor = image_tensor.movedim(-1, 1).contiguous()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        ):
            return tracker_model.forward_video(
                images=image_tensor,
                initial_masks=None,
                text_prompts=[(embedding, mask) for embedding, mask, _max_detections in encode(text)],
                new_det_thresh=0.5,
                max_objects=4,
                detect_interval=1,
                target_device=device,
                target_dtype=dtype,
            )

    pose_track = track(pose, args.pose_prompt)
    reference_track = track(reference[None], args.reference_prompt)
    pose_track["orig_size"] = tuple(pose.shape[1:3])
    pose_track["n_frames"] = int(pose.shape[0])
    reference_track["orig_size"] = tuple(reference.shape[:2])
    reference_track["n_frames"] = 1
    pose_masks = _render(pose_track, "white" if args.replacement else "black")
    reference_masks = _render(reference_track, "black" if args.replacement else "white")
    np.savez(args.output, pose_mask=pose_masks, reference_mask=reference_masks)
    print(f"Comfy SAM3.1 masks ready: pose={pose_masks.shape}, reference={reference_masks.shape}", flush=True)


if __name__ == "__main__":
    main()
