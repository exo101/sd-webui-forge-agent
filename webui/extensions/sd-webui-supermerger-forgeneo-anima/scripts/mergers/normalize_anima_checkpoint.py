import argparse
import os
from pathlib import Path
import sys

import safetensors.torch
import torch
from safetensors import safe_open


DIFFUSION_PREFIX = "model.diffusion_model."


def output_key_for_anima(key):
    if key.startswith(DIFFUSION_PREFIX):
        return "net." + key[len(DIFFUSION_PREFIX):], 3
    if key.startswith("llm_adapter."):
        return "net." + key, 3
    if key.startswith("net."):
        return key, 2
    if ".llm_adapter." in key and (
        key.startswith("text_encoders.") or key.startswith("cond_stage_model.")
    ):
        _, _, suffix = key.partition(".llm_adapter.")
        if suffix:
            return "net.llm_adapter." + suffix, 1
    return None, 0


def cast_tensor(tensor, dtype_name):
    if dtype_name == "keep" or not tensor.is_floating_point():
        return tensor
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return tensor.to(dtype=dtype)


def default_output_path(input_path, dtype_name):
    source = Path(input_path)
    suffix = ".net" if dtype_name == "keep" else f".net.{dtype_name}"
    return source.with_name(source.stem + suffix + source.suffix)


def make_output_path(input_path, output_arg, output_dir, dtype_name, multiple):
    if output_arg:
        if multiple:
            raise ValueError("--output can be used only when normalizing a single file. Use --output-dir for multiple files.")
        return output_arg

    output_path = default_output_path(input_path, dtype_name)
    if output_dir:
        output_path = Path(output_dir) / output_path.name
    return str(output_path)


def select_input_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"Could not open file dialog: {exc}") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    files = filedialog.askopenfilenames(
        title="Select Anima checkpoints to normalize",
        filetypes=(("Safetensors checkpoints", "*.safetensors"), ("All files", "*.*")),
    )
    root.destroy()
    return list(files)


def normalize_anima_checkpoint(input_path, output_path, dtype_name, overwrite):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    tensors = {}
    priorities = {}
    skipped = 0

    with safe_open(input_path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        for key in source.keys():
            output_key, priority = output_key_for_anima(key)
            if output_key is None:
                skipped += 1
                continue
            if output_key in priorities and priorities[output_key] > priority:
                skipped += 1
                continue

            tensor = cast_tensor(source.get_tensor(key), dtype_name).contiguous()
            tensors[output_key] = tensor
            priorities[output_key] = priority

    if not tensors:
        raise ValueError("No Anima net-compatible tensors were found.")
    if not any(key.startswith("net.blocks.") for key in tensors):
        raise ValueError("No net.blocks.* tensors were found. This does not look like an Anima checkpoint.")
    if not any(key.startswith("net.llm_adapter.") for key in tensors):
        raise ValueError("No net.llm_adapter.* tensors were found. This does not look like a complete Anima checkpoint.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = {str(k): str(v) for k, v in metadata.items()}
    safetensors.torch.save_file(tensors, output_path, metadata=metadata)

    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"Saved: {output_path}")
    print(f"Kept tensors: {len(tensors)}")
    print(f"Skipped tensors: {skipped}")
    print(f"Tensor bytes: {tensor_bytes:,}")


def main():
    parser = argparse.ArgumentParser(description="Normalize a Forge-internal Anima checkpoint to base-compatible net.* format.")
    parser.add_argument("input", nargs="*", help="Input .safetensors checkpoint(s)")
    parser.add_argument("-o", "--output", help="Output .safetensors checkpoint")
    parser.add_argument("--output-dir", help="Directory for normalized checkpoints")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "keep"), default="bf16", help="Output floating-point dtype")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    parser.add_argument("--dialog", action="store_true", help="Select one or more input checkpoints with a file dialog")
    args = parser.parse_args()

    inputs = list(args.input)
    if args.dialog or not inputs:
        inputs.extend(select_input_files())

    if not inputs:
        print("No input files selected.")
        return 1

    failed = 0
    multiple = len(inputs) > 1
    for input_path in inputs:
        try:
            output = make_output_path(input_path, args.output, args.output_dir, args.dtype, multiple)
            normalize_anima_checkpoint(input_path, output, args.dtype, args.overwrite)
        except Exception as exc:
            failed += 1
            print(f"ERROR: {input_path}: {exc}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
