# sd-webui-seedvr2 for Forge-Neo

SeedVR2 image restoration / upscaling extension for **SD WebUI Forge Neo**.

This extension integrates the official SeedVR2 core directly into Forge-Neo's post-processing pipeline. It does **not** require ComfyUI or ComfyUI custom nodes.

> Current plugin version: **v0.3.1**

---

## Features

- Direct SeedVR2 integration in Forge-Neo post-processing
- Automatic SeedVR2 model discovery
- FP16 / FP8 / GGUF DiT model selection
- Beginner-friendly **2× / 3× / 4×** upscale presets
- Custom target long-edge resolution
- Original ~6K workflow compatibility preset
- Automatic aspect-ratio preservation
- Automatic image tiling and feather blending
- Optional Forge model unloading before SeedVR2 processing
- Separate DiT / VAE / intermediate-tensor offload controls
- VAE tiled encode/decode options for lower VRAM usage
- Advanced settings hidden by default

---

## Quick Start

For normal use, you only need to change **two settings**:

1. **SeedVR2 Model**
2. **Upscale Size**

Everything else can stay at its default value.

Example:

```text
Input: 1024 × 1536

2×  → 2048 × 3072
3×  → 3072 × 4608
4×  → 4096 × 6144
```

The aspect ratio is preserved automatically.

### Which upscale mode should I use?

| Mode | Recommended use |
|---|---|
| **2×** | Recommended default. Good balance of quality, speed and VRAM |
| **3×** | Higher-resolution output when 2× is not enough |
| **4×** | Very large output; requires more processing time and memory |
| **~6K Long Edge** | Compatibility with the original ComfyUI workflow |
| **Custom Long Edge** | Enter the exact maximum edge you want, e.g. 4096 / 6144 / 8192 |

The original ComfyUI workflow used:

```text
6 × 1024 - 64 = 6080 px
```

The **~6K Long Edge** preset keeps this behavior automatically. You no longer need to calculate tile counts manually.

---

## Model Support

The DiT model is **not hardcoded to FP16**.

The extension scans:

```text
<Forge root>/models/SEEDVR2/
```

and lists all detected `.safetensors` and `.gguf` models in the UI.

Typical examples:

```text
seedvr2_ema_3b_fp16.safetensors
seedvr2_ema_3b_fp8_e4m3fn.safetensors
seedvr2_ema_3b-Q8_0.gguf
seedvr2_ema_3b-Q4_K_M.gguf
ema_vae_fp16.safetensors
```

### Model selection guide

| Model type | VRAM usage | Recommendation |
|---|---:|---|
| **FP16** | Highest | Best default for GPUs with enough VRAM |
| **FP8** | Lower | Recommended when FP16 is too memory-heavy |
| **GGUF Q8** | Lower | Good low-VRAM alternative |
| **GGUF Q4** | Lowest | Use when VRAM is very limited |

For most users, keep the VAE as:

```text
ema_vae_fp16.safetensors
```

### RTX 3090 24GB

Recommended starting point:

```text
DiT Model: seedvr2_ema_3b_fp16.safetensors
VAE: ema_vae_fp16.safetensors
Upscale: 2×
Attention: SDPA
Tile Size: 1024
Blend Padding: 64
Color Correction: LAB
Frame Batch: 1
BlockSwap: 0
Intermediate Tensor Offload: CPU
Unload Forge Model Before SeedVR2: Enabled
Keep SeedVR2 Model Cached: Disabled
```

If FP16 causes an out-of-memory error, try **FP8 first** before changing BlockSwap or VAE tiling.

---

## Installation

Extract or clone the extension into:

```text
Forge-Neo/
└── extensions/
    └── sd-webui-seedvr2/
```

The final structure should look like:

```text
sd-webui-seedvr2/
├── install.py
├── requirements.txt
├── README.md
├── scripts/
│   └── seedvr2_postprocessing.py
├── seedvr2_forge/
└── vendor/
```

Do **not** create an extra nested folder such as:

```text
extensions/sd-webui-seedvr2/sd-webui-seedvr2/
```

Restart Forge-Neo after installation.

### Git installation

From your Forge-Neo `extensions` directory:

```bash
git clone <your-repository-url> sd-webui-seedvr2
```

Then restart Forge-Neo.

---

## Model Directory

Create the following directory if it does not already exist:

```text
Forge-Neo/
└── models/
    └── SEEDVR2/
```

Example:

```text
models/SEEDVR2/
├── seedvr2_ema_3b_fp16.safetensors
├── seedvr2_ema_3b_fp8_e4m3fn.safetensors
├── seedvr2_ema_3b-Q8_0.gguf
└── ema_vae_fp16.safetensors
```

You do **not** need every model. Only place the models you actually intend to use.

The model dropdown is populated when Forge builds the extension UI. If you add a new model while Forge is already running, restart Forge-Neo so the list is refreshed.

---

## Basic UI

The default interface is intentionally simple:

```text
SeedVR2 高清放大

[✓] Enable SeedVR2

SeedVR2 Model
[ 3B FP16 / FP8 / GGUF ... ]

Upscale Size
[ 2× / 3× / 4× / ~6K / Custom ]

Advanced Settings ▼
```

Beginners normally do **not** need to open Advanced Settings.

---

## Advanced Settings

Advanced controls include:

### Processing

- Attention mode
- Device
- Tile size
- Blend padding
- Color correction
- Frame batch
- Seed
- Input noise
- Latent noise

### VRAM / Offload

- DiT offload device
- VAE offload device
- Intermediate tensor offload
- BlockSwap
- Swap I/O components
- Unload Forge model before SeedVR2
- Keep SeedVR2 model cached
- Target free VRAM

### VAE Tiling

- VAE tiled encode
- Encode tile size / overlap
- VAE tiled decode
- Decode tile size / overlap

Unless you already understand SeedVR2 memory management, leave these values at their defaults.

---

## How Processing Works

```text
Input Image
    ↓
Calculate Target Resolution
    ↓
Lanczos Pre-Resize
    ↓
Automatic Tile Split
    ↓
SeedVR2 Restoration
    ↓
Feather Tile Blending
    ↓
Final Image
```

The external tiling layer allows large images to be processed without sending the entire final-resolution image through SeedVR2 at once.

---

## VRAM Management

Large Forge checkpoints and SeedVR2 can easily compete for GPU memory.

For that reason, the extension enables this option by default:

```text
Unload Forge model before SeedVR2 = ON
```

Before SeedVR2 starts, the plugin asks Forge's own memory manager to release GPU memory, then performs CUDA cache cleanup.

Default behavior also keeps:

```text
Keep SeedVR2 Model Cached = OFF
```

This prioritizes stability over repeated-run loading speed.

### If you get CUDA Out of Memory

Try these steps in order:

1. Change the DiT model from **FP16 → FP8**.
2. Keep **Unload Forge model before SeedVR2** enabled.
3. Reduce **Tile Size** from `1024` to `768` or `512`.
4. Enable **VAE tiled encode/decode**.
5. Only then consider **BlockSwap**.

Avoid changing several memory settings at once; otherwise it becomes difficult to identify which setting solved the problem.

---

## Dependencies

The installer intentionally does **not** automatically replace or upgrade Forge's:

```text
torch
torchvision
numpy
```

Replacing these packages can break an otherwise working Forge installation.

The installer only adds missing SeedVR2-related dependencies where possible.

The official SeedVR2 core source used by the adapter is bootstrapped into:

```text
extensions/sd-webui-seedvr2/vendor/seedvr2/
```

If automatic source download fails because GitHub is unavailable, the source can be copied manually into that directory.

---

## Current Scope

v0.3.0 currently focuses on:

> **Single-image SeedVR2 restoration and upscaling inside Forge-Neo.**

Video processing is not the primary target of this release.

---

## Troubleshooting

### SeedVR2 model does not appear in the dropdown

Check that the model is inside:

```text
models/SEEDVR2/
```

and restart Forge-Neo.

### `SeedVR2 model not found`

The selected filename no longer exists in the model directory. Re-select the model or restore the file.

### First startup cannot download SeedVR2 source

Check your GitHub/network access. You can also manually copy the supported SeedVR2 source tree into:

```text
extensions/sd-webui-seedvr2/vendor/seedvr2/
```

### Forge runs out of VRAM after image generation

Keep:

```text
Unload Forge model before SeedVR2 = Enabled
```

Then try FP8.

### Image seams are visible

Try keeping the defaults first:

```text
Tile Size: 1024
Blend Padding: 64
```

If you change tile size, avoid setting blend padding to zero unless you are intentionally testing raw tile boundaries.

---

## Changelog — v0.3.1

- **Final-output-only mode is now enabled by default**
- SeedVR2 now runs before Forge's built-in Upscale by default
- When final-output-only mode is enabled, later Forge postprocessors are skipped to prevent a second resize
- The original input image is preserved during Forge's first-pass stage, so SeedVR2倍率 is calculated from the true source image even when the user customizes postprocessing order
- Stale Forge Upscale metadata is removed from the final SeedVR2 result
- Renamed the basic control from “放大倍率 / 输出尺寸” to **“最终输出倍率”**
- Added a clear UI hint that Forge Upscale is not required when SeedVR2 handles the final output

### v0.3.0

- Added automatic SeedVR2 model scanning
- Added FP16 / FP8 / GGUF model selection
- Added readable model labels
- Added 2× / 3× / 4× upscale presets
- Added custom long-edge output mode
- Added original ~6K workflow compatibility mode
- Simplified the main UI for beginners
- Moved technical parameters into Advanced Settings
- Improved automatic tile layout
- Improved feather tile blending
- Improved Forge-Neo VRAM cleanup behavior
- Separated DiT / VAE / intermediate tensor offload controls
- Added VAE tiled encode/decode controls
- Improved model-path validation
- Improved configuration validation and error handling

---

## Credits

This extension is a Forge-Neo integration layer for **SeedVR2**. SeedVR2 itself and its model weights belong to their respective authors and upstream projects.

Please follow the upstream project's license and model license when redistributing code or weights.

---

## Feedback

Bug reports, compatibility reports, feature requests and pull requests are welcome.

When reporting an issue, please include:

- GPU model and VRAM
- Forge-Neo version / commit
- SeedVR2 DiT model filename
- VAE filename
- Upscale mode
- Relevant console log / traceback

## Why do I get two different output sizes?

Starting with **v0.3.1**, keep **“仅输出最终 SeedVR2 图像 / Final SeedVR2 output only”** enabled.

SeedVR2 will then use the original input image, produce the selected final resolution, and stop Forge's later Upscale/postprocessing chain. You do **not** need to enable Forge's built-in Upscale at the same time.

Example:

```text
Input: 1024 × 1536
Final output: 4×
Result: 4096 × 6144 only
```
