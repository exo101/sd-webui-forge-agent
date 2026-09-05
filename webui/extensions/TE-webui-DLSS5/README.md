# TE WebUI DLSS5

This extension exposes the external TE-ComfyUI-DLSS5 native DLSSNR backend in
Stable Diffusion WebUI Forge's Extras / postprocessing page.

## Bundled Runtime

The extension loads its own local runtime package from:

`webui/extensions/TE-webui-DLSS5/te_dlss5`

This directory contains the native bridge DLLs, DLSSNR runtime files, frame
guidance files, and TE-local Depth Anything V2 code copied from the reference
TE-ComfyUI-DLSS5 package.

## Usage

Open WebUI's Extras page, expand `TE DLSS5 神经画质处理`, then enable
`启用 TE DLSS5`.

- Use `单图 / 独立帧` for still images and batch images.
- Use `视频 / 连续帧` when processing video input, so the DLSSNR backend and
  guidance state are preserved across sequential frames.
- `无引导` is the most compatible mode.
- `NVOF 光流` and `NVOF + Depth Anything V2` require the optional guidance DLLs,
  models, and Python dependencies from the TE-ComfyUI-DLSS5 package.

## Runtime Notes

This is same-resolution neural image/video enhancement. It is not a normal
resize upscaler. If you need resolution changes, run Forge's resize/upscale
operation separately in the desired order.

The native backend requires a supported NVIDIA GPU, Windows DLL loading support,
the bundled TE native bridge DLL, and the bundled DLSSNR runtime.
