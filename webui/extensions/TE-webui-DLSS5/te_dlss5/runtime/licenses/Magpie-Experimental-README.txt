Magpie 0.5.6 experimental x64 build

This directory is updated in place. The ZIP name remains stable across builds.
Built-in update checking is temporarily disabled in 0.5.3-experimental.

DLSSNR runtime note: this package uses the maintainer-provided Ada/RTX 40
patched nvngx_dlssnr.dll version 310.8.0.0. Because the file is binary-patched,
Windows reports an Authenticode hash mismatch; use it only for this experimental
RTX 40 test build.

Experimental effects:
- DLSS SR_Experimental (stored as DLSS\DLSS_ZeroMV for profile compatibility)
- DLSS_ZeroMV_Jitter and the legacy DLSS_OpticalFlow alias
- DLSSNR\DLSSNR_AI_Filter (same-resolution SDR AI filter)
- Diagnostics\FrameGuidance_Motion, FrameGuidance_Confidence,
  FrameGuidance_Depth, FrameGuidance_DepthResidual
- FSR2\FSR2_ZeroMV, FSR2_OpticalFlow
- FSR3\FSR3_ZeroMV, FSR3_OpticalFlow (upscaler only; no frame generation)
- XeSS\XeSS_ZeroMV, XeSS_OpticalFlow
- RTXVideo denoise and VSR quality levels
- DLSS FG_Experimental 2x/3x/4x (do not combine with Smooth Motion)
- XeSSFG cross-vendor x2 frame generation and Intel Arc x2/x3/x4
  multi-frame generation (do not combine with Smooth Motion)

XeSS uses the cross-vendor XeSS 3.0.1 D3D12 DP4a path through D3D11/D3D12
shared resources. XeSS_OpticalFlow estimates motion on a 50% grid and supplies
low-resolution motion vectors plus flat depth. XeSS_ZeroMV explicitly supplies
zero motion vectors plus flat depth. It does not use the Intel Arc-only native
D3D11 path.

These integrations still do not have engine depth, engine motion vectors,
reactive masks, camera matrices, or projection jitter. DLSS SR_Experimental and
DLSS FG_Experimental can instead consume Renderer-owned NVIDIA Optical Flow and
Depth Anything V2 estimated depth. These inputs are experimental and may still
produce ghosting or unstable detail. See logs\magpie.log when reporting errors.

DLSS SR_Experimental usage:
- Use Motion Vectors defaults to On. It supplies current-to-previous optical
  flow in source-pixel units.
- Use Estimated Depth defaults to Off. Enabling it also runs optical flow
  internally to stabilize depth, even if motion-vector delivery is disabled.
- Real guidance is used only when the DLSS input and captured source dimensions
  match. Any unavailable or incompatible channel safely falls back to Zero.
- The Jitter effect remains a separate legacy experiment and does not use this
  new guidance path.

DLSS FG_Experimental usage:
- Select x2, x3, or x4 with Frame Multiplier, subject to GPU/driver support.
- Use Motion Vectors defaults to On; Use Estimated Depth defaults to Off.
- Final effect-chain colour remains at backbuffer resolution. Motion and depth
  remain at captured-source resolution and are synchronized into D3D12 with the
  same captured base-frame ID.
- Do not combine DLSS FG with Smooth Motion or another frame generator.

DLSSNR uses the bundled nvngx_dlssnr.dll Feature 18 exports through D3D12 and consumes one Renderer-owned Frame
Guidance group. NVIDIA Optical Flow supplies current-to-previous source-pixel
motion plus confidence. Depth Anything V2 Small supplies normalized inverse
relative depth through TensorRT FP16, with ONNX Runtime DirectML as hot standby.
Any missing or failed provider falls back to the coherent Zero group without
stopping rendering. The Effect's Guidance Mode selects Available, Force Zero,
Motion Only, or Depth Only for A/B testing. It does not upscale and the current
experiment does not support HDR. NGX Core only allocates and destroys the
parameter block; Core CreateFeature(18) is not a prerequisite.

Provider execution is demand-driven. For the DLSS SR/FG switches, estimated
depth may request NVOF internally for temporal stabilization even when real
motion vectors are not bound to the SDK. Depth Inference Interval defaults to
4; intervening real frames reproject the prior filtered depth. Search
logs\magpie.log for
"DLSSNR STATUS" to verify Feature 18 creation and Evaluate results independently
of NVIDIA's optional on-screen Indicator.

TensorRT requires a compatible local CUDA/cuDNN installation. Engines are built
and cached under LocalAppData for this GPU/model/input configuration and are not
part of this directory. NVIDIA's nvofapi64.dll is loaded from the installed
display driver and is not distributed here. Consult logs\magpie.log to confirm
the selected depth backend, model hash, NVOF grid, reset, and fallback reason.

DLSSFG keeps CPU/GPU fence synchronization. Guidance binding logs distinguish
requested, produced, bound, and Zero-fallback channels. Repeated evaluation failures fall
back to real captured frames and disable frame generation for that scaling
session instead of retrying forever.

Frame Generation now forces exact captured-frame duplicate filtering and no
longer synthesizes repeated base frames to satisfy Magpie's minimum-FPS timer.
XeSSFG also ignores frontend presents caused only by the software cursor or
overlay, preventing mouse movement from inflating the SDK input frame rate.

Smooth Motion compatibility restarts preserve window, minimized, or tray state,
release shortcuts before replacement, and wait for the old process to exit.

Native upscalers and RTX Video are dispatched through NativeEffectBackend and
NativeEffectBackendFactory. Frame-generation presenters remain terminal stages.

This package includes build-manifest.json and THIRD-PARTY-NOTICES.md. The
manifest identifies the source commit, enabled optional backends, and hashes of
the packaged files. Third-party redistribution and GPL compatibility must be
reviewed before publishing a binary package.

Source fork: https://github.com/SAOG0721/Magpie
Upstream: https://github.com/Blinue/Magpie
