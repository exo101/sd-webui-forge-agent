# Third-party components and redistribution / 第三方组件与再分发

This document is an engineering inventory, not legal advice. A component being downloadable or usable locally does not by itself grant permission to redistribute it in this project's source tree or binary releases.

本文是工程侧清单，不构成法律意见。某个组件可以下载或在本机使用，并不自动表示可以把它再次放进本项目源码或二进制 Release。

## Project source / 项目源码

- Magpie and this fork's derived source are licensed under GPLv3. Keep the upstream copyright notices, the root `LICENSE`, the corresponding source, and build instructions with every source distribution.
- The repository must not contain `src/BuildOptions.props.user`, local SDK directories, models, wheels, credentials, logs, `bin/`, `obj/`, or generated packages.
- `dependencies/` is a local development cache outside the source repository. It is not part of the distributable source.

- Magpie 及本 Fork 的派生源码采用 GPLv3。分发源码时应保留上游版权声明、根目录 `LICENSE`、对应源码和构建说明。
- 仓库不得提交 `src/BuildOptions.props.user`、本机 SDK 目录、模型、wheel、凭据、日志、`bin/`、`obj/` 或生成的分发包。
- 工作区中的 `dependencies/` 只是本机开发依赖缓存，不属于可分发源码。

## Component matrix / 组件清单

| Component | Local use | Source repository | Public binary redistribution status |
| --- | --- | --- | --- |
| Magpie-derived code | GPLv3 | Include source and GPLv3 notices | Allowed only while satisfying GPLv3 obligations |
| AMD FSR 2.2 source and community D3D11 backend | MIT-style notices in the local SDK/fork | Do not vendor unless intentionally reviewed; retain notices if copied | Generally notice-based, but verify every shipped file |
| AMD FSR 3/4 API runtime | AMD binary redistribution terms | Do not commit SDK/runtime binaries | Reproduce AMD notices and verify the exact runtime license |
| FSR 4.1.1 capability-check override | Experimental code in this fork | Source can be reviewed under this repository's license | **Do not publish a binary using the override until AMD terms and permission for this behavior have been reviewed** |
| Intel XeSS 3.0.1 | Intel Simplified Software License | Do not commit the binary SDK | Redistribution is for unmodified binaries and requires the Intel notice; verify every shipped file |
| NVIDIA DLSS/NGX and DLSS Frame Generation | NVIDIA RTX SDK license | Do not commit SDK headers, libraries, models, or runtime DLLs | **Unresolved/high risk when combined with GPLv3 Magpie; do not publish such a combined binary without a dedicated review or permission** |
| NVIDIA DLSSNR experimental runtime | NVIDIA proprietary signed runtime; no public DLSSNR SDK contract | Do not commit `nvngx_dlssnr.dll`; configure it as a local build input | **Internal testing only until NVIDIA redistribution permission and GPL compatibility are reviewed** |
| NVIDIA Optical Flow driver API | NVIDIA driver component plus locally supplied SDK-compatible headers | Do not commit or package `nvofapi64.dll`; load the installed display driver's copy from System32 | The driver DLL is not redistributed by this project; review any header provenance before publishing source |
| Depth Anything V2 Small ONNX | Apache-2.0 model family; exact converted artifact and upstream notices must be verified | Keep the model in the local `dependencies/` cache and copy it only for explicitly enabled test builds | Do not publish the model until the exact ONNX artifact's license and notice chain have been reviewed |
| ONNX Runtime DirectML | Microsoft-distributed runtime with MIT and bundled third-party notices | Acquire through the pinned NuGet package; do not commit runtime binaries | Ship only the native files permitted by the package terms and include all supplied notices |
| ONNX Runtime TensorRT, TensorRT, CUDA and cuDNN | Mixed Microsoft/NVIDIA runtime terms | Keep all runtimes local; generate TensorRT engine caches per machine and never commit them | **Internal testing only until every binary's redistribution terms and GPL compatibility are reviewed** |
| NVIDIA Video Effects / Maxine runtime and models | NVIDIA proprietary and AI-product terms, plus bundled third-party notices | Do not commit wheel, models, SDK, or runtime binaries | **Unresolved/high risk; review the exact runtime/model terms and GPL compatibility before release** |
| OptiScaler reference checkout | Reference only; not linked into Magpie | Do not copy its source or binaries without a separate review | Not part of the package |
| Microsoft/Windows redistributable runtime files | Per Microsoft redistribution terms | Do not vendor development SDKs | Ship only files Microsoft marks redistributable and retain required notices |

The controlling texts are the exact license files supplied with each SDK/runtime. The local development cache currently contains, among others:

- `dependencies/DLSS-main/LICENSE.txt`
- `dependencies/FSR2-DX11-source/.../LICENSE.txt`
- `dependencies/FSR-SDK-v2.3.0/docs/license.md`
- `dependencies/XeSS-SDK-3.0.1/LICENSE.txt`
- `dependencies/DepthAnythingV2/onnx-community-depth-anything-v2-small/LICENSE.Depth-Anything-V2.txt`
- `dependencies/onnxruntime/*/LICENSE*`
- `dependencies/nvidia-vfx-python/.../licenses/packaging/`

各组件最终以对应 SDK/运行库随附的完整许可证原文为准，不能只依据本表摘要作发布决定。

## Release policy / 发布策略

The safe default for this experimental fork is:

1. Publish the GPLv3 source and dependency acquisition/build instructions.
2. Keep optional proprietary backends disabled in public reproducible builds unless their redistribution and GPL compatibility have been cleared.
3. Never upload entire SDK folders, wheels, models, import libraries, symbols, or local configuration.
4. For every binary package, retain `LICENSE-Magpie.txt`, this notice, each required vendor notice, and `build-manifest.json`.
5. Review the manifest's file hashes and feature switches before upload. A DLL appearing in `bin/` is not proof that it is redistributable.

本实验 Fork 的安全默认策略是：

1. 优先发布 GPLv3 源码，以及依赖获取和构建说明。
2. 专有可选后端在完成再分发权限和 GPL 兼容性核对前，不进入公开的可复现构建。
3. 不上传完整 SDK、wheel、模型、导入库、符号文件或本机配置。
4. 每个二进制包保留 `LICENSE-Magpie.txt`、本文、各厂商要求的声明和 `build-manifest.json`。
5. 上传前检查清单中的文件哈希与功能开关；某 DLL 出现在 `bin/` 中不代表它天然可以再分发。

## Pre-release checklist / 发布前检查

- [ ] `git status` and `git ls-files` contain no local SDK/runtime/model/configuration files.
- [ ] The package was produced by `scripts/Build-Release.ps1` from a named commit.
- [ ] `build-manifest.json` records the expected commit, version, feature switches, and file hashes.
- [ ] Every shipped third-party binary has an identified redistributable license and required notice.
- [ ] GPLv3 corresponding-source obligations are satisfied for the exact binary.
- [ ] NVIDIA-enabled and FSR4-override binaries have received a separate compatibility/permission review.
- [ ] The ZIP hash is recorded in the GitHub Release notes.

- [ ] `git status` 和 `git ls-files` 中没有本机 SDK、运行库、模型或配置文件。
- [ ] 分发包由 `scripts/Build-Release.ps1` 从明确的提交构建。
- [ ] `build-manifest.json` 中的提交、版本、功能开关和文件哈希符合预期。
- [ ] 每个第三方二进制都已确认可再分发条款及必须携带的声明。
- [ ] 对该二进制履行 GPLv3 对应源码义务。
- [ ] 含 NVIDIA 后端或 FSR4 绕过逻辑的包已经单独完成兼容性/权限审核。
- [ ] GitHub Release 说明中记录 ZIP 的 SHA-256。
