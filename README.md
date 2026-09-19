# sd-webui-agent-forge

<div align="center">

**面向 Agent 智能体时代的 Stable Diffusion WebUI Forge 中文整合平台 | 多模态模型 | MCP / Skill | 智能 GUI 启动器**

[![GitHub stars](https://img.shields.io/github/stars/exo101/sd-webui-agent-forge)](https://github.com/exo101/sd-webui-agent-forge/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/exo101/sd-webui-agent-forge)](https://github.com/exo101/sd-webui-agent-forge/network)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13.12-blue.svg)](https://www.python.org/)

[📖 使用教程](https://www.bilibili.com/video/BV1Zxu86uE6k?spm_id_from=333.788.recommend_more_video.-1&trackid=web_related_0.router-related-2589621-dpmnd.1786250847122.213&vd_source=343e49b703fb5b4137cd6c1987846f37) | [🚀 快速开始](#快速开始)

</div>

---

## 📋 项目介绍

**sd-webui-agent-forge** 是基于 **Stable Diffusion WebUI Forge** 的中文智能体增强整合版本，面向国内用户、多模态模型和对话式 AI 工作流进行改良。项目目标不再只是提供一个复杂参数面板，而是让用户通过聊天就能调用 WebUI 能力、组合模型、扩写提示词、处理图像/视频、诊断问题并完成复杂任务。


> [!NOTE]
> 本版本为改良版本，部分插件直接安装会发生兼容性错误，为了适应众多新旧插件做了些许修改，如需要安装旧插件直接让智能体帮忙安装排除修改bug和不兼容性。
>
> **版本路线说明**：从 **v3.6.8** 之后，项目名称和发展方向调整为 `sd-webui-agent-forge`，不再继续沿用 `neo-v3` 分支路线，而是转向以智能体（Agent）、MCP 和 Skill 为核心的持续演进路线。后续版本将优先围绕自然语言操作、模型编排、多模态理解、工具调用和易用性进行设计。

---

## 🚀 新路线：Agent 智能体与 MCP / Skill

本项目正在从“手动操作大量参数的 WebUI”逐步进化为“可以通过对话完成复杂任务的 AI 工作台”。用户不必记住所有参数和菜单位置，只需描述目标，智能体就可以理解任务、调用 WebUI 功能，并在必要时向用户确认关键操作。

### 智能体可以做什么

| 能力 | 说明 |
|------|------|
| **自然语言生图** | 根据用户描述生成图片，自动选择或调整模型、尺寸、采样器、步数、提示词和相关参数 |
| **记忆与上下文感知** | 在启用记忆功能后，记录用户偏好、常用模型、风格习惯和当前任务上下文；用户也可以清理或关闭记忆 |
| **复杂参数设置** | 根据显存、模型类型、画面比例和任务目标，自动组合生成参数，减少手动调整负担 |
| **模型组合调配** | 读取可用模型、VAE、文本编码器、LoRA、ControlNet 和预处理器，并按任务需要组合使用 |
| **提示词扩写与整理** | 将简单描述扩展为更完整的正向提示词，整理负面提示词、权重、风格、构图和主体细节 |
| **批量任务处理** | 执行批量生图、批量放大、批量抠图、批量格式处理和多步骤图像工作流 |
| **多模态理解** | 在支持相应工具和模型的情况下，读取并分析文档、图像、代码、视频、网页内容和工作区文件 |
| **图像与视频处理** | 调用图生图、后期处理、放大、抠图、图层分离、视频抽帧、视频生成 |
| **扩展与 WebUI 检查** | 查看已安装扩展、调研扩展功能、诊断 WebUI 文件和定位常见内部错误 |
| **对话式任务编排** | 将“读取资料 → 分析 → 生成提示词 → 选择模型 → 批量生成 → 对比结果”串成连续任务 |

> 智能体的实际能力取决于已安装的扩展、可用模型、显存、操作系统权限以及配置的 AI 模型/API。涉及删除文件、执行命令、修改代码或外部服务调用时，应先确认范围并做好备份；智能体不会替代人工审核生成结果和代码修改。

### MCP 与 Skill

- **MCP**：为智能体提供标准化工具接口，使其能够访问 WebUI 功能、模型管理、工作区文件、网页内容、图像/视频处理和其他外部工具。
- **Skill**：以 `SKILL.md` 形式描述某类任务的使用规则和步骤。智能体可以发现可用 Skill，读取对应说明后按规范执行任务。
- **工具调用**：智能体可以调用文生图、图生图、模型切换、参数更新、模型列表、后期处理、视频处理、文档分析、网页读取、扩展诊断等工具。
- **可扩展性**：用户可以根据自己的工作流增加模型、扩展、MCP 服务或 Skill，使智能体逐步适应个人的生产流程。

### 智能体大脑

项目支持使用云端 API 模型或本地模型作为智能体大脑：

- **API 模型**：可配置兼容 OpenAI 接口的云端模型，也可以使用项目中支持的图像/视频 API 服务。
- **本地模型**：可通过本地 OpenAI 兼容服务或 `llama.cpp` 运行 GGUF 模型，适合重视隐私、希望离线使用或网络条件有限的用户。
- **混合工作流**：使用本地模型负责对话、参数规划和工具调用，使用云端模型负责复杂推理或图像/视频生成；具体组合取决于用户的配置。

### 面向新手的 UI 方向

后续版本将全面简化高频 UI 参数设置：保留必要的高级选项，同时让智能体承担参数解释、自动配置和任务编排工作。新手可以直接描述“生成一张某种风格、某种比例的图片”，熟悉 WebUI 的用户仍然可以继续使用传统界面和高级参数。

---

## 🖼️ 图像对比模块

项目内置图像对比功能，用于快速查看生成前后或两次生成结果之间的差异：

- **文生图**：第一次生成单张图片时不显示对比；再次生成后自动比较上一次和本次结果。
- **图生图**：生成完成后自动对比原图和生成结果。
- **后期处理**：处理完成后自动对比原图和处理结果。
- **PNG 图片信息**：上传图片后自动显示可对比结果。
- **滑动查看**：通过滑动条调整分界线，适合查看细节、放大效果、重绘差异和风格变化。

---

## 🇨🇳 国内用户与 ModelScope（魔搭社区）支持

项目集成了面向国内用户的 ModelScope（魔搭社区）模型下载能力。用户可以在 WebUI 的模型下载器中浏览或填写模型仓库 ID，下载完整仓库或指定文件，并保存到对应模型目录。

支持的主要方式包括：

- ModelScope（魔搭社区）模型仓库下载；
- Hugging Face 模型仓库下载；
- 完整仓库下载和单文件下载；
- 多文件选择、下载历史和自定义保存路径；
- 国内网络环境下优先使用 ModelScope，减少跨境网络访问带来的不稳定；
- 已下载模型可继续由 WebUI 或智能体发现、切换和调用。

> ModelScope 下载速度和可用性仍取决于网络、仓库权限、模型大小和平台服务状态。部分模型还需要额外的文本编码器、VAE、LoRA 或其他组件，下载后请按照模型说明放入正确目录。

---

## � 部署区

> 部署区涵盖从零开始部署本项目所需的所有信息，包括系统要求、安装步骤、启动器使用和常见问题。

### 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10/11 (64位) | Windows 11 |
| **Python** | 3.13.12 | 3.13.12 |
| **GPU** | NVIDIA GPU 8GB+ 显存 | NVIDIA GPU 12GB+ 显存 |
| **内存** | 16GB | 32GB |
| **磁盘空间** | 50GB 可用空间 | 100GB+ SSD |

### 快速开始

#### 第一步：安装 Python

1. 下载 Python 3.13.12：[官方下载地址](https://www.python.org/downloads/release/python-31312/)
2. 运行安装程序，**务必勾选** "Add Python to PATH"
3. 安装完成后，打开命令行输入 `python --version` 验证

#### 第二步：下载项目代码

```bash
git clone https://github.com/exo101/sd-webui-agent-forge.git
cd sd-webui-agent-forge
```

或直接下载 ZIP 解压到任意目录（**路径不要包含中文和空格**）。

#### 第三步：启动

**方式一：使用智能启动器（推荐）**

1. 进入项目目录，双击运行 `启动器.exe`
2. 点击 **⚡ 启动** 按钮
3. 首次启动会自动安装依赖，等待进度完成
4. 浏览器自动打开 WebUI 界面

**方式二：使用批处理文件**

1. 进入 `webui` 目录，双击运行 `webui-user.bat`
2. 等待依赖安装和 WebUI 启动
3. 浏览器打开 `http://127.0.0.1:7860`

> [!NOTE]
> 首次启动需要 5-15 分钟安装依赖，请保持网络畅通，不要关闭窗口。

### 智能启动器

启动器采用 PyQt6 开发，提供以下功能模块：

| 功能 | 说明 |
|------|------|
| **⚡ 启动/停止** | 一键启动或停止 WebUI |
| **🌐 页面** | 打开 WebUI 的浏览器界面 |
| **🔄 检查更新** | 自动检测 GitHub 最新版本 |
| **🛑 全部停止** | 强制停止所有相关进程 |

| 标签页 | 功能 |
|--------|------|
| **环境检测** | GPU 型号、显存大小、驱动版本、系统资源 |
| **模型管理** | 模型目录结构说明、下载指南 |
| **扩展管理** | 已安装扩展列表、启用/禁用 |
| **参数设置** | 代理配置、启动参数、性能选项 |
| **运行日志** | 实时日志输出、错误诊断 |

### 常见问题

| 问题 | 解答 |
|------|------|
| **启动器显示"环境检测失败"？** | 检查是否安装了 Python 3.13.12 并勾选了 "Add Python to PATH"，重启电脑后重试 |
| **首次启动卡在"安装依赖"？** | 首次安装需要 10-30 分钟，请检查网络连接，可在启动器中配置代理 |
| **浏览器显示"无法访问此网站"？** | 检查启动器日志，确认端口 7860 未被占用，或点击启动器"页面"按钮手动打开 |
| **生成图片时提示"显存不足"？** | 启用"显存防溢出保护"，降低分辨率，使用 FP8 量化模型 |
| **生成的图片全黑或质量差？** | 确认选择了正确的模型，检查提示词，尝试换采样器（如 `Euler a`），增加采样步数 |
| **如何更新到最新版本？** | 在启动器主控台点击"检查启动器更新"，或重新 `git pull` |

---

## 🧠 内核区

> 内核区介绍本项目基于 Forge 框架所做的底层优化，包括性能加速、显存优化和功能增强。

### 核心优化

- **ComfyUI 后端重写** — 内存管理、模型补丁、注意力机制全面优化
- **模型加载优化** — 加速启动和模型切换
- **内存泄漏修复** — 切换 checkpoint 时的内存问题
- **uv 包管理器支持** — 大幅加速依赖安装

### 注意力加速

| 优化项 | 说明 |
|--------|------|
| **SageAttention** | 新一代注意力优化，显存占用极低 |
| **FlashAttention** | 高速注意力计算 |
| **xFormers** | 内存高效的注意力实现 |
| **Triton 内核** | int8 矩阵乘法加速 |

### 推理加速

- **Spectrum** — 免训练加速所有模型，即开即用
- **TAESD 实时预览** — 所有模型支持实时预览
- **半精度上采样器** — 加速上采样过程
- **GPU 瓦片合成** — 加速高分辨率图像合成
- **PyTorch 编译加速** — 使用 `torch.compile` 加速推理

### 显存优化策略

| 显存大小 | 推荐配置 | 支持模型 |
|---------|---------|---------|
| **4-6 GB** | TAESD + 分块处理 | SD1.5, SD2.1 |
| **8 GB** | 默认配置 | SDXL, Flux.2-Klein 4B |
| **12 GB** | 全功能 | Flux, Anima,  |
| **16 GB+** | 无限制 | Qwen-Image, Flux.2-Klein 9B |

### 其他增强

- **显存防溢出保护** — 防止显存溢出导致的崩溃，支持 UNet/VAE 分块处理
- **多图拼接参考** — 多图像拼接与参考功能
- **种子多样性增强** — 改善蒸馏模型的种子多样性
- **调制引导控制** — 改善 Anima 模型的生成质量
- **支持更多图像格式** — .avif、.heif、.jxl
- **X/Y/Z 图自动行计数优化**

---

## 🎨 模型区

> 模型区介绍本项目支持的各类模型及其文件结构，助你快速了解如何下载和配置模型。

### 模型目录结构

```
webui/models/
├── Stable-diffusion/          # 所有主模型（SD1.5、SDXL、Flux、Anima、Qwen-Image、Wan 等）
├── Lora/                      # LoRA 微调模型
├── VAE/                       # 变分自编码器
├── text_encoder/              # 文本编码器 ⚠️ 需手动下载
├── CLIP/                      # CLIP 文本编码器
├── ControlNet/                # ControlNet 控制网络
├── ControlNetPreprocessor/    # ControlNet 预处理器
├── ESRGAN/                    # 超分辨率放大模型
└── RealESRGAN/                # 超分辨率模型
```

### 模型类型与文件结构

#### SD1.5 / SDXL 模型

传统模型，只需一个 checkpoint 文件即可运行：

```
models/Stable-diffusion/
└── your_model.safetensors    # 包含 UNet + VAE + 文本编码器
```

#### Flux 模型

Flux 采用 DiT 架构，组件分离存储：

```
models/Stable-diffusion/
└── flux1-dev-fp8.safetensors

models/text_encoder/          # T5 编码器（必需，约 4.7GB）
└── t5xxl_fp8_e4m3fn.safetensors

models/clip/                  # CLIP 编码器（必需，约 235MB）
└── clip_l.safetensors
```

#### Flux.2-Klein 模型

多模态编辑模型，支持图像编辑与生成：

```
models/Stable-diffusion/
└── flux-2-klein-9b-fp8.safetensors

models/text_encoder/
└── qwen_3_4b.safetensors

models/vae/
└── flux2-vae.safetensors
```

#### Anima 模型

二次元高质量专用模型：

```
models/Stable-diffusion/
└── anima.safetensors

models/text_encoder/
└── qwen_3_06b_base.safetensors

models/VAE/
└── qwen_image_vae.safetensors
```

#### Qwen-Image 模型

通义千问图像生成/编辑模型：

```
models/Stable-diffusion/
└── qwen_image_edit_2511_int8_convrot.safetensors

models/text_encoder/
└── qwen_2.5_vl_7b_fp8_scaled.safetensors

models/VAE/
└── qwen_image_vae.safetensors
```


> [!TIP]
> 导出视频需要安装 **[FFmpeg](https://ffmpeg.org/)**

### 模型下载

| 模型 | 下载方式 | 说明 |
|------|---------|------|
| **SD1.5 / SDXL** | Civitai Helper 插件一键下载 | Stable-diffusion |
| **LoRA** | Civitai Helper 插件一键下载 | 风格/角色微调 |
| **ControlNet** | Civitai Helper 插件一键下载 | 控制网络 |
| **Flux CLIP** | [HuggingFace 手动下载](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main/clip_l.safetensors) | 放入 `models/clip/` |
| **Flux T5** | [HuggingFace 手动下载](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main/t5xxl_fp8_e4m3fn.safetensors) | 放入 `models/text_encoder/` |

---

## 🔌 插件区

> 插件区介绍本项目已集成和优化的各类扩展插件，按功能分类展示。

### 新增插件

| 插件名称 | 功能说明 |
|---------|---------|
| **🎨 美学提升** | Qwen3.5 图像与视频美学质量分析 |
| **📷 相机角度选择器** | 3D 可视化多角度提示词选择，支持方位角、高程角、距离调整 |
| **🎥 多媒体处理** | Qwen3-TTS 语音合成、唇形同步多媒体处理 |
| **👁️ 图像识别与对话** | 基于 Qwen3.5 视觉模型的图像识别与对话功能 |
| **✂️ 图像分割与抠图** | SAM 模型一键抠图、背景替换、图像清理 |
| **🔍 图层分离** | 动漫风格图像的图层分解与透明化处理，支持深度估计和3D效果生成 |
| **🌄 无边图像浏览** | 快速浏览和管理历史生成图片 |
| **🖼️ 图像对比** | 并排对比两张生成图片的差异 |


### 优化插件

| 插件名称 | 优化说明 |
|---------|---------|
| **🔧 ADetailer** | 兼容性优化，修复人脸修复问题 |
| **🔧 Photoshop 插件** | Auto-Photoshop-StableDiffusion-Plugin 增强 |
| **🏷️ WD 1.4 标签器** | 自动生成图像标签，支持中文 |
| **🌐 Civitai Helper** | 模型下载与管理，支持一键下载、元数据同步、批量操作 |
| **🎯 LoRA Prompt Tool** | LoRA 提示词智能推荐与权重调节 |
| **🔄 区域提示词** | 图像分区控制，不同区域使用不同提示词 |
| **📦 SuperMerger** | 模型合并与融合工具 |
| **🔤 标签补全** | 提示词自动补全，提高输入效率 |
| **🇨🇳 中文界面** | 完全汉化的界面语言包 |

### 内置功能

| 功能名称 | 说明 |
|---------|------|
| **🛡️ 显存防溢出保护** | 防止显存溢出导致的崩溃，支持 UNet/VAE 分块处理 |
| **🖼️ 多图拼接参考** | 多图像拼接与参考功能 |
| **🌱 种子多样性增强** | 改善蒸馏模型的种子多样性 |
| **⚡ 频谱预测加速** | 免训练加速所有模型 |
| **🔥 PyTorch 编译加速** | 使用 torch.compile 加速推理 |
| **🎛️ 调制引导控制** | 改善 Anima 模型的生成质量 |

### 已集成扩展列表

> 以下是 `webui/extensions/` 目录下已集成的全部扩展，开箱即用，无需自行安装。

| 扩展目录 | 用途简述 |
|---------|---------|
| **aadetailer-neoforge** | ADetailer 人脸/手部自动修复 |
| **forge-h3-studio** | H3 Studio 综合工具集 |
| **infinite-browsing** | 无限图像浏览，快速查看历史生成图 |
| **sd-civitai-browser-neo** | Civitai 模型在线浏览与一键下载 |
| **sd-dynamic-prompts_neo** | 动态提示词模板，支持变量与逻辑分支 |
| **sd-forge-regional-prompter-neo** | 区域提示词，分区域控制画面内容 |
| **sd-forge-tutorial** | 内置新手图文教程 |
| **sd-webui-AestheticEnhancement-llama.cpp** | 基于 Qwen3.5 的图像/视频美学分析 |
| **sd-webui-agent-dev-2.4** | 智能体 Agent，支持自然语言对话式生图 |
| **sd-webui-bsk-camera-control-forge-neo** | 3D 相机角度选择器，多角度提示词 |
| **sd-webui-forge-neo-seedvr2** | SeedVR2 视频超分辨率增强 |
| **sd-webui-model-keyword** | 自动识别模型关键词，避免漏触发 |
| **sd-webui-model_downloader** | 模型批量下载工具 |
| **sd-webui-multimodal-media** | 多媒体处理：TTS 语音、唇形同步、Qwen 视频、ACE-Step 音乐 |
| **sd-webui-openpose-editor** | OpenPose 姿态可视化编辑器 |
| **sd-webui-prompt-all-in-one-neo** | 提示词管理一体化（历史记录、翻译、权重） |
| **sd-webui-ps-plugin-api** | Photoshop 插件对接 API，PS 内直接生图 |
| **sd-webui-see-through-sam** | SAM 一键抠图与图层分离 |
| **sd-webui-supermerger-forgeneo-anima** | SuperMerger 模型合并/融合 |
| **sd-webui-tagcomplete-neo** | 提示词自动补全 |
| **sd-webui-trellis2** | TRELLIS.2 单图生成 3D 模型 |
| **stable-diffusion-webui-localization-zh_Hans** | 中文汉化语言包 |
| **stable-diffusion-webui-wd14-tagger** | WD 1.4 反向标签器，自动给图像打标 |
| **TE-webui-DLSS5** | DLSS 5 视频超分与插帧 |

> [!TIP]
> 所有扩展均位于 `webui/extensions/`，可在 WebUI 的 **Extensions** 页面单独启用或禁用；删除对应文件夹即可彻底卸载。

---

## 📚 学习资源

- **视频教程**: [B站教程合集](https://www.bilibili.com/video/BV1KfXyBTEXb)
- **Wiki**: [Haoming02 Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)

---

## 🤝 社区支持

### QQ 交流群

<img src="launcher/qq群ai交流群.jpg" alt="QQ交流群" width="200"/>

扫码加入 AI 交流群，获取最新整合包、使用技巧和问题解答。

### B站频道

关注 [哔哩哔哩（鸡肉爱土豆）](https://space.bilibili.com/403361177) 获取最新教程和更新通知。

---

## 🙏 致谢

- **Haoming02** — [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) 分支作者
- **AUTOMATIC1111** — Stable Diffusion WebUI 原始项目
- **lllyasviel** — Forge 优化框架
- **comfyanonymous** — ComfyUI 项目
- **kijai**、**city96** — 社区贡献者
- 所有开源图像生成社区的贡献者

---

## 📄 许可证

本项目遵循 AGPL-3.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

> [!NOTE]
> 此版本整合包通过秋叶aaaki、张吕敏、Haoming02 等多位大佬技术总结做出的版本，不属于任何个人、企业，是非盈利性质的开源软件。

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star 支持一下！⭐**

Made with ❤️ by exo101

</div>
