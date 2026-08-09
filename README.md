# Stable Diffusion WebUI Forge - Neo (中文改良版)

<div align="center">

**基于 AUTOMATIC1111 的现代化 AI 图像生成平台 | 支持最新多模态模型 | 智能 GUI 启动器**

[![GitHub stars](https://img.shields.io/github/stars/exo101/sd-webui-forge-neo-v3)](https://github.com/exo101/sd-webui-forge-neo-v3/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/exo101/sd-webui-forge-neo-v3)](https://github.com/exo101/sd-webui-forge-neo-v3/network)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13.12-blue.svg)](https://www.python.org/)

[📖 使用教程](https://www.bilibili.com/video/BV1KfXyBTEXb) | [💬 QQ交流群](#社区支持) | [🚀 快速开始](#快速开始)

</div>

---

## 📋 项目介绍

本项目是基于 **Stable Diffusion WebUI Forge** 的中文改良版本，专注于优化和多模态插件融合，目标是通过简单易用的 GUI 运行最新的流行模型。

**原作者**：[Haoming02](https://github.com/Haoming02) · [原项目链接](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

> [!NOTE]
> 本版本为改良版本，部分插件直接安装会发生兼容性错误，为了适应众多新旧插件做了些许修改。

---

## � 部署区

> 部署区涵盖从零开始部署本项目所需的所有信息，包括系统要求、安装步骤、启动器使用和常见问题。

### 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10/11 (64位) | Windows 11 |
| **Python** | 3.13.12 | 3.13.12 |
| **GPU** | NVIDIA GPU 4GB+ 显存 | NVIDIA GPU 12GB+ 显存 |
| **内存** | 16GB | 32GB |
| **磁盘空间** | 50GB 可用空间 | 100GB+ SSD |

### 快速开始

#### 第一步：安装 Python

1. 下载 Python 3.13.12：[官方下载地址](https://www.python.org/downloads/release/python-31312/)
2. 运行安装程序，**务必勾选** "Add Python to PATH"
3. 安装完成后，打开命令行输入 `python --version` 验证

#### 第二步：下载项目代码

```bash
git clone https://github.com/exo101/sd-webui-forge-neo-v3.git
cd sd-webui-forge-neo-v3
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
| **12 GB** | 全功能 | Flux, Anima, Qwen-Image |
| **16 GB+** | 无限制 | Wan 2.2, Flux.2-Klein 9B |

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
├── Stable-diffusion/          # Stable-diffusion（SD1.5 / SDXL）
├── diffusion_models/          # DiT 架构模型（Flux、Anima、Qwen-Image、Wan）
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
models/diffusion_models/
└── flux1-dev-fp8.safetensors

models/text_encoder/          # T5 编码器（必需，约 4.7GB）
└── t5xxl_fp8_e4m3fn.safetensors

models/clip/                  # CLIP 编码器（必需，约 235MB）
└── clip_l.safetensors
```

#### Flux.2-Klein 模型

多模态编辑模型，支持图像编辑与生成：

```
models/diffusion_models/
└── flux-2-klein-9b-fp8.safetensors

models/text_encoder/
└── qwen_3_4b.safetensors

models/vae/
└── flux2-vae.safetensors
```

#### Anima 模型

二次元高质量专用模型：

```
models/diffusion_models/
└── anima.safetensors

models/text_encoder/
└── qwen_3_06b_base.safetensors

models/VAE/
└── qwen_image_vae.safetensors
```

#### Qwen-Image 模型

通义千问图像生成/编辑模型：

```
models/diffusion_models/
├── svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors
└── svdq-fp4_r128-qwen-image-lightningv1.1-8steps.safetensors

models/text_encoder/
└── qwen_2.5_vl_7b_fp8_scaled.safetensors

models/VAE/
└── qwen_image_vae.safetensors
```

#### Wan 2.2 视频模型

视频生成模型：

```
models/diffusion_models/
├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
└── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors

models/text_encoder/
├── umt5-xx-fp8-scaled.safetensors
└── umt5-xxl-enc-bf16.safetensors

models/VAE/
└── wan_2.2_vae.safetensors
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

- **Haoming02** — [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) 原作者
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
