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

本项目是基于 **Stable Diffusion WebUI Forge Neo** 的中文改良版本，专注于优化和多模态插件融合，目标是通过简单易用的 GUI 运行最新的流行模型。

### ✨ 核心特性

- 🎯 **极简显存占用** - 所有项目与插件最高均不超过 16GB 显存，以 12GB-8GB 居多
- 🚀 **极速推理** - 集成多种注意力优化（SageAttention、FlashAttention、xFormers）
- 🌐 **智能启动器** - PyQt6 开发的现代化 GUI，支持一键启动、环境检测、模型管理
- 🔧 **ComfyUI 后端重写** - 内存管理、模型补丁、注意力机制全面优化
- 📦 **多模型支持** - Anima、Flux.2-Klein、Qwen-Image、Wan 2.2 等最新模型
- 🔄 **自动更新** - 内置 GitHub 自动更新机制，保持最新版本
- 🇨🇳 **完全汉化** - 界面、文档、提示全部中文化

**原作者**：[Haoming02](https://github.com/Haoming02)  
**原项目链接**：[sd-webui-forge-classic/neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

> [!NOTE]
> 本版本为改良版本，部分插件直接安装会发生兼容性错误，为了适应众多新旧插件做了些许修改。

---

## 🚀 快速开始

### 系统要求

- **操作系统**: Windows 10/11 (64位)
- **Python**: 3.13.12
- **GPU**: NVIDIA GPU (推荐 8GB+ 显存)
- **磁盘空间**: 至少 50GB 可用空间

### 安装步骤

#### 方法一：下载整合包（推荐）

1. 加入 QQ 群下载最新整合包（见底部社区支持）
2. 解压到任意目录（路径不要包含中文或空格）
3. 双击运行 `webui（启动器）.bat`
4. 首次启动会自动安装依赖，完成后浏览器自动打开

#### 方法二：从源码安装

```bash
# 1. 克隆仓库
git clone https://github.com/exo101/sd-webui-forge-neo-v3.git
cd sd-webui-forge-neo-v3

# 2. 安装 Python 3.13.12
# 从 https://www.python.org/downloads/release/python-31312/ 下载
# 安装时勾选 "Add Python to PATH"

# 3. 启动
双击运行 webui（启动器）.bat
```

### 首次使用

1. **等待依赖安装** - 首次启动会自动下载并安装所有依赖（约 5-15 分钟）
2. **访问 WebUI** - 浏览器自动打开 `http://127.0.0.1:7860`
3. **下载模型** - 根据需求下载对应模型（见下方模型配置）

---

## 🎮 智能启动器功能

启动器采用 PyQt6 开发，提供以下功能模块：

### 🏠 主控台
- ⚡ 一键启动/停止 WebUI
- 🔄 **检查启动器更新** - 自动检测 GitHub 最新版本
- 🌐 快速打开 Web 界面
- 🛑 强制停止全部进程

### 🔧 环境检测
- 📊 GPU 信息检测（型号、显存、驱动版本）
- 💾 系统资源监控（CPU、内存、磁盘）
- ✅ 依赖完整性检查
- 🔍 环境变量验证

### 📦 模型管理
- 📂 模型目录结构说明
- 📥 模型下载指南
- 🗂️ 模型共享配置（ComfyUI 双向同步）
- 💡 最佳实践建议

### 🧩 扩展管理
- 📋 已安装扩展列表
- 🔌 扩展启用/禁用
- 📝 扩展详细信息

### ⚙️ 参数设置
- 🌐 代理配置
- 🚀 启动参数自定义
- 💻 性能优化选项
- 📁 路径配置

### 📜 运行日志
- 📝 实时日志输出
- 🔍 错误诊断
- 📊 性能统计

---

## 📦 支持的模型

### 模型配置要求

不同类型的模型需要不同的组件文件，以下是各类模型的组件结构：

#### SD1.5 / SD2.1 模型
只需一个 checkpoint 文件即可运行：
```
models/Stable-diffusion/
└── your_model.safetensors  # 包含 UNet + VAE + CLIP
```

#### SDXL 模型
同样只需一个 checkpoint 文件：
```
models/Stable-diffusion/
└── sdxl_model.safetensors  # 包含 UNet + VAE + 双文本编码器
```

#### Flux 模型
Flux 模型采用 DiT 架构，组件分离存储：
```
models/diffusion_models/
└── flux1-dev-fp8.safetensors    # 或 flux-schnell.safetensors

models/text_encoder/         # T5 文本编码器（必需）
└── t5xxl_fp8_e4m3fn.safetensors  # ~4.7GB

models/clip/                 # CLIP 文本编码器（必需）
└── clip_l.safetensors       # ~235MB

models/VAE/                  # Flux 专用 VAE（可选）
└── flux_vae.safetensors
```

> [!IMPORTANT]
> **文本编码器必须手动下载！** 不会自动下载，请从 HuggingFace 下载后放入 `models/text_encoder/` 和 `models/clip/` 目录。

**下载地址**：
- CLIP-L: https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main/clip_l.safetensors
- T5-XXL fp8: https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main/t5xxl_fp8_e4m3fn.safetensors

#### Flux.2-Klein 模型
Flux.2-Klein 是多模态编辑模型：
```
models/diffusion_models/
├── flux2-klein-4b.safetensors   # 4B 轻量版
└── flux-2-klein-9b-fp8.safetensors   # 9B 标准版

models/text_encoder/              # 文本编码器（必需）
└── qwen_3_8b_fp8mixed.safetensors
└── qwen_3_4b.safetensors

models/vae/
└── flux2-vae.safetensors
```

#### Anima 模型
Anima 是二次元高质量专用模型：
```
models/diffusion_models/
└── anima.safetensors            # Anima 主模型

models/text_encoder/              # 文本编码器（必需）
└── qwen_3_06b_base.safetensors  # 通义千问编码器

models/VAE/
└── qwen_image_vae.safetensors   # Qwen-Image 专用 VAE
```

#### Qwen-Image 模型
```
models/diffusion_models/
├── svdq-fp4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors  # 编辑模型
└── svdq-fp4_r128-qwen-image-lightningv1.1-8steps.safetensors  # 文生图模型

models/text_encoder/              # 文本编码器（必需）
└── qwen_2.5_vl_7b_fp8_scaled.safetensors

models/VAE/
└── qwen_image_vae.safetensors
```

#### Wan 2.2 视频模型
```
models/diffusion_models/
├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors  # 高噪声
└── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors   # 低噪声

models/text_encoder/              # 文本编码器（必需）
├── umt5-xx-fp8-scaled.safetensors
└── umt5-xxl-enc-bf16.safetensors

models/VAE/
└── wan_2.2_vae.safetensors
```

> [!TIP]
> 导出视频需要安装 **[FFmpeg](https://ffmpeg.org/)**

### 其他模型组件

#### ControlNet 模型
```
models/ControlNet/
├── controlnet-union-sdxl-1.0_promax.safetensors
└── ...
```

#### ControlNet 预处理器
```
models/ControlNetPreprocessor/
├── dw-ll_ucoco_384.pth
└── ...
```

### 模型目录结构

项目 `models` 目录包含以下子目录：

| 目录名称 | 说明 |
|---------|------|
| **Stable-diffusion** | 主模型目录，存放 SD1.5、SDXL、Flux 等大模型 |
| **Lora** | LoRA 微调模型，用于风格/角色/概念微调 |
| **VAE** | 变分自编码器，影响图像色彩和细节 |
| **text_encoder** | 文本编码器模型（T5、CLIP、Qwen 等）⚠️ **需手动下载** |
| **diffusion_models** | 扩散模型（Flux、SD3 等 DiT 架构） |
| **ControlNet** | ControlNet 控制网络模型 |
| **ControlNetPreprocessor** | ControlNet 预处理器模型 |
| **ESRGAN** | ESRGAN 超分辨率放大模型 |
| **RealESRGAN** | RealESRGAN 超分辨率模型 |
| **diffusers** | Diffusers 格式模型 |

---

## 🧩 插件列表

### 新增插件

| 插件名称 | 功能说明 |
|---------|---------|
| **🎨 美学提升** (sd-webui-AestheticEnhancement) | Qwen3.5 图像与视频美学质量分析 |
| **🎬 分镜助手** (sd-webui-Storyboard Assistant) | 专业的剧本与分镜管理系统，支持多故事、多角色管理，可视化分镜墙编排 |
| **📷 相机角度选择器** (sd-webui-camera-angle-selector) | 3D 可视化多角度提示词选择，支持方位角、高程角、距离调整 |
| **🎥 多媒体处理** (sd-webui-multimodal-media) | Qwen3-TTS，唇形同步多媒体处理 |
| **👁️ 图像识别与对话** (sd-webui-qwen-vision-chat) | 基于 Qwen3.5 视觉模型的图像识别与对话功能 |
| **✂️ 图像分割与清理** (sd-webui-sam-matting) | 图像分割、抠图、背景清理功能 |
| **🔍 图层分离** (sd-webui-see-through) | 动漫风格图像的图层分解与透明化处理，支持深度估计和3D效果生成 |

**使用参考**：[WebUI分镜助手插件与Qwen3-TTS协作教程](https://www.bilibili.com/video/BV1foQzBMErp/)

### 优化的旧插件

| 插件名称 | 优化说明 |
|---------|---------|
| **🔧 ADetailer** | 兼容性优化，修复人脸修复问题 |
| **🔧 Auto-Photoshop-StableDiffusion-Plugin** | PS 插件增强 |
| **🏷️ WD 1.4 标签器** | 自动生成图像标签，支持中文 |
| **🌐 Civitai Helper** | Civitai模型下载与管理工具，支持一键下载、元数据同步和批量操作 |
| **🎯 LoRA Prompt Tool** | LoRA 提示词智能推荐与权重调节工具 |

### 汉化的内置功能

| 功能名称 | 说明 |
|---------|------|
| **🛡️ 显存防溢出保护** | 防止显存溢出导致的崩溃，支持 UNet/VAE 分块处理 |
| **🖼️ 多图拼接参考** | 多图像拼接与参考功能 |
| **🌱 种子多样性增强** | 改善蒸馏模型的种子多样性 |
| **⚡ 频谱预测加速** | 免训练加速所有模型 |
| **🔥 PyTorch 编译加速** | 使用 torch.compile 加速推理 |
| **🎛️ 调制引导控制** | 改善 Anima 模型的生成质量 |

---

## ⚙️ 优化功能

### 性能优化

- **ComfyUI 后端重写** - 内存管理、模型补丁、注意力机制等
- **模型加载优化** - 加速启动和模型切换
- **内存泄漏修复** - 切换 checkpoint 时的内存问题
- **uv 包管理器支持** - 大幅加速依赖安装
- **多种注意力优化** - SageAttention、FlashAttention、xFormers
- **Triton 内核** - int8 矩阵乘法加速
- **Spectrum** - 免训练加速所有模型
- **TAESD 实时预览** - 所有模型支持
- **半精度上采样器** - 加速同时降低质量
- **GPU 瓦片合成** - 加速上采样
- **支持更多图像格式** - .avif、.heif、.jxl
- **X/Y/Z 图自动行计数优化**

### 显存优化策略

| 显存大小 | 推荐配置 | 支持模型 |
|---------|---------|---------|
| **4-6 GB** | TAESD + 分块处理 | SD1.5, SD2.1 |
| **8 GB** | 默认配置 | SDXL, Flux.2-Klein 4B |
| **12 GB** | 全功能 | Flux, Anima, Qwen-Image |
| **16 GB+** | 无限制 | Wan 2.2, Flux.2-Klein 9B |

---

## 🔄 模型共享配置

本项目支持与 ComfyUI 双向模型共享：

### WebUI 共享模型给 ComfyUI

在 ComfyUI 的 `extra_model_paths.yaml` 中添加：
```
a111:
    base_path: D:/ai/sd-webui-forge-neo-v3/webui
    checkpoints: models/Stable-diffusion
    loras: models/Lora
    vae: models/VAE
    text_encoder: models/text_encoder
    unet: models/unet
    diffusion_models: models/diffusion_models
    clip: models/clip
```

### ComfyUI 共享模型给 WebUI

创建 `comfy_paths.yaml`：
```
comfyui:
    base_path: D:/ComfyUI-aki-v2.1/ComfyUI/models
    checkpoints: checkpoints
    diffusion_models: diffusion_models
    unet: unet
    clip: clip
    text_encoders: text_encoders
    loras: loras
    vae: vae
```

---

## 🌐 远程访问配置

### 局域网访问

修改启动参数，添加 `--listen --port 7860`：

```
# 在 webui-user.bat 中添加
set COMMANDLINE_ARGS=--listen --port 7860
```

### SSH 隧道访问

```
# 本地执行
ssh -L 7860:localhost:7860 user@remote-server

# 远程服务器必须使用 --listen 参数
./webui.sh --listen --port 7860
```

> [!WARNING]
> 使用 `--listen` 时，服务将绑定到 `0.0.0.0`，允许外部 IP 访问。确保防火墙已正确配置。

---

## ❓ 常见问题

### Q1: 首次启动卡在依赖安装？
**A**: 这是正常现象，首次安装需要 5-15 分钟。请保持网络连接稳定，不要关闭命令行窗口。

### Q2: 提示缺少文本编码器？
**A**: Flux、Anima、Qwen-Image 等模型需要手动下载文本编码器。请参考上方"模型配置要求"章节，从 HuggingFace 下载后放入对应目录。

### Q3: 生成速度慢？
**A**: 
- 检查是否启用了正确的注意力优化（设置 → 优化 → 选择 FlashAttention 或 xFormers）
- 降低分辨率或使用 FP8 量化模型
- 启用 TAESD 实时预览可提升体验

### Q4: 显存不足？
**A**: 
- 启用"显存防溢出保护"（扩展 → 显存防溢出保护）
- 降低分辨率或使用更小的模型
- 关闭其他占用显存的程序

### Q5: 如何更新启动器？
**A**: 在主控台点击"🔄 检查启动器更新"按钮，如有新版本会自动下载并应用。

### Q6: Linux/macOS 支持？
**A**: 目前仅提供 Windows 官方支持。Linux/macOS 用户可尝试从源码运行，但可能遇到兼容性问题。

---

## 📚 学习资源

- **视频教程**: [B站教程](https://www.bilibili.com/video/BV1KfXyBTEXb)
- **Wiki**: [Haoming02 Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)
- **分镜助手教程**: [B站分镜教程](https://www.bilibili.com/video/BV1foQzBMErp/)

---

## 🤝 社区支持

### QQ 交流群

<img src="launcher/qq群ai交流群.jpg" alt="QQ交流群" width="200"/>

扫码加入 AI 交流群，获取：
- 📦 最新整合包下载
- 💡 使用技巧分享
- 🔧 问题解答与支持
- 🎨 作品交流展示

### B站频道

关注 [哔哩哔哩（鸡肉爱土豆）](https://space.bilibili.com/403361177) 获取最新教程和更新通知。

---

## 🙏 致谢

感谢以下项目和贡献者：

- **Haoming02** - [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) 原作者
- **AUTOMATIC1111** - Stable Diffusion WebUI 原始项目
- **lllyasviel** - Forge 优化框架
- **comfyanonymous** - ComfyUI 项目
- **kijai**、**city96** - 社区贡献者
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