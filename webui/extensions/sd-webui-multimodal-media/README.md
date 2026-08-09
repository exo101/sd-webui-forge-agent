# Multimodal Media - 多模态媒体处理插件

## 🚀 重要更新：系统工具现在是可选依赖！

### ✅ Python 依赖（自动安装）

以下 Python 包会在 WebUI 启动时**自动检查并安装**：

| 包名 | 用途 | 必需 |
|------|------|------|
| **insightface** | 人脸检测和分析 | ✅ 必需 |
| **onnxruntime-gpu** | GPU 加速的 ONNX 运行时 | ✅ 必需 |
| **ffmpeg-python** | FFmpeg Python 绑定 | ✅ 必需 |
| **torchaudio** | PyTorch 音频处理 | ✅ 必需 |
| **qwen-tts** | Qwen3-TTS 语音合成 | ✅ 必需 |
| **soundfile** | 音频文件读写 | ✅ 必需 |
| **resampy** | 高质量音频重采样 | ✅ 必需 |
| **librosa** | 音频分析和处理 | ✅ 必需 |
| **dashscope** | 阿里云百炼 SDK（Qwen Video） | ✅ 必需 |
| **Pillow** | 图像处理 | ✅ 必需 |

---

### ⚠️ 系统工具（可选但推荐）

**重要：** FFmpeg 和 SoX 现在是**可选依赖**，插件可以在没有它们的情况下正常加载和运行。Python 库会作为后备方案自动启用。

#### FFmpeg（推荐安装）⭐

**用途：** 视频编解码、音频提取、多媒体转换

**不安装的影响：**
- ✅ 基本功能仍可使用（使用 `ffmpeg-python` 和其他 Python 库）
- ⚠️  某些高级视频处理功能可能受限
- 💡 建议安装以获得完整功能和更好性能

**安装方法：**
```bash
# Windows (推荐)
winget install Gyan.FFmpeg

# 或手动下载
# https://www.gyan.dev/ffmpeg/builds/
```

---

#### SoX（推荐安装）⭐

**用途：** 音频预处理、格式转换、音频效果处理

**不安装的影响：**
- ✅ 插件可以正常使用 `soundfile`、`librosa` 等 Python 库作为替代
- ⚠️  某些 LatentSync 音频处理功能可能降级到纯 Python 实现
- 💡 建议安装以获得最佳音频处理性能

**安装方法：**
```bash
# Windows (推荐)
winget install DavidHetherington.Sox

# 或手动下载
# https://sourceforge.net/projects/sox/files/sox/
```

**环境变量配置：**
```
❌ 错误：C:\sox-14.4.2
✅ 正确：C:\sox-14.4.2\bin
```

⚠️ **重要：** 必须将 `bin` 子目录添加到 PATH，而不是根目录！

---

## 🔧 快速安装（一键脚本）

### 方式一：自动安装脚本（推荐）

**双击运行：**
```
D:\sd-webui-forge-classic-neo\extensions\sd-webui-multimodal-media\quick_install_tools.bat
```

脚本会自动：
- ✅ 检查 FFmpeg 和 SoX 是否已安装
- ✅ 使用 winget 自动安装缺失的工具
- ✅ 提供清晰的安装状态反馈

---

### 方式二：手动命令行安装

打开 **PowerShell（管理员）**，执行：

```powershell
# 安装 FFmpeg
winget install Gyan.FFmpeg

# 安装 SoX
winget install DavidHetherington.Sox

# 验证安装
ffmpeg -version
sox --version
```

---

## 📊 启动行为说明

### 现在的启动流程：

```
WebUI 启动
  ↓
Multimodal Media 插件加载
  ↓
检查 Python 依赖（自动安装缺失项）
  ↓
✅ 插件成功加载并显示在界面中
  ↓
运行时检测系统工具（FFmpeg/SoX）
  ↓
如果缺失 → 使用 Python 库替代 + 显示提示信息
如果有 → 使用系统工具获得更好性能
```

### 预期控制台输出：

```
✅ Multimodal Media Python dependencies loaded (system tools like FFmpeg/SoX are optional)
✅ Multimodal Media 插件已准备就绪
✅ Multimodal Media plugin is ready

[如果检测到系统工具缺失，会在真正需要使用时提示]
```

---

## 🎬 功能标签页说明

重启 WebUI 后，您将在界面中看到 **Multimodal Media** 标签页，包含以下子功能：

### 1️⃣ Qwen3-TTS 语音合成

**功能：**
- 文本转语音（支持多种语音风格）
- 基础音色选择（中文/英文/日语等）
- 定制声音选择（需指定声音 ID）
- 声音设计器模式（自定义音调、语速、情感）
- 批量处理支持

**依赖：**
- ✅ qwen-tts（Python 包）
- ⚠️  FFmpeg/SoX（可选，用于音频后处理）

---

### 2️⃣ 数字人对口型生成（LatentSync）

**功能：**
- 根据图片 + 音频生成对口型视频
- 支持 LatentSync 模型
- 可调节推理步数、引导系数
- 支持种子控制和 DeepCache 加速

**依赖：**
- ✅ insightface（人脸检测）
- ✅ onnxruntime-gpu（GPU 加速）
- ⚠️  FFmpeg/SoX（推荐安装，提升视频处理性能）

---

### 3️⃣ Qwen Video 万相视频生成（新增）⭐

**功能：**
- **图生视频 (wan2.6-i2v)**: 上传图片生成视频
- **图生视频 (wan2.5-i2v)**: 经典图生视频模型
- **关键帧生视频 (wan2.2-kf2v)**: 从关键帧生成视频
- **文生视频 (wan2.5-t2v)**: 纯文本生成视频
- 支持 API 调用（阿里云百炼平台）
- 支持任务查询和历史记录查看
- 分辨率、时长、音频可配置

**依赖：**
- ✅ dashscope（阿里云百炼 SDK）
- ✅ Pillow（图像处理）
- ✅ requests（HTTP 请求）
- ⚠️  API Key（必需，在界面中设置）

**使用说明：**
1. 获取阿里云百炼 API Key：https://dashscope.console.aliyun.com/
2. 在界面中输入 API Key 并点击"设置"
3. 选择视频生成模式（图生视频/文生视频等）
4. 上传参考图片（如果需要）和输入提示词
5. 配置分辨率、时长等参数
6. 点击"生成视频"按钮
7. 等待任务完成（异步处理）
8. 查看和下载生成的视频

---

### 4️⃣ 视频关键帧提取

**功能：**
- 从视频中提取帧图像
- 支持按时间间隔或帧数提取
- 预览和批量导出
- 支持多种视频格式

**依赖：**
- ✅ ffmpeg-python
- ⚠️  FFmpeg（推荐安装，提升处理速度）

---

## ⚠️ 常见问题排查

### ❌ "SoX could not be found!"

**这不是致命错误！** 插件仍然可以正常工作。

**含义：** SoX 未安装，LatentSync 模块会使用 Python 库作为替代方案。

**解决方案（可选）：**
1. 运行 `quick_install_tools.bat` 自动安装
2. 或手动安装后确保 `sox --version` 可以执行
3. **重启终端和 WebUI**

---

### ❌ "FFmpeg not found"

**这也不是致命错误！** 基本功能不受影响。

**含义：** FFmpeg 未安装，某些视频处理功能可能受限。

**解决方案（推荐）：**
1. 运行 `quick_install_tools.bat` 自动安装
2. 或手动安装后确保 `ffmpeg -version` 可以执行
3. **重启终端和 WebUI**

---

### ✅ 插件能正常显示吗？

**是的！** 即使 FFmpeg 和 SoX 都未安装，插件也能：
- ✅ 正常加载并显示在 WebUI 界面中
- ✅ 使用 Python 库执行基本功能
- ✅ 提供完整的用户界面
- ⚠️  某些高级功能可能提示安装系统工具

---

## 🔍 验证系统工具安装

### 检查当前环境：

打开命令提示符，执行：

```bash
# 检查 FFmpeg
where ffmpeg
ffmpeg -version

# 检查 SoX
where sox
sox --version

# 如果显示路径和版本信息，说明安装成功！
```

---

### 使用检测工具：

```bash
cd D:\sd-webui-forge-classic-neo\extensions\sd-webui-multimodal-media
python scripts\runtime_tools_check.py
```

输出示例：
```
=== Multimodal Media System Tools Check ===

FFmpeg: ✅ Available
SoX: ❌ Not found

⚠️  以下系统工具未安装（可选，但推荐安装以获得更好体验）:
SoX:
  Windows: winget install DavidHetherington.Sox
  或访问：https://sourceforge.net/projects/sox/files/sox/

💡 提示：这些工具是可选的，插件可以使用 Python 库作为替代方案
   但安装后可以获得更好的性能和兼容性
```

---

## 💡 设计哲学

### 为什么改为可选依赖？

1. **用户体验优先**：不让外部工具阻碍插件加载和显示
2. **渐进式增强**：有系统工具时用系统工具，没有时用 Python 库
3. **灵活部署**：适合不同技术水平的用户
4. **降低门槛**：新手可以先用基础功能，再逐步完善环境

---

## 🎯 下一步操作

1. **立即重启 WebUI**：插件现在应该能正常显示了
2. **测试基本功能**：即使没有 FFmpeg/SoX，大部分功能也能工作
3. **（推荐）安装系统工具**：运行 `quick_install_tools.bat` 获得完整体验
4. **验证安装**：运行 `runtime_tools_check.py` 查看当前状态
5. **配置 Qwen Video**：
   - 获取阿里云百炼 API Key
   - 在界面第 3 个标签页设置
   - 开始生成视频！

---

## 🙏 致谢

- **Qwen3-TTS**: 阿里巴巴通义实验室
- **Qwen Video (万相)**: 阿里云百炼平台
- **LatentSync**: MIT 研究者
- **InsightFace**: 开源人脸分析库
- **SoX**: Sound eXchange 项目团队
- **FFmpeg**: 开源多媒体框架
- **Librosa**: Python 音频分析库

## 🎬 Multimodal Media - 多媒体处理工具

一个集成多种 AI 多媒体功能的 Stable Diffusion WebUI 扩展，提供语音合成、视频生成和视频分析功能。

### ✨ 主要功能

#### 1. Qwen3-TTS 语音合成
- 支持 Base、CustomVoice、VoiceDesign 三种模型
- 语音克隆、自定义音色、声音设计
- 多语言支持（中文、英文、日文等）

#### 2. 数字人对口型生成
- 基于 LatentSync 技术
- 音频驱动唇形同步
- 高质量数字人视频生成

#### 3. 视频关键帧提取
- 智能视频帧提取
- 支持多种提取模式
- 批量处理能力

#### 4. Qwen Video 万相视频生成
- 阿里云百炼视频生成 API
- 文本到视频转换
- 高质量视频内容创作

#### 5. **IndexTTS-2 语音合成** ⭐ NEW
- **高保真音色克隆**：通过参考音频精确克隆目标音色
- **情感控制**：4 种情感控制模式
  - 与音色参考音频相同
  - 使用情感参考音频
  - 使用情感向量控制（8维）
  - 使用情感描述文本控制（实验性）
- **多语言支持**：中文、英文
- **高级参数调节**：温度、top-p、top-k、重复惩罚等
- **发送到分镜**：一键将生成的音频添加到分镜助手

---

### 📦 安装说明

#### 自动安装
1. 将本扩展放入 `webui/extensions` 目录
2. 重启 WebUI，插件会自动安装依赖

#### 手动安装依赖
```bash
pip install insightface onnxruntime-gpu ffmpeg-python torchaudio
pip install qwen-tts soundfile resampy librosa
pip install dashscope Pillow omegaconf modelscope
```

---

### 📁 模型下载

#### IndexTTS-2 模型
- **存储位置**：`WebUI根目录/models/indextts-2`
- **下载方式 1（推荐）**：在 UI 中点击"📥 下载/更新 IndexTTS-2 模型"按钮
- **下载方式 2（手动）**：从魔搭社区下载
  - 地址：https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/files
  - 必需文件：
    - `bpe.model`
    - `gpt.pth`
    - `config.yaml`
    - `s2mel.pth`
    - `wav2vec2bert_stats.pt`

#### Qwen3-TTS 模型
- **存储位置**：`WebUI根目录/models/qwen3-tts`
- 首次使用时自动下载

#### Whisper 模型（用于语音识别）
- **存储位置**：`WebUI根目录/models/whisper-tiny`
- 用于 Base 模型的自动语音识别功能

---

### 🚀 使用指南

#### IndexTTS-2 快速开始

1. **准备参考音频**
   - 录制或上传 3-10 秒清晰的人声音频
   - 建议：安静环境、无背景噪音、语速适中

2. **输入文本**
   - 在文本框中输入要合成的文字
   - 支持中文和英文

3. **选择情感模式**
   - **与音色相同**：保持参考音频的中性情感
   - **情感参考音频**：上传另一段带目标情感的音频
   - **情感向量控制**：手动调节 8 维情感向量（需要试验）
   - **情感描述文本**：用文字描述情感（如"开心的"、"悲伤的"）

4. **调整参数**
   - **情感权重**：控制情感强度（推荐 0.8-1.5）
   - **温度**：较低更稳定，较高更多样
   - **Top-p/Top-k**：控制采样多样性

5. **生成音频**
   - 点击"🎵 生成语音"按钮
   - 等待推理完成（首次加载模型较慢）

6. **后续操作**
   - 试听生成的音频
   - 点击"📤 发送到分镜"添加到分镜助手
   - 点击"📁 打开输出目录"查看文件

#### 情感控制技巧

**方法 1：使用情感参考音频（推荐）**
```
步骤：
1. 上传音色参考音频（张三的声音）
2. 选择"使用情感参考音频"模式
3. 上传情感参考音频（李四开心说话的音频）
4. 设置情感权重为 1.0-1.2
5. 生成 → 得到"张三开心说话"的音频
```

**方法 2：使用情感向量控制（高级）**
```
8 维向量代表不同情感特征：
- 维度 1-2：积极/消极
- 维度 3-4：强/弱
- 维度 5-6：快/慢
- 维度 7-8：其他特征

需要多次试验找到合适的组合
```

**方法 3：使用情感描述文本（实验性）**
```
示例描述：
- "开心的、兴奋的"
- "悲伤的、低落的"
- "愤怒的、激动的"
- "温柔的、平静的"
```

---

### ⚙️ 系统要求

- **GPU**：NVIDIA RTX 3060+（推荐显存 ≥8GB）
- **CUDA**：11.8 或更高版本
- **Python**：3.10 - 3.12
- **磁盘空间**：至少 10GB（用于模型文件）

---

### 🐛 常见问题

#### Q1: 模型下载失败怎么办？
**A**: 
1. 检查网络连接
2. 尝试手动下载：https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/files
3. 将下载的文件放到 `models/indextts-2` 目录

#### Q2: 生成时显存不足（OOM）
**A**:
1. 降低 `max_mel_tokens` 参数（减少音频长度）
2. 关闭 FP16（如果已启用）
3. 确保没有其他程序占用显存

#### Q3: 合成效果不理想
**A**:
1. 提高参考音频质量（清晰、无噪音）
2. 调整情感权重（尝试 0.8-1.5 范围）
3. 降低温度参数（0.6-0.8 更稳定）
4. 尝试不同的情感控制模式

#### Q4: 长文本如何处理？
**A**: IndexTTS-2 会自动分段处理长文本，可通过"每段最大文本 Token 数"参数控制分段大小（默认 120）

---

### 📝 更新日志

#### v1.0.0 (2026-05-11)
- ✨ 新增 IndexTTS-2 语音合成功能
- ✨ 支持音色克隆和情感控制
- ✨ 集成到 Multimodal Media 标签页
- ✨ 支持一键下载到分镜助手

---

### 🙏 致谢

- **IndexTTS-2**: [IndexTeam](https://www.modelscope.cn/models/IndexTeam/IndexTTS-2)
- **Qwen3-TTS**: [Alibaba Cloud](https://huggingface.co/Qwen)
- **LatentSync**: ByteDance
- **Stable Diffusion WebUI**: [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

### 📄 许可证

本项目遵循原项目的许可证协议。IndexTTS-2 模型的使用请遵循其官方许可证。
