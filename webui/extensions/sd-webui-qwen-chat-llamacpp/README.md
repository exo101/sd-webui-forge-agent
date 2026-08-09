
# Qwen 图像识别与语言交互插件

本插件为 Stable Diffusion WebUI Forge 提供强大的图像识别和智能对话功能，支持 **Ollama** 和 **llama.cpp** 两种后端。

<img width="1804" height="764" alt="插件界面" src="https://github.com/user-attachments/assets/9179d303-9324-40b3-ae27-862f1264261a" />

## 🌟 新增功能：llama.cpp 支持

插件现已支持 llama.cpp 后端，提供更高的性能和更灵活的模型选择。

### 后端对比

| 特性 | Ollama | llama.cpp |
|------|--------|-----------|
| **易用性** | 一键启动，简单便捷 | 需要手动配置，更灵活 |
| **模型格式** | 自动下载管理 | GGUF 格式，需自行下载 |
| **性能优化** | 自动优化 | 支持自定义参数调优 |
| **GPU 加速** | 自动检测 | 支持 `--n-gpu-layers` 参数 |
| **扩展能力** | 有限 | 支持更多模型和自定义配置 |

## 📦 集成功能

### 1. 图像识别与语言交互
#### 核心功能
- **视觉模型对话**：支持上传图片进行智能识别和对话
- **纯文本对话**：无需图片即可进行文本交流
- **多轮对话**：支持上传一次图片后连续提问
- **模型切换**：视觉模型/语言模型灵活切换
- **后端切换**：Ollama / llama.cpp 一键切换

#### 支持的模型

**Ollama 模型**：
- **视觉模型**（支持图片识别 + 文本聊天）：
  - `qwen3.5:9b` - 高精度版本（推荐 16GB+ 显存）
  - `qwen3.5:4b` - 平衡版本（推荐 12GB 显存）
  - `qwen3-vl:8b` - 视觉语言模型
  - `qwen3-vl:4b` - 中等视觉模型
  - `qwen3-vl:2b` - 轻量级视觉模型（推荐 8GB 显存）

- **语言模型**（仅文本对话）：
  - `qwen3:latest` - 最新版本
  - `qwen3.5:4b` - 平衡版本
  - `deepseek-r1:8b` - DeepSeek 模型

**llama.cpp 模型**（GGUF 格式）：
- **视觉模型**（支持图片识别 + 文本聊天）：
  - `qwen3-vl-2b/4b/8b` - Qwen3 视觉系列
  - `qwen2.5-vl-3b/7b/32b` - Qwen2.5 视觉系列
  - `llava-v1.6-vicuna-7b/13b` - LLaVA 系列
  - `llava-yi-v1.6-6b/9b` - LLaVA-Yi 系列
  - `deepseek-vl-7b/16b` - DeepSeek 视觉系列
  - `cogvlm2-7b/19b` - CogVLM2 系列

- **语言模型**（仅文本对话）：
  - `qwen3.5-0.5b/1.5b/3b/4b/7b/8b/14b/32b/72b` - Qwen3.5 系列（推荐）
  - `qwen2.5-0.5b/1.5b/3b/7b/14b/32b/72b-instruct` - Qwen2.5 系列
  - `hy-mt2-1.8b/7b` - 混元翻译模型
  - `llama-3-8b/70b-instruct` - Llama 3 系列
  - `mistral-7b-instruct-v0.3` - Mistral 系列
  - `mixtral-8x7b-instruct-v0.1` - Mixtral 系列
  - `gemma-2-9b/27b-it` - Gemma 2 系列

### 2. 关键词辅助模板
- 😊 **表情包模板**：角色表情变化描述
- 📖 **创作故事模板**：完整的故事框架生成
- 🎬 **分镜描写模板**：专业分镜脚本描述
- 🎨 **分镜视觉呈现参考**：镜头语言与构图技巧
- 📢 **海报设计模板**：电商海报设计规范
- 👤 **三视图模板**：角色三视图描述

### 3. 标签管理
- 📁 **文件夹管理**：批量处理 txt 文件
- ➕ **关键词添加**：支持开头/结尾/随机位置
- ➖ **关键词删除**：智能匹配删除
- 🔄 **批量操作**：一键对所有文件操作

### 4. 图像管理
- 🖼️ **图片预览**：支持多种格式（PNG/JPG/WebP 等）
- 📂 **目录加载**：快速浏览整个文件夹
- 🎯 **网格展示**：4 列布局，高清预览

### 5. 快捷描述功能
- 📝 **自然语言描述**：正式风格的详细描述
- 🎨 **MidJourney 提示词**：MJ 风格提示词生成
- 🎬 **分镜构图描述**：专业的分镜分析
- 🎥 **图生视频描述**：基于图片的视频生成提示
- 📹 **文生视频描述**：完整的文生视频提示词
- ✨ **简单描述**：快速图像内容概括

## 🚀 安装说明

### 方法一：使用 Ollama（推荐新手）

**步骤 1：安装 Ollama**

1. 访问官网下载：https://ollama.com/
2. 安装 Ollama 到系统

**步骤 2：下载 Qwen 模型**

```bash
# 视觉模型（支持图片识别）
ollama run qwen3.5:4b    # 平衡版（推荐 12GB 显存）
ollama run qwen3-vl:4b   # 视觉模型（推荐 8GB+ 显存）

# 语言模型（仅文本对话）
ollama run qwen3:latest
ollama run qwen3.5:4b
```

**步骤 3：安装插件**

将插件放置在 WebUI 的 `extensions` 目录，重启 WebUI Forge。

---

### 方法二：使用 llama.cpp（高级用户）

**步骤 1：下载 llama.cpp**

1. 访问 GitHub：https://github.com/ggerganov/llama.cpp/releases
2. 下载适合您系统的二进制文件（`llama-server`）
3. 解压到任意目录

**步骤 2：下载 GGUF 模型**

推荐下载源：
- **ModelScope**: https://modelscope.cn/ (国内访问更快)
- **HuggingFace**: https://huggingface.co/

推荐模型：
- 视觉模型：`llava-v1.6-vicuna-7b`、`qwen3-vl-4b`
- 语言模型：`qwen3.5-4b`、`llama-3-8b-instruct`

**步骤 3：启动 llama.cpp 服务器**

**Windows (PowerShell):**
```powershell
./llama-server.exe --model "D:/path/to/model.gguf" --host 0.0.0.0 --port 8080 --n-gpu-layers -1
```

**Linux/macOS:**
```bash
./llama-server --model /path/to/model.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers -1
```

**参数说明:**
- `--model`: 指定 GGUF 模型文件路径
- `--host 0.0.0.0`: 允许外部访问
- `--port 8080`: 服务端口
- `--n-gpu-layers -1`: 将所有层加载到 GPU（0 表示 CPU 模式）
- `--threads`: CPU 线程数（默认 4）
- `--ctx-size`: 上下文窗口大小

**步骤 4：配置插件**

1. 在插件界面选择 **"llama.cpp"** 后端
2. 配置服务器地址（默认 `http://localhost:8080`）
3. 在模型下拉菜单中选择您的模型名称
4. 开始使用！

---

## 📖 使用指南

### 基本操作

1. **选择后端**
   - 新手推荐使用 **Ollama**（简单易用）
   - 高级用户推荐使用 **llama.cpp**（性能更好）

2. **选择模型类型**
   - 需要识别图片：选择"视觉模型"
   - 纯文本对话：选择"语言模型"

3. **上传图片（可选）**
   - 单张图片模式：一次处理一张图片
   - 批量图片模式：同时处理多张图片

4. **输入问题或描述**
   - 可以直接提问关于图片的问题
   - 可以使用快捷描述按钮快速生成提示词

### 功能示例

#### 图像识别示例
```
用户：[上传图片] 这张图片里有什么？
AI: 图片中显示的是一个穿着街头风格服装的动漫角色...
```

#### 提示词生成示例
```
用户：[点击"MidJourney 提示词"按钮]
AI: A futuristic robot holding a glowing drink bottle, 
cyberpunk style, neon lights background, detailed mechanical parts...
```

#### 分镜描述示例
```
用户：[点击"分镜构图描述"按钮]
AI: 第一格：昏暗房间内部，左侧躺地上的人，右侧坐椅子的男人...
```

## ⚙️ API 配置

### Ollama 配置
- **服务器地址**：默认 `http://localhost:11434`
- **超时时间**：默认 300 秒

### llama.cpp 配置
- **服务器地址**：默认 `http://localhost:8080`
- **超时时间**：默认 300 秒

## ❓ 常见问题

### Q1: Ollama 服务无法启动
**A**: 
- 确认已正确安装 Ollama
- 检查端口是否被占用（默认 11434）
- 查看 Ollama 日志获取错误信息

### Q2: llama.cpp 无法加载模型
**A**:
- 确认模型文件路径正确
- 确认模型格式为 GGUF（不支持旧版 GGML）
- 检查 GPU 显存是否足够

### Q3: 显存不足怎么办？
**A**: 
- 选择更小的模型（如 qwen3-vl:2b）
- 关闭其他占用 GPU 的程序
- 降低并发处理数量
- 使用 llama.cpp 时可以减少 `--n-gpu-layers` 参数

### Q4: 图片识别失败
**A**: 
- 确认已下载视觉模型（带 vl 后缀或 llava 前缀）
- 检查图片格式是否支持
- 查看后端服务是否正常运行

### Q5: 如何查看已安装的模型？
**A**:
- **Ollama**: `ollama list`
- **llama.cpp**: 在插件界面点击"刷新模型列表"

### Q6: 可以同时运行 Ollama 和 llama.cpp 吗？
**A**:
- 可以！只要它们使用不同的端口，不会冲突

## 📝 更新日志

### v1.1.0 (2026-06-08)
- ✅ 新增 llama.cpp 后端支持
- ✅ 支持 GGUF 格式模型
- ✅ 添加 llama.cpp API 配置界面
- ✅ 支持后端一键切换（Ollama / llama.cpp）
- ✅ 扩展模型列表，支持更多 GGUF 模型

### v1.0.0 (2026-03-24)
- ✅ 从 MultiModal 插件分离为独立插件
- ✅ 集成 Qwen 系列视觉/语言模型对话
- ✅ 关键词辅助模板功能
- ✅ 标签管理与图像管理
- ✅ 资源汇总与公告板
- ✅ 快捷描述功能

## 📧 开发者信息

- **原插件作者**：鸡肉爱土豆
- **Bilibili**: https://space.bilibili.com/403361177
- **本插件**：从 MultiModal 插件分离的独立版本

## 📄 许可证

本插件遵循原 MultiModal 插件的开源协议。
