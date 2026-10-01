# 📦 更新日志

本文件记录每次内核更新（git 提交）的新增内容。
启动器「Update 内核更新」成功后会自动解析本文件，并弹窗提示本次更新的新增内容。

> 格式约定：每个 `## 日期 (commit: <7位提交号>)` 为一节，启动器按提交号匹配本次 pull 到的新提交。
> 子节用 `### 新增` / `### 优化` / `### 修复` 分类，条目用 `- ` 开头。

## 2026-10-01 (commit: da49890, ...)

### 新增

- **全栈式视听工作台（H3 Studio）**：整合视频生成、视频关键帧提取、声音合成、音乐生成为统一工作台，支持文生视频（FL2VA）、参考生视频（REF2VA）、首尾帧生视频三种模式
- **MiniMax-H3 视频生成**：基于 DiffSynth Pipeline 本地推理，支持 INT8（Comfy-Org）与 NF4（DiffSynth-Studio）两种量化格式，自动匹配对应 VAE、文本编码器
- **视频关键帧提取**：支持从参考视频中提取关键帧用于视频生成
- **声音合成（Breeze-TTS-2）**：支持声音克隆、声音设计、声音引导三种模式，开源权重双语 TTS（中文/英文）
- **音乐生成**：基于 ACE-Step 1.5 的音乐生成能力
- **Qwen-Image-2.1**：本地图像生成与编辑模型，Comfy-Org int8 convrot 引擎
- **开源社区模型下载器**：一键下载 MiniMax-H3（INT8/NF4）、Qwen-Image-2.1、Breeze-TTS-2 等模型组合，自动放置到对应目录
- **多媒体功能整合**：原 `sd-webui-multimodal-media` 插件已合并入 H3 Studio 工作台，统一入口管理视频、音频、音乐能力

### 优化

- **H3 Studio 显存/内存策略**：四档自适应策略（高性能 / 省内存 / 省显存 / 极致省），自动检测 VRAM 与 RAM 选择最优配置
- **H3 Studio LoRA 支持**：扫描 `models/Lora` 目录中的 H3 LoRA，支持一键加载 H3 Turbo、风格或角色 LoRA
- **H3 Studio INT8 组合**：下载器 INT8 组合现已包含原版视频/音频 VAE（fp16/fp32），无需单独下载
- **按需启用功能**：插件功能较多，并非所有人都需要全部功能，可通过启动器「插件管理」按需禁用/启用子功能，避免不必要的内存占用

### 修复

- **VAE 加载兼容性**：修复音频 VAE weight norm 格式差异导致的 missing keys 问题，自动拆分 `weight` → `weight_g` + `weight_v`
- **VAE 多余键过滤**：移除 `latents_mean` / `latents_std` 等非模型键，避免 strict 加载报错
- **SDXL VAE 残留**：Agent 切换模型时自动清除不兼容的 `forge_additional_modules` 和 `sd_vae`，根治 SDXL 生成浆糊问题
- **SDXL 尺寸规范**：本地 SDXL 模型改用标准尺寸（9:16 → 768x1344），避免因尺寸过大导致生成质量下降

### ⚠️ 重要：transformers 版本隔离

由于不同模块对 transformers 版本要求冲突，**音乐生成与声音合成（Breeze-TTS / ACE-Step）必须部署在隔离的 Python 环境中**：

| 模块 | transformers 版本 | 说明 |
|------|------------------|------|
| MiniMax-H3 视频生成 | **5.17.0**（新版） | Qwen3VL 文本编码器依赖新版 API |
| Qwen-Image-2.1 | **5.17.0**（新版） | 同上 |
| Breeze-TTS-2 声音合成 | **4.57.3**（旧版） | 声音克隆模型仅兼容旧版 transformers |
| ACE-Step 音乐生成 | **4.57.3**（旧版） | 同上 |

**部署方式**：
- 主环境（WebUI）使用 transformers 5.17.0，运行视频生成与图像生成
- 为声音合成与音乐生成创建独立虚拟环境（transformers 4.57.3），通过子进程调用

### 💡 使用提示

- **按需启用**：H3 Studio 功能丰富，如不使用某项功能，可在启动器「插件管理」中禁用对应子模块，降低内存压力
- **求助渠道**：遇到问题可加入用户交流群，或直接询问内置的「绘梦智能体助手」——它了解 WebUI 中的一切功能，可协助排查问题、指导操作

## 2026-09-28 (commit: 9e9ceb1, 90fe03f, 81344f3)

### 新增
- Qwen-Image-2.1 本地模型：基于 Comfy-Org int8 convrot 引擎，预设 Euler / 30 步 / CFG 1.0，模型下载器提供「qwen-image-2.1」一键组合
- MiniMax-H3 本地视频模型：INT8 版（Comfy-Org）与 NF4 4bit 量化版（DiffSynth-Studio/MiniMax-H3-NF4），模型下载器提供「MiniMax-H3-INT8」「MiniMax-H3-NF4」一键组合
- H3 Studio 本地后端：为 MiniMax-H3 系列模型提供本地推理后端
- 绘梦智能体助手新增 @图层分离（See-Through）与 @自动发现插件 标签
- 启动器内核更新完成后自动弹窗展示本次更新的新增内容

### 优化
- 绘梦智能体助手 @qwen 标签升级为 Qwen-Image-2.1 上下文编辑模型
- PS 插件模型列表与绘梦智能体助手同步（PS 插件 2.1.0）

### 修复
- 内核更新：拉取前会连同未跟踪文件一起暂存（git stash --include-untracked），更新成功后自动恢复本地改动，避免「untracked working tree files would be overwritten」导致更新失败、本地配置（如 API Key）丢失
