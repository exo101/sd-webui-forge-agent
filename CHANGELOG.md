# 📦 更新日志

本文件记录每次内核更新（git 提交）的新增内容。
启动器「Update 内核更新」成功后会自动解析本文件，并弹窗提示本次更新的新增内容。

> 格式约定：每个 `## 日期 (commit: <7位提交号>)` 为一节，启动器按提交号匹配本次 pull 到的新提交。
> 子节用 `### 新增` / `### 优化` / `### 修复` 分类，条目用 `- ` 开头。

## 2026-09-28 (commit: 9e9ceb1)

### 新增
- Qwen-Image-2.1 本地模型：基于 Comfy-Org int8 convrot 引擎，预设 Euler / 30 步 / CFG 1.0，模型下载器提供「qwen-image-2.1」一键组合
- MiniMax-H3 本地视频模型：INT8 版（Comfy-Org）与 NF4 4bit 量化版（DiffSynth-Studio/MiniMax-H3-NF4），模型下载器提供「MiniMax-H3-INT8」「MiniMax-H3-NF4」一键组合
- H3 Studio 本地后端：为 MiniMax-H3 系列模型提供本地推理后端
- 绘梦智能体助手新增 @图层分离（See-Through）与 @自动发现插件 标签
- 启动器内核更新完成后自动弹窗展示本次更新的新增内容

### 优化
- 绘梦智能体助手 @qwen 标签升级为 Qwen-Image-2.1 上下文编辑模型
- PS 插件模型列表与绘梦智能体助手同步（PS 插件 2.1.0）
