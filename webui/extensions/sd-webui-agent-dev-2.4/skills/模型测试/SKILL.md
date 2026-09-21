# Model Tester Skill（模型测试技能）

## 简介
这是一个用于对比测试不同图像生成模型的智能体技能。支持 Krea2 Turbo、Z-Image Turbo、Anima 三种模型的公平对比测试。

## 功能特点
- **统一测试条件**：所有测试使用相同的提示词和分辨率设置
- **模型专属参数继承**：采样器、调度器和迭代步数使用智能体内部按模型架构定义的 preset，不由技能自行写死
- **图像反推支持**：可以上传参考图片，自动反推提示词进行测试
- **9:16 竖版尺寸**：统一使用 1024×1792 分辨率

## 支持的模型及配置

### Krea2 Turbo
- 文本编码器 (TE): qwen3vl_4b_fp8_scaled.safetensors
- VAE: qwen_image_vae.safetensors  
- 采样器/调度器/迭代步数：使用 Krea2 对应的智能体 preset

### Z-Image Turbo
- 文本编码器 (TE): qwen_3_4b.safetensors
- VAE: flux-ae.safetensors
- 采样器/调度器/迭代步数：使用 Z-Image 对应的智能体 preset

### Anima（二次元专用）
- 文本编码器 (TE): qwen_3_06b_base.safetensors
- VAE: qwen_image_vae.safetensors
- 采样器/调度器/迭代步数：使用 Anima 对应的智能体 preset

## 使用流程

### 方式一：直接提示词测试
1. 向智能体助手提供要测试的提示词
2. 确认测试请求。
3. 依次切换模型并调用 `txt2img`。**不要在 `txt2img` 参数中填写 `steps`、`sampler_name`、`cfg_scale` 或调度器参数**，让智能体根据当前模型的 `preset_arch` 自动使用对应 preset；不要用 WebUI 全局默认值覆盖模型 preset。
4. 三个模型使用统一提示词和尺寸，但采样器、调度器、迭代步数和 CFG 可以按各自模型 preset 不同；最终对比表必须记录每个模型实际返回的参数。
5. 三个模型都生成完成后，必须调用 `stitch_images`，把本轮三张结果按 3 列拼成一张横向对比图；不要只把三张图依次发出。`images` 参数传入三张结果，`labels=["Krea2 Turbo", "Z-Image Turbo", "Anima"]`，`columns=3`，`padding=16`，`background_color=(245,245,245)`。
6. 最终回复必须附带一个 Markdown 对比表，表格至少包含：模型、实际使用的模型/采样参数、输出尺寸、观察结论；对比图作为表格前的总览图展示。
7. 对比图完成后必须调用 `create_model_comparison_html`，传入三张本轮结果、模型名称、提示词和参数备注，生成可点击打开的 HTML 文件；最终回复必须明确给出该 HTML 文件路径/链接。

### 方式二：图像反推测试
1. 上传一张参考图片
2. 技能自动分析图片并反推提示词
3. 向你展示反推的提示词，确认无误后执行测试
4. 使用反推的提示词对三个模型进行测试

## 注意事项
- 所有测试统一使用 9:16 竖版尺寸（1024×1792）
- 只有用户明确指定参数时才覆盖模型 preset；随机种子默认使用 `-1`，尺寸保持本技能规定的 1024×1792，模型专属采样参数由智能体 preset 决定。
- 测试结果可能因模型特性存在差异，这是正常现象
- **对比结果交付格式**：先生成全部图片，再调用 `stitch_images` 生成总览图，最后用表格总结；若某个模型失败，也要在表格中保留该模型并写明失败原因，不能静默跳过。
- HTML 对比页必须使用自包含图片数据，打开文件时不依赖临时图片路径。

## 文件位置
- SKILL.md: 技能说明文档
- model_tester.py: Python脚本模板
