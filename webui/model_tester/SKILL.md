# Model Tester Skill（模型测试技能）

## 简介
这是一个用于对比测试不同图像生成模型的智能体技能。支持 Krea2 Turbo、Z-Image Turbo、Anima 三种模型的公平对比测试。

## 功能特点
- **统一测试条件**：所有测试使用相同的提示词和分辨率设置
- **自动模型配置**：根据所选模型自动匹配最佳的采样器和迭代步数
- **图像反推支持**：可以上传参考图片，自动反推提示词进行测试
- **9:16 竖版尺寸**：统一使用 1024×1792 分辨率

## 支持的模型及配置

### Krea2 Turbo（多风格美学）
- 文本编码器 (TE): qwen3vl_4b_fp8_scaled.safetensors
- VAE: qwen_image_vae.safetensors  
- 采样器: DPM++ 2M Karras
- 迭代步数: 20-30

### Z-Image Turbo（精致人像）
- 文本编码器 (TE): qwen_3_4b.safetensors
- VAE: flux-ae.safetensors
- 采样器: Euler a
- 迭代步数: 15-25

### Anima（二次元动漫）
- 文本编码器 (TE): qwen_3_06b_base.safetensors
- VAE: qwen_image_vae.safetensors
- 采样器: DPM++ 2M Karras
- 迭代步数: 20-30

## 使用流程

### 方式一：直接提示词测试
1. 向智能体助手提供要测试的提示词
2. 确认测试请求
3. 技能自动依次调用三个模型生成图片
4. 结果并排展示供对比

### 方式二：图像反推测试
1. 上传一张参考图片
2. 技能自动分析图片并反推提示词
3. 向你展示反推的提示词，确认无误后执行测试
4. 使用反推的提示词对三个模型进行测试

## 注意事项
- 所有测试统一使用 9:16 竖版尺寸（1024×1792）
- CFG Scale 固定为 7.0
- 随机种子设为 -1（每次随机）
- 测试结果可能因模型特性存在差异，这是正常现象

## 文件位置
- SKILL.md: 技能说明文档
- model_tester.py: Python脚本模板
