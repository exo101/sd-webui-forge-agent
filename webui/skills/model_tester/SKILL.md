# Model Tester Skill（模型测试技能）

---

## 功能描述

本技能用于在 Krea2、Z-Image、Anima 三种模型之间进行公平对比测试，确保使用相同的提示词和分辨率（9:16），但根据各模型的推荐配置使用对应的采样器和迭代步数。

## 支持的模型

- **Krea2 Turbo** - 多风格美学模型，适合插画/概念艺术
- **Z-Image Turbo** - 快速生图模型
- **Anima** - 二次元高质量专用模型

## 各模型推荐配置

| 模型 | 文本编码器 (TE) | VAE | 采样器建议 | 迭代步数建议 |
|------|----------------|-----|-----------|-------------|
| Krea2 Turbo | qwen3vl_4b_fp8_scaled.safetensors | qwen_image_vae.safetensors | DPM++ 2M Karras | 20-30 |
| Z-Image Turbo | qwen_3_4b.safetensors | flux-ae.safetensors | Euler a | 15-25 |
| Anima | qwen_3_06b_base.safetensors | qwen_image_vae.safetensors | DPM++ 2M Karras | 20-30 |

## 使用流程

### 1. 获取用户输入

首先询问用户：
- 想要测试的提示词（或上传图片进行图像反推）
- 是否需要进行模型对比测试

### 2. 确认并执行

根据用户提供的提示词，依次调用三个模型生成图片进行对比。

### 3. 结果展示

将三张生成的图片并排展示，供用户对比效果差异。

## 注意事项

- 所有测试使用统一的 9:16 竖版尺寸（1024x1792）
- 提示词由用户提供或从上传图片反推
- 各模型的采样器和迭代步数根据模型特性自动选择最优配置
