# See-Through WebUI Plugin

## 功能介绍

See-Through 是一个基于 AI 的动漫图像分解插件，可以将单张动漫图像转换为多层 PSD 文件。这是基于 [See-Through: Single-image Layer Decomposition for Anime Characters](https://github.com/shitagaki-lab/see-through) 项目的研究成果。

<img width="895" height="787" alt="QQ20260403-224612" src="https://github.com/user-attachments/assets/a07952c2-7ddb-4002-bd49-adc7a14fc588" />

<img width="1271" height="752" alt="QQ20260403-224809" src="https://github.com/user-attachments/assets/176574fc-9716-4c02-8ef8-21ae9766d78d" />

<img width="1412" height="686" alt="QQ20260403-224635" src="https://github.com/user-attachments/assets/463960bd-fa89-443b-b3b4-2545aaeed34f" />


## 核心功能

- **图像分解** - 将单张动漫图像分解为多个语义层
- **LayerDiff 3D** - 基于扩散的透明层生成
- **Marigold 深度估计** - 专为动漫优化的伪深度估计
- **SAM 分割** - 语义身体部分分割
- **PSD 输出** - 导出为可编辑的 Photoshop 文件
- **自动依赖安装** - WebUI 启动时自动安装缺失的依赖
- **两种分割模式** - 人物分割和场景分割
- **V3 模型** - 23 层身体部位分离，更精细的图层分解
- **批量处理** - 支持批量上传多张图片依次处理
- **预览效果** - 处理完成后自动展示所有分离的图层

## 插件界面说明

### 图像输入

| 输入模式 | 说明 |
|---------|------|
| 使用生成的图像 | 使用 txt2img/img2img 生成的图像（暂不支持） |
| 上传图像 | 上传本地图像进行处理（推荐） |
| 指定图像路径 | 输入图像文件的完整路径 |

### 处理选项

| 选项 | 默认值 | 说明 |
|-----|-------|------|
| 保存为 PSD 文件 | 启用 | 将分解的图层保存为 PSD 格式 |
| 处理分辨率 | 1024 | 较低的分辨率可减少显存占用，提高处理速度 |
| 推理步数 | 30 | 较低的步数可加快速度，但可能降低质量（推荐 20-30） |

### 分割模式

#### 1. 人物分割

适用于动漫人物、写实人物、人物插画等。

**人物分割选项：**

| 选项 | 默认值 | 说明 |
|-----|-------|------|
| 使用 LayerDiff 3D | 启用 | 基于扩散的透明层生成 |
| 使用 Marigold 深度估计 | 启用 | 专为动漫优化的深度估计 |
| 使用 SAM 分割 | 启用 | 语义身体部分分割 |
| 启用左右分离 | 启用 | 将手、脚、耳等部位分为左右两部分 |

> V3 模型支持 23 个身体部位标签：front hair、back hair、headwear、face、irides、eyebrow、eyewhite、eyelash、eyewear、ears、earwear、nose、mouth、neck、neckwear、topwear、handwear、bottomwear、legwear、footwear、tail、wings、objects

#### 2. 场景分割 (SAM)

适用于复杂场景的自动分割。

**场景分割选项：**

| 选项 | 默认值 | 说明 |
|-----|-------|------|
| 最大分割数量 | 10 | 控制分割出的元素数量 |
| 最小区域大小 | 1000 | 过滤小于此面积的分割区域（像素） |
| SAM 模型类型 | vit_b | vit_b: 小模型（~1.2GB）, vit_l: 中模型（~2.5GB）, vit_h: 大模型（~3.9GB） |

### 内存优化选项

| 选项 | 默认值 | 说明 |
|-----|-------|------|
| 使用 NF4 量化 (8GB GPU) | 启用 | 使用 4 位量化模型权重，峰值显存~8GB |
| 缓存文本嵌入 | 启用 | 节省约 2GB 显存且无速度损失 |
| 启用组卸载 | 禁用 | 将显存降至~0.2GB，但速度降低 2-3 倍 |
| 启用 CPU 卸载 | 禁用 | 将模型卸载到 CPU 以节省显存 |

## 安装方法

### 自动安装

1. 将插件目录复制到 WebUI 的 extensions 文件夹中
2. 重启 WebUI
3. 插件会自动检测并安装缺失的依赖项（psd-tools、pycocotools）

### 手动安装依赖

如果自动安装失败，可以手动安装依赖项：

```bash
python -m pip install psd-tools
python -m pip install pycocotools
```

## 使用方法

### 基本使用步骤

1. 在 txt2img 或 img2img 界面中找到 **"See-Through 图层分离为 PSD 文件"** 折叠面板
2. 勾选 **"启用 See-Through"**
3. 选择 **输入模式**：
   - 推荐选择 **"上传图像"**，然后上传本地图像
   - 或选择 **"指定图像路径"**，输入图像文件的完整路径
4. 选择 **分割模式**：
   - **人物分割**：适用于动漫人物、写实人物等
   - **场景分割 (SAM)**：适用于复杂场景的自动分割
5. 根据显存大小调整 **处理分辨率**、**推理步数**
6. 根据需要启用 **内存优化选项**
7. 点击 **"处理"** 按钮或使用 **"批量处理"** 上传多张图片
8. 处理完成后自动展示所有分离的图层预览
9. 点击 **"打开输出目录"** 查看结果

### 显存配置建议

| 显存大小 | 处理分辨率 | 推荐设置 |
|---------|----------|---------|
| 8GB | 512-768 | 启用 NF4 量化、缓存文本嵌入 |
| 12GB | 768-1024 | 启用缓存文本嵌入 |
| 16GB | 1024-1280 | 默认设置即可 |
| 24GB+ | 1280-1536 | 可使用较大分辨率 |


每个处理任务都有独立的输出目录，包含：

- 分层的 PSD 文件（如果启用）
- 各个图层的 PNG 文件
- 深度图（如果启用）
- 分割掩码（如果启用）

## 适用场景

### 最佳场景
- 动漫人物图像
- 写实人物图像
- 人物为主的插画
- 包含人物的电商海报

### 有限支持
- 动物图像：可能只能获得粗略的分割结果
- 电商海报：可以识别海报中的人物部分
- 插画：适合处理人物为主的插画

### 不支持
- 纯风景场景（山石、天空、大地、海面等）
- 复杂场景
- 大量文字的图像

## 依赖项

### 基础依赖
- numpy
- opencv-python
- Pillow
- torchvision
- transformers

### See-Through 核心依赖
- segment-anything
- groundingdino-py
- diffusers
- accelerate
- omegaconf
- einops
- pytorch-lightning

### 图像处理
- opencv-contrib-python
- scikit-image
- imageio

### COCO 数据集工具
- pycocotools

### PSD 文件处理
- psd-tools

## 常见问题

**Q: 处理失败怎么办？**
A: 检查 See-Through 项目是否正确安装，确保所有依赖项都已安装。查看 WebUI 后台日志获取详细错误信息。

**Q: 显存不足怎么办？**
A: 尝试以下方法：
1. 降低处理分辨率（512-768）
2. 启用 NF4 量化
3. 启用缓存文本嵌入
4. 启用组卸载（最后手段）

**Q: 输出的 PSD 文件如何使用？**
A: 可以在 Photoshop 或其他支持 PSD 的软件中打开，每个图层都是独立的，可以进行编辑和动画制作。

**Q: 左右分离功能不起作用？**
A: 确保在插件 UI 中勾选了"启用左右分离"选项。某些图像可能不包含可分离的部位。

**Q: 处理结果会被覆盖吗？**
A: 不会。每次处理都会生成唯一的输出目录，基于时间戳命名，确保不会相互覆盖。

**Q: 人物分割和场景分割有什么区别？**
A: 
- **人物分割**：专门针对人物图像，可以将人物分解为多个语义层（头发、眼睛、衣服等）
- **场景分割 (SAM)**：使用 SAM 模型自动分割场景中的各个元素，适合复杂场景

## 模型位置

### 人物分割模型
- **Marigold 深度估计模型**：`sd-webui-forge-neo-v3\webui\models\diffusers\models--24yearsold--seethroughv0.0.1_marigold_nf4`
- **LayerDiff 3D 模型**：`sd-webui-forge-neo-v3\webui\models\diffusers\models--24yearsold--seethroughv0.0.2_layerdiff3d_nf4`

### 场景分割模型
- **SAM 模型**：`sd-webui-forge-neo-v3\webui\models\sams`
  - `sam_vit_b_01ec64.pth`（小模型）
  - `sam_vit_h_4b8939.pth`（大模型）
  - `sam_vitl0b3195.pth`（中模型）

### 模型下载说明
1. 所有模型均已包含在整合包中
2. 如需更新或单独下载模型，请从群主网盘中获取

## 项目参考

本插件基于以下研究项目：
- [See-Through GitHub](https://github.com/shitagaki-lab/see-through)
- 论文：See-through: Single-image Layer Decomposition for Anime Characters
- 发表于：ACM SIGGRAPH 2026

## 技术支持

如有问题，请参考：
- See-Through 项目的 GitHub 仓库
- WebUI 的扩展文档
- 查看 WebUI 后台日志获取详细错误信息
