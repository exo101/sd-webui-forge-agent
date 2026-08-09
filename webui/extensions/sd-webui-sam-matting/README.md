# sd-webui-sam-matting

集成图像分割，智能抠图，图像定点清理功能
# SAM 智能分割与抠图插件 (SAM Matting Extension)

本插件为 Stable Diffusion WebUI Forge 提供强大的图像分割、智能抠图和图像清理功能。

## 集成功能

### 1. Segment Anything 图像分割
- **功能**：基于 Meta 的 Segment Anything Model (SAM) 进行图像分割
- **支持模式**：
  - 自动分割：随机分割图像中的不同区域
  - 手动分割：通过标记点精确控制分割区域
- **模型选项**：
  - vit_h (2.38G)：最大模型，精度最高
  - vit_l (1.25G)：中等模型，平衡性能和资源消耗
  - <img width="1827" height="881" alt="QQ20260324-200953" src="https://github.com/user-attachments/assets/5be17363-f977-4d12-86fe-45af650a6f83" />


### 2. Rembg 智能抠图
- **功能**：使用 rembg 库进行自动背景移除
- **特性**：
  - 支持批量处理
  - 可自定义背景颜色
  - 支持透明背景输出
  - CUDA 加速支持
<img width="1832" height="818" alt="QQ20260324-200844" src="https://github.com/user-attachments/assets/b19fea61-907a-484c-b4fd-6b6290a2fe46" />

### 3. LiteLama 图像清理
- **功能**：使用 LiteLama 模型进行图像物体清理/移除
- **特性**：
  - 基于深度学习的图像修复
  - 支持 GPU 加速
  - 自动保存处理结果
  - <img width="1823" height="729" alt="QQ20260324-201057" src="https://github.com/user-attachments/assets/eb2f6054-5560-4a23-a19a-68269e5aea5b" />


## 安装说明

### 前置要求
- Stable Diffusion WebUI Forge v5.4 或更高版本
- Python 3.10+
- PyTorch 与 CUDA 环境（可选，用于加速）

### 依赖安装

#### 1. Segment Anything 依赖
```bash
pip install segment-anything
```

下载 SAM 模型文件：
- 访问 https://github.com/facebookresearch/segment-anything
- 下载 `sam_vit_h_4b8939.pth` 或 `sam_vit_l_0b3195.pth`
- 将模型文件放置到 `models/sam/` 目录

#### 2. Rembg 依赖（智能抠图）
```bash
pip install rembg onnxruntime-gpu  # GPU 版本
# 或
pip install rembg onnxruntime  # CPU 版本
```

#### 3. LiteLama 依赖（图像清理）
```bash
pip install litelama
```

模型会自动下载到 `models/cleaner/` 目录

## 使用方法

### 启动插件
1. 将本插件放置在 `extensions/` 目录下
2. 启动 WebUI Forge
3. 在顶部标签页中找到"图像分割与清理"标签

### 功能使用

#### 图像分割
1. 上传需要分割的图片
2. 选择模型类型（vit_h 或 vit_l）
3. **自动分割模式**：点击"运行自动分割"按钮
4. **手动分割模式**：在图片上点击添加标记点，然后点击"根据标记点运行分割"
5. 查看分割结果，最多显示 6 个最佳匹配结果

#### 智能抠图
1. 上传一张或多张图片
2. 选择背景颜色（或使用透明背景）
3. 点击"开始处理"按钮
4. 等待处理完成，结果会自动保存到 `outputs/image-matting/` 目录

#### 图像清理
1. 上传带遮罩的图片（使用涂鸦工具标记需要清理的区域）
2. 点击"清理图像"按钮
3. 查看清理结果，结果会自动保存到 `outputs/cleaner/` 目录

## 输出目录

所有处理结果会保存在以下目录：
- 图像分割：`outputs/segment-anything/`
  - `point_segmentation/`：手动分割结果
  - `random_segmentation/`：自动分割结果
- 智能抠图：`outputs/image-matting/`
- 图像清理：`outputs/cleaner/`

## 常见问题

### Q: Segment Anything 模型加载失败
A: 请确保：
1. 已正确安装 `segment-anything` 库
2. 模型文件已下载到正确的目录（`models/sam/`）
3. 模型文件名正确（`sam_vit_h_4b8939.pth` 或 `sam_vit_l_0b3195.pth`）

### Q: 智能抠图速度慢
A: 
- 安装 `onnxruntime-gpu` 以启用 GPU 加速
- 确保 CUDA 环境配置正确

### Q: 图像清理功能不可用
A: 
- 确保已安装 `litelama` 库
- 首次使用时会自动下载模型，请保持网络连接
- 也可以手动下载模型并放置到 `models/cleaner/big-lama.safetensors`

## 开发者信息

- 原插件作者：鸡肉爱土豆
- Bilibili: https://space.bilibili.com/403361177
- 本插件为 MultiModal 插件的独立拆分版本

## 许可证

请遵守原始项目的许可证和使用条款。

## 更新日志

### v1.0.0
- 从 MultiModal 插件分离为独立插件
- 优化模块导入和依赖管理
- 改进错误处理和用户提示

