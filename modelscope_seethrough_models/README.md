# See-through 图层分离模型

本仓库用于保存 See-through 单图动漫角色图层分离流程所需的两个 Diffusers 模型：

| 模型 | 用途 | 建议目录 |
|---|---|---|
| `seethroughv0.0.1_marigold` | 从输入图像和分离图层估计伪深度 | `seethroughv0.0.1_marigold/` |
| `seethroughv0.0.2_layerdiff3d` | 生成带透明通道的语义图层（SDXL） | `seethroughv0.0.2_layerdiff3d/` |

这两个模型配合使用：LayerDiff 3D 先生成头发、脸部、衣物等透明图层，Marigold 再为图层估计深度，最后由 See-through 项目的后处理代码导出 PSD 或 PNG 图层。

## 模型信息

### `seethroughv0.0.1_marigold`

- Pipeline 类型：`MarigoldDepthPipeline`
- 基础模型：Marigold Depth
- 默认处理分辨率：768
- 权重格式：Safetensors

### `seethroughv0.0.2_layerdiff3d`

- Pipeline 类型：`KDiffusionStableDiffusionXLPipeline`
- 额外组件：`UNetFrameConditionModel`、`TransparentVAE`
- 基础模型：SDXL
- 权重格式：Safetensors

## 使用环境

推荐 Python 3.12、PyTorch 2.8.0 和 CUDA 12.8。完整 See-through 依赖请参考项目中的 `see-through/requirements.txt`。

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
pip install diffusers==0.37.0 transformers accelerate safetensors \
  opencv-python pillow numpy
```

示例代码依赖 See-through 源码中的 `common` 模块。请将 See-through 项目放在本地，并把其 `see-through` 目录传给 `--source-root`。

## 快速运行

```bash
python inference_example.py \
  --model-root . \
  --source-root /path/to/See-through/see-through \
  --input /path/to/input.png \
  --output ./output \
  --device cuda \
  --steps 4 \
  --resolution 768
```

在魔搭模型目录中运行时，`--model-root .` 会自动读取：

```text
./seethroughv0.0.1_marigold
./seethroughv0.0.2_layerdiff3d
```

也可以直接传本地目录：

```bash
python inference_example.py \
  --marigold /data/models/seethroughv0.0.1_marigold \
  --layerdiff /data/models/seethroughv0.0.2_layerdiff3d \
  --source-root /data/See-through/see-through \
  --input input.png
```

## 显存说明

LayerDiff 3D 约 9.5 GB，Marigold 约 3.1 GB。完整图层分离流程还需要额外显存。显存不足时，建议：

1. 先运行 LayerDiff，再释放模型；
2. 再运行 Marigold；
3. 使用 `--cpu-offload`；
4. 降低 `--resolution`。

本仓库中的模型是普通精度 Diffusers 权重，不是 NF4 量化权重。NF4 模型请使用项目中对应的 `*_nf4` 模型和量化示例。

## 目录结构

```text
.
├── README.md
├── inference_example.py
├── seethroughv0.0.1_marigold/
│   ├── model_index.json
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   └── vae/
└── seethroughv0.0.2_layerdiff3d/
    ├── model_index.json
    ├── text_encoder/
    ├── text_encoder_2/
    ├── tokenizer/
    ├── tokenizer_2/
    ├── trans_vae/
    ├── unet/
    └── vae/
```

## 来源与许可

模型名称和模型结构沿用 See-through 项目及其上游模型。发布到魔搭前，请同时遵守上游项目、基础模型和依赖组件的许可证要求。

相关项目：

- [See-through](https://github.com/dmMaze/See-through)
- [Marigold](https://github.com/prs-eth/Marigold)
- [LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse_DiffusersCLI)
