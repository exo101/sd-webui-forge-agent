# TRELLIS.2 Extension for SD WebUI Forge

图像转 3D 模型生成扩展，基于微软 TRELLIS.2 项目。

## 快速开始

### 1. 安装依赖

双击运行 `install.bat` 或在命令行执行：

```bash
cd D:\ai\sd-webui-forge-classic-neo\extensions\sd-webui-trellis2
install.bat
```

**智能检测机制：**
- ✅ 首次运行：自动安装所有缺失的依赖
- ✅ 后续运行：跳过已安装的包，快速完成检查
- ✅ WebUI 启动时：自动运行但不会重复安装

这将自动安装：
- Python 依赖（huggingface-hub, transformers, diffusers 等）
- Git 依赖（utils3d, MoGe）
- 下载 TRELLIS.2 模型（约 16GB，如未下载）

**注意**：以下编译模块需要预先安装到 Python 环境：
- nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, utils3d, o-voxel

如果未安装，请从 TRELLIS.2 项目目录运行 `setup.bat`。

### 2. 启动 WebUI

```bash
cd D:\ai\sd-webui-forge-classic-neo
webui.bat
```

### 3. 使用扩展

1. 在浏览器打开 WebUI（通常 http://127.0.0.1:7860）
2. **在顶部标签栏找到 "TRELLIS.2 3D Generator" 标签**
3. 点击该标签进入 TRELLIS.2 界面
4. 上传一张清晰的物体图片（建议正方形，背景简洁）
5. 调整参数或使用默认值
6. 点击 **"Generate 3D Asset"**
7. 等待生成完成（首次需加载模型 2-5 分钟）
8. 查看和下载 OBJ/GLB 模型

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **Seed** | 随机种子 | 42 |
| **Guidance Scale** | 引导强度（1-10） | 5.0 |
| **Inference Steps** | 推理步数（10-100） | 50 |
| **Octree Resolution** | 八叉树分辨率 | 1024 |
| **Mesh Simplification** | 网格简化比例（0-1） | 0.9 |
| **Texture Resolution** | 纹理分辨率（512-4096） | 2048 |
| **Environment Map** | 环境贴图 | none |

### 预设配置

- **快速预览**: Octree 512, Steps 20, Texture 512（~30秒）
- **标准质量**: Octree 1024, Steps 50, Texture 2048（~2-3分钟）
- **高质量**: Octree 1536, Steps 80, Texture 4096（~5-10分钟）

## 系统要求

- **GPU**: NVIDIA RTX 4090/A100/H100（至少 24GB 显存）
- **Python**: 3.10+
- **CUDA**: 12.4+
- **内存**: 32GB+ RAM

## 常见问题

### Q: 看不到 "TRELLIS.2 3D Generator" 标签？

**检查控制台输出：**
- ✅ 正常：看到 `[TRELLIS.2] UI callbacks registered successfully!`
- ❌ 异常：看到错误信息

**解决步骤：**
1. 确认 TRELLIS.2 项目在 `extensions/sd-webui-trellis2/TRELLIS.2/`
2. 运行 `python check_env.py` 验证环境
3. 重启 WebUI 并刷新浏览器

### Q: 首次生成很慢？

首次点击生成时需要加载模型（2-5 分钟），这是正常的。后续生成会直接使用缓存，速度大幅提升。

### Q: 显存不足？

降低参数：
- Octree Resolution: 1536 → 1024 → 512
- Texture Resolution: 4096 → 2048 → 1024
- 关闭其他占用 GPU 的程序

### Q: 模型加载失败？

确认模型已下载：
```bash
dir D:\ai\sd-webui-forge-classic-neo\models\trellis2
```

应包含 `TRELLIS.2-4B/` 等目录。如缺失，重新运行 `install.bat` 或手动下载。

## 技术说明

#### 懒加载机制

模型**不会**在 WebUI 启动时加载，仅在用户点击"Generate 3D Asset"按钮时才加载。这样可以：
- ✅ 快速启动 WebUI
- ✅ 节省显存（不使用时无占用）
- ✅ 灵活切换其他功能

#### 文件结构

```
extensions/sd-webui-trellis2/
├── scripts/
│   └── trellis2_script.py    # 核心逻辑和 UI 注册
├── TRELLIS.2/                # TRELLIS.2 项目源码
├── install.py                # 自动安装脚本
├── install.bat               # Windows 安装脚本
├── download_model.py         # 模型下载工具
├── check_env.py              # 环境诊断
├── outputs/                  # 生成的 3D 模型
└── README.md                 # 本文档

models/trellis2/              # 模型存储目录
└── TRELLIS.2-4B/             # 主模型（约 16GB）
```

## 相关链接

- **TRELLIS.2 官方**: https://github.com/microsoft/TRELLIS.2
- **Hugging Face 模型**: https://huggingface.co/microsoft/TRELLIS.2-4B
- **项目主页**: https://microsoft.github.io/TRELLIS.2

---

**版本**: 1.0.0  
**最后更新**: 2026-05-22
