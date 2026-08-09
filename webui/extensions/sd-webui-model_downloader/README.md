# 开源社区模型下载器

一个用于 Stable Diffusion WebUI 的模型下载插件，支持从 **HuggingFace** 和 **魔搭社区(ModelScope)** 下载模型。

## 功能特性

- ✅ **双平台支持**: 支持 HuggingFace 和 ModelScope 两大模型平台
- ✅ **完整模型下载**: 一键下载整个模型仓库
- ✅ **单文件下载**: 按需下载模型中的特定文件
- ✅ **文件列表预览**: 获取模型仓库的完整文件列表
- ✅ **多文件选择**: 支持同时选择多个文件下载
- ✅ **下载历史**: 记录所有下载记录，方便追溯
- ✅ **自定义保存路径**: 自由设置模型保存位置

## 安装方法

### 方法一：通过 WebUI 扩展管理器安装

1. 打开 WebUI
2. 点击 **Extensions** → **Install from URL**
3. 在 URL 输入框中粘贴本插件的仓库地址
4. 点击 **Install**
5. 重启 WebUI

### 方法二：手动安装

1. 克隆或下载本插件到 WebUI 的 `extensions` 目录：
   ```bash
   cd /path/to/webui/extensions
   git clone <repository-url>
   ```

2. 安装依赖：
   ```bash
   pip install modelscope>=1.14.0 huggingface-hub>=0.22.0
   ```

3. 重启 WebUI

## 使用说明

### 基本操作

1. 在 WebUI 顶部标签栏点击 **开源社区模型下载器**
2. 选择模型源（HuggingFace 或 ModelScope）
3. 输入模型名称/仓库 ID（格式：`用户名/模型名`）

### 获取文件列表

1. 输入模型名称后，点击 **获取文件列表**
2. 等待加载完成后，下拉菜单会显示该模型的所有文件

### 下载完整模型

1. 输入模型名称和保存路径
2. 点击 **下载完整模型**
3. 等待下载完成，状态栏会显示结果

### 下载单个文件

1. 先点击 **获取文件列表** 获取文件列表
2. 在下拉菜单中选择要下载的文件（支持多选）
3. 点击 **下载选中文件**

### 下载历史

切换到 **下载历史** 标签页，可查看所有下载记录，支持清空历史。

## 模型 ID 格式

模型 ID 必须包含用户名和模型名，格式如下：

- **ModelScope**: `Comfy-Org/Krea-2`、`qwen/Qwen2.5-7B-Instruct`
- **HuggingFace**: `runwayml/stable-diffusion-v1-5`、`Lykon/dreamshaper-8`

## 常见问题

### Q: 模型 ID 格式错误怎么办？

A: 模型 ID 必须是 `用户名/模型名` 的格式，例如 `Comfy-Org/Krea-2`，不能只输入模型名。

### Q: 获取文件列表失败？

A: 可能原因：
- 模型 ID 输入错误
- 网络连接问题
- 模型仓库不存在或权限受限

### Q: 下载速度慢？

A: 国内用户建议优先使用 ModelScope 源，可获得更快的下载速度。

### Q: 下载中断怎么办？

A: 重新点击下载按钮即可，会自动续传。

## 技术细节

- **缓存目录**: `extensions/sd-webui-model_downloader/cache/`
- **下载历史**: 保存在 `cache/download_history.json`，最多保留 100 条记录
- **依赖**: `modelscope`、`huggingface-hub`

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
