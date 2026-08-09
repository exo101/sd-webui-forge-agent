# 💡 打光辅助插件 - Lighting Assistant

一个用于 Stable Diffusion WebUI Forge 的打光辅助插件，帮助你生成专业的打光关键词，或通过 AI 分析图片的打光方式。

## ✨ 功能特点

### 1. 关键词生成器
- **光源方向**：正面光、侧面光、逆光、伦勃朗光、蝴蝶光等
- **光线质量**：硬光、柔光、漫射光、聚光
- **光线颜色**：暖光、冷光、黄金时刻、霓虹光、月光等
- **特殊光效**：丁达尔光、光晕、体积光、轮廓光等
- **光影对比**：高对比、低对比、明暗对照
- **氛围风格**：戏剧性、柔和、神秘、浪漫、恐怖、科幻等

### 2. 图片反推分析
- 上传图片，AI 自动分析打光方式
- 生成对应的打光关键词
- 支持 llama.cpp 视觉模型

### 3. 一键发送
- 将生成的关键词直接发送到文生图/图生图的提示词框

## 📦 安装

### 方法一：直接安装
将插件文件夹放到 WebUI 的 `extensions` 目录下：
```
sd-webui-forge-neo-v3/webui/extensions/sd-webui-lighting-assistant/
```

### 方法二：Git 克隆
```bash
cd sd-webui-forge-neo-v3/webui/extensions
git clone https://github.com/your-repo/sd-webui-lighting-assistant.git
```

## 🚀 使用方法

### 关键词生成器
1. 打开 WebUI，切换到 "💡 打光辅助" 标签页
2. 在 "关键词生成器" 标签中选择你想要的打光参数
3. 点击 "生成关键词" 按钮
4. 复制关键词或点击 "发送到文生图/图生图"
5. 与Flux2-Klein-9B-True-v2编辑模型配合使用
6. 上传到多图拼接参考插件进行生成

<img width="1831" height="894" alt="QQ20260615-021748" src="https://github.com/user-attachments/assets/31e587a9-e2a6-4801-bce9-2cd2cd0f8211" />

<img width="1821" height="871" alt="QQ20260615-021809" src="https://github.com/user-attachments/assets/2aaf962e-1262-43f9-b208-708fa4af651d" />


### 图片反推分析
1. 切换到 "图片反推分析" 标签
2. 上传要分析的图片
3. 配置 llama.cpp 服务地址和模型
4. 点击 "分析打光" 按钮
5. 查看分析结果和提取的关键词

## ⚙️ llama.cpp 配置

### 启动 llama.cpp 服务器
```bash
llama-server --model path/to/your/vision-model.gguf --host 0.0.0.0 --port 8080
```

### 推荐的视觉模型
- LLaVA 系列
- Qwen-VL 系列
- MiniCPM-V 系列
- InternVL 系列

### 测试连接
在插件界面点击 "测试连接" 按钮验证服务是否正常。

## 📚 打光参考手册

插件内置了详细的打光技巧参考，包括：
- 光源方向详解
- 光线质量说明
- 常见布光方案
- 色温参考表

## 🎨 打光关键词示例

### 人像戏剧光
```
rembrandt lighting, hard lighting, warm lighting, cinematic lighting, chiaroscuro, dramatic
```

### 赛博朋克风格
```
rim lighting, hard lighting, neon lighting, cyberpunk lighting, outline lighting, low-key, sci-fi
```

### 梦幻仙境
```
backlight, soft lighting, golden hour, god rays, lens flare, high-key, dreamy
```



## 🔧 常见问题

### Q: 无法连接到 llama.cpp 服务？
A: 请确保：
1. llama.cpp 服务器已启动
2. 服务地址和端口正确
3. 防火墙允许连接

### Q: 分析结果没有关键词？
A: 请确保使用的模型支持视觉输入，并且模型已正确加载。

### Q: 发送到提示词框不工作？
A: 请确保你在正确的页面（文生图或图生图），并且提示词框存在。

## 📝 更新日志

### v1.0.0
- 初始版本
- 关键词生成器
- 图片反推分析
- 发送到提示词框功能

## 📄 许可证

MIT License

## 🙏 致谢

- Stable Diffusion WebUI Forge
- llama.cpp
- 所有贡献者
