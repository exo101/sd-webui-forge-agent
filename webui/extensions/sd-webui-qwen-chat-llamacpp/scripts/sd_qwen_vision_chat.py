import gradio as gr
import numpy as np
import shutil
from modules import script_callbacks
from pathlib import Path
import sys
from modules.llama_port import get_llama_url

_LLAMA_URL = get_llama_url()

# 添加 scripts 目录到系统路径
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# 添加插件根目录到系统路径（用于导入 ollama 模块）
extension_dir = scripts_dir.parent
if str(extension_dir) not in sys.path:
    sys.path.append(str(extension_dir))

# 导入各个功能模块
try:
    from prompt_templates import create_prompt_template_ui
except ImportError:
    create_prompt_template_ui = None
    print("Warning: Could not import prompt_templates")

try:
    from quick_description import create_quick_description
except ImportError:
    create_quick_description = None
    print("Warning: Could not import quick_description")

try:
    from tag_management import create_tag_management_module
except ImportError:
    create_tag_management_module = None
    print("Warning: Could not import tag_management")

try:
    from image_management import create_image_management_module
except ImportError:
    create_image_management_module = None
    print("Warning: Could not import image_management")


def vision_chat_tab():
    """创建图像识别与语言交互标签页"""
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            # 模型下载说明标签页
            with gr.TabItem("1 模型下载说明"):
                gr.Markdown("""
# 📥 Ollama / llama.cpp 与模型下载说明

---

## 📌 选择后端

- **Ollama**: 更简单，一键启动，适合快速开始
- **llama.cpp**: 更灵活，性能优化，适合高级用户

---

## 🦙 Ollama 使用方法

### 步骤 1：安装 Ollama

1. 访问官网下载：[https://ollama.com/](https://ollama.com/)
2. 下载适合您系统的安装包
3. 安装 Ollama 到系统

### 步骤 2：下载 Qwen 模型

根据您的显存大小选择合适的模型：

#### 视觉模型（支持图片识别）

| 模型命令 | 说明 | 推荐显存 |
|---------|------|---------|
| `ollama run qwen3.5:9b` | 高精度版 | 16GB+ |
| `ollama run qwen3.5:4b` | 平衡版（推荐） | 12GB |
| `ollama run qwen3-vl:8b` | 视觉语言模型 | 12GB+ |
| `ollama run qwen3-vl:4b` | 中等视觉模型 | 8GB+ |
| `ollama run qwen3-vl:2b` | 轻量版 | 8GB |

#### 语言模型（仅文本对话）

| 模型命令 | 说明 |
|---------|------|
| `ollama run qwen3:latest` | 最新版语言模型 |
| `ollama run qwen3.5:4b` | 平衡版语言模型 |

### 步骤 3：删除模型（可选）

```
ollama rm 模型名称
```

例如：`ollama rm qwen3.5:4b`

---

## 🐑 llama.cpp 使用方法

### 步骤 1：下载 llama.cpp

1. 访问 GitHub：[https://github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases)
2. 下载适合您系统的二进制文件（`llama-server` 或 `llama.cpp-windows.zip`）
3. 解压到任意目录

### 步骤 2：下载 GGUF 模型

推荐下载源：
- **ModelScope**: [https://modelscope.cn/](https://modelscope.cn/) (国内访问更快)
- **HuggingFace**: [https://huggingface.co/](https://huggingface.co/)

推荐的模型：

#### 视觉模型（支持图片识别）
- `llava-v1.6-vicuna-7b` | 推荐：16GB
- `llava-v1.6-vicuna-13b` | 推荐：32GB
- `llava-yi-v1.6-6b` | 推荐：16GB
- `llava-yi-v1.6-9b` | 推荐：24GB
- `deepseek-vl-7b` | 推荐：24GB
- `cogvlm2-19b` | 推荐：48GB

#### 语言模型（仅文本对话）
- `qwen2.5-7b-instruct` | 推荐：16GB
- `qwen2.5-14b-instruct` | 推荐：32GB
- `llama-3-8b-instruct` | 推荐：16GB
- `llama-3-70b-instruct` | 推荐：80GB
- `mistral-7b-instruct-v0.3` | 推荐：16GB
- `mixtral-8x7b-instruct-v0.1` | 推荐：48GB
- `gemma-2-9b-it` | 推荐：24GB
- `gemma-2-27b-it` | 推荐：64GB

### 步骤 3：启动 llama.cpp 服务器

使用以下命令启动（根据您的模型路径调整）：

**Windows (PowerShell):**
```powershell
./llama-server.exe --model "D:/path/to/your-model.gguf" --host 0.0.0.0 --port 8080 --n-gpu-layers -1
```

**Linux/macOS:**
```bash
./llama-server --model /path/to/your-model.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers -1
```

**参数说明:**
- `--model`: 指定 GGUF 模型文件路径
- `--host 0.0.0.0`: 允许外部访问（如果只在本机使用可用 `127.0.0.1`）
- `--port 8080`: 服务端口（可以修改为其他端口）
- `--n-gpu-layers -1`: 将所有层加载到 GPU（0 表示 CPU 模式）
- `--threads`: CPU 线程数（默认 4）

### 步骤 4：在插件中配置 llama.cpp

1. 在插件界面选择 **"llama.cpp"** 后端
2. 在 "llama.cpp API 配置" 中配置服务器地址（默认 `http://localhost:8080`）
3. 在模型下拉菜单中选择您的模型名称（或直接输入）
4. 开始使用！

---

## 💡 使用提示

**Ollama 用户:**
- 首次运行模型时会自动下载，请耐心等待
- 下载完成后模型会保存在本地，下次使用无需重新下载
- 建议根据您的显卡显存大小选择合适的模型

**llama.cpp 用户:**
- GGUF 模型文件通常比较大，请确保有足够的磁盘空间
- 使用 `--n-gpu-layers -1` 可以利用 GPU 加速推理速度
- 可以通过调整 `--ctx-size` 参数来增加上下文窗口大小

**通用提示:**
- 如果遇到显存不足，可以尝试使用更小的模型
- 批量处理图片时，建议使用较小的模型以提高速度

---

## 🔧 常见问题

**Q: 如何查看已安装的 Ollama 模型？**
```bash
ollama list
```

**Q: 如何停止 llama.cpp 服务器？**
- 在终端中按 `Ctrl+C` 即可

**Q: 我可以同时运行 Ollama 和 llama.cpp 吗？**
- 可以！只要它们使用不同的端口，不会冲突

**Q: llama.cpp 支持的模型格式是什么？**
- GGUF 格式（旧的 GGML 格式不再支持）
""")
            
            # 图像识别与语言交互标签页
            with gr.TabItem("2 图像识别与关键词辅助"):
                with gr.Row():
                    # 左侧区域：标签管理、图像管理、模型选择
                    with gr.Column(scale=1):
                        # 标签管理模块
                        try:
                            if create_tag_management_module is not None:
                                tag_management_components = create_tag_management_module()
                                if tag_management_components:
                                    with gr.Box():
                                        if "refresh_button" in tag_management_components:
                                            tag_management_components["refresh_button"]
                                        if "folder_path" in tag_management_components:
                                            tag_management_components["folder_path"].elem_classes = ["xykc-accordion"]
                            else:
                                gr.Markdown("标签管理模块当前不可用。")
                        except Exception as e:
                            print(f"标签管理模块加载失败：{e}")
                        
                        # 图像管理模块
                        try:
                            if create_image_management_module is not None:
                                image_management_ui = create_image_management_module()
                                if image_management_ui:
                                    with gr.Box():
                                        if "dir_input" in image_management_ui:
                                            image_management_ui["dir_input"]
                                        if "load_dir_btn" in image_management_ui:
                                            image_management_ui["load_dir_btn"]
                                        if "gallery" in image_management_ui:
                                            image_management_ui["gallery"]
                            else:
                                gr.Markdown("图像管理模块当前不可用。")
                        except Exception as e:
                            print(f"图像管理模块加载失败：{e}")
                        
                        # 模型选择区域
                        with gr.Group():
                            gr.Markdown("### 模型选择")
                            gr.Markdown("📌 **模型选择建议**：8GB 显存选择 2B，12GB-16GB 显存可选择 4B-9B 模型")
                            
                            model_type = gr.Radio(
                                [("视觉模型", "vision"), ("语言模型", "text")],
                                value="vision",
                                label="模型类型",
                                interactive=True,
                                info="视觉模型支持图片识别和纯文本聊天，语言模型仅支持文本对话"
                            )
                            
                            with gr.Row():
                                vision_model = gr.Dropdown(
                                    label="视觉模型",
                                    choices=[
                                        # Ollama 视觉模型
                                        "qwen3-vl:8b", "qwen3-vl:4b", "qwen3-vl:2b",
                                        "qwen2.5-vl:3b", "qwen2.5-vl:7b", "qwen2.5-vl:32b",
                                        # llama.cpp 视觉模型
                                        "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
                                        "qwen2.5-vl-3b", "qwen2.5-vl-7b", "qwen2.5-vl-32b",
                                        "llava-v1.6-vicuna-7b", "llava-v1.6-vicuna-13b",
                                        "llava-yi-v1.6-6b", "llava-yi-v1.6-9b",
                                        "deepseek-vl-7b", "deepseek-vl-16b",
                                        "cogvlm2-19b", "cogvlm2-7b"
                                    ],
                                    value="qwen3-vl:4b",
                                    interactive=True,
                                    info="选择视觉模型（支持图片识别 + 文本聊天）",
                                    scale=2,
                                    elem_classes="larger-text",
                                    container=True
                                )
                            
                            with gr.Row():
                                language_model = gr.Dropdown(
                                    label="语言模型",
                                    choices=[
                                        # Ollama 语言模型
                                        "qwen3:latest", "qwen3.5:4b", "qwen3.5:9b",
                                        # llama.cpp 语言模型 - Qwen3.5 系列（推荐）
                                        "qwen3.5-0.5b", "qwen3.5-1.5b", "qwen3.5-3b", "qwen3.5-4b",
                                        "qwen3.5-7b", "qwen3.5-8b", "qwen3.5-14b", "qwen3.5-32b", "qwen3.5-72b",
                                        "qwen3.5-9b-deepseek-v4-flash-mtp",
                                        # Qwen2.5 系列
                                        "qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct",
                                        "qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "qwen2.5-32b-instruct",
                                        # 翻译模型 Hy-MT2
                                        "hy-mt2-1.8b", "hy-mt2-7b",
                                        # 其他常用模型
                                        "llama-3-8b-instruct", "llama-3-70b-instruct",
                                        "mistral-7b-instruct-v0.3", "mixtral-8x7b-instruct-v0.1",
                                        "gemma-2-9b-it", "gemma-2-27b-it"
                                    ],
                                    value="qwen3:latest",
                                    interactive=False,
                                    info="选择语言模型",
                                    scale=2,
                                    elem_classes="larger-text",
                                    container=True
                                )
                            
                            refresh_models_btn = gr.Button(
                                "🔄 刷新模型列表",
                                size="sm",
                                variant="secondary",
                                scale=1,
                                elem_classes="larger-text"
                            )
                        
                        # 图像上传区域（使用 ForgeCanvas 避免权限问题）
                        with gr.Group():
                            gr.Markdown("### 📤 图片上传")
                            gr.Markdown("📌 **使用说明**：Qwen3.5 等多模态模型支持同时上传图片和文字聊天")
                            
                            upload_method = gr.Radio(
                                [("单张图片", "single"), ("批量图片", "batch")],
                                value="single",
                                label="上传方式",
                                interactive=True,
                                scale=2,
                                elem_classes="larger-text",
                                container=True
                            )
                            
                            with gr.Box(visible=True) as image_container:
                                # 使用 ForgeCanvas 代替 gr.Image，避免 Windows 权限问题
                                from modules_forge.forge_canvas.canvas import ForgeCanvas
                                
                                qwen_canvas = ForgeCanvas(
                                    no_upload=False,
                                    no_scribbles=True,  # 关闭涂鸦功能
                                    height=300,
                                    elem_id="qwen_vision_image"
                                )
                                
                                # 使用隐藏的 State 来存储 Canvas 的 background 值
                                image_input = qwen_canvas.background
                                
                                # 批量上传仍然使用 Files（仅在需要时启用）
                                multi_images_input = gr.Files(
                                    type="filepath",
                                    label="多张图片输入",
                                    visible=False,
                                    height=300,
                                    scale=1,
                                    min_width=300,
                                    file_count="multiple",
                                    file_types=["image"]
                                )
                    
                    # 右侧区域：关键词辅助模板和聊天区域
                    with gr.Column(scale=1):
                        # 关键词辅助模板区域
                        with gr.Accordion("关键词辅助模板", open=False):
                            if create_prompt_template_ui is not None:
                                template_ui = create_prompt_template_ui()
                                with gr.Row():
                                    with gr.Column():
                                        template_ui["expression_template"]
                                    with gr.Column():
                                        template_ui["story_template"]
                                    with gr.Column():
                                        template_ui["shot_template"]
                            else:
                                gr.Markdown("关键词辅助模板模块当前不可用。")
                        
                        # 聊天区域
                        chat_history = gr.Chatbot(
                            elem_id="chatbot", 
                            label="聊天记录", 
                            height=300,
                            render=True
                        )
                        
                        # 隐藏的状态组件：保存最后使用的图片路径
                        last_image_path_state = gr.State(value=None)
                        
                        chat_message = gr.Textbox(
                            show_label=False,
                            placeholder="输入消息（支持多轮对话，可上传一次图片后连续提问）",
                            container=True,
                            scale=1,
                            min_width=300,
                            lines=3
                        )
                        with gr.Row(equal_height=True):
                            submit_button = gr.Button(
                                "发送",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )
                            clear_button = gr.Button(
                                "清空聊天",
                                size="lg", 
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )
                            save_button = gr.Button(
                                "保存聊天记录",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )
                            copy_button = gr.Button(
                                "复制最新回复",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button",
                                scale=2
                            )

                        # 快捷描述区域
                        with gr.Group():
                            if create_quick_description is not None:
                                quick_description_buttons = create_quick_description(chat_message)
                            else:
                                quick_description_buttons = {}
                        
                        # 批量识别生成标签区域
                        with gr.Accordion("批量识别生成标签", open=False):
                            gr.Markdown("### 批量处理图片并生成标签文件")
                            
                            # 获取插件目录路径
                            import os
                            from pathlib import Path
                            extension_dir = Path(__file__).parent.parent
                            default_image_dir = str(extension_dir / "images")
                            
                            # 创建默认目录
                            os.makedirs(default_image_dir, exist_ok=True)
                            
                            batch_image_dir = gr.Textbox(
                                label="图片目录路径",
                                value=default_image_dir,
                                placeholder="输入包含图片的文件夹路径",
                                container=True
                            )
                            
                            batch_tag_prompt = gr.Textbox(
                                label="标签生成提示词",
                                value="请识别图片内容，生成详细的标签，使用逗号分隔，不要包含任何解释性文字",
                                placeholder="输入用于生成标签的提示词",
                                lines=2,
                                container=True
                            )
                            
                            batch_start_btn = gr.Button(
                                "开始批量识别",
                                size="lg",
                                variant="primary",
                                elem_classes="orange-button"
                            )
                            
                            batch_result = gr.Textbox(
                                label="批量处理结果",
                                lines=5,
                                container=True
                            )
                
                # 模型后端选择
                backend_type = gr.Radio(
                    [("Ollama", "ollama"), ("llama.cpp", "llamacpp")],
                    value="ollama",
                    label="模型后端",
                    interactive=True,
                    info="选择使用 Ollama 或 llama.cpp 作为模型后端"
                )
                
                # 添加 Ollama API 配置
                with gr.Accordion("⚙️ Ollama API 配置", open=False) as ollama_config:
                    ollama_host = gr.Textbox(
                        label="Ollama 服务器地址",
                        value="http://localhost:11434",
                        placeholder="http://localhost:11434",
                        info="Ollama API 服务器地址"
                    )
                    ollama_timeout = gr.Number(
                        label="超时时间（秒）",
                        value=300,
                        minimum=60,
                        maximum=3600,
                        step=60,
                        info="请求超时时间，默认 300 秒"
                    )
                
                # 添加 API 调用函数
                import sys
                import os
                from pathlib import Path
                
                # 添加 ollama 目录到 Python 路径（放在最前面以确保优先导入）
                extension_dir = Path(__file__).parent.parent
                ollama_dir = extension_dir / "ollama"
                ollama_dir_str = str(ollama_dir)
                # 移除可能存在的其他 ollama 目录路径
                sys.path = [p for p in sys.path if 'ollama' not in p.lower()]
                # 插入到最前面
                sys.path.insert(0, ollama_dir_str)
                
                # 导入 Ollama API
                OLLAMA_AVAILABLE = False
                LLAMACPP_AVAILABLE = False
                get_ollama_models = None
                get_llamacpp_models = None
                
                try:
                    from ollama_api import get_response_lvm_ollama_api, get_response_text_ollama_api, get_ollama_models
                    OLLAMA_AVAILABLE = True
                except ImportError as e:
                    print(f"警告：无法导入 Ollama API 模块：{e}")
                
                # 导入 llama.cpp API（使用绝对路径导入确保导入正确的模块）
                try:
                    # 清除可能已缓存的模块
                    if 'llamacpp_api' in sys.modules:
                        del sys.modules['llamacpp_api']
                    
                    from llamacpp_api import get_response_lvm_llamacpp_api, get_response_text_llamacpp_api, get_llamacpp_models
                    LLAMACPP_AVAILABLE = True
                    print(f"✅ llama.cpp API 导入成功: get_llamacpp_models={get_llamacpp_models}")
                    # 检查实际导入的模块路径
                    import llamacpp_api
                    print(f"✅ 导入的 llamacpp_api 模块路径: {llamacpp_api.__file__}")
                except ImportError as e:
                    print(f"❌ 警告：无法导入 llama.cpp API 模块：{e}")
                    import traceback
                    traceback.print_exc()
                
                # 默认模型列表（用于检测失败时的回退）
                default_ollama_vision_models = [
                    "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3-vl:8b", "qwen3-vl:4b",
                    "huihui_ai/qwen3.5-abliterated:4b", "huihui_ai/qwen3.5-abliterated:9b"
                ]
                default_ollama_language_models = [
                    "qwen3:latest", "qwen3.5:4b", "qwen3.5:9b"
                ]
                default_llamacpp_vision_models = [
                    "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
                    "qwen2.5-vl-3b", "qwen2.5-vl-7b", "qwen2.5-vl-32b",
                    "llava-v1.6-vicuna-7b", "llava-v1.6-vicuna-13b",
                    "llava-yi-v1.6-6b", "llava-yi-v1.6-9b",
                    "deepseek-vl-7b", "deepseek-vl-16b",
                    "cogvlm2-19b", "cogvlm2-7b"
                ]
                default_llamacpp_language_models = [
                    # Qwen3.5 系列（推荐）
                    "qwen3.5-0.5b", "qwen3.5-1.5b", "qwen3.5-3b", "qwen3.5-4b",
                    "qwen3.5-7b", "qwen3.5-8b", "qwen3.5-14b", "qwen3.5-32b", "qwen3.5-72b",
                    "qwen3.5-9b-deepseek-v4-flash-mtp",
                    # Qwen2.5 系列
                    "qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct",
                    "qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "qwen2.5-32b-instruct",
                    "qwen2.5-72b-instruct",
                    # 翻译模型 Hy-MT2
                    "hy-mt2-1.8b", "hy-mt2-7b",
                    # 其他常用模型
                    "llama-3-8b-instruct", "llama-3-70b-instruct",
                    "mistral-7b-instruct-v0.3", "mixtral-8x7b-instruct-v0.1",
                    "gemma-2-9b-it", "gemma-2-27b-it"
                ]
                
                # 刷新模型列表函数
                def refresh_models(backend_type, ollama_host):
                    """根据后端类型刷新模型列表"""
                    if backend_type == "ollama":
                        if OLLAMA_AVAILABLE and get_ollama_models:
                            models = get_ollama_models(ollama_host)
                            if models:
                                # 对于 Ollama，我们暂时无法区分视觉/语言模型
                                # 所以所有模型都放在两个列表中
                                return models, models
                        return default_ollama_vision_models, default_ollama_language_models
                    else:  # llamacpp
                        print(f"🔍 [llama.cpp] 尝试刷新模型列表...")
                        print(f"   LLAMACPP_AVAILABLE: {LLAMACPP_AVAILABLE}")
                        print(f"   get_llamacpp_models: {get_llamacpp_models}")
                        print(f"   llamacpp_host: {_LLAMA_URL}")
                        if LLAMACPP_AVAILABLE and get_llamacpp_models:
                            try:
                                models = get_llamacpp_models(_LLAMA_URL)
                                print(f"   获取到的模型列表: {models}")
                                if models:
                                    # 同样暂时无法区分
                                    return models, models
                            except Exception as e:
                                print(f"❌ 获取模型列表失败: {e}")
                                import traceback
                                traceback.print_exc()
                        return default_llamacpp_vision_models, default_llamacpp_language_models
                
                def base64_to_image_file(base64_str, output_path):
                    """将 Base64 字符串保存为图片文件"""
                    import base64
                    if base64_str.startswith("data:image/png;base64,"):
                        base64_str = base64_str.replace("data:image/png;base64,", "")
                    
                    image_data = base64.b64decode(base64_str)
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    
                    return output_path
                
                # 简化的聊天处理函数（集成 Ollama 和 llama.cpp API）
                def on_chat(message, chat_history, vision_model, language_model, model_type, 
                           upload_method, image_input, multi_images_input, 
                           ollama_host, ollama_timeout, backend_type):
                    """聊天处理函数（集成 Ollama 和 llama.cpp API）"""
                    # 获取插件目录路径
                    from pathlib import Path
                    import os
                    extension_dir = Path(__file__).parent.parent
                    
                    print(f"\n=== 调试信息 ===")
                    print(f"backend_type: {backend_type}")
                    print(f"message: {message}")
                    print(f"upload_method: {upload_method}")
                    print(f"image_input type: {type(image_input)}")
                    print(f"multi_images_input: {multi_images_input}")
                    print(f"model_type: {model_type}")
                    
                    # 检查是否有输入
                    has_input = False
                    if message:
                        has_input = True
                    if image_input is not None:
                        # 处理 numpy 数组的情况
                        if isinstance(image_input, np.ndarray):
                            if image_input.size > 0:
                                has_input = True
                        else:
                            has_input = True
                    if multi_images_input and len(multi_images_input) > 0:
                        has_input = True
                    
                    if not has_input:
                        return "", chat_history
                    
                    # 判断是否有图片
                    has_image = False
                    image_count = 0
                    temp_image_paths = []
                    
                    # 处理 ForgeCanvas 的图片（可能是 PIL 对象或 Base64 字符串）
                    if upload_method == "single" and image_input:
                        has_image = True
                        image_count = 1
                        
                        # 将图片保存为临时文件
                        try:
                            temp_dir = os.path.join(extension_dir, "tmp", "qwen_uploads")
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_path = os.path.join(temp_dir, f"temp_{os.urandom(4).hex()}.png")
                            
                            # 如果是 PIL 对象
                            if hasattr(image_input, 'save'):
                                # 转换为 RGB 模式（移除 alpha 通道）
                                if image_input.mode == 'RGBA':
                                    image_input = image_input.convert('RGB')
                                image_input.save(temp_path)
                                temp_image_paths.append(temp_path)
                                print(f"✓ 已保存 PIL 图片：{temp_path}")
                            # 如果是 Base64 字符串
                            elif isinstance(image_input, str) and image_input.startswith("data:image"):
                                base64_to_image_file(image_input, temp_path)
                                temp_image_paths.append(temp_path)
                                print(f"✓ 已保存 Base64 图片：{temp_path}")
                            else:
                                print(f"⚠ image_input 格式不正确：{type(image_input)}")
                                has_image = False
                        except Exception as e:
                            print(f"❌ 保存图片失败：{e}")
                            has_image = False
                    
                    # 处理批量上传
                    elif upload_method == "batch" and multi_images_input:
                        has_image = True
                        if isinstance(multi_images_input, list):
                            image_count = len(multi_images_input)
                            temp_image_paths.extend([f for f in multi_images_input if isinstance(f, str)])
                        else:
                            image_count = 1
                            if isinstance(multi_images_input, str):
                                temp_image_paths.append(multi_images_input)
                    
                    print(f"has_image: {has_image}, image_count: {image_count}, temp_paths: {len(temp_image_paths)}")
                    
                    # 构建消息
                    user_message = message
                    if has_image:
                        if image_count == 1:
                            user_message = f"[图片] {message}" if message else "[图片]"
                        else:
                            user_message = f"[{image_count}张图片] {message}" if message else f"[{image_count}张图片]"
                    
                    # 添加到聊天记录（先显示用户消息）
                    if user_message:
                        chat_history.append((user_message, ""))
                    
                    # 调用 API
                    ai_response = ""
                    
                    if backend_type == "ollama" and OLLAMA_AVAILABLE:
                        try:
                            # 选择模型
                            model_name = vision_model if model_type == "vision" else language_model
                            
                            # 根据是否有图片选择函数
                            if has_image and temp_image_paths and model_type == "vision":
                                # 视觉模型 - 使用第一张图片
                                print(f"📷 [Ollama] 调用视觉模型：{model_name}, 图片路径：{temp_image_paths[0]}")
                                ai_response = get_response_lvm_ollama_api(
                                    input_model_name=model_name,
                                    input_content=message or "请描述这张图片",
                                    input_image_path=temp_image_paths[0],
                                    ollama_host=ollama_host,
                                    timeout=ollama_timeout
                                )
                            else:
                                # 语言模型或纯文本
                                print(f"💬 [Ollama] 调用语言模型：{model_name}")
                                ai_response = get_response_text_ollama_api(
                                    input_model_name=model_name,
                                    input_content=message or "你好！有什么可以帮助你的？",
                                    ollama_host=ollama_host,
                                    timeout=ollama_timeout
                                )
                            
                            if not ai_response:
                                ai_response = "[错误] Ollama API 返回空响应"
                        
                        except Exception as e:
                            ai_response = f"[错误] {str(e)}"
                            print(f"❌ Ollama API 调用失败：{e}")
                            import traceback
                            traceback.print_exc()
                    elif backend_type == "llamacpp" and LLAMACPP_AVAILABLE:
                        try:
                            # 选择模型
                            model_name = vision_model if model_type == "vision" else language_model
                            
                            # 根据是否有图片选择函数
                            if has_image and temp_image_paths and model_type == "vision":
                                # 视觉模型 - 使用第一张图片
                                print(f"📷 [llama.cpp] 调用视觉模型：{model_name}, 图片路径：{temp_image_paths[0]}")
                                ai_response = get_response_lvm_llamacpp_api(
                                    input_model_name=model_name,
                                    input_content=message or "请描述这张图片",
                                    input_image_path=temp_image_paths[0],
                                    llamacpp_host=_LLAMA_URL,
                                    timeout=300
                                )
                            else:
                                # 语言模型或纯文本
                                print(f"💬 [llama.cpp] 调用语言模型：{model_name}")
                                ai_response = get_response_text_llamacpp_api(
                                    input_model_name=model_name,
                                    input_content=message or "你好！有什么可以帮助你的？",
                                    llamacpp_host=_LLAMA_URL,
                                    timeout=300
                                )
                            
                            if not ai_response:
                                ai_response = "[错误] llama.cpp API 返回空响应"
                        
                        except Exception as e:
                            ai_response = f"[错误] {str(e)}"
                            print(f"❌ llama.cpp API 调用失败：{e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        if backend_type == "ollama":
                            ai_response = "[未安装] Ollama API 模块不可用"
                        else:
                            ai_response = "[未安装] llama.cpp API 模块不可用"
                    
                    print(f"✅ AI 回复：{ai_response[:100]}...")
                    print("=================\n")
                    
                    # 更新聊天记录中的 AI 回复
                    if chat_history and chat_history[-1][0] == user_message:
                        chat_history[-1] = (user_message, ai_response)
                    
                    # 为批量上传的图片生成标签文件
                    if upload_method == "batch" and multi_images_input:
                        # 获取插件目录路径
                        extension_dir = Path(__file__).parent.parent
                        images_dir = os.path.join(extension_dir, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        
                        for image_path in temp_image_paths:
                            try:
                                # 获取原始文件名
                                original_filename = os.path.basename(image_path)
                                
                                # 复制图片到插件目录
                                dest_image_path = os.path.join(images_dir, original_filename)
                                shutil.copy2(image_path, dest_image_path)
                                
                                # 生成标签文件路径（保存在插件目录，使用相同的文件名）
                                txt_file_path = os.path.splitext(dest_image_path)[0] + ".txt"
                                
                                # 保存标签到文件
                                os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
                                with open(txt_file_path, 'w', encoding='utf-8') as f:
                                    f.write(ai_response)
                                print(f"✅ 已生成标签文件: {txt_file_path}")
                                print(f"✅ 已保存图片: {dest_image_path}")
                            except Exception as e:
                                print(f"❌ 生成标签文件失败: {str(e)}")
                    
                    # 清理临时文件
                    for temp_path in temp_image_paths:
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except:
                            pass
                    
                    return "", chat_history
                
                def batch_process_images(image_dir, tag_prompt, vision_model, 
                                       ollama_host, ollama_timeout, backend_type):
                    """批量处理图片并生成标签文件"""
                    if not image_dir or not os.path.isdir(image_dir):
                        return "错误：请提供有效的图片目录路径"
                    
                    # 检查后端是否可用
                    if backend_type == "ollama" and not OLLAMA_AVAILABLE:
                        return "错误：Ollama API 模块不可用，请检查安装"
                    if backend_type == "llamacpp" and not LLAMACPP_AVAILABLE:
                        return "错误：llama.cpp API 模块不可用，请检查安装"
                    
                    # 支持的图片格式
                    supported_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
                    
                    # 获取所有图片文件
                    image_files = []
                    for file_name in os.listdir(image_dir):
                        file_path = os.path.join(image_dir, file_name)
                        if os.path.isfile(file_path):
                            ext = os.path.splitext(file_name)[1].lower()
                            if ext in supported_extensions:
                                image_files.append(file_path)
                    
                    if not image_files:
                        return "错误：指定目录中没有找到支持的图片文件"
                    
                    # 处理结果
                    results = []
                    success_count = 0
                    failed_count = 0
                    
                    # 处理每张图片
                    for image_path in image_files:
                        try:
                            # 生成标签文件路径（保存在图片同目录，使用相同的文件名）
                            txt_file_path = os.path.splitext(image_path)[0] + ".txt"
                            
                            # 调用视觉模型生成标签
                            print(f"正在处理: {os.path.basename(image_path)}")
                            
                            if backend_type == "ollama":
                                tags = get_response_lvm_ollama_api(
                                    input_model_name=vision_model,
                                    input_content=tag_prompt,
                                    input_image_path=image_path,
                                    ollama_host=ollama_host,
                                    timeout=ollama_timeout
                                )
                            else:  # llamacpp
                                tags = get_response_lvm_llamacpp_api(
                                    input_model_name=vision_model,
                                    input_content=tag_prompt,
                                    input_image_path=image_path,
                                    llamacpp_host=_LLAMA_URL,
                                    timeout=300
                                )
                            
                            if tags:
                                # 保存标签到文件
                                os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
                                with open(txt_file_path, 'w', encoding='utf-8') as f:
                                    f.write(tags)
                                results.append(f"✅ 成功: {os.path.basename(image_path)} → {os.path.basename(txt_file_path)}")
                                success_count += 1
                            else:
                                results.append(f"❌ 失败: {os.path.basename(image_path)} - 模型返回空结果")
                                failed_count += 1
                            
                        except Exception as e:
                            results.append(f"❌ 错误: {os.path.basename(image_path)} - {str(e)}")
                            failed_count += 1
                    
                    # 生成总结
                    summary = f"批量处理完成: 成功 {success_count} 个, 失败 {failed_count} 个\n"
                    return summary + "\n".join(results)
                
                # 聊天事件绑定
                chat_inputs = [chat_message, chat_history, vision_model, language_model, 
                              model_type, upload_method, image_input, multi_images_input, 
                              ollama_host, ollama_timeout, backend_type]
                chat_outputs = [chat_message, chat_history]

                chat_message.submit(on_chat, inputs=chat_inputs, outputs=chat_outputs)
                submit_button.click(on_chat, inputs=chat_inputs, outputs=chat_outputs)
                clear_button.click(lambda: [], outputs=[chat_history])
                
                # 批量处理事件绑定
                batch_inputs = [batch_image_dir, batch_tag_prompt, vision_model, 
                               ollama_host, ollama_timeout, backend_type]
                batch_outputs = [batch_result]
                batch_start_btn.click(batch_process_images, inputs=batch_inputs, outputs=batch_outputs)
                
                # 刷新模型按钮事件
                def on_refresh_models(backend_type, ollama_host):
                    """刷新模型列表"""
                    vision_choices, language_choices = refresh_models(backend_type, ollama_host)
                    return gr.update(choices=vision_choices), gr.update(choices=language_choices)
                
                refresh_models_btn.click(
                    fn=on_refresh_models,
                    inputs=[backend_type, ollama_host],
                    outputs=[vision_model, language_model]
                )
                
                # 后端切换时自动刷新模型
                def on_backend_switch(backend_type, ollama_host):
                    """切换后端时刷新模型列表"""
                    vision_choices, language_choices = refresh_models(backend_type, ollama_host)
                    # 根据后端设置默认值
                    default_vision = vision_choices[0] if vision_choices else "qwen3.5:4b"
                    default_language = language_choices[0] if language_choices else "qwen3:latest"
                    return (
                        gr.update(choices=vision_choices, value=default_vision),
                        gr.update(choices=language_choices, value=default_language)
                    )
                
                backend_type.change(
                    fn=on_backend_switch,
                    inputs=[backend_type, ollama_host],
                    outputs=[vision_model, language_model]
                )
                
                # 上传方式切换事件
                def switch_upload(method):
                    """切换上传方式"""
                    if method == "single":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)
                
                upload_method.change(
                    fn=switch_upload,
                    inputs=[upload_method],
                    outputs=[image_input, multi_images_input]
                )
                
                # 使用 JavaScript 实现复制功能
                copy_button.click(
                    None,
                    inputs=[chat_history],
                    outputs=[],
                    _js="""
                    (chat_history) => {
                        if (chat_history && chat_history.length > 0) {
                            const lastMessage = chat_history[chat_history.length - 1];
                            if (lastMessage && lastMessage.length >= 2) {
                                const aiResponse = lastMessage[1];
                                if (aiResponse && aiResponse.length > 0) {
                                    navigator.clipboard.writeText(aiResponse).then(() => {
                                        alert("最新回复已复制到剪贴板！");
                                    }).catch(err => {
                                        console.error('复制失败：', err);
                                        alert("复制失败，请手动复制");
                                    });
                                    return;
                                }
                            }
                        }
                        alert("没有可复制的回复内容");
                    }
                    """
                )
    
    return [(ui, "图像识别与语言交互", "Vision_Chat_Tab")]


# 注册标签页
script_callbacks.on_ui_tabs(vision_chat_tab)
