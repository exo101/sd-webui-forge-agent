import gradio as gr
import numpy as np
from modules import scripts
from pathlib import Path
import sys
import os
from modules.llama_port import get_llama_url

_LLAMA_URL = get_llama_url()

# Add scripts directory to system path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# Add extension root directory to system path
extension_dir = scripts_dir.parent
if str(extension_dir) not in sys.path:
    sys.path.append(str(extension_dir))

# Import function modules
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


class VisionChatScript(scripts.Script):

    def __init__(self):
        super().__init__()
        self._injected = False

    def title(self):
        return "图像识别"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def after_component(self, component, **kwargs):
        elem_id = getattr(component, "elem_id", None)
        if elem_id == "txt2img_results_panel" and not self._injected:
            self._injected = True
            self._create_vision_chat_ui()

    def _create_vision_chat_ui(self):
        with gr.Accordion("图像识别", open=False):
            with gr.Row():
                # Left area: tag management, model selection, image upload
                with gr.Column(scale=1):
                    # Tag management module
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
                        print(f"Tag management module load failed: {e}")

                    # Vision model selection
                    with gr.Group():
                        gr.Markdown("### 视觉模型")
                        gr.Markdown("📌 **模型建议**: 8GB显存 -> 2B, 12GB-16GB显存 -> 4B-9B")

                        vision_model = gr.Dropdown(
                            label="视觉模型",
                            choices=[
                                "Qwen3.5-2B-Q6_K.gguf",
                            ],
                            value="Qwen3.5-2B-Q6_K.gguf",
                            interactive=True,
                            info="选择视觉模型（支持图片识别+文字对话）",
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

                    # Image upload area
                    with gr.Group():
                        gr.Markdown("### 📤 图片上传")
                        gr.Markdown("📌 **使用说明**: 视觉模型支持上传图片并附带文字对话")

                        with gr.Box(visible=True) as image_container:
                            from modules_forge.forge_canvas.canvas import ForgeCanvas

                            qwen_canvas = ForgeCanvas(
                                no_upload=False,
                                no_scribbles=True,
                                height=300,
                                elem_id="qwen_vision_image"
                            )

                            image_input = qwen_canvas.background

                # Right area: chat area
                with gr.Column(scale=1):
                    chat_history = gr.Chatbot(
                        elem_id="chatbot",
                        label="聊天历史",
                        height=300,
                        render=True
                    )

                    last_image_path_state = gr.State(value=None)

                    chat_message = gr.Textbox(
                        show_label=False,
                        placeholder="输入消息（支持多轮对话，上传一次图片后可连续提问）",
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
                            "清空对话",
                            size="lg",
                            variant="primary",
                            elem_classes="orange-button",
                            scale=2
                        )
                        save_button = gr.Button(
                            "保存对话",
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

                    # Quick description area
                    with gr.Group():
                        if create_quick_description is not None:
                            quick_description_buttons = create_quick_description(chat_message)
                        else:
                            quick_description_buttons = {}

                    # Batch recognition tag generation area
                    with gr.Accordion("批量识别与标签生成", open=False):
                        # Add ollama directory to Python path
                        ext_dir = Path(__file__).parent.parent
                        ollama_dir = ext_dir / "ollama"
                        ollama_dir_str = str(ollama_dir)
                        sys.path = [p for p in sys.path if 'ollama' not in p.lower()]
                        sys.path.insert(0, ollama_dir_str)

                        # Import llama.cpp API
                        LLAMACPP_AVAILABLE = False
                        get_llamacpp_models = None

                        try:
                            if 'llamacpp_api' in sys.modules:
                                del sys.modules['llamacpp_api']

                            from llamacpp_api import get_response_lvm_llamacpp_api, get_llamacpp_models
                            LLAMACPP_AVAILABLE = True
                            print(f"✅ llama.cpp API imported successfully")
                        except ImportError as e:
                            print(f"❌ Warning: Could not import llama.cpp API module: {e}")
                            import traceback
                            traceback.print_exc()

                        # Default vision model list
                        default_llamacpp_vision_models = [
                            "Qwen3.5-2B-Q6_K.gguf",
                        ]

                        # Batch processing UI
                        batch_image_dir = gr.Textbox(
                            label="图片目录路径",
                            value=str(ext_dir / "images"),
                            placeholder="输入包含图片的文件夹路径",
                            container=True
                        )

                        batch_tag_prompt = gr.Textbox(
                            label="标签生成提示词",
                            value="请识别图片内容并生成详细标签，用逗号分隔，不要包含解释性文字",
                            placeholder="输入标签生成提示词",
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

                        # Refresh model list function
                        def refresh_models():
                            print(f"🔍 [llama.cpp] Refreshing model list...")
                            if LLAMACPP_AVAILABLE and get_llamacpp_models:
                                try:
                                    models = get_llamacpp_models(_LLAMA_URL)
                                    print(f"   Retrieved models: {models}")
                                    if models:
                                        return gr.update(choices=models, value=models[0])
                                except Exception as e:
                                    print(f"❌ Failed to get model list: {e}")
                                    import traceback
                                    traceback.print_exc()
                            return gr.update(choices=default_llamacpp_vision_models, value=default_llamacpp_vision_models[0])

                        def base64_to_image_file(base64_str, output_path):
                            import base64
                            if base64_str.startswith("data:image/png;base64,"):
                                base64_str = base64_str.replace("data:image/png;base64,", "")
                            image_data = base64.b64decode(base64_str)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, 'wb') as f:
                                f.write(image_data)
                            return output_path

                        def on_chat(message, chat_history, vision_model, image_input):
                            print(f"\n=== Debug Info ===")
                            print(f"message: {message}")
                            print(f"vision_model: {vision_model}")

                            if not message and image_input is None:
                                return "", chat_history

                            has_image = False
                            temp_image_paths = []

                            if image_input is not None:
                                has_image = True
                                try:
                                    temp_dir = os.path.join(ext_dir, "tmp", "qwen_uploads")
                                    os.makedirs(temp_dir, exist_ok=True)
                                    temp_path = os.path.join(temp_dir, f"temp_{os.urandom(4).hex()}.png")
                                    if hasattr(image_input, 'save'):
                                        if image_input.mode == 'RGBA':
                                            image_input = image_input.convert('RGB')
                                        image_input.save(temp_path)
                                        temp_image_paths.append(temp_path)
                                        print(f"✓ Saved PIL image: {temp_path}")
                                    elif isinstance(image_input, str) and image_input.startswith("data:image"):
                                        base64_to_image_file(image_input, temp_path)
                                        temp_image_paths.append(temp_path)
                                        print(f"✓ Saved Base64 image: {temp_path}")
                                    else:
                                        print(f"⚠ image_input format incorrect: {type(image_input)}")
                                        has_image = False
                                except Exception as e:
                                    print(f"❌ Failed to save image: {e}")
                                    has_image = False

                            print(f"has_image: {has_image}, temp_paths: {len(temp_image_paths)}")

                            user_message = message
                            if has_image:
                                user_message = f"[Image] {message}" if message else "[Image]"

                            if user_message:
                                chat_history.append((user_message, ""))

                            ai_response = ""

                            if LLAMACPP_AVAILABLE:
                                try:
                                    model_name = vision_model
                                    if has_image and temp_image_paths:
                                        print(f"📷 [llama.cpp] Calling vision model: {model_name}, image: {temp_image_paths[0]}")
                                        ai_response = get_response_lvm_llamacpp_api(
                                            input_model_name=model_name,
                                            input_content=message or "请描述这张图片",
                                            input_image_path=temp_image_paths[0],
                                            llamacpp_host=_LLAMA_URL,
                                            timeout=300
                                        )
                                    else:
                                        print(f"💬 [llama.cpp] Calling vision model (text): {model_name}")
                                        ai_response = get_response_lvm_llamacpp_api(
                                            input_model_name=model_name,
                                            input_content=message or "你好！请问有什么可以帮你的？",
                                            input_image_path=None,
                                            llamacpp_host=_LLAMA_URL,
                                            timeout=300
                                        )
                                    if not ai_response:
                                        ai_response = "[错误] llama.cpp API 返回空结果"
                                except Exception as e:
                                    ai_response = f"[错误] {str(e)}"
                                    print(f"❌ llama.cpp API call failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                ai_response = "[未安装] llama.cpp API 模块不可用"

                            print(f"✅ AI reply: {ai_response[:100]}...")
                            print("=================\n")

                            if chat_history and chat_history[-1][0] == user_message:
                                chat_history[-1] = (user_message, ai_response)

                            for temp_path in temp_image_paths:
                                try:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                except:
                                    pass

                            return "", chat_history

                        def batch_process_images(image_dir, tag_prompt, vision_model):
                            if not image_dir or not os.path.isdir(image_dir):
                                return "错误：请提供有效的图片目录路径"
                            if not LLAMACPP_AVAILABLE:
                                return "错误：llama.cpp API 模块不可用，请检查安装"

                            supported_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
                            image_files = []
                            for file_name in os.listdir(image_dir):
                                file_path = os.path.join(image_dir, file_name)
                                if os.path.isfile(file_path):
                                    ext = os.path.splitext(file_name)[1].lower()
                                    if ext in supported_extensions:
                                        image_files.append(file_path)

                            if not image_files:
                                return "错误：指定目录中未找到支持的图片文件"

                            results = []
                            success_count = 0
                            failed_count = 0

                            for image_path in image_files:
                                try:
                                    txt_file_path = os.path.splitext(image_path)[0] + ".txt"
                                    print(f"正在处理: {os.path.basename(image_path)}")
                                    tags = get_response_lvm_llamacpp_api(
                                        input_model_name=vision_model,
                                        input_content=tag_prompt,
                                        input_image_path=image_path,
                                        llamacpp_host=_LLAMA_URL,
                                        timeout=300
                                    )
                                    if tags:
                                        os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
                                        with open(txt_file_path, 'w', encoding='utf-8') as f:
                                            f.write(tags)
                                        results.append(f"✅ 成功: {os.path.basename(image_path)} -> {os.path.basename(txt_file_path)}")
                                        success_count += 1
                                    else:
                                        results.append(f"❌ 失败: {os.path.basename(image_path)} - 模型返回空结果")
                                        failed_count += 1
                                except Exception as e:
                                    results.append(f"❌ 错误: {os.path.basename(image_path)} - {str(e)}")
                                    failed_count += 1

                            summary = f"批量处理完成：{success_count} 个成功，{failed_count} 个失败\n"
                            return summary + "\n".join(results)

                        # Chat event bindings
                        chat_inputs = [chat_message, chat_history, vision_model, image_input]
                        chat_outputs = [chat_message, chat_history]

                        chat_message.submit(on_chat, inputs=chat_inputs, outputs=chat_outputs)
                        submit_button.click(on_chat, inputs=chat_inputs, outputs=chat_outputs)
                        clear_button.click(lambda: [], outputs=[chat_history])

                        # Batch processing event bindings
                        batch_inputs = [batch_image_dir, batch_tag_prompt, vision_model]
                        batch_outputs = [batch_result]
                        batch_start_btn.click(batch_process_images, inputs=batch_inputs, outputs=batch_outputs)

                        # Refresh model button event
                        refresh_models_btn.click(
                            fn=refresh_models,
                            inputs=[],
                            outputs=[vision_model]
                        )

                # Copy functionality using JavaScript
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
                                        alert("Latest reply copied to clipboard!");
                                    }).catch(err => {
                                        console.error('Copy failed:', err);
                                        alert("Copy failed, please copy manually");
                                    });
                                    return;
                                }
                            }
                        }
                        alert("No reply content to copy");
                    }
                    """
                )