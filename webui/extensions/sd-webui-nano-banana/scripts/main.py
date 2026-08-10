"""Nano Banana - AI image generation via Pixapi API."""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
from modules import script_callbacks

# Add scripts directory to sys.path for local imports
_scripts_dir = Path(__file__).parent
if str(_scripts_dir) not in sys.path:
    sys.path.append(str(_scripts_dir))

from pixapi_client import PixapiClient, PixapiError

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_client = PixapiClient()

# Size slider limits (pixels)
MIN_SIZE = 256
MAX_SIZE = 2048
STEP_SIZE = 32

# Aspect ratio map for Pixapi API
ASPECT_MAP = {
    "1:1": (1.0, "1:1"),
    "3:2": (3 / 2, "3:2"),
    "2:3": (2 / 3, "2:3"),
    "4:3": (4 / 3, "4:3"),
    "3:4": (3 / 4, "3:4"),
    "5:4": (5 / 4, "5:4"),
    "4:5": (4 / 5, "4:5"),
    "16:9": (16 / 9, "16:9"),
    "9:16": (9 / 16, "9:16"),
}

QUALITY_OPTIONS = ["auto", "low", "medium", "high"]


def _size_to_aspect_ratio(width: int, height: int) -> str:
    """Convert pixel dimensions to the closest Pixapi aspect ratio string."""
    if height == 0:
        return "1:1"
    ratio = width / height
    closest = min(ASPECT_MAP.values(), key=lambda v: abs(v[0] - ratio))
    return closest[1]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def _refresh_models(api_key: str) -> tuple[gr.Dropdown, gr.Dropdown, str]:
    """Fetch models from Pixapi and populate dropdowns."""
    if not api_key.strip():
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None),
            "请先输入 API Key",
        )

    _client.set_api_key(api_key.strip())
    try:
        models = _client.list_models()
    except PixapiError as e:
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None),
            f"获取模型列表失败: {e}",
        )
    except Exception as e:
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None),
            f"连接错误: {e}",
        )

    image_models = [m for m in models if m.get("category") == "image"]
    if not image_models:
        return (
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None),
            "当前账户没有可用的图像模型",
        )

    # Separate text-to-image and image-to-image models
    t2i = [m["id"] for m in image_models if any(
        c in m.get("capabilities", []) for c in ("text-to-image", "image-to-image")
    )]
    i2i = [m["id"] for m in image_models if "image-to-image" in m.get("capabilities", [])]

    gen_choices = t2i if t2i else [m["id"] for m in image_models]
    edit_choices = i2i if i2i else [m["id"] for m in image_models]

    return (
        gr.Dropdown(choices=gen_choices, value=gen_choices[0] if gen_choices else None),
        gr.Dropdown(choices=edit_choices, value=edit_choices[0] if edit_choices else None),
        f"已加载 {len(image_models)} 个模型",
    )


def _on_generate(
    api_key: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    quality: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Text-to-image generation."""
    if not api_key.strip():
        return None, "请先输入 API Key"

    if not prompt.strip():
        return None, "请输入提示词"

    if width < MIN_SIZE or height < MIN_SIZE:
        return None, f"尺寸过小，宽高均需≥{MIN_SIZE}px"
    if width > MAX_SIZE or height > MAX_SIZE:
        return None, f"尺寸过大，宽高均需≤{MAX_SIZE}px"

    _client.set_api_key(api_key.strip())
    progress(0, desc="正在生成图像...")

    try:
        q = quality if quality != "auto" else None
        aspect = _size_to_aspect_ratio(width, height)
        data = _client.generate_image(model=model, prompt=prompt, size=aspect, quality=q)
    except PixapiError as e:
        return None, str(e)
    except Exception as e:
        return None, f"错误: {e}"

    if not data:
        return None, "API 未返回图像数据"

    url = data[0].get("url")
    if not url:
        return None, "响应中未找到图像 URL"

    # Download and return as local image
    try:
        img_bytes = _client.download_image(url)
        suffix = Path(url).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name, f"完成! URL: {url}"
    except Exception as e:
        return None, f"下载失败: {e}"


def _on_edit(
    api_key: str,
    model: str,
    prompt: str,
    input_image: str,  # filepath from gr.Image
    width: int,
    height: int,
    quality: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Image editing with file upload support."""
    if not api_key.strip():
        return None, "请先输入 API Key"

    if not prompt.strip():
        return None, "请输入编辑指令"

    if width < MIN_SIZE or height < MIN_SIZE:
        return None, f"尺寸过小，宽高均需≥{MIN_SIZE}px"
    if width > MAX_SIZE or height > MAX_SIZE:
        return None, f"尺寸过大，宽高均需≤{MAX_SIZE}px"

    _client.set_api_key(api_key.strip())
    progress(0, desc="正在编辑图像...")

    # Determine image URL: either uploaded file or direct URL
    try:
        if input_image and os.path.isfile(input_image):
            progress(0.1, desc="正在上传图片到临时托管...")
            image_url = _client.upload_to_hosting(input_image)
        elif input_image and input_image.startswith("http"):
            image_url = input_image.strip()
        else:
            return None, "请上传图片或输入图片 URL"
    except PixapiError as e:
        return None, str(e)
    except Exception as e:
        return None, f"上传失败: {e}"

    progress(0.3, desc="正在编辑图像...")
    try:
        q = quality if quality != "auto" else None
        aspect = _size_to_aspect_ratio(width, height)
        data = _client.edit_image(
            model=model, prompt=prompt, image_url=image_url, size=aspect, quality=q
        )
    except PixapiError as e:
        return None, str(e)
    except Exception as e:
        return None, f"错误: {e}"

    if not data:
        return None, "API 未返回图像数据"

    url = data[0].get("url")
    if not url:
        return None, "响应中未找到图像 URL"

    try:
        img_bytes = _client.download_image(url)
        suffix = Path(url).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name, f"完成! URL: {url}"
    except Exception as e:
        return None, f"下载失败: {e}"


def _on_batch_edit(
    api_key: str,
    model: str,
    prompt: str,
    input_images: list,  # list of filepaths from gr.Files
    width: int,
    height: int,
    quality: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Batch edit - process multiple uploaded images with the same instruction."""
    if not api_key.strip():
        return [], "请先输入 API Key"

    if not prompt.strip():
        return [], "请输入编辑指令"

    if not input_images or len(input_images) == 0:
        return [], "请上传至少一张图片"

    if width < MIN_SIZE or height < MIN_SIZE:
        return [], f"尺寸过小，宽高均需≥{MIN_SIZE}px"
    if width > MAX_SIZE or height > MAX_SIZE:
        return [], f"尺寸过大，宽高均需≤{MAX_SIZE}px"

    _client.set_api_key(api_key.strip())
    q = quality if quality != "auto" else None
    aspect = _size_to_aspect_ratio(width, height)

    result_paths = []
    log_lines = []
    total = len(input_images)

    for i, file_path in enumerate(input_images):
        progress((i + 1) / total, desc=f"正在处理第 {i+1}/{total} 张...")
        try:
            # Upload to temporary hosting
            progress((i + 0.3) / total, desc=f"正在上传第 {i+1}/{total} 张...")
            image_url = _client.upload_to_hosting(file_path)
            # Edit via Pixapi
            progress((i + 0.6) / total, desc=f"正在编辑第 {i+1}/{total} 张...")
            data = _client.edit_image(
                model=model, prompt=prompt, image_url=image_url, size=aspect, quality=q
            )
            if not data:
                log_lines.append(f"[{i+1}] ✗ API 未返回数据")
                continue
            result_url = data[0].get("url")
            if not result_url:
                log_lines.append(f"[{i+1}] ✗ 响应中无 URL")
                continue
            img_bytes = _client.download_image(result_url)
            suffix = Path(result_url).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(img_bytes)
            tmp.close()
            result_paths.append(tmp.name)
            log_lines.append(f"[{i+1}] ✓ 完成")
        except PixapiError as e:
            log_lines.append(f"[{i+1}] ✗ {e}")
        except Exception as e:
            log_lines.append(f"[{i+1}] ✗ 错误: {e}")

    summary = f"批量编辑完成: {len(result_paths)}/{total} 成功"
    log_lines.insert(0, summary)
    return result_paths, "\n".join(log_lines)


def _on_async_generate(
    api_key: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    quality: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Submit async generation task and return task ID."""
    if not api_key.strip():
        return "请先输入 API Key"

    if width < MIN_SIZE or height < MIN_SIZE:
        return f"尺寸过小，宽高均需≥{MIN_SIZE}px"
    if width > MAX_SIZE or height > MAX_SIZE:
        return f"尺寸过大，宽高均需≤{MAX_SIZE}px"

    _client.set_api_key(api_key.strip())
    try:
        q = quality if quality != "auto" else None
        aspect = _size_to_aspect_ratio(width, height)
        task_id = _client.submit_async_generation(model=model, prompt=prompt, size=aspect, quality=q)
        return f"任务已提交: {task_id}"
    except PixapiError as e:
        return str(e)
    except Exception as e:
        return f"错误: {e}"


def _on_check_task(api_key: str, task_id: str) -> str:
    """Check async task status."""
    if not api_key.strip():
        return "请先输入 API Key"
    if not task_id.strip():
        return "请输入任务 ID"

    _client.set_api_key(api_key.strip())
    try:
        task = _client.get_task(task_id.strip())
        status = task.get("status", "unknown")
        progress = task.get("progress", 0)
        result = task.get("result", {})
        status_map = {"submitted": "已提交", "processing": "处理中", "completed": "已完成", "failed": "失败"}
        status_cn = status_map.get(status, status)
        msg = f"状态: {status_cn} | 进度: {progress}%"
        if result and result.get("data"):
            urls = [d.get("url", "") for d in result["data"]]
            msg += f"\n结果 URL: {', '.join(urls)}"
        if task.get("credits_cost"):
            msg += f" | 消耗积分: {task['credits_cost']}"
        return msg
    except PixapiError as e:
        return str(e)
    except Exception as e:
        return f"错误: {e}"


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------
def on_ui_tabs():
    """Register the Nano Banana tab in WebUI."""
    with gr.Blocks(analytics_enabled=False, title="Nano Banana") as ui:
        gr.Markdown(
            "# Nano Banana\n"
            "由 Pixapi (Gemini / GPT Image) 驱动的 AI 图像生成。\n\n"
            "前往 [pixapi.ai](https://pixapi.ai) 获取 API Key"
        )

        # ---- API Key input ----
        with gr.Row():
            api_key = gr.Textbox(
                label="Pixapi API Key",
                placeholder="输入你的 Pixapi API Key",
                type="password",
                scale=3,
            )
            refresh_btn = gr.Button("刷新模型列表", variant="primary", scale=1)
        model_status = gr.Textbox(label="状态", interactive=False)

        # ---- Model selection ----
        with gr.Row():
            gen_model = gr.Dropdown(label="生图模型", choices=[], scale=1)
            edit_model = gr.Dropdown(label="编辑模型", choices=[], scale=1)

        # ---- Model descriptions ----
        gr.Markdown(
            "---\n"
            "### 模型简介\n\n"
            "| 模型 | 说明 |\n"
            "|------|------|\n"
            "| `gemini-3.1-flash-image-preview` | **主力模型** — 兼具速度与质量，适合日常文生图和图像编辑任务 |\n"
            "| `gemini-3.1-flash-lite-image` | **轻量模型** — 处理速度最快，适合快速预览和批量处理 |\n"
            "| `gemini-3-pro-image-preview` | **旗舰模型** — 质量最高，适合复杂构图和精细编辑需求 |\n"
        )

        with gr.Tabs():
            # ================================================================
            # Tab 1: Text-to-Image
            # ================================================================
            with gr.TabItem("文生图"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述你想要生成的图像...",
                            lines=4,
                        )
                        with gr.Row():
                            with gr.Column():
                                width = gr.Slider(
                                    label="宽度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                            with gr.Column():
                                height = gr.Slider(
                                    label="高度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                        with gr.Row():
                            quality = gr.Dropdown(
                                label="质量", choices=QUALITY_OPTIONS, value="auto"
                            )
                        gen_btn = gr.Button("生成图像", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        output_img = gr.Image(label="生成结果", type="filepath")
                        output_log = gr.Textbox(label="输出信息", interactive=False)

                gen_btn.click(
                    fn=_on_generate,
                    inputs=[api_key, gen_model, prompt, width, height, quality],
                    outputs=[output_img, output_log],
                )

            # ================================================================
            # Tab 2: Image Editing
            # ================================================================
            with gr.TabItem("图像编辑"):
                with gr.Row():
                    with gr.Column(scale=2):
                        edit_prompt = gr.Textbox(
                            label="编辑指令",
                            placeholder="描述如何编辑图像...",
                            lines=4,
                        )
                        edit_image = gr.Image(
                            label="上传图片",
                            type="filepath",
                            height=280,
                        )
                        with gr.Row():
                            with gr.Column():
                                edit_width = gr.Slider(
                                    label="宽度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                            with gr.Column():
                                edit_height = gr.Slider(
                                    label="高度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                        with gr.Row():
                            edit_quality = gr.Dropdown(
                                label="质量", choices=QUALITY_OPTIONS, value="auto"
                            )
                        edit_btn = gr.Button("编辑图像", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        edit_output_img = gr.Image(label="编辑结果", type="filepath")
                        edit_output_log = gr.Textbox(label="输出信息", interactive=False)

                edit_btn.click(
                    fn=_on_edit,
                    inputs=[api_key, edit_model, edit_prompt, edit_image, edit_width, edit_height, edit_quality],
                    outputs=[edit_output_img, edit_output_log],
                )

            # ================================================================
            # Tab 3: Batch Edit
            # ================================================================
            with gr.TabItem("批量编辑"):
                gr.Markdown("上传多张图片并对它们应用相同的编辑指令。")
                with gr.Row():
                    with gr.Column(scale=2):
                        batch_prompt = gr.Textbox(
                            label="编辑指令",
                            placeholder="描述如何编辑所有图片...",
                            lines=4,
                        )
                        batch_files = gr.File(
                            label="上传图片（支持多选）",
                            file_count="multiple",
                            file_types=["image"],
                            height=200,
                        )
                        with gr.Row():
                            with gr.Column():
                                batch_width = gr.Slider(
                                    label="宽度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                            with gr.Column():
                                batch_height = gr.Slider(
                                    label="高度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                        with gr.Row():
                            batch_quality = gr.Dropdown(
                                label="质量", choices=QUALITY_OPTIONS, value="auto"
                            )
                        batch_btn = gr.Button("批量编辑", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        batch_gallery = gr.Gallery(label="编辑结果", columns=3, height=400)
                        batch_log = gr.Textbox(label="处理日志", interactive=False, lines=8)

                batch_btn.click(
                    fn=_on_batch_edit,
                    inputs=[api_key, edit_model, batch_prompt, batch_files, batch_width, batch_height, batch_quality],
                    outputs=[batch_gallery, batch_log],
                )

            # ================================================================
            # Tab 4: Async Tasks
            # ================================================================
            with gr.TabItem("异步任务"):
                gr.Markdown("提交耗时任务并轮询获取结果。")
                with gr.Row():
                    with gr.Column():
                        async_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="描述图像...",
                            lines=3,
                        )
                        with gr.Row():
                            with gr.Column():
                                async_width = gr.Slider(
                                    label="宽度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                            with gr.Column():
                                async_height = gr.Slider(
                                    label="高度 (px)", minimum=MIN_SIZE, maximum=MAX_SIZE,
                                    value=1024, step=STEP_SIZE
                                )
                        with gr.Row():
                            async_quality = gr.Dropdown(
                                label="质量", choices=QUALITY_OPTIONS, value="auto"
                            )
                        async_submit_btn = gr.Button("提交异步任务", variant="primary")
                        async_log = gr.Textbox(label="结果", interactive=False, lines=4)

                    with gr.Column():
                        task_id_input = gr.Textbox(
                            label="任务 ID",
                            placeholder="粘贴任务 ID 查询状态",
                        )
                        check_task_btn = gr.Button("查询任务状态")
                        task_result = gr.Textbox(
                            label="任务详情", interactive=False, lines=6
                        )

                async_submit_btn.click(
                    fn=_on_async_generate,
                    inputs=[api_key, gen_model, async_prompt, async_width, async_height, async_quality],
                    outputs=[async_log],
                )
                check_task_btn.click(
                    fn=_on_check_task,
                    inputs=[api_key, task_id_input],
                    outputs=[task_result],
                )

        # ---- Refresh models ----
        refresh_btn.click(
            fn=_refresh_models,
            inputs=[api_key],
            outputs=[gen_model, edit_model, model_status],
        )

    return [(ui, "Nano Banana", "nano_banana_tab")]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
script_callbacks.on_ui_tabs(on_ui_tabs, name="nano_banana")