import gradio as gr
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Optional
import os
import glob
import time
import threading
import base64
import io
import uuid
import queue as queue_module

from backend.args import dynamic_args
from modules import images, scripts, sd_models, shared
from modules.api import api
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.sd_samplers_common import images_tensor_to_samples
from modules.shared import device, opts, sd_model
from modules.ui_components import InputAccordion

# 全局变量用于存储主界面组件
txt2img_w_slider = None
txt2img_h_slider = None
img2img_w_slider = None
img2img_h_slider = None
txt2img_prompt = None  # 文生图正面提示词
img2img_prompt = None  # 图生图正面提示词

# 注册组件
def on_ui_tabs():
    """注册组件到A1111Context"""
    return []

# 关闭8的倍数
def closesteight(num):
    rem = num % 8
    if rem <= 4:
        return round(num - rem)
    else:
        return round(num + (8 - rem))

t2i_info = """
插件仅支持编辑模型：Klein，Qwen-Image-Edit，NanoBanana，gpt-image-2，Krea 2
"""

i2i_info = """
插件仅支持编辑模型：Klein，Qwen-Image-Edit，NanoBanana，gpt-image-2，Krea 2
"""

# ==================== 批量任务管理器 ====================

@dataclass
class BatchTask:
    """单个批量任务的数据结构"""
    id: str
    prompt: str
    reference_image: Image.Image
    status: str = "waiting"  # waiting, processing, completed, failed
    result_image: Optional[Image.Image] = None
    error_message: str = ""


class BatchTaskManager:
    """批量任务队列管理器"""

    def __init__(self):
        self.tasks: list[BatchTask] = []
        self._lock = threading.Lock()
        self._is_running = False
        self._stop_requested = False

    def add_task(self, prompt: str, reference_image: Image.Image) -> BatchTask:
        """添加一个任务到队列"""
        task = BatchTask(
            id=str(uuid.uuid4())[:8],
            prompt=prompt,
            reference_image=reference_image,
        )
        with self._lock:
            self.tasks.append(task)
        print(f"[Image Stitch Batch] 添加任务 {task.id}: {prompt[:40]}...")
        return task

    def get_status(self) -> list[dict]:
        """获取所有任务的状态"""
        with self._lock:
            return [
                {
                    "id": t.id,
                    "prompt": t.prompt[:40],
                    "status": t.status,
                    "error": t.error_message[:100] if t.error_message else "",
                    "has_result": t.result_image is not None,
                }
                for t in self.tasks
            ]

    def get_statistics(self) -> dict:
        """获取队列统计信息"""
        with self._lock:
            waiting = sum(1 for t in self.tasks if t.status == "waiting")
            processing = sum(1 for t in self.tasks if t.status == "processing")
            completed = sum(1 for t in self.tasks if t.status == "completed")
            failed = sum(1 for t in self.tasks if t.status == "failed")
            return {
                "total": len(self.tasks),
                "waiting": waiting,
                "processing": processing,
                "completed": completed,
                "failed": failed,
            }

    def clear_completed(self):
        """清除已完成和失败的任务"""
        with self._lock:
            self.tasks = [t for t in self.tasks if t.status in ("waiting", "processing")]

    def clear_all(self):
        """清除所有任务"""
        with self._lock:
            self.tasks = []

    def get_next_pending(self) -> Optional[BatchTask]:
        """获取下一个等待中的任务"""
        with self._lock:
            for t in self.tasks:
                if t.status == "waiting":
                    t.status = "processing"
                    return t
            return None

    def mark_completed(self, task_id: str, result_image: Image.Image):
        """标记任务为已完成"""
        with self._lock:
            for t in self.tasks:
                if t.id == task_id:
                    t.status = "completed"
                    t.result_image = result_image
                    break

    def mark_failed(self, task_id: str, error_message: str):
        """标记任务为失败"""
        with self._lock:
            for t in self.tasks:
                if t.id == task_id:
                    t.status = "failed"
                    t.error_message = error_message
                    break

    def is_running(self) -> bool:
        return self._is_running

    def stop(self):
        """请求停止批量处理"""
        self._stop_requested = True

    def start_processing(self, progress: gr.Progress = gr.Progress()):
        """开始批量处理所有等待中的任务"""
        self._is_running = True
        self._stop_requested = False

        all_tasks = []
        with self._lock:
            # 收集所有等待中的任务
            for t in self.tasks:
                if t.status == "waiting":
                    all_tasks.append(t)
                    t.status = "processing"

        total = len(all_tasks)
        print(f"[Image Stitch Batch] 开始批量处理，共 {total} 个任务")

        completed_results = []
        failed_results = []

        for i, task in enumerate(all_tasks):
            if self._stop_requested:
                print(f"[Image Stitch Batch] 批量处理已停止（第 {i+1}/{total} 个任务）")
                # 将未处理的任务恢复为 waiting
                with self._lock:
                    for remaining in all_tasks[i:]:
                        if remaining.status == "processing":
                            remaining.status = "waiting"
                break

            progress((i + 1) / total, desc=f"正在处理任务 {i+1}/{total}: {task.prompt[:30]}...")
            print(f"[Image Stitch Batch] 处理任务 {task.id}: {task.prompt[:40]}...")

            try:
                result = self._process_single_task(task)
                if result:
                    task.result_image = result
                    task.status = "completed"
                    completed_results.append(result)
                    print(f"[Image Stitch Batch] ✅ 任务 {task.id} 完成")
                else:
                    raise RuntimeError("生成失败，未返回图片")
            except Exception as e:
                task.status = "failed"
                task.error_message = str(e)
                failed_results.append(task)
                print(f"[Image Stitch Batch] ❌ 任务 {task.id} 失败: {e}")

        self._is_running = False
        print(f"[Image Stitch Batch] 批量处理完成: {len(completed_results)} 成功, {len(failed_results)} 失败")
        return completed_results, failed_results

    def _process_single_task(self, task: BatchTask) -> Optional[Image.Image]:
        """处理单个任务：使用 txt2img + image_stitch 参考图生成"""
        from modules.sd_models import FakeInitialModel
        forge_model_mode = getattr(shared.opts, 'forge_model_mode', 'local')
        is_api_mode = forge_model_mode == "api" or isinstance(shared.sd_model, FakeInitialModel)

        if not is_api_mode and isinstance(shared.sd_model, FakeInitialModel):
            # 本地模式但模型未加载：尝试加载模型
            from modules.sd_models import forge_model_reload
            try:
                from modules_forge import main_entry
                checkpoint = getattr(shared.opts, 'sd_model_checkpoint', '')
                if checkpoint:
                    main_entry.checkpoint_change(checkpoint, preset=None, save=False, refresh=True)
                forge_model_reload()
            except Exception as e:
                print(f"[Image Stitch Batch] 尝试加载模型失败: {e}")
                raise RuntimeError("模型未加载，请先在 WebUI 中选择并加载模型")

        # 获取当前主界面参数
        width = getattr(txt2img_w_slider, "value", 512) if txt2img_w_slider else 512
        height = getattr(txt2img_h_slider, "value", 512) if txt2img_h_slider else 512
        if callable(width):
            width = width()
        if callable(height):
            height = height()

        # 创建处理对象
        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=task.prompt,
            negative_prompt=shared.opts.negative_prompt if hasattr(shared.opts, 'negative_prompt') else "",
            width=width,
            height=height,
            do_not_save_samples=True,
            do_not_save_grid=True,
            outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "",
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids or "",
        )

        if is_api_mode:
            # API 模式：通过 api_providers 传参考图，不用本地编码
            from modules_forge.api_providers import set_reference_images, get_session_api_key
            api_key = get_session_api_key()
            print(f"[Image Stitch Batch] API Key check: key='{api_key[:4] if api_key else '(empty)'}...', forge_model_mode='{forge_model_mode}'")
            if not api_key:
                # 尝试从 shared.opts 读取
                fallback_key = getattr(shared.opts, 'forge_api_key', '')
                if fallback_key:
                    from modules_forge.api_providers import set_session_api_key
                    set_session_api_key(fallback_key)
                    print(f"[Image Stitch Batch] 已从 opts 恢复 API Key")
                    api_key = fallback_key
            set_reference_images([task.reference_image])
            processed = process_images(p)
            set_reference_images([])  # 清理
        else:
            # 本地模式：编码参考图到模型
            p.clear_prompt_cache()
            p.sd_model.clear_references()
            dynamic_args.is_referencing = True

            ref = ImageStitch.preprocess(ImageStitch, task.reference_image, 1024)
            image = images.flatten(ref, opts.img2img_background_color)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).to(device=device).unsqueeze(0)
            images_tensor_to_samples(image, 0, p.sd_model)

            dynamic_args.is_referencing = False

            processed = process_images(p)

        # 提取结果图片
        for img in processed.images:
            if isinstance(img, Image.Image):
                return img
        return None


# 全局批量任务管理器实例
batch_manager = BatchTaskManager()


class ImageStitch(scripts.Script):
    sorting_priority = 529

    def __init__(self):
        self.cached_parameters: list[int] = None

    def title(self):
        return "多图参考"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(i2i_info if is_img2img else t2i_info)

            # 使用 State 存储当前图片列表
            current_images_state = gr.State(value=[])

            # 图片上传区域
            with gr.Row():
                upload_btn = gr.UploadButton("📤 上传图片", file_types=["image"], type="binary", size="sm", scale=1)
                delete_btn = gr.Button("🗑️ 删除选中", size="sm", variant="stop", scale=1)
                clear_btn = gr.Button("清空", size="sm", scale=1)

            # 图片选择器
            image_selector = gr.Dropdown(
                label="选择要删除的图片",
                choices=[],
                value=None,
                interactive=True,
                allow_custom_value=False,
                scale=2
            )

            references = gr.Gallery(
                value=None,
                type="pil",
                interactive=False,
                show_label=False,
                container=False,
                label="参考潜空间",
                min_width=384,
                height=384,
                columns=3,
                rows=1,
                allow_preview=False,
                object_fit="contain",
                elem_id="image_stitch_ref_latent",
            )

            # 最大边长限制
            max_dim = gr.Slider(
                minimum=0,
                maximum=2048,
                value=1024,
                step=256,
                label="最大边长限制",
                info="降低编码时的显存占用；设为 0 表示不限制",
            )

            # 自动设置尺寸
            auto_size_btn = gr.Button("📏 从首图设置尺寸", size="sm", visible=False)

            # Pose 素材库 - 折叠面板
            with gr.Accordion("📚 Pose 素材库", open=False):
                pose_gallery = gr.Gallery(
                    label="点击选择图片添加到上方",
                    value=[],
                    type="filepath",
                    interactive=False,
                    show_label=True,
                    container=True,
                    min_width=384,
                    height=400,
                    columns=6,
                    rows=3,
                    allow_preview=True,
                    object_fit="cover",
                    elem_id="pose_material_library",
                )

                with gr.Row():
                    refresh_pose_btn = gr.Button("🔄 刷新素材库", size="sm")
                    pose_count_info = gr.Textbox(
                        label="素材数量",
                        value="加载中...",
                        interactive=False,
                        scale=2
                    )

            # 常用编辑提示词 - 折叠面板
            with gr.Accordion("✏️ 常用编辑提示词", open=False):
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt_input = gr.Textbox(
                            label="输入新提示词",
                            placeholder="例如：改为3d灰模，生成三视图，正面，侧面，背面",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        add_prompt_btn = gr.Button("➕ 添加", variant="primary", size="sm")

                preset_prompts_dropdown = gr.Dropdown(
                    label="选择预设提示词",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=False,
                    info="从下拉列表中选择常用提示词"
                )

                with gr.Row():
                    delete_preset_btn = gr.Button("🗑️ 删除选中提示词", variant="stop", size="sm")
                    refresh_presets_btn = gr.Button("🔄 刷新列表", size="sm")

                preset_info = gr.Textbox(
                    label="提示词管理说明",
                    value="💡 使用方法：从下拉列表选择提示词后点击'🗑️ 删除选中提示词'即可删除",
                    interactive=False
                )

            # ========== 批量任务 ==========
            with gr.Accordion("📋 批量任务", open=False):
                gr.HTML(
                    """
                    <div style="margin-bottom: 8px; font-size: 13px; color: #666;">
                    每组任务包含：一张参考图 + 一条关键词。点击"添加到队列"后，任务会进入队列等待处理。
                    </div>
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_image_upload = gr.UploadButton(
                            "📤 上传参考图", 
                            file_types=["image"], 
                            type="binary",
                            size="sm",
                        )
                        batch_image_preview = gr.Image(
                            label="参考图预览",
                            type="pil",
                            height=200,
                            interactive=False,
                        )

                    with gr.Column(scale=2):
                        batch_prompt = gr.Textbox(
                            label="关键词",
                            placeholder="输入编辑关键词，例如：改为3d灰模，去除背景，生成三视图...",
                            lines=3,
                        )
                        with gr.Row():
                            add_batch_btn = gr.Button("➕ 添加到队列", variant="primary", size="sm", scale=2)
                            batch_clear_input_btn = gr.Button("清空输入", size="sm", scale=1)

                # 任务队列状态
                with gr.Row():
                    batch_stats = gr.HTML(
                        value="<div style='font-size:13px; color:#888;'>队列为空</div>",
                    )
                    batch_refresh_btn = gr.Button("🔄 刷新状态", size="sm", scale=1)
                    batch_start_btn = gr.Button("▶ 开始批量生成", variant="primary", size="sm", scale=1)
                    batch_stop_btn = gr.Button("⏹ 停止", variant="stop", size="sm", scale=1)
                    batch_clear_btn = gr.Button("🗑️ 清空已完成", size="sm", scale=1)

                # 任务列表
                batch_task_list = gr.HTML(
                    value="<div style='font-size:13px; color:#888;'>暂无任务</div>",
                )

                # 结果画廊
                batch_gallery = gr.Gallery(
                    label="生成结果",
                    value=[],
                    type="pil",
                    columns=4,
                    rows=2,
                    height=400,
                    object_fit="contain",
                    interactive=False,

                )

                # 用于存储任务结果的 State
                batch_results_state = gr.State(value=[])

            # ========== 事件绑定 ==========

            # ===== 提示词管理功能 =====

            def get_preset_prompts_dir():
                """Get preset prompts storage directory."""
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                presets_dir = os.path.join(plugin_dir, "..", "presets")
                if not os.path.exists(presets_dir):
                    os.makedirs(presets_dir)
                return presets_dir

            def load_preset_prompts():
                """Load all preset prompts."""
                presets_dir = get_preset_prompts_dir()
                prompts = []

                print(f"[Image Stitch] 开始扫描目录: {presets_dir}")

                for file in sorted(os.listdir(presets_dir)):
                    if file.endswith('.txt'):
                        filepath = os.path.join(presets_dir, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                prompt_text = f.read().strip()
                                print(f"[Image Stitch] 读取文件 {file}: '{prompt_text}' (长度: {len(prompt_text)})")
                                if prompt_text:
                                    prompts.append(prompt_text)
                        except Exception as e:
                            print(f"[Image Stitch] 加载提示词失败 {file}: {e}")

                print(f"[Image Stitch] ✅ 成功加载 {len(prompts)} 个预设提示词: {prompts}")
                return gr.update(choices=prompts, value=(prompts[0] if prompts else None))

            def save_preset_prompt(prompt_text):
                """Save a new preset prompt."""
                if not prompt_text or not prompt_text.strip():
                    return gr.update(), load_preset_prompts(), "❌ 提示词不能为空"

                presets_dir = get_preset_prompts_dir()
                prompt_text = prompt_text.strip()

                filename = prompt_text[:20].replace('/', '_').replace('\\', '_').replace(':', '_')
                filepath = os.path.join(presets_dir, f"{filename}.txt")

                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(presets_dir, f"{filename}_{counter}.txt")
                    counter += 1

                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(prompt_text)
                    print(f"[Image Stitch] 已保存提示词: {filepath}")
                    return gr.update(value=""), load_preset_prompts(), f"✅ 已保存: {os.path.basename(filepath)}"
                except Exception as e:
                    print(f"[Image Stitch] 保存提示词失败: {e}")
                    return gr.update(), load_preset_prompts(), f"❌ 保存失败: {str(e)}"

            def delete_selected_preset(selected_prompt):
                """Delete the selected preset prompt."""
                if not selected_prompt:
                    print("[Image Stitch] ❌ 删除失败：未选择提示词")
                    return load_preset_prompts(), "❌ 请先从下拉列表选择要删除的提示词"

                presets_dir = get_preset_prompts_dir()
                print(f"[Image Stitch] 开始删除提示词: '{selected_prompt}'")

                deleted = False
                deleted_file = None
                target_filepath = None

                for file in sorted(os.listdir(presets_dir)):
                    if file.endswith('.txt'):
                        filepath = os.path.join(presets_dir, file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content == selected_prompt:
                                    target_filepath = filepath
                                    deleted_file = file
                                    print(f"[Image Stitch] 🎯 找到目标文件: {file}")
                                    break
                        except Exception as e:
                            print(f"[Image Stitch] ⚠️ 读取文件失败 {file}: {e}")

                if target_filepath:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            os.remove(target_filepath)
                            deleted = True
                            print(f"[Image Stitch] ✅ 成功删除文件: {deleted_file}")
                            break
                        except PermissionError as e:
                            if attempt < max_retries - 1:
                                wait_time = 0.5 * (attempt + 1)
                                print(f"[Image Stitch] ⚠️ 文件被占用，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                                time.sleep(wait_time)
                            else:
                                print(f"[Image Stitch] ❌ 删除失败（重试{max_retries}次后仍被占用）: {e}")
                                return load_preset_prompts(), f"❌ 文件被占用，请关闭相关程序后重试: {deleted_file}"
                        except Exception as e:
                            print(f"[Image Stitch] ❌ 删除失败: {e}")
                            return load_preset_prompts(), f"❌ 删除失败: {str(e)}"
                else:
                    print(f"[Image Stitch] ❌ 未找到匹配的文件")
                    return load_preset_prompts(), f"❌ 未找到提示词: {selected_prompt[:30]}..."

                if deleted:
                    print(f"[Image Stitch] ✅ 删除成功，刷新列表")
                    return load_preset_prompts(), f"✅ 已删除: {deleted_file}"
                else:
                    return load_preset_prompts(), f"❌ 删除失败: {deleted_file}"

            def use_selected_preset(selected_prompt):
                """Use the selected preset prompt."""
                if not selected_prompt:
                    return "", "❌ 请先从下拉列表选择提示词"
                print(f"[Image Stitch] 使用提示词: {selected_prompt[:50]}...")
                return selected_prompt, f"✅ 已加载提示词"

            # ===== Pose 素材库 =====

            def scan_pose_library():
                """Scan pose directory and return image list."""
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                pose_dir = os.path.join(plugin_dir, "..", "pose")

                if not os.path.exists(pose_dir):
                    print(f"[Image Stitch] Pose 目录不存在: {pose_dir}")
                    return [], "❌ 目录不存在"

                supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.bmp"]
                image_files = []

                for ext in supported_extensions:
                    pattern = os.path.join(pose_dir, ext)
                    image_files.extend(glob.glob(pattern))

                image_files.sort()

                print(f"[Image Stitch] 找到 {len(image_files)} 个 Pose 素材")
                return image_files, f"✅ 共 {len(image_files)} 个素材"

            def init_default_prompts():
                """Initialize default prompts on first run."""
                presets_dir = get_preset_prompts_dir()

                existing_files = [f for f in os.listdir(presets_dir) if f.endswith('.txt')]
                if existing_files:
                    print(f"[Image Stitch] 已存在 {len(existing_files)} 个预设提示词，跳过初始化")
                    return

                default_prompts = [
                    "改为3d灰模",
                    "生成三视图，正面，侧面，背面",
                    "只参考姿势",
                    "去除背景",
                    "去除文字",
                    "4k画质",
                    "二次元动漫风格"
                ]

                for prompt in default_prompts:
                    filename = prompt[:20].replace('/', '_').replace('\\', '_').replace(':', '_')
                    filepath = os.path.join(presets_dir, f"{filename}.txt")

                    counter = 1
                    while os.path.exists(filepath):
                        filepath = os.path.join(presets_dir, f"{filename}_{counter}.txt")
                        counter += 1

                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(prompt)
                        print(f"[Image Stitch] 创建默认提示词: {prompt}")
                    except Exception as e:
                        print(f"[Image Stitch] 创建默认提示词失败: {e}")

                print(f"[Image Stitch] 默认提示词初始化完成")

            # 从素材库添加图片到主列表
            def add_pose_to_gallery(current_images, pose_path):
                """Add selected pose image to main list."""
                if not pose_path or not os.path.exists(pose_path):
                    return current_images

                try:
                    new_image = Image.open(pose_path)
                    if new_image.mode == 'RGBA':
                        new_image = new_image.convert('RGB')

                    if not current_images:
                        updated_images = [new_image]
                    else:
                        updated_images = current_images.copy()
                        updated_images.append(new_image)

                    print(f"[Image Stitch] 已添加 Pose: {os.path.basename(pose_path)}")
                    return updated_images
                except Exception as e:
                    print(f"[Image Stitch] 添加 Pose 失败: {e}")
                    return current_images

            # 自动设置尺寸函数
            def auto_set_dimensions(gallery):
                if not gallery:
                    return gr.skip(), gr.skip()

                first_image = None
                if isinstance(gallery[0], str):
                    import base64
                    from io import BytesIO
                    if gallery[0].startswith('data:image'):
                        base64_str = gallery[0].split(',')[1]
                        img_data = base64.b64decode(base64_str)
                        first_image = Image.open(BytesIO(img_data))
                elif isinstance(gallery[0], tuple) and len(gallery[0]) > 0:
                    if isinstance(gallery[0][0], Image.Image):
                        first_image = gallery[0][0]
                    elif isinstance(gallery[0][0], str):
                        import base64
                        from io import BytesIO
                        if gallery[0][0].startswith('data:image'):
                            base64_str = gallery[0][0].split(',')[1]
                            img_data = base64.b64decode(base64_str)
                            first_image = Image.open(BytesIO(img_data))

                if first_image:
                    width, height = first_image.size
                    width = closesteight(width)
                    height = closesteight(height)
                    print(f"[Image Stitch] 从上传图像设置尺寸: {width}x{height}")
                    return width, height
                else:
                    print("[Image Stitch] 无法获取图像尺寸")
                    return gr.skip(), gr.skip()
            
            # ===== 图片上传/删除/清空事件 =====

            def _sync_to_api(images):
                """Sync the given image list to the global API reference variable."""
                try:
                    from modules_forge.api_providers import set_reference_images
                    pil_images = []
                    if images:
                        if isinstance(images[0], str):
                            from modules.api import api
                            pil_images = [api.decode_base64_to_image(img) for img in images]
                        elif isinstance(images[0], tuple):
                            pil_images = [img for (img, _) in images]
                        else:
                            pil_images = images
                    set_reference_images(pil_images)
                    print(f"[Image Stitch] Synced {len(pil_images)} reference image(s) to global API variable")
                except Exception as e:
                    print(f"[Image Stitch] Sync to global failed: {e}")

            def _add_image(new_img, current_list):
                """Add one uploaded image to the list."""
                from io import BytesIO
                updated = list(current_list) if current_list else []
                if new_img is not None:
                    if isinstance(new_img, bytes):
                        updated.append(Image.open(BytesIO(new_img)))
                    elif isinstance(new_img, str):
                        updated.append(Image.open(new_img))
                    elif isinstance(new_img, tuple) and len(new_img) >= 2 and isinstance(new_img[1], bytes):
                        updated.append(Image.open(BytesIO(new_img[1])))
                    elif isinstance(new_img, tuple) and len(new_img) >= 1 and isinstance(new_img[0], str):
                        updated.append(Image.open(new_img[0]))
                    else:
                        # Try to convert directly
                        try:
                            updated.append(Image.open(new_img) if isinstance(new_img, (str, bytes)) else Image.open(BytesIO(new_img)))
                        except Exception:
                            print(f"[Image Stitch] Unknown upload format: {type(new_img)}")
                _sync_to_api(updated)
                choices = [f"图片 {i+1}" for i in range(len(updated))]
                return updated, updated, gr.update(choices=choices, value=None)

            def _delete_selected(selected_label, current_list):
                """Delete the selected image from the list."""
                if selected_label is None or not current_list:
                    return current_list, current_list, gr.update()
                try:
                    idx = int(selected_label.split(" ")[1]) - 1
                    if 0 <= idx < len(current_list):
                        updated = [img for i, img in enumerate(current_list) if i != idx]
                        _sync_to_api(updated)
                        choices = [f"图片 {i+1}" for i in range(len(updated))]
                        return updated, updated, gr.update(choices=choices, value=None)
                except (ValueError, IndexError):
                    pass
                return current_list, current_list, gr.update()

            def _clear_all():
                """Clear all images."""
                _sync_to_api([])
                return [], [], gr.update(choices=[], value=None)

            upload_btn.upload(
                fn=_add_image,
                inputs=[upload_btn, current_images_state],
                outputs=[current_images_state, references, image_selector],
                show_progress=False
            )

            delete_btn.click(
                fn=_delete_selected,
                inputs=[image_selector, current_images_state],
                outputs=[current_images_state, references, image_selector],
                show_progress=False
            )

            clear_btn.click(
                fn=_clear_all,
                outputs=[current_images_state, references, image_selector],
                show_progress=False
            )
            
            # 页面加载时自动扫描素材库
            try:
                from modules.shared import demo
                demo.load(
                    fn=scan_pose_library,
                    inputs=[],
                    outputs=[pose_gallery, pose_count_info],
                    show_progress=False
                )
                
                # 初始化默认提示词
                demo.load(
                    fn=init_default_prompts,
                    inputs=[],
                    outputs=[],
                    show_progress=False
                )
                
                # 加载预设提示词列表
                demo.load(
                    fn=load_preset_prompts,
                    inputs=[],
                    outputs=[preset_prompts_dropdown],
                    show_progress=False
                )
            except Exception:
                pass
            
            # ===== 提示词管理事件绑定 =====
            
            # 绑定添加提示词按钮
            add_prompt_btn.click(
                fn=save_preset_prompt,
                inputs=[prompt_input],
                outputs=[prompt_input, preset_prompts_dropdown, preset_info],
                show_progress=False
            )
            
            # 绑定删除选中提示词按钮
            delete_preset_btn.click(
                fn=delete_selected_preset,
                inputs=[preset_prompts_dropdown],
                outputs=[preset_prompts_dropdown, preset_info],
                show_progress=False
            )
            
            # 绑定刷新预设列表按钮
            refresh_presets_btn.click(
                fn=load_preset_prompts,
                inputs=[],
                outputs=[preset_prompts_dropdown],
                show_progress=False
            ).then(
                fn=lambda: "✅ 已刷新",
                inputs=[],
                outputs=[preset_info],
                show_progress=False
            )

        

        # ===== 批量任务事件绑定 =====

            # 渲染帮助函数
            def _render_stats(stats: dict, extra_msg: str = "") -> str:
                parts = [
                    f"<span style='color:#666;'>总计: {stats['total']}</span>",
                ]
                if stats['waiting'] > 0:
                    parts.append(f"<span style='color:#f0ad4e;'>⏳ 等待: {stats['waiting']}</span>")
                if stats['processing'] > 0:
                    parts.append(f"<span style='color:#5bc0de;'>🔄 处理中: {stats['processing']}</span>")
                if stats['completed'] > 0:
                    parts.append(f"<span style='color:#5cb85c;'>✅ 已完成: {stats['completed']}</span>")
                if stats['failed'] > 0:
                    parts.append(f"<span style='color:#d9534f;'>❌ 失败: {stats['failed']}</span>")
                if extra_msg:
                    parts.append(f"<span>{extra_msg}</span>")
                return f"<div style='font-size:13px; display:flex; gap:12px;'>{' | '.join(parts)}</div>"

            def _render_task_list(tasks: list[dict]) -> str:
                if not tasks:
                    return "<div style='font-size:13px; color:#888;'>暂无任务</div>"
                rows = []
                for t in tasks:
                    status_icon = {
                        "waiting": "⏳",
                        "processing": "🔄",
                        "completed": "✅",
                        "failed": "❌",
                    }.get(t["status"], "❓")
                    error_text = f"<span style='color:red; font-size:11px;'>{t['error']}</span>" if t["error"] else ""
                    rows.append(
                        f"<div style='display:flex; gap:8px; padding:4px 0; border-bottom:1px solid #eee; font-size:13px;'>"
                        f"<span>{status_icon}</span>"
                        f"<span style='color:#888; width:60px;'>{t['id']}</span>"
                        f"<span style='flex:1;'>{t['prompt']}</span>"
                        f"<span style='color:#666;'>{t['status']}</span>"
                        f"{error_text}"
                        f"</div>"
                    )
                return f"<div style='max-height:300px; overflow-y:auto;'>{''.join(rows)}</div>"

            # 批量任务 - 上传图片预览
            def _upload_batch_image(img_bytes):
                if img_bytes is None:
                    return None
                from io import BytesIO
                try:
                    return Image.open(BytesIO(img_bytes))
                except Exception:
                    return None

            batch_image_upload.upload(
                fn=_upload_batch_image,
                inputs=[batch_image_upload],
                outputs=[batch_image_preview],
                show_progress=False,
            )

            # 批量任务 - 添加到队列
            def _add_batch_task(image, prompt_text):
                if image is None:
                    return (
                        "<div style='font-size:13px; color:red;'>请先上传参考图</div>",
                        "<div style='font-size:13px; color:#888;'>暂无任务</div>",
                        None,
                        [],
                    )
                if not prompt_text or not prompt_text.strip():
                    return (
                        "<div style='font-size:13px; color:red;'>请输入关键词</div>",
                        "<div style='font-size:13px; color:#888;'>暂无任务</div>",
                        None,
                        [],
                    )
                batch_manager.add_task(prompt_text.strip(), image)

                stats = batch_manager.get_statistics()
                tasks_html = _render_task_list(batch_manager.get_status())
                stats_html = _render_stats(stats)
                return stats_html, tasks_html, None, []

            add_batch_btn.click(
                fn=_add_batch_task,
                inputs=[batch_image_preview, batch_prompt],
                outputs=[batch_stats, batch_task_list, batch_image_preview, batch_gallery],
                show_progress=False,
            )

            # 批量任务 - 清空输入
            batch_clear_input_btn.click(
                fn=lambda: (None, ""),
                outputs=[batch_image_preview, batch_prompt],
                show_progress=False,
            )

            # 批量任务 - 刷新状态
            def _refresh_batch():
                stats = batch_manager.get_statistics()
                tasks = batch_manager.get_status()
                # 收集结果图片
                results = []
                for t in batch_manager.tasks:
                    if t.status == "completed" and t.result_image:
                        results.append(t.result_image)
                return _render_stats(stats), _render_task_list(tasks), results

            batch_refresh_btn.click(
                fn=_refresh_batch,
                inputs=[],
                outputs=[batch_stats, batch_task_list, batch_gallery],
                show_progress=False,
            )

            # 批量任务 - 开始批量生成
            def _start_batch(progress: gr.Progress = gr.Progress()):
                if batch_manager.is_running():
                    return (
                        "<div style='font-size:13px; color:orange;'>批量处理正在进行中...</div>",
                        "<div style='font-size:13px; color:#888;'>暂无任务</div>",
                        [],
                    )
                completed, failed = batch_manager.start_processing(progress)
                stats = batch_manager.get_statistics()
                tasks = batch_manager.get_status()
                results = []
                for t in batch_manager.tasks:
                    if t.status == "completed" and t.result_image:
                        results.append(t.result_image)
                msg = f"✅ 完成: {len(completed)} 个, ❌ 失败: {len(failed)} 个"
                return _render_stats(stats, msg), _render_task_list(tasks), results

            batch_start_btn.click(
                fn=_start_batch,
                inputs=[],
                outputs=[batch_stats, batch_task_list, batch_gallery],
                show_progress=False,
            )

            # 批量任务 - 停止
            def _stop_batch():
                if batch_manager.is_running():
                    batch_manager.stop()
                    return "<div style='font-size:13px; color:orange;'>正在停止...</div>"
                return "<div style='font-size:13px; color:#888;'>没有正在进行的批量任务</div>"

            batch_stop_btn.click(
                fn=_stop_batch,
                inputs=[],
                outputs=[batch_stats],
                show_progress=False,
            )

            # 批量任务 - 清空已完成
            def _clear_completed():
                batch_manager.clear_completed()
                stats = batch_manager.get_statistics()
                tasks = batch_manager.get_status()
                results = []
                for t in batch_manager.tasks:
                    if t.status == "completed" and t.result_image:
                        results.append(t.result_image)
                return _render_stats(stats), _render_task_list(tasks), results

            batch_clear_btn.click(
                fn=_clear_completed,
                inputs=[],
                outputs=[batch_stats, batch_task_list, batch_gallery],
                show_progress=False,
            )

            # 定时刷新任务状态
            batch_refresh_btn.click(
                fn=_refresh_batch,
                inputs=[],
                outputs=[batch_stats, batch_task_list, batch_gallery],
                show_progress=False,
            )

        

        return [enable, references, max_dim]

    @staticmethod
    def reset_references(p: StableDiffusionProcessing):
        # re-encode conditioning
        p.clear_prompt_cache()
        p.sd_model.clear_references()

    def process(self, p: StableDiffusionProcessing, enable: bool, references: list[str | tuple[Image.Image, str]], max_dim: int):
        if not (enable and references and any(getattr(dynamic_args, key) for key in ("kontext", "edit", "klein", "wan", "krea2"))):
            if self.cached_parameters is None:
                return

            # if previously enabled, clear out the ref_latents
            self.cached_parameters = None
            self.reset_references(p)
            return

        references = self.extract_images(references)

        # cache is based on reference inputs & model
        cache: list[str | int | bool] = [str(sd_models.model_data.forge_loading_parameters), *(self.hash_image(ref) for ref in references), (dynamic_args.wan and isinstance(p, StableDiffusionProcessingTxt2Img))]
        if self.cached_parameters == cache:
            return

        self.cached_parameters = cache
        self.reset_references(p)

        _batch_size: int = None

        if dynamic_args.wan:
            if isinstance(p, StableDiffusionProcessingTxt2Img):
                _batch_size = p.batch_size
                if _batch_size == 1:
                    shared.log.error("Wan 2.2 需要超过一帧...")
                    return
            if len(references) > 1:
                shared.log.warning("Wan 2.2 只使用第一张参考图片...")
                references = [references[0]]

        # Krea2 supports up to 3 reference images
        if dynamic_args.krea2 and len(references) > 3:
            shared.log.warning("Krea2 最多支持 3 张参考图片，将使用前 3 张...")
            references = references[:3]

        dynamic_args.is_referencing = True

        for reference in references:
            reference = self.preprocess(reference, max_dim)
            if _batch_size:
                reference = images.resize_image(1, reference, p.width, p.height)
            image = images.flatten(reference, opts.img2img_background_color)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).to(device=device).unsqueeze(0)

            if _batch_size:
                dim = [_batch_size - 1] + list(image.shape)[1:]
                empty = torch.empty(dim, dtype=torch.float32, device=device)
                image = torch.cat([image, empty], dim=0)

            images_tensor_to_samples(image, 0, p.sd_model)  # calls encode_first_stage

        dynamic_args.is_referencing = False

    @staticmethod
    def extract_images(gallery: list[str | tuple[Image.Image, str] | Image.Image]) -> list[Image.Image]:
        if not gallery:
            return []
        if isinstance(gallery[0], str):
            return [api.decode_base64_to_image(img) for img in gallery]
        if isinstance(gallery[0], tuple):
            return [img for (img, _) in gallery]
        # Already PIL Images
        return [img for img in gallery]

    @staticmethod
    def preprocess(img: Image.Image, limit: int) -> Image.Image:
        w, h = img.size

        if limit > 0 and max(w, h) > limit:
            ratio = limit / max(w, h)
            _w, _h = int(w * ratio), int(h * ratio)
        else:
            _w, _h = w, h

        if _w % 64 != 0 or _h % 64 != 0:
            _w = round(_w / 64) * 64
            _h = round(_h / 64) * 64

        if w != _w or h != _h:
            return images.resize_image(1, img, _w, _h)
        else:
            return img

    @staticmethod
    def hash_image(img: Image.Image) -> int:
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        img = img.convert("L")
        return hash(str(list(img.getdata())))
    
    def after_component(self, component: gr.components.Component, **kwargs):
        """注册主界面组件"""
        global txt2img_w_slider, txt2img_h_slider, img2img_w_slider, img2img_h_slider
        global txt2img_prompt, img2img_prompt
        
        elem_id = getattr(component, "elem_id", None)
        if elem_id == "txt2img_width":
            txt2img_w_slider = component
        elif elem_id == "txt2img_height":
            txt2img_h_slider = component
        elif elem_id == "img2img_width":
            img2img_w_slider = component
        elif elem_id == "img2img_height":
            img2img_h_slider = component
        elif elem_id == "txt2img_prompt":
            txt2img_prompt = component
        elif elem_id == "img2img_prompt":
            img2img_prompt = component








































































































