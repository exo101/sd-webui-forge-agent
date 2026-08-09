import gradio as gr
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
import os
import glob
import time

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
用于 <b>Flux-Kontext</b> / <b>Flux.2-Klein</b> / <b>Qwen-Image-Edit</b> / <b>Krea 2</b><br>
在 <b>文生图</b> 中使用以实现自定义分辨率的空潜空间效果<br>
用于 <b>Wan 2.2 I2V</b>：在 <b>文生图</b> 中设置为最后一帧以实现 LastFrameToVideo<br>
<b>注意:</b> 这实际上并不会拼接图像
"""

i2i_info = """
用于 <b>Flux-Kontext</b> / <b>Flux.2-Klein</b> / <b>Qwen-Image-Edit</b> / <b>Krea 2</b><br>
在 <b>图生图</b> 中使用以实现多图输入效果<br>
用于 <b>Wan 2.2 I2V</b>：在 <b>图生图</b> 中设置为最后一帧以实现 FirstLastFrameToVideo<br>
<b>注意:</b> 这实际上并不会拼接图像
"""


class ImageStitch(scripts.Script):
    sorting_priority = 529

    def __init__(self):
        self.cached_parameters: list[int] = None

    def title(self):
        return "多图拼接参考"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(i2i_info if is_img2img else t2i_info)
            
            # 使用 State 存储当前图片列表，支持追加模式
            current_images_state = gr.State(value=[])
            select_index = gr.State(-1)  # 选中的图片索引
            
            references = gr.Gallery(
                value=None,
                type="pil",
                interactive=True,
                show_label=False,
                container=False,
                show_download_button=False,
                show_share_button=False,
                label="参考潜空间",
                min_width=384,
                height=384,
                columns=3,
                rows=1,
                allow_preview=False,
                object_fit="contain",
                elem_id="image_stitch_ref_latent",
            )
            
            # Pose 素材库 - 折叠面板
            with gr.Accordion("📚 Pose 素材库", open=False):
                pose_gallery = gr.Gallery(
                    label="点击选择图片添加到上方",
                    value=[],
                    type="filepath",
                    interactive=False,
                    show_label=True,
                    container=True,
                    show_download_button=False,
                    show_share_button=False,
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
                
                # 预设提示词列表 - 使用 Dropdown 下拉选择
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
                    value="💡 使用方法：从下拉列表选择提示词后点击'🗑️ 删除选中提示词'即可删除，或点击'🔄 刷新列表'重新加载",
                    interactive=False
                )
            
            # 添加图片上传区域 - 紧凑布局
            with gr.Row():
                add_image_input = gr.Image(
                    label="拖拽或点击上传图片",
                    type="pil",
                    height=150,
                    show_label=True,
                    scale=3
                )
                with gr.Column(scale=1):
                    add_btn = gr.Button("➕ 添加", variant="primary", size="sm")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="sm")
            
            # 交换位置控制
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**调整顺序：**")
                with gr.Column(scale=3):
                    with gr.Row():
                        position1 = gr.Number(label="位置1", value=0, minimum=0, step=1)
                        position2 = gr.Number(label="位置2", value=1, minimum=0, step=1)
                    swap_btn = gr.Button("🔄 交换", size="sm")
            
            # 删除选中图片按钮
            with gr.Row():
                delete_selected_btn = gr.Button("🗑️ 删除选中图片", variant="stop", size="sm")
                selected_info = gr.Textbox(
                    label="选中状态",
                    value="未选中任何图片",
                    interactive=False,
                    scale=3
                )
            
            # 最大边长限制 - 防止爆显存
            max_dim = gr.Slider(
                minimum=0,
                maximum=2048,
                value=1024,
                step=256,
                label="最大边长限制",
                info="降低编码时的显存占用；应用于所有参考图片；设为 0 表示不限制",
            )
            
            # 自动设置尺寸控制
            with gr.Row():
                auto_size_btn = gr.Button("📏 从首图设置尺寸", size="sm")
            
            # ========== 事件绑定 ==========
            
            # ===== 提示词管理功能 =====
            
            def get_preset_prompts_dir():
                """获取预设提示词存储目录"""
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                presets_dir = os.path.join(plugin_dir, "..", "presets")
                if not os.path.exists(presets_dir):
                    os.makedirs(presets_dir)
                return presets_dir
            
            def load_preset_prompts():
                """加载所有预设提示词"""
                presets_dir = get_preset_prompts_dir()
                prompts = []
                
                print(f"[Image Stitch] 开始扫描目录: {presets_dir}")
                
                # 读取所有 .txt 文件
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
                # 返回 gr.update 以正确更新 Dropdown 的 choices
                return gr.update(choices=prompts, value=(prompts[0] if prompts else None))
            
            def save_preset_prompt(prompt_text):
                """保存新的预设提示词"""
                if not prompt_text or not prompt_text.strip():
                    return gr.update(), load_preset_prompts(), "❌ 提示词不能为空"
                
                presets_dir = get_preset_prompts_dir()
                prompt_text = prompt_text.strip()
                
                # 生成文件名（使用前20个字符）
                filename = prompt_text[:20].replace('/', '_').replace('\\', '_').replace(':', '_')
                filepath = os.path.join(presets_dir, f"{filename}.txt")
                
                # 如果文件已存在，添加序号
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
                """删除选中的预设提示词"""
                if not selected_prompt:
                    print("[Image Stitch] ❌ 删除失败：未选择提示词")
                    return load_preset_prompts(), "❌ 请先从下拉列表选择要删除的提示词"
                
                presets_dir = get_preset_prompts_dir()
                print(f"[Image Stitch] 开始删除提示词: '{selected_prompt}'")
                
                # 查找对应的文件
                deleted = False
                deleted_file = None
                target_filepath = None
                
                # 第一次遍历：找到目标文件路径
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
                
                # 如果找到目标文件，尝试删除（带重试机制）
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
                """使用选中的预设提示词"""
                if not selected_prompt:
                    return "", "❌ 请先从下拉列表选择提示词"
                
                print(f"[Image Stitch] 使用提示词: {selected_prompt[:50]}...")
                return selected_prompt, f"✅ 已加载提示词"
            
            # ===== Pose 素材库 =====
            
            # 扫描并加载 Pose 素材库
            def scan_pose_library():
                """扫描 pose 目录并返回图片列表"""
                # 获取插件目录下的 pose 文件夹
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                pose_dir = os.path.join(plugin_dir, "..", "pose")
                
                if not os.path.exists(pose_dir):
                    print(f"[Image Stitch] Pose 目录不存在: {pose_dir}")
                    return [], "❌ 目录不存在"
                
                # 支持的图片格式
                supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.bmp"]
                image_files = []
                
                for ext in supported_extensions:
                    pattern = os.path.join(pose_dir, ext)
                    image_files.extend(glob.glob(pattern))
                
                # 按文件名排序
                image_files.sort()
                
                print(f"[Image Stitch] 找到 {len(image_files)} 个 Pose 素材")
                return image_files, f"✅ 共 {len(image_files)} 个素材"
            
            # 初始化默认提示词
            def init_default_prompts():
                """初始化默认提示词（仅在首次运行时执行）"""
                presets_dir = get_preset_prompts_dir()
                
                # 检查是否已有提示词
                existing_files = [f for f in os.listdir(presets_dir) if f.endswith('.txt')]
                if existing_files:
                    print(f"[Image Stitch] 已存在 {len(existing_files)} 个预设提示词，跳过初始化")
                    return
                
                # 默认提示词列表
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
                    
                    # 避免覆盖已存在的文件
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
                """将选中的 pose 图片添加到主列表"""
                if not pose_path or not os.path.exists(pose_path):
                    return current_images
                
                try:
                    # 加载图片
                    new_image = Image.open(pose_path)
                    if new_image.mode == 'RGBA':
                        new_image = new_image.convert('RGB')
                    
                    # 追加到现有列表
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
            
            # 添加图片函数 - 追加模式
            def add_image_to_gallery(current_images, new_image):
                """将新图片追加到现有列表中"""
                if new_image is None:
                    return current_images, gr.update(value=None)
                
                # 如果当前没有图片，直接返回新图片
                if not current_images:
                    return [new_image], gr.update(value=None)
                
                # 追加到现有列表
                updated_images = current_images.copy()
                updated_images.append(new_image)
                print(f"[Image Stitch] 已添加图片，当前共 {len(updated_images)} 张")
                
                return updated_images, gr.update(value=None)
            
            # 清空图片函数
            def clear_gallery():
                """清空所有图片"""
                print("[Image Stitch] 已清空图片列表")
                return [], gr.update(value=None)
            
            # 交换位置函数
            def swap_positions(gallery, pos1, pos2):
                if not gallery or len(gallery) < 2:
                    return gallery
                
                pos1 = int(pos1)
                pos2 = int(pos2)
                
                if pos1 < 0 or pos1 >= len(gallery) or pos2 < 0 or pos2 >= len(gallery):
                    return gallery
                
                # 创建新的列表并交换元素
                new_gallery = gallery.copy()
                new_gallery[pos1], new_gallery[pos2] = new_gallery[pos2], new_gallery[pos1]
                return new_gallery
            
            # 自动设置尺寸函数
            def auto_set_dimensions(gallery):
                if not gallery:
                    return gr.skip(), gr.skip()
                
                # 获取第一张图像
                first_image = None
                if isinstance(gallery[0], str):
                    # 处理base64编码的图像
                    import base64
                    from io import BytesIO
                    if gallery[0].startswith('data:image'):
                        # 移除data:image前缀
                        base64_str = gallery[0].split(',')[1]
                        # 解码base64
                        img_data = base64.b64decode(base64_str)
                        # 转换为PIL图像
                        first_image = Image.open(BytesIO(img_data))
                elif isinstance(gallery[0], tuple) and len(gallery[0]) > 0:
                    # 处理元组形式的图像
                    if isinstance(gallery[0][0], Image.Image):
                        first_image = gallery[0][0]
                    elif isinstance(gallery[0][0], str):
                        # 处理元组中的base64图像
                        import base64
                        from io import BytesIO
                        if gallery[0][0].startswith('data:image'):
                            base64_str = gallery[0][0].split(',')[1]
                            img_data = base64.b64decode(base64_str)
                            first_image = Image.open(BytesIO(img_data))
                
                if first_image:
                    width, height = first_image.size
                    # 调整为8的倍数
                    width = closesteight(width)
                    height = closesteight(height)
                    print(f"[Image Stitch] 从上传图像设置尺寸: {width}x{height}")
                    return width, height
                else:
                    print("[Image Stitch] 无法获取图像尺寸")
                    return gr.skip(), gr.skip()
            
            # Gallery 选择事件 - 记录选中索引
            def on_select(evt: gr.SelectData) -> int:
                return evt.index
            
            references.select(
                fn=on_select,
                outputs=[select_index],
                queue=False,
                show_progress=False
            ).then(
                fn=lambda idx: f"已选中第 {idx + 1} 张图片" if idx >= 0 else "未选中任何图片",
                inputs=[select_index],
                outputs=[selected_info],
                show_progress=False
            )
            
            # 删除选中图片函数
            def delete_selected_image(current_images, index):
                """删除选中的图片"""
                if not current_images or index < 0 or index >= len(current_images):
                    print("[Image Stitch] 删除失败：未选中有效图片")
                    return current_images, -1, "❌ 请先点击选择要删除的图片"
                
                # 创建新列表并移除指定索引的图片
                updated_images = current_images.copy()
                removed_image = updated_images.pop(index)
                
                print(f"[Image Stitch] 已删除第 {index + 1} 张图片，剩余 {len(updated_images)} 张")
                return updated_images, -1, f"✅ 已删除第 {index + 1} 张图片"
            
            # 绑定刷新素材库按钮
            refresh_pose_btn.click(
                fn=scan_pose_library,
                inputs=[],
                outputs=[pose_gallery, pose_count_info],
                show_progress=False
            )
            
            # 绑定 Pose 画廊选择事件 (点击添加)
            pose_gallery.select(
                fn=add_pose_to_gallery,
                inputs=[current_images_state, pose_gallery],
                outputs=[current_images_state],
                show_progress=False
            ).then(
                fn=lambda x: x,
                inputs=[current_images_state],
                outputs=[references],
                show_progress=False
            )
            
            # 绑定添加图片按钮
            add_btn.click(
                fn=add_image_to_gallery,
                inputs=[current_images_state, add_image_input],
                outputs=[current_images_state, add_image_input],
                show_progress=False
            ).then(
                fn=lambda x: x,
                inputs=[current_images_state],
                outputs=[references],
                show_progress=False
            )
            
            # 绑定清空按钮
            clear_btn.click(
                fn=clear_gallery,
                inputs=[],
                outputs=[current_images_state, add_image_input],
                show_progress=False
            ).then(
                fn=lambda x: x,
                inputs=[current_images_state],
                outputs=[references],
                show_progress=False
            )
            
            # 绑定交换按钮
            swap_btn.click(
                fn=swap_positions,
                inputs=[references, position1, position2],
                outputs=[references],
                show_progress=False
            ).then(
                fn=lambda x: x,
                inputs=[references],
                outputs=[current_images_state],
                show_progress=False
            )
            
            # 绑定自动设置尺寸按钮
            # 获取主界面的宽度和高度滑块
            width_slider = txt2img_w_slider if not is_img2img else img2img_w_slider
            height_slider = txt2img_h_slider if not is_img2img else img2img_h_slider
            
            # 绑定按钮点击事件
            auto_size_btn.click(
                fn=auto_set_dimensions,
                inputs=[references],
                outputs=[width_slider, height_slider],
                show_progress=False
            )
            
            # 绑定删除选中图片按钮
            delete_selected_btn.click(
                fn=delete_selected_image,
                inputs=[current_images_state, select_index],
                outputs=[current_images_state, select_index, selected_info],
                show_progress=False
            ).then(
                fn=lambda x: x,
                inputs=[current_images_state],
                outputs=[references],
                show_progress=False
            )
            
            # Gallery 变化时同步更新 State
            references.change(
                fn=lambda x: x,
                inputs=[references],
                outputs=[current_images_state],
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
    def extract_images(gallery: list[str | tuple[Image.Image, str]]) -> list[Image.Image]:
        if isinstance(gallery[0], str):
            return [api.decode_base64_to_image(img) for img in gallery]
        return [img for (img, _) in gallery]

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








































































































