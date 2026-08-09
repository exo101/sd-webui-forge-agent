"""
Qwen视频生成主UI模块
整合所有子模块，提供统一的UI界面
"""

import gradio as gr
import os
import tempfile
from PIL import Image
from modules import shared

from .api_handler import set_api_key
from .video_models import (
    generate_video_with_wan27_t2v,
    generate_video_with_wan27_i2v,
    generate_video_with_wan27_r2v,
    generate_video_with_wan27_videoedit
)
from .task_query import query_video_task, get_recent_tasks
from .utils import open_video_output_dir, create_html_video_player, download_video_to_local


def save_image_to_temp(im):
    """将PIL图像保存到临时文件并返回路径"""
    print(f"[DEBUG save_image_to_temp] 输入类型: {type(im)}")
    print(f"[DEBUG save_image_to_temp] 输入值: {im}")
    
    if im is None:
        print("[DEBUG save_image_to_temp] 输入为None，返回None")
        return None
    if isinstance(im, str):
        print(f"[DEBUG save_image_to_temp] 输入是字符串路径: {im}")
        return im
    if hasattr(im, 'name'):
        print(f"[DEBUG save_image_to_temp] 输入有name属性: {im.name}")
        return im.name
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_img_{os.urandom(8).hex()}.png")
    
    try:
        if isinstance(im, Image.Image):
            print(f"[DEBUG save_image_to_temp] 输入是PIL图像，保存到: {temp_path}")
            im.save(temp_path)
            print(f"[DEBUG save_image_to_temp] 保存成功，文件存在: {os.path.exists(temp_path)}")
            return temp_path
        elif isinstance(im, dict) and 'image' in im:
            print(f"[DEBUG save_image_to_temp] 输入是字典，keys: {im.keys()}")
            print(f"[DEBUG save_image_to_temp] image字段类型: {type(im['image'])}")
            img_array = im['image']
            if img_array is not None:
                pil_img = Image.fromarray(img_array)
                pil_img.save(temp_path)
                print(f"[DEBUG save_image_to_temp] 从字典保存成功，文件存在: {os.path.exists(temp_path)}")
                return temp_path
            else:
                print("[DEBUG save_image_to_temp] image字段为None")
                return None
        else:
            print(f"[DEBUG save_image_to_temp] 无法处理的类型: {type(im)}")
            return None
    except Exception as e:
        print(f"[DEBUG save_image_to_temp] 保存临时图像失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_qwen_video_gen_ui():
    """
    创建Qwen视频生成UI界面
    """
    with gr.Blocks() as qwen_video_interface:
        gr.Markdown("# Qwen视频生成 API")
        gr.Markdown("使用阿里云百炼平台的wan2.7模型进行视频生成")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="请输入您的百炼API Key",
                    info="输入API Key后点击下方按钮设置"
                )
                set_api_key_btn = gr.Button("设置API Key", variant="secondary")
                api_key_status = gr.Textbox(label="状态", interactive=False)
                
                set_api_key_btn.click(
                    fn=set_api_key,
                    inputs=api_key_input,
                    outputs=api_key_status
                )
        
        with gr.Tabs():
            with gr.TabItem("文生视频 (wan2.7)"):
                with gr.Row():
                    with gr.Column():
                        wan27_t2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...支持多镜头叙事，如：第1个镜头[0-3秒] 全景：雨夜的纽约街头",
                            lines=5,
                            max_lines=8
                        )
                        
                        wan27_t2v_negative_prompt = gr.Textbox(
                            label="反向提示词",
                            placeholder="输入不希望出现在视频中的内容，如：低质量、模糊、多余手指",
                            lines=2,
                            max_lines=4
                        )
                        
                        wan27_t2v_audio_file = gr.Audio(
                            label="音频文件（可选）",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            wan27_t2v_resolution = gr.Dropdown(
                                label="分辨率",
                                choices=["720P", "1080P"],
                                value="720P",
                                interactive=True
                            )
                            
                            wan27_t2v_duration = gr.Slider(
                                label="时长（秒）",
                                minimum=2,
                                maximum=15,
                                step=1,
                                value=10
                            )
                        
                        wan27_t2v_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            wan27_t2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            wan27_t2v_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def process_wan27_t2v_request(audio_file, prompt, negative_prompt, resolution, duration):
                    result = generate_video_with_wan27_t2v(prompt, audio_file, resolution, duration, negative_prompt)
                    
                    if "任务ID:" in result:
                        return result, None
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                wan27_t2v_gen_btn.click(
                    fn=process_wan27_t2v_request,
                    inputs=[wan27_t2v_audio_file, wan27_t2v_prompt, wan27_t2v_negative_prompt, wan27_t2v_resolution, wan27_t2v_duration],
                    outputs=[wan27_t2v_progress, wan27_t2v_output]
                )
            
            with gr.TabItem("图生视频 (wan2.7)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 输入类型选择")
                        wan27_i2v_mode = gr.Dropdown(
                            label="生成模式",
                            choices=[
                                ("首帧生视频", "first_frame"),
                                ("首尾帧生视频", "first_last_frame"),
                                ("视频续写", "video_continuation"),
                                ("视频续写+尾帧", "video_continuation_last_frame")
                            ],
                            value="first_frame",
                            interactive=True
                        )
                        
                        wan27_i2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...",
                            lines=4,
                            max_lines=6
                        )
                        
                        wan27_i2v_negative_prompt = gr.Textbox(
                            label="反向提示词",
                            placeholder="输入不希望出现在视频中的内容",
                            lines=2,
                            max_lines=4
                        )
                        
                        with gr.Group(visible=True) as first_frame_group:
                            wan27_i2v_first_frame = gr.Image(
                                label="首帧图像",
                                interactive=True
                            )
                        
                        with gr.Group(visible=False) as last_frame_group:
                            wan27_i2v_last_frame = gr.Image(
                                label="尾帧图像",
                                interactive=True
                            )
                        
                        with gr.Group(visible=False) as first_clip_group:
                            wan27_i2v_first_clip = gr.Video(
                                label="首段视频",
                                interactive=True
                            )
                        
                        wan27_i2v_audio_file = gr.Audio(
                            label="驱动音频（可选）",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            wan27_i2v_resolution = gr.Dropdown(
                                label="分辨率",
                                choices=["720P", "1080P"],
                                value="720P",
                                interactive=True
                            )
                            
                            wan27_i2v_duration = gr.Slider(
                                label="时长（秒）",
                                minimum=2,
                                maximum=15,
                                step=1,
                                value=10
                            )
                        
                        wan27_i2v_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            wan27_i2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            wan27_i2v_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def update_i2v_visibility(mode):
                    first_frame_visible = mode in ["first_frame", "first_last_frame"]
                    last_frame_visible = mode in ["first_last_frame", "video_continuation_last_frame"]
                    first_clip_visible = mode in ["video_continuation", "video_continuation_last_frame"]
                    return gr.update(visible=first_frame_visible), gr.update(visible=last_frame_visible), gr.update(visible=first_clip_visible)
                
                wan27_i2v_mode.change(
                    fn=update_i2v_visibility,
                    inputs=[wan27_i2v_mode],
                    outputs=[first_frame_group, last_frame_group, first_clip_group]
                )
                
                def process_wan27_i2v_request(mode, prompt, negative_prompt, first_frame, last_frame, first_clip, audio_file, resolution, duration):
                    first_frame_path = save_image_to_temp(first_frame)
                    last_frame_path = save_image_to_temp(last_frame)
                    first_clip_path = None
                    
                    if isinstance(first_clip, str):
                        first_clip_path = first_clip
                    elif hasattr(first_clip, 'name'):
                        first_clip_path = first_clip.name
                    
                    if mode == "first_frame":
                        if not first_frame_path:
                            return "❌ 请上传首帧图像。", None
                        result = generate_video_with_wan27_i2v(prompt, first_frame_path, None, None, audio_file, resolution, duration, negative_prompt)
                    elif mode == "first_last_frame":
                        if not first_frame_path:
                            return "❌ 请上传首帧图像。", None
                        if not last_frame_path:
                            return "❌ 请上传尾帧图像。", None
                        result = generate_video_with_wan27_i2v(prompt, first_frame_path, last_frame_path, None, audio_file, resolution, duration, negative_prompt)
                    elif mode == "video_continuation":
                        if not first_clip_path:
                            return "❌ 请上传首段视频。", None
                        result = generate_video_with_wan27_i2v(prompt, None, None, first_clip_path, audio_file, resolution, duration, negative_prompt)
                    else:
                        if not first_clip_path:
                            return "❌ 请上传首段视频。", None
                        if not last_frame_path:
                            return "❌ 请上传尾帧图像。", None
                        result = generate_video_with_wan27_i2v(prompt, None, last_frame_path, first_clip_path, audio_file, resolution, duration, negative_prompt)
                    
                    if "任务ID:" in result:
                        return result, None
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                wan27_i2v_gen_btn.click(
                    fn=process_wan27_i2v_request,
                    inputs=[wan27_i2v_mode, wan27_i2v_prompt, wan27_i2v_negative_prompt, wan27_i2v_first_frame, wan27_i2v_last_frame, wan27_i2v_first_clip, wan27_i2v_audio_file, wan27_i2v_resolution, wan27_i2v_duration],
                    outputs=[wan27_i2v_progress, wan27_i2v_output]
                )
            
            with gr.TabItem("参考生视频 (wan2.7)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 参考生视频")
                        gr.Markdown("上传参考图像/视频，生成保持角色形象一致性的视频")
                        
                        wan27_r2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...\n支持参考指代：图1、图2、视频1、视频2",
                            lines=5,
                            max_lines=8
                        )
                        
                        wan27_r2v_negative_prompt = gr.Textbox(
                            label="反向提示词",
                            placeholder="输入不希望出现在视频中的内容",
                            lines=2,
                            max_lines=4
                        )
                        
                        gr.Markdown("#### 参考图像（最多5张）")
                        with gr.Row():
                            wan27_r2v_ref_img1 = gr.Image(
                                label="参考图像1",
                                interactive=True
                            )
                            wan27_r2v_ref_img2 = gr.Image(
                                label="参考图像2",
                                interactive=True
                            )
                        with gr.Row():
                            wan27_r2v_ref_img3 = gr.Image(
                                label="参考图像3",
                                interactive=True
                            )
                            wan27_r2v_ref_img4 = gr.Image(
                                label="参考图像4",
                                interactive=True
                            )
                        wan27_r2v_ref_img5 = gr.Image(
                            label="参考图像5",
                            interactive=True
                        )
                        
                        gr.Markdown("#### 参考视频（可选）")
                        wan27_r2v_ref_video = gr.Video(
                            label="参考视频",
                            interactive=True
                        )
                        
                        wan27_r2v_audio_file = gr.Audio(
                            label="驱动音频（可选）",
                            type="filepath",
                            interactive=True
                        )
                        
                        with gr.Row():
                            wan27_r2v_resolution = gr.Dropdown(
                                label="分辨率",
                                choices=["720P", "1080P"],
                                value="720P",
                                interactive=True
                            )
                            
                            wan27_r2v_duration = gr.Slider(
                                label="时长（秒）",
                                minimum=2,
                                maximum=15,
                                step=1,
                                value=10
                            )
                        
                        wan27_r2v_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            wan27_r2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            wan27_r2v_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def process_wan27_r2v_request(prompt, negative_prompt, ref_img1, ref_img2, ref_img3, ref_img4, ref_img5, ref_video, audio_file, resolution, duration):
                    print(f"[DEBUG process_wan27_r2v_request] 开始处理请求")
                    print(f"[DEBUG process_wan27_r2v_request] prompt: {prompt}")
                    print(f"[DEBUG process_wan27_r2v_request] ref_img1类型: {type(ref_img1)}")
                    print(f"[DEBUG process_wan27_r2v_request] ref_img2类型: {type(ref_img2)}")
                    print(f"[DEBUG process_wan27_r2v_request] ref_img3类型: {type(ref_img3)}")
                    print(f"[DEBUG process_wan27_r2v_request] ref_img4类型: {type(ref_img4)}")
                    print(f"[DEBUG process_wan27_r2v_request] ref_img5类型: {type(ref_img5)}")
                    print(f"[DEBUG process_wan27_r2v_request] ref_video类型: {type(ref_video)}")
                    print(f"[DEBUG process_wan27_r2v_request] audio_file类型: {type(audio_file)}")
                    
                    ref_images = []
                    for idx, img in enumerate([ref_img1, ref_img2, ref_img3, ref_img4, ref_img5], 1):
                        print(f"[DEBUG] 处理参考图像{idx}...")
                        path = save_image_to_temp(img)
                        print(f"[DEBUG] 参考图像{idx}处理后路径: {path}")
                        if path:
                            ref_images.append(path)
                    
                    print(f"[DEBUG] 最终参考图像列表: {ref_images}")
                    
                    ref_videos = []
                    if ref_video:
                        print(f"[DEBUG] 处理参考视频: {ref_video}")
                        if isinstance(ref_video, str):
                            ref_videos.append(ref_video)
                            print(f"[DEBUG] 视频是字符串路径: {ref_video}")
                        elif hasattr(ref_video, 'name'):
                            ref_videos.append(ref_video.name)
                            print(f"[DEBUG] 视频有name属性: {ref_video.name}")
                        else:
                            print(f"[DEBUG] 视频类型未知: {type(ref_video)}")
                    
                    print(f"[DEBUG] 最终参考视频列表: {ref_videos}")
                    
                    if not ref_images and not ref_videos:
                        return "❌ 请至少上传一张参考图像或一个参考视频。", None
                    
                    result = generate_video_with_wan27_r2v(
                        prompt, 
                        reference_images=ref_images,
                        reference_videos=ref_videos,
                        audio_file=audio_file,
                        resolution=resolution,
                        duration=duration,
                        negative_prompt=negative_prompt
                    )
                    
                    if "任务ID:" in result:
                        return result, None
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频生成完成！", html_player
                            else:
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                wan27_r2v_gen_btn.click(
                    fn=process_wan27_r2v_request,
                    inputs=[wan27_r2v_prompt, wan27_r2v_negative_prompt, wan27_r2v_ref_img1, wan27_r2v_ref_img2, wan27_r2v_ref_img3, wan27_r2v_ref_img4, wan27_r2v_ref_img5, wan27_r2v_ref_video, wan27_r2v_audio_file, wan27_r2v_resolution, wan27_r2v_duration],
                    outputs=[wan27_r2v_progress, wan27_r2v_output]
                )
            
            with gr.TabItem("视频编辑 (wan2.7)"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 视频编辑")
                        gr.Markdown("上传视频，使用自然语言指令进行局部编辑或风格转换")
                        
                        wan27_ve_prompt = gr.Textbox(
                            label="编辑指令",
                            placeholder="请输入编辑指令...\n如：将整个画面转换为黏土风格\n或：将视频中女孩的衣服替换为图片中的衣服",
                            lines=5,
                            max_lines=8
                        )
                        
                        wan27_ve_negative_prompt = gr.Textbox(
                            label="反向提示词",
                            placeholder="输入不希望出现在视频中的内容",
                            lines=2,
                            max_lines=4
                        )
                        
                        gr.Markdown("#### 要编辑的视频")
                        wan27_ve_video = gr.Video(
                            label="原始视频",
                            interactive=True
                        )
                        
                        gr.Markdown("#### 参考图像（可选，用于局部替换）")
                        wan27_ve_ref_image = gr.Image(
                            label="参考图像",
                            interactive=True
                        )
                        
                        wan27_ve_resolution = gr.Dropdown(
                            label="分辨率",
                            choices=["720P", "1080P"],
                            value="720P",
                            interactive=True
                        )
                        
                        wan27_ve_gen_btn = gr.Button("开始编辑", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            wan27_ve_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 编辑结果")
                            wan27_ve_output = gr.HTML(
                                label="视频预览",
                                visible=True,
                                elem_id="video_preview_container"
                            )
                
                def process_wan27_ve_request(prompt, negative_prompt, video, ref_image, resolution):
                    video_path = None
                    if isinstance(video, str):
                        video_path = video
                    elif hasattr(video, 'name'):
                        video_path = video.name
                    
                    if not video_path:
                        return "❌ 请上传要编辑的视频文件。", None
                    
                    ref_image_path = save_image_to_temp(ref_image)
                    
                    result = generate_video_with_wan27_videoedit(
                        prompt,
                        video_path=video_path,
                        reference_image_path=ref_image_path,
                        resolution=resolution,
                        negative_prompt=negative_prompt
                    )
                    
                    if "任务ID:" in result:
                        return result, None
                    elif "视频URL:" in result:
                        lines = result.split('\n')
                        video_url = None
                        for line in lines:
                            if line.startswith("视频URL:"):
                                video_url = line.replace("视频URL:", "").strip()
                                break
                        
                        if video_url:
                            local_path = download_video_to_local(video_url)
                            if local_path and os.path.exists(local_path):
                                html_player = create_html_video_player(local_path)
                                return "✅ 视频编辑完成！", html_player
                            else:
                                html_player = create_html_video_player(video_url)
                                return f"⚠️ 视频下载到本地失败，但可以通过下面链接访问：\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                wan27_ve_gen_btn.click(
                    fn=process_wan27_ve_request,
                    inputs=[wan27_ve_prompt, wan27_ve_negative_prompt, wan27_ve_video, wan27_ve_ref_image, wan27_ve_resolution],
                    outputs=[wan27_ve_progress, wan27_ve_output]
                )
            
            with gr.TabItem("任务状态监控"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 任务状态监控")
                        gr.Markdown("实时查看视频生成任务的进度和状态")
                        
                        task_id_monitor = gr.Textbox(
                            label="任务ID",
                            placeholder="请输入要监控的任务ID...",
                            lines=1
                        )
                        
                        monitor_start_btn = gr.Button("开始监控", variant="primary")
                        monitor_stop_btn = gr.Button("停止监控", variant="secondary")
                    
                    with gr.Column():
                        monitor_result = gr.Textbox(
                            label="监控结果",
                            lines=10,
                            interactive=False
                        )
                
                def start_monitoring(task_id):
                    if not task_id or not task_id.strip():
                        return "❌ 请输入有效的任务ID。"
                    
                    return query_video_task(task_id)
                
                monitor_start_btn.click(
                    fn=start_monitoring,
                    inputs=task_id_monitor,
                    outputs=monitor_result
                )
                
                # 添加自动刷新功能
                refresh_btn = gr.Button("刷新状态", variant="secondary")
                
                refresh_btn.click(
                    fn=query_video_task,
                    inputs=task_id_monitor,
                    outputs=monitor_result
                )
                
                # 显示最近任务
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 最近的任务")
                        recent_tasks_btn = gr.Button("加载最近任务", variant="secondary")
                    
                    with gr.Column():
                        recent_tasks_display = gr.Dataframe(
                            headers=["任务ID", "状态", "提交时间", "模型"],
                            datatype=["str", "str", "str", "str"],
                            label="最近任务列表",
                            interactive=False
                        )
                
                def load_recent_tasks():
                    tasks = get_recent_tasks()
                    data = []
                    for task in tasks:
                        data.append([
                            task['task_id'],
                            task['status'],
                            task['submit_time'],
                            task['model']
                        ])
                    return data
                
                recent_tasks_btn.click(
                    fn=load_recent_tasks,
                    inputs=[],
                    outputs=recent_tasks_display
                )
        
        # 添加打开输出目录按钮
        open_output_dir_btn = gr.Button("打开输出目录", variant="secondary")
        open_output_dir_btn.click(
            fn=open_video_output_dir,
            inputs=[],
            outputs=[]
        )

    return qwen_video_interface


# 定义模块可用性标志
QWEN_VIDEO_GEN_AVAILABLE = True  # 根据实际需求设定，如果功能完整则为True
