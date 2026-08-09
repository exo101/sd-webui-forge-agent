"""
Kling视频生成主UI模块
整合所有子模块，提供统一的UI界面
"""

import gradio as gr
import os
import tempfile
from PIL import Image
from pathlib import Path

from .api_handler import set_kling_api_key, get_kling_task_result
from .models import (
    generate_video_with_kling_t2v,
    generate_video_with_kling_i2v,
    generate_video_with_kling_omni
)

# 获取 webui 根目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent.parent


def save_image_to_temp(im):
    """将PIL图像保存到临时文件并返回路径"""
    if im is None:
        return None
    if isinstance(im, str):
        return im
    if hasattr(im, 'name'):
        return im.name
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_img_{os.urandom(8).hex()}.png")
    
    try:
        if isinstance(im, Image.Image):
            im.save(temp_path)
            return temp_path
        elif isinstance(im, dict) and 'image' in im:
            Image.fromarray(im['image']).save(temp_path)
            return temp_path
        else:
            return None
    except Exception as e:
        print(f"保存临时图像失败: {e}")
        return None


def open_kling_output_dir():
    """打开Kling视频输出目录"""
    save_dir = os.path.join(WEBUI_ROOT, "output", "kling-video")
    os.makedirs(save_dir, exist_ok=True)
    try:
        if os.name == 'nt':
            os.startfile(save_dir)
    except Exception as e:
        print(f"打开目录失败: {str(e)}")


def create_html_video_player(video_url: str):
    """创建HTML视频播放器"""
    return f'''
    <div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
        <video width="100%" controls style="max-width: 800px; height: auto;">
            <source src="{video_url}" type="video/mp4">
            您的浏览器不支持视频播放。
        </video>
    </div>
    <div style="text-align: center; margin-top: 10px;">
        <a href="{video_url}" target="_blank" download>💾 下载视频</a>
    </div>
    '''


def download_video_to_local(video_url: str, filename: str = None):
    """将远程视频下载到本地并返回本地路径"""
    import requests
    import time
    try:
        save_dir = os.path.join(WEBUI_ROOT, "output", "kling-video")
        os.makedirs(save_dir, exist_ok=True)
        
        if not filename:
            filename = os.path.basename(video_url).split('?')[0]
            if not filename or '.' not in filename:
                filename = f"video_{int(time.time())}.mp4"
        
        local_path = os.path.join(save_dir, filename)
        
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        else:
            return None
    except Exception as e:
        print(f"下载视频失败: {str(e)}")
        return None


def get_recent_kling_tasks():
    """获取最近的可灵任务"""
    import json
    save_dir = os.path.join(WEBUI_ROOT, "output", "kling-video")
    tasks = []
    
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            if filename.endswith('.json') and filename.startswith('task_'):
                try:
                    task_path = os.path.join(save_dir, filename)
                    with open(task_path, 'r', encoding='utf-8') as f:
                        task = json.load(f)
                        tasks.append(task)
                except Exception:
                    pass
    
    tasks.sort(key=lambda x: x.get('submit_time', 0), reverse=True)
    return tasks[:20]


def query_kling_task(task_id: str) -> str:
    """查询Kling任务状态"""
    if not task_id or not task_id.strip():
        return "❌ 请输入有效的任务ID。"
    
    result = get_kling_task_result(task_id)
    
    if "error" in result:
        return f"❌ {result['error']}"
    
    try:
        task_status = result.get("output", {}).get("task_status", "UNKNOWN")
        video_url = result.get("output", {}).get("video_url", None)
        
        status_text = f"任务状态: {task_status}\n"
        
        if video_url:
            status_text += f"视频URL: {video_url}\n"
            local_path = download_video_to_local(video_url)
            if local_path:
                html_player = create_html_video_player(local_path)
                return f"{status_text}\n✅ 视频已下载到本地！\n{html_player}"
            else:
                status_text += "\n⚠️ 视频下载失败，请点击上方链接查看"
                return status_text
        
        return status_text
        
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}\n完整响应: {result}"


def create_kling_video_gen_ui():
    """创建Kling视频生成UI界面"""
    with gr.Blocks() as kling_video_interface:
        gr.Markdown("# Kling 可灵视频生成 API")
        gr.Markdown("使用阿里云百炼平台的可灵AI模型进行视频生成")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="请输入您的阿里云百炼API Key",
                    info="输入API Key后点击下方按钮设置"
                )
                set_api_key_btn = gr.Button("设置API Key", variant="secondary")
                api_key_status = gr.Textbox(label="状态", interactive=False)
                
                set_api_key_btn.click(
                    fn=set_kling_api_key,
                    inputs=api_key_input,
                    outputs=api_key_status
                )
        
        with gr.Tabs():
            with gr.TabItem("文生视频"):
                with gr.Row():
                    with gr.Column():
                        kling_t2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...",
                            lines=4,
                            max_lines=6
                        )
                        
                        kling_t2v_negative = gr.Textbox(
                            label="反向提示词",
                            placeholder="输入不希望出现在视频中的内容",
                            lines=2
                        )
                        
                        with gr.Row():
                            kling_t2v_model = gr.Dropdown(
                                label="模型",
                                choices=[
                                    ("Kling V3 Omni (推荐)", "kling/kling-v3-omni-video-generation"),
                                    ("Kling V3", "kling/kling-v3-video-generation")
                                ],
                                value="kling/kling-v3-omni-video-generation"
                            )
                            
                            kling_t2v_mode = gr.Dropdown(
                                label="模式",
                                choices=[
                                    ("专家模式(高品质)", "pro"),
                                    ("标准模式", "std"),
                                    ("4K模式", "4k")
                                ],
                                value="pro"
                            )
                        
                        with gr.Row():
                            kling_t2v_ratio = gr.Dropdown(
                                label="画面比例",
                                choices=["16:9", "9:16", "1:1"],
                                value="16:9"
                            )
                            
                            kling_t2v_duration = gr.Dropdown(
                                label="时长",
                                choices=["5秒", "10秒"],
                                value="5秒"
                            )
                        
                        kling_t2v_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            kling_t2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            kling_t2v_output = gr.HTML(
                                label="视频预览",
                                visible=True
                            )
                
                def process_kling_t2v(model, mode, ratio, duration, prompt, negative):
                    duration_int = 5 if duration == "5秒" else 10
                    result = generate_video_with_kling_t2v(
                        prompt, model, mode, ratio, duration_int, negative
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
                                return f"⚠️ 视频下载失败\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                kling_t2v_gen_btn.click(
                    fn=process_kling_t2v,
                    inputs=[kling_t2v_model, kling_t2v_mode, kling_t2v_ratio, kling_t2v_duration, kling_t2v_prompt, kling_t2v_negative],
                    outputs=[kling_t2v_progress, kling_t2v_output]
                )
            
            with gr.TabItem("图生视频"):
                with gr.Row():
                    with gr.Column():
                        kling_i2v_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...",
                            lines=4,
                            max_lines=6
                        )
                        
                        kling_i2v_negative = gr.Textbox(
                            label="反向提示词",
                            placeholder="输入不希望出现在视频中的内容",
                            lines=2
                        )
                        
                        kling_i2v_mode_select = gr.Dropdown(
                            label="生成模式",
                            choices=[
                                ("首帧生视频", "first_frame"),
                                ("首尾帧生视频", "first_last_frame")
                            ],
                            value="first_frame"
                        )
                        
                        with gr.Group(visible=True) as kling_i2v_first_group:
                            kling_i2v_first_frame = gr.Image(
                                label="首帧图像"
                            )
                        
                        with gr.Group(visible=False) as kling_i2v_last_group:
                            kling_i2v_last_frame = gr.Image(
                                label="尾帧图像（首尾帧模式时使用）"
                            )
                        
                        with gr.Row():
                            kling_i2v_model = gr.Dropdown(
                                label="模型",
                                choices=[
                                    ("Kling V3 Omni (推荐)", "kling/kling-v3-omni-video-generation"),
                                    ("Kling V3", "kling/kling-v3-video-generation")
                                ],
                                value="kling/kling-v3-omni-video-generation"
                            )
                            
                            kling_i2v_mode = gr.Dropdown(
                                label="模式",
                                choices=[
                                    ("专家模式(高品质)", "pro"),
                                    ("标准模式", "std"),
                                    ("4K模式", "4k")
                                ],
                                value="pro"
                            )
                        
                        kling_i2v_duration = gr.Dropdown(
                            label="时长",
                            choices=["5秒", "10秒"],
                            value="5秒"
                        )
                        
                        kling_i2v_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            kling_i2v_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            kling_i2v_output = gr.HTML(
                                label="视频预览",
                                visible=True
                            )
                
                def update_kling_i2v_visibility(mode):
                    first_visible = mode in ["first_frame", "first_last_frame"]
                    last_visible = mode in ["first_last_frame"]
                    return gr.update(visible=first_visible), gr.update(visible=last_visible)
                
                kling_i2v_mode_select.change(
                    fn=update_kling_i2v_visibility,
                    inputs=[kling_i2v_mode_select],
                    outputs=[kling_i2v_first_group, kling_i2v_last_group]
                )
                
                def process_kling_i2v(mode, model, mode_quality, duration, prompt, negative, first_frame, last_frame):
                    duration_int = 5 if duration == "5秒" else 10
                    first_frame_path = save_image_to_temp(first_frame)
                    last_frame_path = save_image_to_temp(last_frame)
                    
                    if mode == "first_frame":
                        if not first_frame_path:
                            return "❌ 请上传首帧图像。", None
                        result = generate_video_with_kling_i2v(
                            prompt, first_frame_path, None, model, mode_quality, duration_int, negative
                        )
                    else:
                        if not first_frame_path:
                            return "❌ 请上传首帧图像。", None
                        if not last_frame_path:
                            return "❌ 请上传尾帧图像。", None
                        result = generate_video_with_kling_i2v(
                            prompt, first_frame_path, last_frame_path, model, mode_quality, duration_int, negative
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
                                return f"⚠️ 视频下载失败\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                kling_i2v_gen_btn.click(
                    fn=process_kling_i2v,
                    inputs=[kling_i2v_mode_select, kling_i2v_model, kling_i2v_mode, kling_i2v_duration, kling_i2v_prompt, kling_i2v_negative, kling_i2v_first_frame, kling_i2v_last_frame],
                    outputs=[kling_i2v_progress, kling_i2v_output]
                )
            
            with gr.TabItem("Omni视频生成"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Omni模型 - 多种生成方式组合")
                        
                        kling_omni_prompt = gr.Textbox(
                            label="提示词",
                            placeholder="请输入视频描述...可引用图片：<<<image_1>>>",
                            lines=4,
                            max_lines=6
                        )
                        
                        kling_omni_negative = gr.Textbox(
                            label="反向提示词",
                            lines=2
                        )
                        
                        kling_omni_first_frame = gr.Image(
                            label="首帧图像（可选）"
                        )
                        
                        kling_omni_last_frame = gr.Image(
                            label="尾帧图像（可选）"
                        )
                        
                        with gr.Row():
                            kling_omni_model = gr.Dropdown(
                                label="模型",
                                choices=[
                                    ("Kling V3 Omni (推荐)", "kling/kling-v3-omni-video-generation")
                                ],
                                value="kling/kling-v3-omni-video-generation"
                            )
                            
                            kling_omni_mode = gr.Dropdown(
                                label="模式",
                                choices=[
                                    ("专家模式(高品质)", "pro"),
                                    ("标准模式", "std"),
                                    ("4K模式", "4k")
                                ],
                                value="pro"
                            )
                        
                        with gr.Row():
                            kling_omni_ratio = gr.Dropdown(
                                label="画面比例",
                                choices=["16:9", "9:16", "1:1"],
                                value="16:9"
                            )
                            
                            kling_omni_duration = gr.Dropdown(
                                label="时长",
                                choices=["5秒", "10秒"],
                                value="5秒"
                            )
                        
                        kling_omni_gen_btn = gr.Button("生成视频", variant="primary")
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 任务进度")
                            kling_omni_progress = gr.Textbox(
                                label="进度信息",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 生成结果")
                            kling_omni_output = gr.HTML(
                                label="视频预览",
                                visible=True
                            )
                
                def process_kling_omni(model, mode, ratio, duration, prompt, negative, first_frame, last_frame):
                    duration_int = 5 if duration == "5秒" else 10
                    first_frame_path = save_image_to_temp(first_frame)
                    last_frame_path = save_image_to_temp(last_frame)
                    
                    result = generate_video_with_kling_omni(
                        prompt, first_frame_path, last_frame_path, None, model, mode, ratio, duration_int, negative
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
                                return f"⚠️ 视频下载失败\n{result}", html_player
                        else:
                            return result, None
                    else:
                        return result, None
                
                kling_omni_gen_btn.click(
                    fn=process_kling_omni,
                    inputs=[kling_omni_model, kling_omni_mode, kling_omni_ratio, kling_omni_duration, kling_omni_prompt, kling_omni_negative, kling_omni_first_frame, kling_omni_last_frame],
                    outputs=[kling_omni_progress, kling_omni_output]
                )
            
            with gr.TabItem("任务状态监控"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 任务状态监控")
                        task_id_monitor = gr.Textbox(
                            label="任务ID",
                            placeholder="请输入要监控的任务ID...",
                            lines=1
                        )
                        monitor_btn = gr.Button("查询状态", variant="primary")
                        refresh_btn = gr.Button("刷新", variant="secondary")
                    
                    with gr.Column():
                        monitor_result = gr.Textbox(
                            label="监控结果",
                            lines=10,
                            interactive=False
                        )
                
                monitor_btn.click(
                    fn=query_kling_task,
                    inputs=task_id_monitor,
                    outputs=monitor_result
                )
                
                refresh_btn.click(
                    fn=query_kling_task,
                    inputs=task_id_monitor,
                    outputs=monitor_result
                )
                
                with gr.Row():
                    recent_tasks_btn = gr.Button("加载最近任务", variant="secondary")
                    recent_tasks_display = gr.Dataframe(
                        headers=["任务ID", "模型", "提交时间"],
                        datatype=["str", "str", "str"],
                        label="最近任务列表",
                        interactive=False
                    )
                
                def load_recent_tasks():
                    tasks = get_recent_kling_tasks()
                    data = []
                    for task in tasks:
                        from datetime import datetime
                        submit_time = datetime.fromtimestamp(task.get('submit_time', 0)).strftime('%Y-%m-%d %H:%M:%S')
                        data.append([
                            task.get('task_id', ''),
                            task.get('model', ''),
                            submit_time
                        ])
                    return data
                
                recent_tasks_btn.click(
                    fn=load_recent_tasks,
                    inputs=[],
                    outputs=[recent_tasks_display]
                )
        
        open_output_dir_btn = gr.Button("打开输出目录", variant="secondary")
        open_output_dir_btn.click(
            fn=open_kling_output_dir,
            inputs=[],
            outputs=[]
        )

    return kling_video_interface


KLING_VIDEO_GEN_AVAILABLE = True
