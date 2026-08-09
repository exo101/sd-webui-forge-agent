import gradio as gr
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from modules import shared

# 获取 webui 根目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent

# 添加自定义CSS样式来控制视频组件尺寸
custom_css = """
.video-frame-extractor-video video {
    max-height: 300px !important;
    object-fit: contain !important;
}

.video-frame-extractor-video .file-preview {
    max-height: 300px !important;
}
"""

def create_video_frame_extractor():
    """创建视频分帧提取功能组件"""
    
    # 注入自定义CSS样式
    gr.Markdown(f"<style>{custom_css}</style>", visible=False)
    
    def extract_video_frames(video, quality, mode):
        """提取视频关键帧并保存为图片"""
        if video is None:
            return [], []
        
        # 创建保存目录 - 使用 webui/output/video-frames
        save_dir = os.path.join(WEBUI_ROOT, "output", "video-frames")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join(save_dir, f"video_{timestamp}")
        os.makedirs(video_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"[Video Frame Extractor] 视频信息:")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - FPS: {fps}")
        print(f"  - 时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")
        
        # 根据视频时长自动计算关键帧数量
        # 新规则：每2秒提取1帧，最少10帧，最多60帧（更适合长视频）
        auto_num_frames = max(10, min(60, int(duration / 2)))
        print(f"  - 计划提取帧数: {auto_num_frames} (每2秒1帧)")
        
        # 计算要提取的帧位置
        frames_to_extract = []
        if mode == "uniform":
            for i in range(auto_num_frames):
                frame_pos = int(i * total_frames / auto_num_frames)
                frames_to_extract.append(frame_pos)
            print(f"  - 模式: 均匀分布")
        elif mode == "interval":
            interval = int(total_frames / auto_num_frames)
            frames_to_extract = [i * interval for i in range(auto_num_frames)]
            print(f"  - 模式: 固定间隔 (间隔={interval}帧)")
        elif mode == "change_detection":
            print(f"  - 模式: 变化检测")
            prev_frame = None
            change_frames = []
            frame_count = 0
            while True and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    non_zero = np.count_nonzero(diff)
                    if non_zero > 1000:  # 阈值可根据需求调整
                        change_frames.append(frame_count)
                
                prev_frame = frame.copy()
                frame_count += 1
            
            print(f"  - 检测到变化帧数: {len(change_frames)}")
            # 取变化帧的前 auto_num_frames 帧作为关键帧
            frames_to_extract = change_frames[:auto_num_frames]
        
        print(f"  - 实际待提取帧数: {len(frames_to_extract)}")
        
        # 重置视频读取器
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 提取并保存帧
        extracted_images = []
        preview_images = []
        saved_count = 0
        failed_count = 0
        for idx, frame_pos in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                # 构建文件路径
                filename = os.path.join(video_dir, f"frame_{saved_count:04d}.jpg")
                
                # 保存帧（OpenCV imwrite 需要 BGR 格式，frame 本身就是 BGR）
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                # 添加到结果列表
                extracted_images.append(filename)
                
                # 转换为 RGB 用于预览显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 缩小用于预览
                height, width = frame_rgb.shape[:2]
                preview_size = (800, int(800 * height / width)) if width > height else (int(800 * width / height), 800)
                preview = cv2.resize(frame_rgb, preview_size)
                preview_images.append(preview)
                
                saved_count += 1
                if (idx + 1) % 10 == 0 or idx == len(frames_to_extract) - 1:
                    print(f"  - 进度: {idx + 1}/{len(frames_to_extract)}, 已保存: {saved_count}")
            else:
                failed_count += 1
                print(f"  - ⚠️ 警告: 无法读取第 {idx + 1} 帧 (位置={frame_pos})")
        
        cap.release()
        
        print(f"[Video Frame Extractor] 提取完成!")
        print(f"  - 成功: {saved_count} 帧")
        print(f"  - 失败: {failed_count} 帧")
        print(f"  - 保存目录: {video_dir}")
        
        return extracted_images, preview_images
    
    # 创建左右分栏布局，参数在左，结果在右
    with gr.Row():
        with gr.Column(scale=1):
            # 创建视频输入组件
            video_input = gr.Video(
                label="上传视频", 
                height=300, 
                elem_classes=["video-frame-extractor-video"]  # 添加自定义CSS类
            )
            
            # 创建参数输入区域（已移除关键帧数量，改为自动计算）
            frame_quality = gr.Slider(label="帧质量", minimum=1, maximum=100, value=85, step=1)
            
            # 创建模式选择
            frame_mode = gr.Radio(
                label="提取模式",
                choices=[
                    ("均匀分布（根据视频时长自动计算）", "uniform"), 
                    ("固定间隔（根据视频时长自动计算）", "interval"), 
                    ("变化检测（智能识别场景切换）", "change_detection")
                ],
                value="uniform"
            )
            
            # 添加提示信息
            info_text = gr.Markdown(
                """
                💡 **提示**：关键帧数量将根据视频时长自动计算
                - 规则：**每2秒提取1帧**，最少10帧，最多60帧
                - 例如：10秒视频 → 提取10帧；2分钟视频 → 提取60帧
                - **推荐使用"均匀分布"模式**以获得稳定的提取效果
                """
            )
            
            # 添加提取按钮
            extract_btn = gr.Button("提取关键帧", variant="primary")
            
            # 添加打开输出目录按钮
            open_output_dir_btn = gr.Button("打开输出目录")
            
            # 添加文件输出组件，用于展示/下载提取的帧
            frame_output = gr.File(label="提取的帧文件", file_count="multiple")
            
        with gr.Column(scale=1):
            # 创建预览区域
            frame_preview = gr.Gallery(label="帧预览", columns=5, height=400, visible=True)
    
    # 绑定提取事件
    extract_btn.click(
        fn=extract_video_frames,
        inputs=[video_input, frame_quality, frame_mode],
        outputs=[frame_output, frame_preview]
    )
    
    def open_video_frames_output_dir():
        """打开视频帧输出目录"""
        output_dir = os.path.join(WEBUI_ROOT, "output", "video-frames")
        os.makedirs(output_dir, exist_ok=True)
        import subprocess
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            print(f"打开目录失败: {e}")
    
    open_output_dir_btn.click(fn=open_video_frames_output_dir, inputs=[], outputs=[])
    
    # 返回所有创建的组件和函数
    return {
        "video_input": video_input,
        "frame_quality": frame_quality,
        "frame_mode": frame_mode,
        "frame_preview": frame_preview,
        "frame_output": frame_output,
        "extract_video_frames": extract_video_frames
    }