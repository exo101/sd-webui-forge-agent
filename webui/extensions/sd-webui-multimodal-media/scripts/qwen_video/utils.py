"""
工具函数模块
包含文件处理、目录操作等通用功能
"""

import os
import tempfile
import requests
import base64
import mimetypes
from pathlib import Path
from modules import shared
import subprocess
import time

# 获取 webui 根目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent.parent


def handle_file_input(file_path: str, file_type: str):
    """
    处理文件输入，支持本地文件和URL
    """
    try:
        if file_path.startswith(('http://', 'https://')):
            # 如果是URL，直接返回
            return {"success": True, "url": file_path, "type": "url"}
        elif file_path.startswith('data:'):
            # 如果已经是Base64编码的数据，直接返回
            return {"success": True, "url": file_path, "type": "base64"}
        else:
            # 本地文件，转换为Base64
            if not os.path.exists(file_path):
                return {"success": False, "error": f"文件不存在: {file_path}"}
            
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or not mime_type.startswith((f"{file_type}/", "image/", "audio/", "video/")):
                return {"success": False, "error": f"不支持或无法识别的{file_type}格式"}
            
            with open(file_path, "rb") as file:
                encoded_string = base64.b64encode(file.read()).decode('utf-8')
            
            # 根据文件类型返回对应的MIME类型
            if file_type == 'image':
                return {"success": True, "url": f"data:{mime_type};base64,{encoded_string}", "type": "base64"}
            elif file_type == 'audio':
                return {"success": True, "url": f"data:{mime_type};base64,{encoded_string}", "type": "base64"}
            else:
                return {"success": True, "url": f"data:{mime_type};base64,{encoded_string}", "type": "base64"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def open_video_output_dir():
    """
    打开视频输出目录
    """
    save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"视频输出目录路径: {save_dir}")  # 调试信息
    
    try:
        if os.name == 'nt':  # Windows
            os.startfile(save_dir)
        elif os.name == 'posix':  # Linux/Mac
            subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', save_dir])
    except Exception as e:
        print(f"打开目录失败: {str(e)}")


def create_html_video_player(video_url: str):
    """
    创建HTML视频播放器
    """
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
    """
    将远程视频下载到本地并返回本地路径
    """
    try:
        # 确保输出目录存在
        save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"视频保存目录: {save_dir}")  # 调试信息
        
        # 生成文件名
        if not filename:
            # 从URL提取文件名或生成唯一文件名
            filename = os.path.basename(video_url).split('?')[0]  # 移除URL参数
            if not filename or '.' not in filename:
                filename = f"video_{int(time.time())}.mp4"
        
        local_path = os.path.join(save_dir, filename)
        print(f"视频将保存到: {local_path}")  # 调试信息
        
        # 下载视频
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        print(f"HTTP响应状态码: {response.status_code}")  # 调试信息
        print(f"内容长度: {response.headers.get('content-length', 'Unknown')}")  # 调试信息
        
        # 以二进制写入模式打开文件
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉keep-alive的新块
                    f.write(chunk)
        
        # 验证下载的文件是否有效
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"视频下载完成: {local_path}")  # 调试信息
            print(f"文件大小: {os.path.getsize(local_path)} 字节")  # 调试信息
            return local_path
        else:
            print(f"下载的文件无效或为空")  # 调试信息
            return None
    except Exception as e:
        print(f"下载视频失败: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return None