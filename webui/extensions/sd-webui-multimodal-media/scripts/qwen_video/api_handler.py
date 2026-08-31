"""
Qwen视频生成API处理模块
负责处理与API相关的所有请求
"""

import os
import json
import requests
import time
import base64
import mimetypes
from modules import shared
import urllib.request
import subprocess
import platform
from typing import Dict, Any
from PIL import Image
import io


# API基础URL（中国站）
BASE_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis'


def set_api_key(api_key: str) -> str:
    """
    设置API Key到环境变量
    """
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
        return "API Key已设置成功！"
    else:
        return "请输入有效的API Key"


def get_task_result(task_id: str) -> Dict[Any, Any]:
    """
    获取任务结果
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return {"error": "未找到DASHSCOPE_API_KEY环境变量，请先设置API密钥"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # 使用正确的任务查询API端点
        query_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        response = requests.get(query_url, headers=headers)
        
        # 检查HTTP状态码
        if response.status_code == 404:
            return {"error": f"任务不存在: {task_id}，请确认任务ID是否正确"}
        elif response.status_code == 401:
            return {"error": "API密钥无效，请检查DASHSCOPE_API_KEY是否正确设置"}
        elif response.status_code == 429:
            return {"error": "请求过于频繁，请稍后再试"}
        elif response.status_code >= 400:
            return {"error": f"查询失败，HTTP状态码: {response.status_code}, 详情: {response.text}"}
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"查询任务失败: {str(e)}"}
    except Exception as e:
        return {"error": f"处理查询响应时出错: {str(e)}"}


def encode_file_to_base64(file_path: str) -> str:
    """
    将文件编码为Base64格式
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        # 默认使用常见图像格式
        mime_type = 'application/octet-stream'
    
    try:
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"文件Base64编码失败: {str(e)}")
        return ""


def process_image_transparency(file_path: str) -> str:
    """
    处理图像透明通道，将带透明通道的PNG转换为RGB模式
    """
    try:
        with Image.open(file_path) as img:
            img_format = img.format.upper() if img.format else 'JPEG'
            
            # 如果是PNG格式且有透明通道，则转换为RGB模式
            if img.mode in ('RGBA', 'LA', 'P') and img_format == 'PNG':
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 如果原图有alpha通道，合并到白色背景上
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为掩码
                else:
                    background.paste(img)
                
                # 保存转换后的图像
                temp_path = file_path.rsplit('.', 1)[0] + '_no_alpha.jpg'
                background.save(temp_path, format='JPEG')
                return temp_path
            else:
                # 如果没有透明通道，直接返回原文件
                return file_path
    except Exception as e:
        print(f"图像透明通道处理失败: {str(e)}")
        return file_path  # 如果处理失败，返回原文件


def validate_and_process_audio(file_path: str, max_duration: int = 30) -> str:
    """
    验证和处理音频文件以符合API要求
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return file_path

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path.lower())
        
        # 检查文件格式
        supported_formats = ['.mp3', '.wav', '.flac', '.aac', '.m4a']
        if ext not in supported_formats:
            print(f"警告: 音频格式 {ext} 可能不被支持，尝试继续处理...")
        
        # 检查音频时长和大小
        try:
            import librosa
            audio_data, sr = librosa.load(file_path, sr=None, duration=max_duration)
            duration = librosa.get_duration(y=audio_data, sr=sr)
            
            if duration > max_duration:
                print(f"音频时长 {duration}s 超过最大限制 {max_duration}s，将被截断")
                # 截取前max_duration秒的音频
                samples_per_second = len(audio_data) / duration
                target_samples = int(samples_per_second * max_duration)
                audio_data = audio_data[:target_samples]
                
                # 保存截断后的音频
                temp_path = file_path.rsplit('.', 1)[0] + '_truncated.wav'
                import soundfile as sf
                sf.write(temp_path, audio_data, sr)
                return temp_path
            else:
                return file_path
        except ImportError:
            # 如果没有安装librosa，跳过时长检查
            print("提示: 未安装librosa库，无法检查音频时长和大小")
            return file_path
        except Exception as e:
            print(f"音频处理失败: {str(e)}，使用原始文件")
            return file_path

    except Exception as e:
        print(f"音频验证失败: {str(e)}")
        return file_path  # 如果验证失败，返回原文件


def handle_file_input(file_path: str, filetype: str) -> dict:
    """
    处理文件输入：只使用Base64编码进行传输
    返回格式: {"success": bool, "url": str, "error": str}
    """
    if not file_path:
        return {"success": False, "url": "", "error": "文件路径为空"}

    # 验证是否为字符串路径
    if not isinstance(file_path, str):
        # 可能是gradio组件或其他对象，尝试提取路径
        if hasattr(file_path, 'name'):
            file_path = file_path.name
        elif str(file_path).startswith('<'):
            return {"success": False, "url": "", "error": f"无法处理的文件类型: {type(file_path)}"}
        else:
            file_path = str(file_path)

    # 验证本地文件是否存在
    if not os.path.exists(file_path):
        return {"success": False, "url": "", "error": f"文件不存在: {file_path}"}

    # 如果是图像文件，检查格式并处理透明通道
    if filetype == 'image':
        file_path = process_image_transparency(file_path)
    elif filetype == 'audio':
        # 如果是音频文件，进行预处理
        file_path = validate_and_process_audio(file_path, 30)  # 最大支持30秒

    # 对于所有本地文件，使用Base64编码
    encoded_data = encode_file_to_base64(file_path)
    if encoded_data:
        return {"success": True, "url": encoded_data, "error": ""}
    else:
        return {"success": False, "url": "", "error": "文件Base64编码失败"}