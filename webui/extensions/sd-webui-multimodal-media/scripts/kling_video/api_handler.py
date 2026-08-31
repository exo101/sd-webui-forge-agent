"""
Kling视频生成API处理模块
负责处理与Kling API相关的所有请求
"""

import os
import json
import requests
import time
import base64
import mimetypes
from pathlib import Path
from PIL import Image
import io


# Kling API基础URL（阿里云百炼）
KLING_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis'


def set_kling_api_key(api_key: str) -> str:
    """
    设置Kling API Key到环境变量
    """
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
        return "API Key已设置成功！"
    else:
        return "请输入有效的API Key"


def get_kling_task_result(task_id: str) -> dict:
    """
    获取Kling任务结果
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return {"error": "未找到DASHSCOPE_API_KEY环境变量，请先设置API密钥"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        query_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        response = requests.get(query_url, headers=headers)
        
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
            
            if img.mode in ('RGBA', 'LA', 'P') and img_format == 'PNG':
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                
                temp_path = file_path.rsplit('.', 1)[0] + '_no_alpha.jpg'
                background.save(temp_path, format='JPEG')
                return temp_path
            else:
                return file_path
    except Exception as e:
        print(f"图像透明通道处理失败: {str(e)}")
        return file_path


def handle_file_input(file_path: str, filetype: str) -> dict:
    """
    处理文件输入：只使用Base64编码进行传输
    返回格式: {"success": bool, "url": str, "error": str}
    """
    if not file_path:
        return {"success": False, "url": "", "error": "文件路径为空"}

    if not isinstance(file_path, str):
        if hasattr(file_path, 'name'):
            file_path = file_path.name
        elif str(file_path).startswith('<'):
            return {"success": False, "url": "", "error": f"无法处理的文件类型: {type(file_path)}"}
        else:
            file_path = str(file_path)

    if not os.path.exists(file_path):
        return {"success": False, "url": "", "error": f"文件不存在: {file_path}"}

    if filetype == 'image':
        file_path = process_image_transparency(file_path)
    elif filetype == 'video':
        pass  # 视频文件直接处理

    encoded_data = encode_file_to_base64(file_path)
    if encoded_data:
        return {"success": True, "url": encoded_data, "error": ""}
    else:
        return {"success": False, "url": "", "error": "文件Base64编码失败"}
