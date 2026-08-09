"""
Kling视频生成模型模块
负责处理各种Kling视频生成模型的调用
"""

import os
import json
import requests
import time
import base64
import mimetypes
from pathlib import Path
from modules import shared
from .api_handler import handle_file_input, KLING_BASE_URL

# 获取 webui 根目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent.parent


def generate_video_with_kling_t2v(prompt: str, model: str, mode: str, aspect_ratio: str, duration: int, 
                                   negative_prompt: str = "", callback_url: str = "") -> str:
    """
    使用Kling模型生成视频（文生视频）
    模型: kling-v3-video-generation, kling-v3-omni-video-generation
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": {
            "prompt": prompt
        },
        "parameters": {
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    if callback_url and callback_url.strip():
        payload["parameters"]["callback_url"] = callback_url

    try:
        response = requests.post(KLING_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "kling-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": model,
                "mode": mode,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "submit_time": time.time()
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_kling_t2v.json"
            task_path = os.path.join(save_dir, task_filename)
            
            with open(task_path, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            
            return result_text
        else:
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                return f"❌ API调用失败: {error_msg}\n请检查API密钥和模型权限"
            else:
                return f"❌ API响应中未找到任务ID: {result}"
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return f"❌ 请求错误 (400): 请检查输入参数是否正确"
        else:
            return f"❌ HTTP错误: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ 请求失败: {str(e)}"
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_kling_i2v(prompt: str, first_frame_path: str, last_frame_path: str = None,
                                   model: str = "kling/kling-v3-video-generation", mode: str = "pro",
                                   duration: int = 5, negative_prompt: str = "") -> str:
    """
    使用Kling模型生成视频（图生视频 - 首帧或首尾帧）
    模型: kling/kling-v3-video-generation, kling/kling-v3-omni-video-generation
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    media = []

    if first_frame_path:
        image_result = handle_file_input(first_frame_path, 'image')
        if not image_result["success"]:
            return f"❌ 首帧图像文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "first_frame",
            "url": image_result["url"]
        })
    else:
        return "❌ 请上传首帧图像。"

    if last_frame_path:
        last_image_result = handle_file_input(last_frame_path, 'image')
        if not last_image_result["success"]:
            return f"❌ 尾帧图像文件处理失败: {last_image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "last_frame",
            "url": last_image_result["url"]
        })

    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "media": media
        },
        "parameters": {
            "mode": mode,
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    try:
        response = requests.post(KLING_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "kling-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "media": media,
                "model": model,
                "mode": mode,
                "duration": duration,
                "submit_time": time.time()
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_kling_i2v.json"
            task_path = os.path.join(save_dir, task_filename)
            
            with open(task_path, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            
            return result_text
        else:
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                return f"❌ API调用失败: {error_msg}\n请检查API密钥和模型权限"
            else:
                return f"❌ API响应中未找到任务ID: {result}"
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return f"❌ 请求错误 (400): 请检查输入参数是否正确"
        else:
            return f"❌ HTTP错误: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ 请求失败: {str(e)}"
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_kling_omni(prompt: str, first_frame_path: str = None, last_frame_path: str = None,
                                   reference_images: list = None, model: str = "kling/kling-v3-omni-video-generation",
                                   mode: str = "pro", aspect_ratio: str = "16:9", duration: int = 5,
                                   negative_prompt: str = "") -> str:
    """
    使用Kling Omni模型生成视频（支持更丰富的输入组合）
    模型: kling/kling-v3-omni-video-generation
    支持：首帧、首尾帧、参考图片等多种组合
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    media = []

    if first_frame_path:
        image_result = handle_file_input(first_frame_path, 'image')
        if not image_result["success"]:
            return f"❌ 首帧图像文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "first_frame",
            "url": image_result["url"]
        })

    if last_frame_path:
        last_image_result = handle_file_input(last_frame_path, 'image')
        if not last_image_result["success"]:
            return f"❌ 尾帧图像文件处理失败: {last_image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "last_frame",
            "url": last_image_result["url"]
        })

    if reference_images:
        for ref_img in reference_images:
            if ref_img:
                ref_result = handle_file_input(ref_img, 'image')
                if ref_result["success"]:
                    media.append({
                        "type": "reference_image",
                        "url": ref_result["url"]
                    })

    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "media": media if media else None
        },
        "parameters": {
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    try:
        response = requests.post(KLING_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "kling-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "media": media,
                "model": model,
                "mode": mode,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "submit_time": time.time()
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_kling_omni.json"
            task_path = os.path.join(save_dir, task_filename)
            
            with open(task_path, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            
            return result_text
        else:
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                return f"❌ API调用失败: {error_msg}\n请检查API密钥和模型权限"
            else:
                return f"❌ API响应中未找到任务ID: {result}"
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return f"❌ 请求错误 (400): 请检查输入参数是否正确"
        else:
            return f"❌ HTTP错误: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ 请求失败: {str(e)}"
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"
