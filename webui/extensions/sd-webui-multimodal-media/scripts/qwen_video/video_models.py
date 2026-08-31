"""
Qwen视频生成模型模块
负责处理各种视频生成模型的调用
"""

import os
import json
import requests
import time
import base64
import mimetypes
from pathlib import Path
from modules import shared
import urllib.request
from typing import Dict, Any
from .api_handler import handle_file_input, BASE_URL

# 获取 webui 根目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent.parent


def generate_video_with_wan26(prompt: str, image_path: str, audio_file, resolution: str, duration: int, audio_enabled: bool, shot_type: str) -> str:
    """
    使用wan2.6-i2v模型生成视频（通过HTTP API）
    注意：根据最新信息，wan2.6-i2v暂不支持SDK调用，使用HTTP API方式
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    headers = {
        "X-DashScope-Async": "enable",  # 启用异步处理
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 处理图像文件（只使用Base64编码）
    image_result = handle_file_input(image_path, 'image')
    if not image_result["success"]:
        return f"❌ 图像文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
    
    image_url = image_result["url"]
    
    # 处理音频文件（只使用Base64编码）（如果提供了的话）
    audio_url = ""
    if audio_file is not None:
        audio_file_path = None
        if hasattr(audio_file, 'name'):  # 如果是Audio组件
            audio_file_path = audio_file.name
        elif isinstance(audio_file, str) and audio_file:  # 如果是路径字符串
            audio_file_path = audio_file
        
        if audio_file_path:
            audio_result = handle_file_input(audio_file_path, 'audio')
            if not audio_result["success"]:
                return f"❌ 音频文件处理失败: {audio_result['error']}\n请检查音频文件是否存在。"
            
            audio_url = audio_result["url"]

    # 构建API请求体，使用官方文档中的参数名称
    payload = {
        "model": "wan2.6-i2v",  # 根据文档使用正确的模型名
        "input": {
            "prompt": prompt,
            "img_url": image_url  # 根据文档使用img_url而不是image_url
        },
        "parameters": {
            "resolution": resolution,
            "prompt_extend": True,  # 根据文档添加此参数
            "watermark": False,     # 根据文档添加此参数
            "duration": duration,   # 根据官方文档添加此参数
            "audio": audio_enabled,
            "shot_type": shot_type
        }
    }
    
    # 如果提供了音频URL，则添加到输入中
    if audio_url and audio_url.strip():
        payload["input"]["audio_url"] = audio_url

    try:
        # 发送API请求
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            # 根据规范，使用task_id构建标准查询URL
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            # 返回任务ID，用户可以稍后查询结果
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
            result_text += "注意：wan2.6-i2v暂不支持SDK调用，使用HTTP API方式"
            
            # 保存任务信息到本地
            save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "image_url": image_url,  # 保持记录原始image_url
                "audio_url": audio_url,
                "resolution": resolution,
                "duration": duration,
                "audio_enabled": audio_enabled,
                "shot_type": shot_type,
                "submit_time": time.time()
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}.json"
            task_path = os.path.join(save_dir, task_filename)
            
            with open(task_path, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            
            return result_text
        else:
            # 检查错误信息
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                return f"❌ API调用失败: {error_msg}\n请检查API密钥和模型权限"
            else:
                return f"❌ API响应中未找到任务ID: {result}"
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return f"❌ 请求错误 (400): 请检查输入参数是否正确，特别是图像和音频文件是否已成功上传到服务器"
        else:
            return f"❌ HTTP错误: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ 请求失败: {str(e)}"
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_wan25_i2v(prompt: str, image_path: str, audio_file, resolution: str, duration: int, audio_enabled: bool) -> str:
    """
    使用wan2.5-i2v模型生成视频（通过SDK同步调用）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    # 处理图像文件（使用Base64编码）
    image_result = handle_file_input(image_path, 'image')
    if not image_result["success"]:
        return f"❌ 图像文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
    
    image_url = image_result["url"]
    
    # 处理音频文件（使用Base64编码）（如果提供了的话）
    audio_url = ""
    if audio_enabled and audio_file is not None:
        audio_file_path = None
        if hasattr(audio_file, 'name'):  # 如果是Audio组件
            audio_file_path = audio_file.name
        elif isinstance(audio_file, str) and audio_file:  # 如果是路径字符串
            audio_file_path = audio_file
        
        if audio_file_path:
            # 验证音频文件格式
            if not any(audio_file_path.lower().endswith(ext) for ext in ['.mp3', '.wav']):
                return "❌ 音频文件格式不支持，请使用MP3或WAV格式的音频文件。"
            
            # 检查文件大小
            if os.path.exists(audio_file_path):
                file_size = os.path.getsize(audio_file_path)
                max_size = 15 * 1024 * 1024  # 15MB
                if file_size > max_size:
                    return f"❌ 音频文件过大 ({file_size / (1024*1024):.2f} MB)，超过15MB限制，请压缩音频文件。"
            
            audio_result = handle_file_input(audio_file_path, 'audio')
            if not audio_result["success"]:
                return f"❌ 音频文件处理失败: {audio_result['error']}\n请检查音频文件是否存在。"
            
            audio_url = audio_result["url"]

    try:
        # 使用DashScope SDK进行调用
        from dashscope import VideoSynthesis
        from http import HTTPStatus
        
        # 构建参数，使用官方示例中的参数
        params = {
            'api_key': api_key,
            'model': 'wan2.5-i2v-preview',
            'prompt': prompt,
            'img_url': image_url,  # 使用官方示例中的参数名
            'resolution': resolution,
            'duration': duration,
            'prompt_extend': True,
            'watermark': False,
            'negative_prompt': "",
            'seed': 12345,
            'audio': audio_enabled  # 根据文档，这应该是用来启用音频功能的
        }
        
        # 如果提供了音频URL，则添加到参数中
        if audio_url and audio_url.strip():
            params['audio_url'] = audio_url

        # 同步调用方式 - 直接返回结果
        response = VideoSynthesis.call(**params)
        
        # 调试：打印完整的响应对象信息
        print("Debug: Full response object:", response)
        print("Debug: Response status code:", response.status_code)
        if hasattr(response, 'output') and response.output:
            print("Debug: Response output:", response.output.__dict__ if hasattr(response.output, '__dict__') else response.output)
        
        if response.status_code == HTTPStatus.OK:
            # 检查是否包含视频URL（同步调用应该直接返回视频URL）
            if hasattr(response.output, 'video_url') and response.output.video_url:
                # 同步调用，直接返回视频URL
                video_url = response.output.video_url
                result_text = f"✅ 视频生成任务已完成！\n"
                result_text += f"视频URL: {video_url}\n"
                result_text += "注意：视频URL有效期24小时，请及时下载。\n"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "video_url": video_url,
                    "prompt": prompt,
                    "image_url": image_url,  # 保持记录原始image_url
                    "audio_url": audio_url,
                    "resolution": resolution,
                    "duration": duration,
                    "audio_enabled": audio_enabled,
                    "submit_time": time.time(),
                    "model": "wan2.5-i2v-preview"
                }
                
                timestamp = int(time.time())
                task_filename = f"task_sync_{timestamp}_wan25.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            elif hasattr(response.output, 'task_id') and response.output.task_id:
                # 如果同步调用返回了任务ID，说明可能仍在处理中
                task_id = response.output.task_id
                status_uri = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"  # 构造查询URL
                
                result_text = f"⏳ 视频生成任务已提交，正在处理中！\n"
                result_text += f"任务ID: {task_id}\n"
                result_text += f"状态查询URL: {status_uri}\n"
                result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "task_id": task_id,
                    "status_uri": status_uri,
                    "prompt": prompt,
                    "image_url": image_url,  # 保持记录原始image_url
                    "audio_url": audio_url,
                    "resolution": resolution,
                    "duration": duration,
                    "audio_enabled": audio_enabled,
                    "submit_time": time.time(),
                    "model": "wan2.5-i2v-preview"
                }
                
                task_filename = f"task_{task_id}_{int(time.time())}_wan25.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            else:
                # 输出错误信息，帮助诊断问题
                print(f"Error: No video_url or task_id in response.output. Output: {response.output}")
                return f"❌ 同步调用未返回视频URL或任务ID: {response}"
        else:
            # 输出错误信息，帮助诊断问题
            print(f"Error: API call failed. Status: {response.status_code}, Code: {response.code}, Message: {response.message}")
            return f"❌ API调用失败: status_code={response.status_code}, code={response.code}, message={response.message}"
            
    except Exception as e:
        # 调试：捕获异常时也打印完整的错误信息
        import traceback
        print("Debug: Exception occurred:")
        traceback.print_exc()
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_wan22_kf2v(prompt: str, first_frame_path: str, last_frame_path: str, resolution: str) -> str:
    """
    使用wan2.2-kf2v-flash模型生成视频（通过SDK调用）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    # 验证输入参数
    if not prompt or len(prompt.strip()) == 0:
        return "❌ 提示词不能为空，请输入有效的提示词。"
    
    if not first_frame_path or len(first_frame_path.strip()) == 0:
        return "❌ 首帧图像路径不能为空，请上传图像或输入有效的图像URL。"
    
    if not last_frame_path or len(last_frame_path.strip()) == 0:
        return "❌ 尾帧图像路径不能为空，请上传图像或输入有效的图像URL。"

    # 检查输入是否为URL
    if first_frame_path.startswith(('http://', 'https://')):
        # 如果是URL，直接使用
        first_frame_url = first_frame_path
    else:
        # 如果是本地文件，进行Base64编码
        if not os.path.exists(first_frame_path):
            return f"❌ 首帧图像文件不存在: {first_frame_path}，请检查文件路径是否正确。"
        
        # 检查文件大小
        file_size = os.path.getsize(first_frame_path)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            return f"❌ 首帧图像文件过大 ({file_size / (1024*1024):.2f} MB)，超过{max_size / (1024*1024):.2f} MB限制，请压缩图像或使用更小的图像。"
        
        try:
            with open(first_frame_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(first_frame_path)
                if not mime_type or not mime_type.startswith("image/"):
                    mime_type = 'image/png'  # 默认图像类型
                first_frame_url = f"data:{mime_type};base64,{encoded_image}"
        except Exception as e:
            return f"❌ 首帧图像文件读取失败: {str(e)}\n请检查第一帧图像文件是否存在且可访问。"
    
    # 检查输入是否为URL
    if last_frame_path.startswith(('http://', 'https://')):
        # 如果是URL，直接使用
        last_frame_url = last_frame_path
    else:
        # 如果是本地文件，进行Base64编码
        if not os.path.exists(last_frame_path):
            return f"❌ 尾帧图像文件不存在: {last_frame_path}，请检查文件路径是否正确。"
        
        # 检查文件大小
        file_size = os.path.getsize(last_frame_path)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            return f"❌ 尾帧图像文件过大 ({file_size / (1024*1024):.2f} MB)，超过{max_size / (1024*1024):.2f} MB限制，请压缩图像或使用更小的图像。"
        
        try:
            with open(last_frame_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(last_frame_path)
                if not mime_type or not mime_type.startswith("image/"):
                    mime_type = 'image/png'  # 默认图像类型
                last_frame_url = f"data:{mime_type};base64,{encoded_image}"
        except Exception as e:
            return f"❌ 尾帧图像文件读取失败: {str(e)}\n请检查最后一帧图像文件是否存在且可访问。"

    try:
        # 使用DashScope SDK进行调用
        from dashscope import VideoSynthesis
        from http import HTTPStatus
        
        # 同步调用方式
        rsp = VideoSynthesis.call(
            api_key=api_key,
            model="wan2.2-kf2v-flash",
            prompt=prompt,
            first_frame_url=first_frame_url,
            last_frame_url=last_frame_url,
            resolution=resolution,
            prompt_extend=True
        )
        
        if rsp.status_code == HTTPStatus.OK:
            # 返回视频URL，用户可以稍后查询结果
            result_text = f"✅ 视频生成任务已完成！\n"
            result_text += f"视频URL: {rsp.output.video_url}\n"
            result_text += "注意：视频URL有效期24小时，请及时下载。\n"
            
            # 保存任务信息到本地
            save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "video_url": rsp.output.video_url,
                "prompt": prompt,
                "first_frame_url": first_frame_url,  # 保持记录原始first_frame_url
                "last_frame_url": last_frame_url,   # 保持记录原始last_frame_url
                "resolution": resolution,
                "submit_time": time.time(),
                "model": "wan2.2-kf2v-flash"
            }
            
            timestamp = int(time.time())
            task_filename = f"task_sync_kf2v_{timestamp}.json"
            task_path = os.path.join(save_dir, task_filename)
            
            with open(task_path, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
            
            return result_text
        else:
            return f"❌ API调用失败: status_code={rsp.status_code}, code={rsp.code}, message={rsp.message}"
            
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_wan25_t2v(prompt: str, audio_file, resolution: str, duration: int, audio_enabled: bool) -> str:
    """
    使用wan2.5-t2v-preview模型生成视频（通过SDK调用）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    # 处理音频文件（支持本地路径或URL）
    audio_url = ""
    if audio_enabled and audio_file is not None:
        audio_file_path = None
        if hasattr(audio_file, 'name'):  # 如果是Audio组件
            audio_file_path = audio_file.name
        elif isinstance(audio_file, str) and audio_file:  # 如果是路径字符串
            audio_file_path = audio_file
        
        if audio_file_path:
            # 检查音频路径是否为本地路径
            if audio_file_path.startswith("file://") or not audio_file_path.startswith("http"):
                # 如果是本地路径，检查文件是否存在
                local_audio_path = audio_file_path.replace("file://", "", 1) if audio_file_path.startswith("file://") else audio_file_path
                if not os.path.exists(local_audio_path):
                    return f"❌ 音频文件不存在: {local_audio_path}\n请检查音频文件是否存在。"
                audio_url = f"file://{os.path.abspath(local_audio_path)}"
            else:
                # 如果是URL，直接使用
                audio_url = audio_file_path

    try:
        # 使用DashScope SDK进行调用
        from dashscope import VideoSynthesis
        from http import HTTPStatus
        
        # 构建参数
        params = {
            'model': 'wan2.5-t2v-preview',
            'prompt': prompt,
            'size': resolution,
            'prompt_extend': True,
            'watermark': False,
            'duration': duration,
            'audio': audio_enabled
        }
        
        # 如果提供了音频URL，则添加到参数中
        if audio_url and audio_url.strip():
            params['audio_url'] = audio_url

        # 同步调用方式
        response = VideoSynthesis.call(
            api_key=api_key,
            **params
        )
        
        if response.status_code == HTTPStatus.OK:
            # 检查是否包含任务ID（异步调用）或视频URL（同步调用）
            if hasattr(response.output, 'task_id') and response.output.task_id:
                # 异步调用，返回任务ID
                task_id = response.output.task_id
                # 根据规范，使用task_id构建标准查询URL
                status_uri = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                
                result_text = f"✅ 视频生成任务已成功提交！\n"
                result_text += f"任务ID: {task_id}\n"
                result_text += f"状态查询URL: {status_uri}\n"
                result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
                result_text += "注意：wan2.5-t2v-preview使用SDK方式调用"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "task_id": task_id,
                    "status_uri": status_uri,
                    "prompt": prompt,
                    "audio_url": audio_url,
                    "resolution": resolution,
                    "duration": duration,
                    "audio_enabled": audio_enabled,
                    "submit_time": time.time(),
                    "model": "wan2.5-t2v-preview"
                }
                
                task_filename = f"task_{task_id}_{int(time.time())}_t2v.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            elif hasattr(response.output, 'video_url') and response.output.video_url:
                # 同步调用，直接返回视频URL
                video_url = response.output.video_url
                result_text = f"✅ 视频生成任务已完成！\n"
                result_text += f"视频URL: {video_url}\n"
                result_text += "注意：视频URL有效期24小时，请及时下载。\n"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "video_url": video_url,
                    "prompt": prompt,
                    "audio_url": audio_url,
                    "resolution": resolution,
                    "duration": duration,
                    "audio_enabled": audio_enabled,
                    "submit_time": time.time(),
                    "model": "wan2.5-t2v-preview"
                }
                
                timestamp = int(time.time())
                task_filename = f"task_sync_{timestamp}_t2v.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            else:
                return f"❌ SDK响应中未找到任务ID或视频URL: {response}"
        else:
            return f"❌ API调用失败: status_code={response.status_code}, code={response.code}, message={response.message}"
            
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_wan22_animate_mix(image_url: str, video_url: str, check_image: bool = True, mode: str = "wan-std") -> str:
    """
    使用wan2.2-animate-mix模型生成视频（通过SDK调用）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    # 检查输入是否为URL
    if not image_url.startswith(('http://', 'https://')):
        return "❌ 图像URL必须是有效的公网URL。"
    
    if not video_url.startswith(('http://', 'https://')):
        return "❌ 视频URL必须是有效的公网URL。"

    # 对OSS URL进行特殊处理：下载并转换为Base64格式（如果文件大小允许）
    import requests
    import base64
    import mimetypes
    
    def download_with_retry(url, timeout=30, retries=3):
        """下载URL内容，带重试机制"""
        last_exception = None
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response
                else:
                    print(f"下载失败，状态码: {response.status_code}，尝试: {attempt + 1}/{retries}")
            except requests.exceptions.RequestException as e:
                print(f"请求异常: {str(e)}，尝试: {attempt + 1}/{retries}")
                last_exception = e
                if attempt < retries - 1:  # 不在最后一次尝试后sleep
                    import time
                    time.sleep(2)  # 重试前等待2秒
        
        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException(f"下载失败，最终状态码: {response.status_code}")

    def get_content_size(url):
        """获取内容大小，先尝试HEAD请求，再尝试GET请求"""
        try:
            response = requests.head(url, timeout=30)
            if 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
            
            # 如果HEAD请求没有返回大小，使用GET请求获取
            response = requests.get(url, timeout=30, stream=True)
            if 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
        except Exception:
            pass
        return None

    # 检查图像和视频的大小
    image_size = get_content_size(image_url)
    video_size = get_content_size(video_url)
    
    # API对数据URI大小的限制（10MB）
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    
    # 确定使用哪种格式的URL
    if image_size and image_size > max_size:
        # 文件太大，直接使用URL
        final_image_url = image_url
    else:
        # 文件大小合适，转换为Base64
        try:
            image_response = download_with_retry(image_url)
            
            # 获取图像的MIME类型
            content_type = image_response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                # 尝试从URL后缀判断文件类型
                _, ext = os.path.splitext(image_url.lower())
                valid_image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
                if ext not in valid_image_exts:
                    return f"❌ 图像URL指向的资源类型不正确，期望图像类型，实际为: {content_type}，请确保URL指向有效的图像文件。"
            
            # 将图像数据转换为Base64
            image_data = image_response.content
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            mime_type = content_type if content_type.startswith('image/') else mimetypes.guess_type(image_url)[0] or 'image/png'
            final_image_url = f"data:{mime_type};base64,{encoded_image}"
        except requests.exceptions.ReadTimeout:
            return f"❌ 访问图像URL超时，请检查图像URL是否有效且可公开访问。OSS可能需要更长时间响应，请稍后重试或更换URL。"
        except requests.exceptions.RequestException as e:
            return f"❌ 网络请求错误，请检查图像URL格式是否正确: {str(e)}"
        except Exception as e:
            return f"❌ 无法处理图像URL，请确保URL可访问: {str(e)}"
    
    if video_size and video_size > max_size:
        # 文件太大，直接使用URL
        final_video_url = video_url
    else:
        # 文件大小合适，转换为Base64
        try:
            video_response = download_with_retry(video_url)
            
            # 获取视频的MIME类型
            video_content_type = video_response.headers.get('Content-Type', '').lower()
            if not video_content_type.startswith('video/'):
                # 尝试从URL后缀判断文件类型
                _, ext = os.path.splitext(video_url.lower())
                valid_video_exts = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
                if ext not in valid_video_exts:
                    return f"❌ 视频URL指向的资源类型不正确，期望视频类型，实际为: {video_content_type}，请确保URL指向有效的视频文件。"
            
            # 将视频数据转换为Base64
            video_data = video_response.content
            encoded_video = base64.b64encode(video_data).decode('utf-8')
            video_mime_type = video_content_type if video_content_type.startswith('video/') else mimetypes.guess_type(video_url)[0] or 'video/mp4'
            final_video_url = f"data:{video_mime_type};base64,{encoded_video}"
        except requests.exceptions.ReadTimeout:
            return f"❌ 访问视频URL超时，请检查视频URL是否有效且可公开访问。OSS可能需要更长时间响应，请稍后重试或更换URL。"
        except requests.exceptions.RequestException as e:
            return f"❌ 网络请求错误，请检查视频URL格式是否正确: {str(e)}"
        except Exception as e:
            return f"❌ 无法处理视频URL，请确保URL可访问: {str(e)}"

    try:
        # 使用DashScope SDK进行调用
        from dashscope import VideoSynthesis
        from http import HTTPStatus
        
        # 同步调用方式，根据文件大小选择URL或Base64数据
        response = VideoSynthesis.call(
            api_key=api_key,
            model='wan2.2-animate-mix',
            image_url=final_image_url,  # 根据大小选择URL或Base64
            video_url=final_video_url,  # 根据大小选择URL或Base64
            check_image=check_image,
            mode=mode
        )
        
        if response.status_code == HTTPStatus.OK:
            # 检查是否包含任务ID（异步调用）或视频URL（同步调用）
            if hasattr(response.output, 'task_id') and response.output.task_id:
                # 异步调用，返回任务ID
                task_id = response.output.task_id
                # 根据规范，使用task_id构建标准查询URL
                status_uri = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                
                result_text = f"✅ 视频生成任务已成功提交！\n"
                result_text += f"任务ID: {task_id}\n"
                result_text += f"状态查询URL: {status_uri}\n"
                result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
                result_text += "注意：wan2.2-animate-mix使用SDK方式调用"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "task_id": task_id,
                    "status_uri": status_uri,
                    "image_url": image_url,
                    "video_url": video_url,
                    "check_image": check_image,
                    "mode": mode,
                    "submit_time": time.time(),
                    "model": "wan2.2-animate-mix"
                }
                
                task_filename = f"task_{task_id}_{int(time.time())}_animate_mix.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            elif hasattr(response.output, 'video_url') and response.output.video_url:
                # 同步调用，直接返回视频URL
                video_url_result = response.output.video_url
                result_text = f"✅ 视频生成任务已完成！\n"
                result_text += f"视频URL: {video_url_result}\n"
                result_text += "注意：视频URL有效期24小时，请及时下载。\n"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "video_url": video_url_result,
                    "image_url": image_url,
                    "video_url_param": video_url,
                    "check_image": check_image,
                    "mode": mode,
                    "submit_time": time.time(),
                    "model": "wan2.2-animate-mix"
                }
                
                timestamp = int(time.time())
                task_filename = f"task_sync_{timestamp}_animate_mix.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            else:
                return f"❌ SDK响应中未找到任务ID或视频URL: {response}"
        else:
            return f"❌ API调用失败: status_code={response.status_code}, code={response.code}, message={response.message}"
            
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_wan22_animate_move(image_url: str, video_url: str, check_image: bool = True, mode: str = "wan-std") -> str:
    """
    使用wan2.2-animate-move模型生成视频（通过SDK调用）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    # 检查输入是否为URL
    if not image_url.startswith(('http://', 'https://')):
        return "❌ 图像URL必须是有效的公网URL。"
    
    if not video_url.startswith(('http://', 'https://')):
        return "❌ 视频URL必须是有效的公网URL。"

    # 对OSS URL进行特殊处理：下载并转换为Base64格式（如果文件大小允许）
    import requests
    import base64
    import mimetypes
    
    def download_with_retry(url, timeout=30, retries=3):
        """下载URL内容，带重试机制"""
        last_exception = None
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response
                else:
                    print(f"下载失败，状态码: {response.status_code}，尝试: {attempt + 1}/{retries}")
            except requests.exceptions.RequestException as e:
                print(f"请求异常: {str(e)}，尝试: {attempt + 1}/{retries}")
                last_exception = e
                if attempt < retries - 1:  # 不在最后一次尝试后sleep
                    import time
                    time.sleep(2)  # 重试前等待2秒
        
        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException(f"下载失败，最终状态码: {response.status_code}")

    def get_content_size(url):
        """获取内容大小，先尝试HEAD请求，再尝试GET请求"""
        try:
            response = requests.head(url, timeout=30)
            if 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
            
            # 如果HEAD请求没有返回大小，使用GET请求获取
            response = requests.get(url, timeout=30, stream=True)
            if 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
        except Exception:
            pass
        return None

    # 检查图像和视频的大小
    image_size = get_content_size(image_url)
    video_size = get_content_size(video_url)
    
    # API对数据URI大小的限制（10MB）
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    
    # 确定使用哪种格式的URL
    if image_size and image_size > max_size:
        # 文件太大，直接使用URL
        final_image_url = image_url
    else:
        # 文件大小合适，转换为Base64
        try:
            image_response = download_with_retry(image_url)
            
            # 获取图像的MIME类型
            content_type = image_response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                # 尝试从URL后缀判断文件类型
                _, ext = os.path.splitext(image_url.lower())
                valid_image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
                if ext not in valid_image_exts:
                    return f"❌ 图像URL指向的资源类型不正确，期望图像类型，实际为: {content_type}，请确保URL指向有效的图像文件。"
            
            # 将图像数据转换为Base64
            image_data = image_response.content
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            mime_type = content_type if content_type.startswith('image/') else mimetypes.guess_type(image_url)[0] or 'image/png'
            final_image_url = f"data:{mime_type};base64,{encoded_image}"
        except requests.exceptions.ReadTimeout:
            return f"❌ 访问图像URL超时，请检查图像URL是否有效且可公开访问。OSS可能需要更长时间响应，请稍后重试或更换URL。"
        except requests.exceptions.RequestException as e:
            return f"❌ 网络请求错误，请检查图像URL格式是否正确: {str(e)}"
        except Exception as e:
            return f"❌ 无法处理图像URL，请确保URL可访问: {str(e)}"
    
    if video_size and video_size > max_size:
        # 文件太大，直接使用URL
        final_video_url = video_url
    else:
        # 文件大小合适，转换为Base64
        try:
            video_response = download_with_retry(video_url)
            
            # 获取视频的MIME类型
            video_content_type = video_response.headers.get('Content-Type', '').lower()
            if not video_content_type.startswith('video/'):
                # 尝试从URL后缀判断文件类型
                _, ext = os.path.splitext(video_url.lower())
                valid_video_exts = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
                if ext not in valid_video_exts:
                    return f"❌ 视频URL指向的资源类型不正确，期望视频类型，实际为: {video_content_type}，请确保URL指向有效的视频文件。"
            
            # 将视频数据转换为Base64
            video_data = video_response.content
            encoded_video = base64.b64encode(video_data).decode('utf-8')
            video_mime_type = video_content_type if video_content_type.startswith('video/') else mimetypes.guess_type(video_url)[0] or 'video/mp4'
            final_video_url = f"data:{video_mime_type};base64,{encoded_video}"
        except requests.exceptions.ReadTimeout:
            return f"❌ 访问视频URL超时，请检查视频URL是否有效且可公开访问。OSS可能需要更长时间响应，请稍后重试或更换URL。"
        except requests.exceptions.RequestException as e:
            return f"❌ 网络请求错误，请检查视频URL格式是否正确: {str(e)}"
        except Exception as e:
            return f"❌ 无法处理视频URL，请确保URL可访问: {str(e)}"

    try:
        # 使用DashScope SDK进行调用
        from dashscope import VideoSynthesis
        from http import HTTPStatus
        
        # 同步调用方式，根据文件大小选择URL或Base64数据
        response = VideoSynthesis.call(
            api_key=api_key,
            model='wan2.2-animate-move',
            image_url=final_image_url,  # 根据大小选择URL或Base64
            video_url=final_video_url,  # 根据大小选择URL或Base64
            check_image=check_image,
            mode=mode
        )
        
        if response.status_code == HTTPStatus.OK:
            # 检查是否包含任务ID（异步调用）或视频URL（同步调用）
            if hasattr(response.output, 'task_id') and response.output.task_id:
                # 异步调用，返回任务ID
                task_id = response.output.task_id
                # 根据规范，使用task_id构建标准查询URL
                status_uri = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                
                result_text = f"✅ 视频生成任务已成功提交！\n"
                result_text += f"任务ID: {task_id}\n"
                result_text += f"状态查询URL: {status_uri}\n"
                result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
                result_text += "注意：wan2.2-animate-move使用SDK方式调用"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "task_id": task_id,
                    "status_uri": status_uri,
                    "image_url": image_url,
                    "video_url": video_url,
                    "check_image": check_image,
                    "mode": mode,
                    "submit_time": time.time(),
                    "model": "wan2.2-animate-move"
                }
                
                task_filename = f"task_{task_id}_{int(time.time())}_animate_move.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            elif hasattr(response.output, 'video_url') and response.output.video_url:
                # 同步调用，直接返回视频URL
                video_url_result = response.output.video_url
                result_text = f"✅ 视频生成任务已完成！\n"
                result_text += f"视频URL: {video_url_result}\n"
                result_text += "注意：视频URL有效期24小时，请及时下载。\n"
                
                # 保存任务信息到本地
                save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                os.makedirs(save_dir, exist_ok=True)
                
                task_info = {
                    "video_url": video_url_result,
                    "image_url": image_url,
                    "video_url_param": video_url,
                    "check_image": check_image,
                    "mode": mode,
                    "submit_time": time.time(),
                    "model": "wan2.2-animate-move"
                }
                
                timestamp = int(time.time())
                task_filename = f"task_sync_{timestamp}_animate_move.json"
                task_path = os.path.join(save_dir, task_filename)
                
                with open(task_path, 'w', encoding='utf-8') as f:
                    json.dump(task_info, f, ensure_ascii=False, indent=2)
                
                return result_text
            else:
                return f"❌ SDK响应中未找到任务ID或视频URL: {response}"
        else:
            return f"❌ API调用失败: status_code={response.status_code}, code={response.code}, message={response.message}"
            
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def generate_video_with_wan27_t2v(prompt: str, audio_file, resolution: str, duration: int, negative_prompt: str = "") -> str:
    """
    使用wan2.7-t2v模型生成视频（通过HTTP API异步调用）
    支持多镜头叙事、音频输入、反向提示词
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
    
    if audio_file is not None:
        audio_file_path = None
        if hasattr(audio_file, 'name'):
            audio_file_path = audio_file.name
        elif isinstance(audio_file, str) and audio_file:
            audio_file_path = audio_file
        
        if audio_file_path:
            audio_result = handle_file_input(audio_file_path, 'audio')
            if not audio_result["success"]:
                return f"❌ 音频文件处理失败: {audio_result['error']}\n请检查音频文件是否存在。"
            media.append({
                "type": "driving_audio",
                "url": audio_result["url"]
            })

    payload = {
        "model": "wan2.7-t2v",
        "input": {
            "prompt": prompt,
            "media": media
        },
        "parameters": {
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
            result_text += "注意：wan2.7-t2v使用HTTP API异步调用"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "media": media,
                "resolution": resolution,
                "duration": duration,
                "submit_time": time.time(),
                "model": "wan2.7-t2v"
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_wan27_t2v.json"
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


def generate_video_with_wan27_i2v(prompt: str, first_frame_path: str, last_frame_path: str = None, 
                                  first_clip_path: str = None, audio_file=None, 
                                  resolution: str = "720P", duration: int = 10, 
                                  negative_prompt: str = "") -> str:
    """
    使用wan2.7-i2v模型生成视频（通过HTTP API异步调用）
    支持：首帧生视频、首尾帧生视频、视频续写（首视频片段/首视频片段+尾帧）
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

    if first_clip_path:
        clip_result = handle_file_input(first_clip_path, 'video')
        if not clip_result["success"]:
            return f"❌ 视频片段文件处理失败: {clip_result['error']}\n请检查视频文件是否存在。"
        media.append({
            "type": "first_clip",
            "url": clip_result["url"]
        })
    elif first_frame_path:
        image_result = handle_file_input(first_frame_path, 'image')
        if not image_result["success"]:
            return f"❌ 首帧图像文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "first_frame",
            "url": image_result["url"]
        })
    else:
        return "❌ 请提供首帧图像或首段视频片段。"

    if last_frame_path:
        last_image_result = handle_file_input(last_frame_path, 'image')
        if not last_image_result["success"]:
            return f"❌ 尾帧图像文件处理失败: {last_image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "last_frame",
            "url": last_image_result["url"]
        })

    if audio_file is not None:
        audio_file_path = None
        if hasattr(audio_file, 'name'):
            audio_file_path = audio_file.name
        elif isinstance(audio_file, str) and audio_file:
            audio_file_path = audio_file
        
        if audio_file_path:
            audio_result = handle_file_input(audio_file_path, 'audio')
            if not audio_result["success"]:
                return f"❌ 音频文件处理失败: {audio_result['error']}\n请检查音频文件是否存在。"
            media.append({
                "type": "driving_audio",
                "url": audio_result["url"]
            })

    payload = {
        "model": "wan2.7-i2v",
        "input": {
            "prompt": prompt,
            "media": media
        },
        "parameters": {
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
            result_text += "注意：wan2.7-i2v使用HTTP API异步调用"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "media": media,
                "resolution": resolution,
                "duration": duration,
                "submit_time": time.time(),
                "model": "wan2.7-i2v"
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_wan27_i2v.json"
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


def generate_video_with_wan27_r2v(prompt: str, reference_images: list = None, 
                                   reference_videos: list = None, audio_file=None,
                                   resolution: str = "720P", duration: int = 10, 
                                   negative_prompt: str = "") -> str:
    """
    使用wan2.7-r2v模型生成视频（通过HTTP API异步调用）
    参考生视频：支持多模态输入（图像、视频、音频），生成保持角色形象和音色一致性的视频
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

    # 处理音频文件
    audio_url = None
    if audio_file is not None:
        audio_file_path = None
        if hasattr(audio_file, 'name'):
            audio_file_path = audio_file.name
        elif isinstance(audio_file, str) and audio_file:
            audio_file_path = audio_file
        
        if audio_file_path:
            audio_result = handle_file_input(audio_file_path, 'audio')
            if audio_result["success"]:
                audio_url = audio_result["url"]

    # 处理参考图像
    if reference_images and isinstance(reference_images, list):
        for idx, img_path in enumerate(reference_images):
            if img_path:
                print(f"[DEBUG] 处理参考图像{idx+1}: {img_path}")
                image_result = handle_file_input(img_path, 'image')
                if not image_result["success"]:
                    return f"❌ 参考图像{idx+1}文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
                
                media_item = {
                    "type": "reference_image",
                    "url": image_result["url"]
                }
                # 如果是第一张图像且有音频，将音频附加到该图像
                if idx == 0 and audio_url:
                    media_item["reference_voice"] = audio_url
                
                media.append(media_item)
                print(f"[DEBUG] 参考图像{idx+1}处理成功")

    # 处理参考视频
    if reference_videos and isinstance(reference_videos, list):
        for idx, vid_path in enumerate(reference_videos):
            if vid_path:
                print(f"[DEBUG] 处理参考视频{idx+1}: {vid_path}")
                video_result = handle_file_input(vid_path, 'video')
                if not video_result["success"]:
                    return f"❌ 参考视频{idx+1}文件处理失败: {video_result['error']}\n请检查视频文件是否存在。"
                
                media_item = {
                    "type": "reference_video",
                    "url": video_result["url"]
                }
                # 如果没有参考图像但有音频，将音频附加到第一个视频
                if not reference_images and idx == 0 and audio_url:
                    media_item["reference_voice"] = audio_url
                
                media.append(media_item)
                print(f"[DEBUG] 参考视频{idx+1}处理成功")

    if not media:
        return "❌ 请至少上传一张参考图像或一个参考视频。"

    print(f"[DEBUG] 最终media数组: {media}")
    print(f"[DEBUG] 提示词: {prompt}")

    payload = {
        "model": "wan2.7-r2v",
        "input": {
            "prompt": prompt,
            "media": media
        },
        "parameters": {
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    print(f"[DEBUG] 请求payload: {payload}")

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"[DEBUG] API响应: {result}")
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频生成任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频生成可能需要一些时间。\n"
            result_text += "注意：wan2.7-r2v使用HTTP API异步调用"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "media": media,
                "resolution": resolution,
                "duration": duration,
                "submit_time": time.time(),
                "model": "wan2.7-r2v"
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_wan27_r2v.json"
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


def generate_video_with_wan27_videoedit(prompt: str, video_path: str, 
                                        reference_image_path: str = None,
                                        resolution: str = "720P", 
                                        negative_prompt: str = "") -> str:
    """
    使用wan2.7-videoedit模型编辑视频（通过HTTP API异步调用）
    视频编辑：支持指令编辑和视频迁移，可完成局部增删改、风格转换等任务
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

    if video_path:
        video_result = handle_file_input(video_path, 'video')
        if not video_result["success"]:
            return f"❌ 视频文件处理失败: {video_result['error']}\n请检查视频文件是否存在。"
        media.append({
            "type": "video",
            "url": video_result["url"]
        })
    else:
        return "❌ 请提供要编辑的视频文件。"

    if reference_image_path:
        image_result = handle_file_input(reference_image_path, 'image')
        if not image_result["success"]:
            return f"❌ 参考图像文件处理失败: {image_result['error']}\n请检查图像文件是否存在。"
        media.append({
            "type": "reference_image",
            "url": image_result["url"]
        })

    payload = {
        "model": "wan2.7-videoedit",
        "input": {
            "prompt": prompt,
            "media": media
        },
        "parameters": {
            "resolution": resolution,
            "prompt_extend": True,
            "watermark": False
        }
    }

    if negative_prompt and negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "task_id" in result["output"]:
            task_id = result["output"]["task_id"]
            status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            result_text = f"✅ 视频编辑任务已成功提交！\n"
            result_text += f"任务ID: {task_id}\n"
            result_text += f"状态查询URL: {status_url}\n"
            result_text += "请稍后使用任务ID查询结果，视频编辑可能需要一些时间。\n"
            result_text += "注意：wan2.7-videoedit使用HTTP API异步调用"
            
            save_dir = os.path.join(WEBUI_ROOT, "output", "qwen-video")
            os.makedirs(save_dir, exist_ok=True)
            
            task_info = {
                "task_id": task_id,
                "status_uri": status_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "media": media,
                "resolution": resolution,
                "submit_time": time.time(),
                "model": "wan2.7-videoedit"
            }
            
            task_filename = f"task_{task_id}_{int(time.time())}_wan27_videoedit.json"
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

