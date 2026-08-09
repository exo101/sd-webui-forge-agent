"""
Qwen视频生成任务查询模块
负责处理任务查询和结果展示
"""

import os
import json
import time
from modules import shared
import gradio as gr
import requests


def query_video_task(task_id: str) -> str:
    """
    查询视频生成任务的状态
    增加了对任务刚提交时的特殊处理，当任务正在初始化时提供友好提示
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "⚠️ 未设置DASHSCOPE_API_KEY环境变量，请先设置API密钥。"

    if not task_id or len(task_id.strip()) == 0:
        return "❌ 请输入有效的任务ID。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 任务查询URL
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

    try:
        response = requests.get(url, headers=headers)
        print(f"Debug: Querying task {task_id}, Status Code: {response.status_code}")  # 调试信息
        print(f"Debug: Response Content: {response.text}")  # 调试信息
        
        response.raise_for_status()
        result = response.json()

        if "output" in result and "task_status" in result["output"]:
            task_info = result["output"]
            status = task_info["task_status"]
            
            # 格式化输出任务状态信息
            status_descriptions = {
                "PENDING": "⏳ 任务排队中",
                "RUNNING": "🔄 任务处理中", 
                "SUCCEEDED": "✅ 任务执行成功",
                "FAILED": "❌ 任务执行失败",
                "CANCELED": "⏹️ 任务已取消",
                "UNKNOWN": "❓ 任务不存在或状态未知"
            }
            
            result_text = f"📋 任务ID: {task_id}\n"
            result_text += f"📊 任务状态: {status_descriptions.get(status, status)}\n"
            
            if "submit_time" in task_info:
                result_text += f"⏰ 提交时间: {task_info['submit_time']}\n"
            
            if "scheduled_time" in task_info:
                result_text += f"⏱️ 执行时间: {task_info['scheduled_time']}\n"
                
            if "end_time" in task_info:
                result_text += f"🏁 完成时间: {task_info['end_time']}\n"
                
            if "orig_prompt" in task_info:
                result_text += f"📝 原始提示词: {task_info['orig_prompt']}\n"
            
            if status == "SUCCEEDED":
                if "video_url" in task_info:
                    video_url = task_info["video_url"]
                    result_text += f"🎬 视频URL: {video_url}\n"
                    result_text += "🔗 链接有效期24小时，请及时下载。\n"
                    
                    # 保存视频信息到本地
                    save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    video_info = {
                        "task_id": task_id,
                        "video_url": video_url,
                        "prompt": task_info.get("orig_prompt", ""),
                        "submit_time": task_info.get("submit_time", ""),
                        "end_time": task_info.get("end_time", ""),
                        "status": status
                    }
                    
                    # 生成文件名，基于结束时间或提交时间，避免使用无效字符
                    # 尝试从结束时间或提交时间提取时间戳，如果都不可用则使用当前时间
                    time_str = task_info.get("end_time") or task_info.get("submit_time", "")
                    if time_str:
                        # 移除日期时间中的特殊字符用于文件名
                        filename_time = time_str.replace("-", "").replace(":", "").replace(" ", "")
                    else:
                        # 如果时间信息都不可用，使用当前时间戳
                        filename_time = str(int(time.time()))
                    video_filename = f"video_{task_id}_{filename_time}.json"
                    video_path = os.path.join(save_dir, video_filename)
                    
                    with open(video_path, 'w', encoding='utf-8') as f:
                        json.dump(video_info, f, ensure_ascii=False, indent=2)
                    
                    result_text += f"💾 视频信息已保存至: {video_path}"
                else:
                    result_text += "⚠️ 任务成功但未返回视频URL"
            elif status == "FAILED":
                result_text += "❌ 任务执行失败，请检查错误信息并重试。"
                # 添加错误详情
                error_code = result.get('code', result["output"].get('code', 'N/A'))
                error_message = result.get('message', result["output"].get('message', 'N/A'))
                if error_code != 'N/A':
                    result_text += f"\n错误代码: {error_code}"
                if error_message != 'N/A':
                    result_text += f"\n错误信息: {error_message}"
                    
                # 提供可能的失败原因
                result_text += "\n\n💡 可能的失败原因：\n"
                result_text += "   • 提示词包含违规内容，触发内容安全审核\n"
                result_text += "   • 图像或音频文件格式不支持或损坏\n"
                result_text += "   • API密钥权限不足或余额不足\n"
                result_text += "   • 模型服务暂时不可用\n"
                result_text += "   • 网络连接问题导致API调用失败\n"
                result_text += "\n建议：\n"
                result_text += "   • 检查提示词是否包含敏感内容\n"
                result_text += "   • 确认图像和音频文件格式正确且未损坏\n"
                result_text += "   • 验证API密钥是否有效且有足够权限\n"
                result_text += "   • 稍后重试，可能是临时服务问题\n"
            elif status == "PENDING":
                result_text += "⏳ 任务正在排队等待处理，请稍后再查询。"
            elif status == "RUNNING":
                result_text += "🔄 任务正在处理中，请耐心等待..."
                # 添加预估等待时间提示
                result_text += "\n⏱️ 视频生成通常需要几分钟时间，请耐心等待。"
            elif status == "CANCELED":
                result_text += "⏹️ 任务已被取消。"
            elif status == "UNKNOWN":
                result_text += "❓ 任务状态未知，可能任务ID无效或已过期。"
        else:
            # 检查错误信息
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                # 如果是404错误或包含"not found"、"不存在"的信息，可能是任务刚开始处理
                if "not found" in error_msg.lower() or "不存在" in error_msg:
                    result_text = f"📋 任务ID: {task_id}\n"
                    result_text += "📊 任务状态: ⏳ 任务可能正在初始化\n"
                    result_text += "🔄 系统正在接收任务，请稍等片刻后重试查询。\n"
                    result_text += "💡 有时任务刚提交时需要一点时间才能被系统记录，请稍等10-30秒后重试。\n"
                    return result_text
                else:
                    return f"❌ API调用失败: {error_msg}\n请检查API密钥和任务ID是否有效"
            else:
                # 对于没有明确错误信息但缺少必要字段的情况，视为任务不存在
                result_text = f"📋 任务ID: {task_id}\n"
                result_text += "📊 任务状态: ❌ 任务不存在\n"
                result_text += "🔍 可能的原因：\n"
                result_text += "   • 任务ID输入错误\n"
                result_text += "   • 任务已过期（查询有效期24小时）\n"
                result_text += "   • API调用时出现异常，任务未成功提交\n"
                result_text += "💡 建议：\n"
                result_text += "   • 检查任务ID是否正确\n"
                result_text += "   • 重新提交视频生成任务\n"
                return result_text
                
        return result_text
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # 404错误表示任务不存在
            result_text = f"📋 任务ID: {task_id}\n"
            result_text += "📊 任务状态: ❌ 任务不存在\n"
            result_text += "🔍 可能的原因：\n"
            result_text += "   • 任务ID输入错误\n"
            result_text += "   • 任务已过期（查询有效期24小时）\n"
            result_text += "   • API调用时出现异常，任务未成功提交\n"
            result_text += "💡 建议：\n"
            result_text += "   • 检查任务ID是否正确\n"
            result_text += "   • 重新提交视频生成任务\n"
            return result_text
        elif e.response.status_code == 400:
            return f"❌ 请求错误 (400): 请检查输入参数是否正确"
        else:
            return f"❌ HTTP错误: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ 请求失败: {str(e)}"
    except Exception as e:
        return f"❌ 处理响应时出错: {str(e)}"


def get_recent_tasks():
    """
    获取最近的任务列表
    """
    save_dir = os.path.join(shared.data_path, "outputs", "qwen-video")
    if not os.path.exists(save_dir):
        return []
    
    task_files = []
    for filename in os.listdir(save_dir):
        if filename.startswith("task_") and filename.endswith(".json"):
            filepath = os.path.join(save_dir, filename)
            task_files.append(filepath)
    
    # 按修改时间排序，最新的在前
    task_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    recent_tasks = []
    for filepath in task_files[:10]:  # 只返回最近10个任务
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                task_info = json.load(f)
                recent_tasks.append({
                    'task_id': task_info.get('task_id', ''),
                    'status': task_info.get('status', 'UNKNOWN'),
                    'submit_time': task_info.get('submit_time', ''),
                    'model': task_info.get('model', 'Unknown')
                })
        except Exception:
            continue
    
    return recent_tasks