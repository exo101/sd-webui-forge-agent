# coding=utf-8

import sys
import base64
import os
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

def get_ollama_models(ollama_host=None):
    """获取已安装的 Ollama 模型列表"""
    try:
        base_url = ollama_host if ollama_host else OLLAMA_BASE_URL
        tags_url = f"{base_url}/api/tags"
        
        response = requests.get(tags_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        
        return models
    except requests.exceptions.RequestException as e:
        print(f"获取 Ollama 模型列表失败: {str(e)}")
        return []
    except Exception as e:
        print(f"获取 Ollama 模型列表出错: {str(e)}")
        return []

def get_response_lvm_ollama_api(input_model_name, input_content, input_image_path, ollama_host=None, timeout=120):
    try:
        # 检查图片文件是否存在
        if not os.path.exists(input_image_path):
            print(f"错误：图片文件不存在 '{input_image_path}'")
            return
        
        # 读取图片并编码为Base64
        with open(input_image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 构造消息 - 确保包含图片和文本指令
        messages = [{
            'role': 'user',
            'content': input_content,
            'images': [base64_image]
        }]
        
        # 调用 Ollama API
        base_url = ollama_host if ollama_host else OLLAMA_BASE_URL
        api_url = f"{base_url}/api/chat"
        
        payload = {
            'model': input_model_name,
            'messages': messages,
            'stream': False
        }
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['message']['content']
    
    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}")
        import traceback
        traceback.print_exc()
    except requests.exceptions.RequestException as e:
        print(f"Ollama API 调用失败: {str(e)}")
        print("请确保 Ollama 服务正在运行 (ollama serve)")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"系统错误: {str(e)}")
        import traceback
        traceback.print_exc()

def get_response_llm_ollama_api(input_model_name, input_content, input_file_path, ollama_host=None, timeout=120):
    try:
        # 检查文本文件是否存在
        if not os.path.exists(input_file_path):
            print(f"错误：文件不存在 '{input_file_path}'")
            return
        
        # 读取文件内容
        with open(input_file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # 构造更清晰的提示词
        prompt = f"""请根据以下文本内容执行任务：
        [文本开始]
        {file_content}
        [文本结束]
        
        任务要求：{input_content}"""
        
        # 调用 Ollama API
        base_url = ollama_host if ollama_host else OLLAMA_BASE_URL
        api_url = f"{base_url}/api/chat"
        
        payload = {
            'model': input_model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False
        }
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['message']['content']
    
    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}")
        import traceback
        traceback.print_exc()
    except requests.exceptions.RequestException as e:
        print(f"Ollama API 调用失败: {str(e)}")
        print("请确保 Ollama 服务正在运行 (ollama serve)")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"系统错误: {str(e)}")
        import traceback
        traceback.print_exc()

def get_response_text_ollama_api(input_model_name, input_content, ollama_host=None, timeout=120):
    try:
        # 直接构造消息
        messages = [{
            'role': 'user',
            'content': input_content
        }]
        
        # 调用 Ollama API
        base_url = ollama_host if ollama_host else OLLAMA_BASE_URL
        api_url = f"{base_url}/api/chat"
        
        payload = {
            'model': input_model_name,
            'messages': messages,
            'stream': False
        }
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"Ollama API 调用失败: {str(e)}")
        print("请确保 Ollama 服务正在运行 (ollama serve)")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"系统错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    arguments = sys.argv
    
    # 基本参数检查
    if len(arguments) < 4:  # 修改为4，因为text模式不需要file_path
        print("参数不足！使用方法:")
        print(f"python {arguments[0]} <model_type> <model_name> <user_input> [file_path]")
        print("示例 (视觉模型):")
        print(f"python {arguments[0]} vision llava '请描述图片内容' image.png")
        print("示例 (语言模型):")
        print(f"python {arguments[0]} language mistral '总结内容' document.txt")
        print("示例 (纯文本模型):")
        print(f"python {arguments[0]} text mistral '你的问题'")
        sys.exit(1)

    model_type = arguments[1]
    model_name = arguments[2]
    user_input = arguments[3]
    file_path = arguments[4] if len(arguments) > 4 else None

    # 验证模型类型
    if model_type not in ["vision", "language", "text"]:
        print(f"错误：未知模型类型 '{model_type}' (应为 vision、language 或 text)")
        sys.exit(1)
    
    # 验证文件路径（仅对vision和language类型）
    if model_type in ["vision", "language"]:
        if not file_path or not os.path.exists(file_path):
            print(f"错误：文件不存在 '{file_path}'")
            sys.exit(1)
    
    # 根据模型类型调用相应函数
    try:
        if model_type == "vision":
            response_content = get_response_lvm_ollama_api(model_name, user_input, file_path)
        elif model_type == "language":
            response_content = get_response_llm_ollama_api(model_name, user_input, file_path)
        else:  # text
            response_content = get_response_text_ollama_api(model_name, user_input)
        print(response_content) # 只打印最终响应内容
    except Exception as e:
        print(f"调用模型时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
