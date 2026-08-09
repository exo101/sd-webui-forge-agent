# coding=utf-8

import sys
import base64
import os
import requests
import json

# 读取启动器共享端口配置，自动跟随启动器的 llama 端口
try:
    from modules.llama_port import get_llama_url as _get_llama_url
    DEFAULT_LLAMACPP_URL = _get_llama_url()
except Exception:
    DEFAULT_LLAMACPP_URL = "http://localhost:8080"

def get_llamacpp_models(llamacpp_host=DEFAULT_LLAMACPP_URL):
    """获取 llama.cpp 加载的模型列表"""
    try:
        models_url = f"{llamacpp_host.rstrip('/')}/v1/models"
        
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = []
        
        # 兼容不同的响应格式，安全提取模型
        if "models" in data and isinstance(data["models"], list):
            for m in data["models"]:
                if isinstance(m, dict):
                    # 尝试 'id' 或 'name' 字段
                    if "id" in m:
                        models.append(m["id"])
                    elif "name" in m:
                        models.append(m["name"])
                    elif "model" in m:
                        models.append(m["model"])
                elif isinstance(m, str):
                    models.append(m)
        if "data" in data and isinstance(data["data"], list):
            for m in data["data"]:
                if isinstance(m, dict):
                    # 尝试 'id' 或 'name' 字段
                    if "id" in m:
                        models.append(m["id"])
                    elif "name" in m:
                        models.append(m["name"])
                    elif "model" in m:
                        models.append(m["model"])
                elif isinstance(m, str):
                    models.append(m)
        
        # 去重并返回
        return list(set(models)) if models else []
    except requests.exceptions.RequestException:
        # 静默处理连接错误，不打印冗余日志
        return []
    except Exception as e:
        # 记录但不打印冗余的错误信息
        return []

def get_response_lvm_llamacpp_api(input_model_name, input_content, input_image_path,
                                   llamacpp_host=DEFAULT_LLAMACPP_URL, timeout=300):
    try:
        # 检查图片文件是否存在
        if not os.path.exists(input_image_path):
            print(f"❌ 错误：图片文件不存在 '{input_image_path}'")
            return f"[错误] 图片文件不存在: {input_image_path}"
        
        # 获取图片信息
        file_size = os.path.getsize(input_image_path)
        print(f"📷 图片路径: {input_image_path}, 大小: {file_size} bytes")
        
        # 读取图片并编码为Base64
        with open(input_image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        print(f"✅ Base64编码完成，长度: {len(base64_image)}")
        
        # 构造消息 - 使用 OpenAI 格式的视觉消息
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': input_content},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }]
        
        # llama.cpp /v1/chat/completions API
        api_url = f"{llamacpp_host.rstrip('/')}/v1/chat/completions"
        
        # 调用 llama.cpp API
        payload = {
            'model': input_model_name,
            'messages': messages,
            'stream': False
        }
        
        print(f"🔄 正在调用 llama.cpp API: {api_url}")
        print(f"📝 模型: {input_model_name}")
        print(f"📄 提示词长度: {len(input_content)}")
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        
        # 检查HTTP错误并获取详细信息
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # 获取服务器返回的详细错误信息
            error_details = ""
            try:
                error_json = response.json()
                if 'error' in error_json:
                    if isinstance(error_json['error'], dict):
                        error_details = f", 错误详情: {error_json['error'].get('message', '')}"
                    else:
                        error_details = f", 错误详情: {error_json['error']}"
            except:
                # 如果不是JSON格式，尝试获取文本内容
                try:
                    error_details = f", 响应内容: {response.text[:500]}"
                except:
                    pass
            
            error_msg = f"llama.cpp API 调用失败: {str(e)}{error_details}"
            print(f"❌ {error_msg}")
            return error_msg
        
        print(f"✅ API调用成功，响应状态: {response.status_code}")
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except FileNotFoundError as e:
        print(f"❌ 文件错误: {str(e)}")
        return f"[错误] 文件不存在: {input_image_path}"
    except requests.exceptions.RequestException as e:
        print(f"❌ llama.cpp API 调用失败: {str(e)}")
        print(f"请确保 llama.cpp 服务正在运行 (llama-server --model your_model.gguf --host 0.0.0.0 --port 8080)")
        return f"[错误] 无法连接到 llama.cpp 服务: {str(e)}"
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"[错误] 系统错误: {str(e)}"

def get_response_text_llamacpp_api(input_model_name, input_content,
                                  llamacpp_host=DEFAULT_LLAMACPP_URL, timeout=300):
    try:
        # llama.cpp /v1/chat/completions API
        api_url = f"{llamacpp_host.rstrip('/')}/v1/chat/completions"
        
        # 直接构造消息
        messages = [{
            'role': 'user',
            'content': input_content
        }]
        
        # 调用 llama.cpp API
        payload = {
            'model': input_model_name,
            'messages': messages,
            'stream': False
        }
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"llama.cpp API 调用失败: {str(e)}")
        print(f"请确保 llama.cpp 服务正在运行 (llama-server --model your_model.gguf --host 0.0.0.0 --port 8080)")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"系统错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    arguments = sys.argv
    
    # 基本参数检查
    if len(arguments) < 4:
        print("参数不足！使用方法:")
        print(f"python {arguments[0]} <model_type> <model_name> <user_input> [file_path] [llamacpp_host]")
        print("示例 (视觉模型):")
        print(f"python {arguments[0]} vision llava '请描述图片内容' image.png http://localhost:8080")
        print("示例 (纯文本模型):")
        print(f"python {arguments[0]} text mistral '你的问题'")
        sys.exit(1)

    model_type = arguments[1]
    model_name = arguments[2]
    user_input = arguments[3]
    file_path = arguments[4] if len(arguments) > 4 else None
    llamacpp_host = arguments[5] if len(arguments) > 5 else DEFAULT_LLAMACPP_URL
    
    # 根据模型类型调用相应函数
    try:
        if model_type == "vision":
            response_content = get_response_lvm_llamacpp_api(model_name, user_input, file_path, llamacpp_host)
        else:  # text
            response_content = get_response_text_llamacpp_api(model_name, user_input, llamacpp_host)
        print(response_content)
    except Exception as e:
        print(f"调用模型时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
