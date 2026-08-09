import requests
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 读取启动器共享端口配置，自动跟随启动器的 llama 端口
try:
    from modules.llama_port import get_llama_port
    _port = get_llama_port()
except Exception:
    _port = 8080

# llama.cpp 配置
DEFAULT_LLAMACPP_HOST = "localhost"
DEFAULT_LLAMACPP_PORT = _port
DEFAULT_LLAMACPP_URL = f"http://{DEFAULT_LLAMACPP_HOST}:{DEFAULT_LLAMACPP_PORT}"


def encode_image_to_base64(image_data):
    """将图片转换为 base64 编码（支持文件路径或 PIL Image）"""
    try:
        from io import BytesIO
        
        if isinstance(image_data, str):
            with open(image_data, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif hasattr(image_data, 'save'):
            buffered = BytesIO()
            image_data.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            logger.error("不支持的图片格式")
            return None
    except Exception as e:
        logger.error(f"图片编码失败：{e}")
        return None


def analyze_with_llamacpp(image_data, prompt, model, llamacpp_host=DEFAULT_LLAMACPP_URL, timeout=120):
    """
    使用 llama.cpp 视觉模型进行分析

    Args:
        image_data: 图片路径或 PIL Image
        prompt: 分析提示词
        model: 模型名称
        llamacpp_host: llama.cpp 服务器地址
        timeout: 超时时间（秒）

    Returns:
        dict: 分析结果
    """
    try:
        # 获取文件名（用于日志）
        if isinstance(image_data, str):
            filename = Path(image_data).name
        else:
            filename = "uploaded_image.png"

        # 编码图片
        image_base64 = encode_image_to_base64(image_data)
        if not image_base64:
            return {"success": False, "analysis": "❌ 图片编码失败"}

        # 构造请求
        # 兼容 llama.cpp 服务器的 OpenAI 格式 API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7
        }

        logger.info(f"正在调用 llama.cpp 模型：{model}")
        logger.info(f"分析图片：{filename}")

        # 发送请求到 llama.cpp 服务器
        response = requests.post(
            f"{llamacpp_host.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        # 解析响应
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            analysis_text = result["choices"][0]["message"]["content"]
            logger.info(f"分析完成，图片：{filename}")

            return {
                "success": True,
                "analysis": analysis_text,
                "model": model,
                "image_path": filename
            }
        else:
            logger.warning(f"llama.cpp 响应格式异常：{result}")
            return {
                "success": False,
                "analysis": f"⚠️ llama.cpp 响应格式异常\n\n原始响应：{result}",
                "error": "响应格式错误"
            }

    except requests.exceptions.ConnectionError as e:
        error_msg = f"""❌ **无法连接到 llama.cpp 服务**

可能原因：
1. llama.cpp 服务器未启动
2. llama.cpp 地址不是 {llamacpp_host}
3. 防火墙阻止了连接

解决方法：
1. 启动 llama.cpp 服务器：
   ```bash
   llama-server --model path/to/your/model.gguf --host 0.0.0.0 --port {DEFAULT_LLAMACPP_PORT}
   ```
2. 检查服务器地址设置
3. 关闭防火墙或添加例外规则

技术细节：{str(e)}
"""
        logger.error(f"llama.cpp 连接失败：{e}")
        return {
            "success": False,
            "analysis": error_msg,
            "error": str(e)
        }

    except requests.exceptions.Timeout:
        error_msg = f"""❌ **请求超时**

分析时间超过 {timeout} 秒，可能的原因：
1. 模型正在加载（首次运行需要加载模型到显存）
2. 图片分辨率过高
3. GPU 性能不足

建议：
1. 等待片刻后重试（首次加载模型需要时间）
2. 使用更小的图片尺寸（建议 1024x1024 以内）
3. 确保 GPU 有足够显存（推荐 8GB+）
"""
        logger.error("llama.cpp 请求超时")
        return {
            "success": False,
            "analysis": error_msg,
            "error": "timeout"
        }

    except Exception as e:
        logger.error(f"llama.cpp 分析失败：{e}")
        return {
            "success": False,
            "analysis": f"❌ 分析过程出错：{str(e)}\n\n请检查：\n1. llama.cpp 服务是否正常运行\n2. 模型是否已加载\n3. 图片文件是否正常",
            "error": str(e)
        }


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
        logger.debug(f"获取 llama.cpp 模型列表出错: {str(e)}")
        return []


def test_llamacpp_connection(llamacpp_host=DEFAULT_LLAMACPP_URL):
    """测试 llama.cpp 服务连接状态"""
    try:
        host = llamacpp_host.rstrip('/')
        # 尝试多个端点以兼容不同版本的 llama.cpp
        endpoints = [
            f"{host}/v1/models",
            f"{host}/health",
            f"{host}/",
        ]

        response = None
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    break
            except Exception:
                continue

        if response is not None and response.status_code == 200:
            # 尝试获取模型列表
            models = get_llamacpp_models(llamacpp_host)
            if models:
                return True, f"✅ llama.cpp 连接成功\n\n已加载的模型:\n" + "\n".join(models)
            else:
                return True, f"⚠️ llama.cpp 连接成功，但未检测到模型\n\n请确保已使用 --model 参数启动服务器并加载了视觉模型"
        else:
            return False, f"❌ llama.cpp 响应异常：{response.status_code if response else '无法连接'}"
    except Exception as e:
        return False, f"❌ 无法连接到 llama.cpp 服务\n\n错误信息：{str(e)}\n\n请确保:\n1. llama.cpp 服务器已启动\n2. 地址设置为 {llamacpp_host}"


# 为了兼容其他插件的导入，添加函数别名
def get_response_lvm_llamacpp_api(model_name, prompt, image_path, llamacpp_host=DEFAULT_LLAMACPP_URL, timeout=120):
    """兼容函数：调用视觉模型（get_response_lvm_llamacpp_api）"""
    result = analyze_with_llamacpp(image_path, prompt, model_name, llamacpp_host, timeout)
    return result.get("analysis", "")


def get_response_text_llamacpp_api(model_name, prompt, llamacpp_host=DEFAULT_LLAMACPP_URL, timeout=120):
    """兼容函数：调用语言模型（get_response_text_llamacpp_api）"""
    try:
        # 构造请求
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0.7
        }

        # 发送请求到 llama.cpp 服务器
        response = requests.post(
            f"{llamacpp_host.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        # 解析响应
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return ""
    except Exception as e:
        logger.error(f"llama.cpp 文本分析失败：{e}")
        return ""
