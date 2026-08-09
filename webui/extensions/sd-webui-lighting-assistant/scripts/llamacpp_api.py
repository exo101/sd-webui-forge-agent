"""
llama.cpp API 调用模块
支持视觉模型进行图片打光分析
"""

import requests
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)

# 默认配置（读取启动器共享端口配置，自动跟随启动器的 llama 端口）
try:
    from modules.llama_port import get_llama_port
    _port = get_llama_port()
except Exception:
    _port = 8080

DEFAULT_LLAMACPP_HOST = "localhost"
DEFAULT_LLAMACPP_PORT = _port
DEFAULT_LLAMACPP_URL = f"http://{DEFAULT_LLAMACPP_HOST}:{DEFAULT_LLAMACPP_PORT}"


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图片转换为 base64 编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"图片编码失败：{e}")
        return None


def analyze_lighting_with_llamacpp(
    image_path: str,
    prompt: str,
    model: str = "local-model",
    llamacpp_host: str = DEFAULT_LLAMACPP_URL,
    timeout: int = 120
) -> Dict[str, Any]:
    """
    使用 llama.cpp 视觉模型分析图片打光
    
    Args:
        image_path: 图片路径
        prompt: 分析提示词
        model: 模型名称
        llamacpp_host: llama.cpp 服务器地址
        timeout: 超时时间（秒）
    
    Returns:
        dict: 分析结果
    """
    try:
        # 编码图片
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return {"success": False, "analysis": "图片编码失败", "keywords": ""}

        # 构造请求 - OpenAI 兼容格式
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
            "temperature": 0.7,
            "max_tokens": 2000
        }

        logger.info(f"正在调用 llama.cpp 分析图片打光：{Path(image_path).name}")

        # 发送请求
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
            logger.info(f"打光分析完成")
            
            return {
                "success": True,
                "analysis": analysis_text,
                "model": model,
                "image_path": image_path
            }
        else:
            logger.warning(f"llama.cpp 响应格式异常：{result}")
            return {
                "success": False,
                "analysis": f"响应格式异常",
                "keywords": ""
            }

    except requests.exceptions.ConnectionError as e:
        error_msg = f"""无法连接到 llama.cpp 服务

可能原因：
1. llama.cpp 服务器未启动
2. 服务地址不是 {llamacpp_host}
3. 防火墙阻止了连接

解决方法：
启动 llama.cpp 服务器：
llama-server --model path/to/your/vision-model.gguf --host 0.0.0.0 --port {DEFAULT_LLAMACPP_PORT}
"""
        logger.error(f"llama.cpp 连接失败：{e}")
        return {"success": False, "analysis": error_msg, "keywords": ""}

    except requests.exceptions.Timeout:
        error_msg = f"请求超时（{timeout}秒），请检查模型是否已加载或使用更小的图片"
        logger.error("llama.cpp 请求超时")
        return {"success": False, "analysis": error_msg, "keywords": ""}

    except Exception as e:
        logger.error(f"llama.cpp 分析失败：{e}")
        return {"success": False, "analysis": f"分析出错：{str(e)}", "keywords": ""}


def get_llamacpp_models(llamacpp_host: str = DEFAULT_LLAMACPP_URL) -> List[str]:
    """获取 llama.cpp 加载的模型列表"""
    try:
        models_url = f"{llamacpp_host.rstrip('/')}/v1/models"
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = []
        
        # 兼容不同的响应格式
        if "models" in data and isinstance(data["models"], list):
            for m in data["models"]:
                if isinstance(m, dict):
                    if "id" in m:
                        models.append(m["id"])
                    elif "name" in m:
                        models.append(m["name"])
                elif isinstance(m, str):
                    models.append(m)
        
        if "data" in data and isinstance(data["data"], list):
            for m in data["data"]:
                if isinstance(m, dict):
                    if "id" in m:
                        models.append(m["id"])
                    elif "name" in m:
                        models.append(m["name"])
        
        return list(set(models)) if models else []
    except Exception:
        return []


def test_llamacpp_connection(llamacpp_host: str = DEFAULT_LLAMACPP_URL) -> tuple:
    """测试 llama.cpp 服务连接状态"""
    try:
        host = llamacpp_host.rstrip('/')
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
            models = get_llamacpp_models(llamacpp_host)
            if models:
                return True, f"连接成功\n已加载模型:\n" + "\n".join(models)
            else:
                return True, "连接成功，但未检测到模型"
        else:
            return False, f"响应异常：{response.status_code if response else '无法连接'}"
    except Exception as e:
        return False, f"无法连接到 llama.cpp 服务\n错误：{str(e)}"


def get_lighting_analysis_prompt() -> str:
    """获取打光分析专用提示词"""
    return """你是一位专业的摄影灯光师和视觉艺术家。请分析这张图片的打光方式，并提供详细的打光关键词。

请按以下格式输出：

## 打光分析

### 1. 主光方向
分析主光源的方向（正面光/侧光/逆光/顶光/底光/伦勃朗光等）

### 2. 光线质量
分析光线是硬光还是柔光，是否有漫射

### 3. 光影效果
分析阴影的位置、形状、软硬度

### 4. 光线颜色
分析光线的色温（暖色/冷色/中性色）

### 5. 特殊光效
是否有丁达尔光、光晕、轮廓光等特殊效果

### 6. 氛围感
光线营造的整体氛围（戏剧性/柔和/神秘/明亮等）

## 打光关键词（Prompt）

请生成适合 Stable Diffusion 使用的纯中文打光关键词，用逗号分隔，格式如下：

```
正面光，侧向打光，柔和光照，暖光，体积光，电影光，高对比，明暗对照
```

关键词应包含：
- 光源方向：如 "正面光，侧向打光，逆光，侧逆光"
- 光线质量：如 "柔和光照，硬光，柔光，聚光"
- 光影效果：如 "戏剧阴影，高对比，低对比，明暗对照"
- 光线颜色：如 "暖光，冷光，霓虹光，烛光，月光"
- 特殊效果：如 "丁达尔光，体积光，光晕，轮廓光，眼神光"
- 氛围：如 "电影光，舞台光，赛博朋克光，戏剧性"

请确保关键词简洁、专业、适合 AI 绘画使用，使用纯中文关键词。"""


def extract_keywords_from_analysis(analysis_text: str) -> str:
    """从分析结果中提取关键词"""
    try:
        # 尝试提取 ``` 代码块中的内容
        if "```" in analysis_text:
            parts = analysis_text.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # 奇数索引是代码块内容
                    # 移除可能的语言标识符
                    lines = part.strip().split('\n')
                    if lines[0] in ['python', 'text', 'plaintext', '']:
                        keywords = '\n'.join(lines[1:])
                    else:
                        keywords = part.strip()
                    return keywords.strip()
        
        # 如果没有代码块，尝试提取 "关键词" 相关部分
        if "关键词" in analysis_text or "Keywords" in analysis_text.lower():
            lines = analysis_text.split('\n')
            for line in lines:
                if "关键词" in line or "Keywords" in line.lower() or "prompt" in line.lower():
                    # 返回下一行或当前行冒号后的内容
                    idx = lines.index(line)
                    if idx + 1 < len(lines):
                        return lines[idx + 1].strip()
                    elif ':' in line:
                        return line.split(':', 1)[1].strip()
        
        return ""
    except Exception as e:
        logger.error(f"提取关键词失败：{e}")
        return ""
