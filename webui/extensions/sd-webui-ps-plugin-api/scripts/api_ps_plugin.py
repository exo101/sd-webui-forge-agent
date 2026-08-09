'''
SD WebUI PS Plugin API
为 PS 插件提供 REST API 端点，支持 LoRA 查询、图像分析、抠图等功能
通过 script_callbacks.on_app_started 注册到 FastAPI
'''
import base64
import copy
import io
import json
import os
import subprocess
import sys
import time
from typing import Optional
import gradio as gr
import numpy as np
import requests
import torch
from fastapi import FastAPI
from PIL import Image, ImageFilter
from modules import script_callbacks, shared

LORA_DIR = os.path.join(shared.models_path, 'Lora')


def decode_base64_to_image(base64_str: str = None) -> Image.Image:
    '''将 base64 字符串解码为 PIL Image'''
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def encode_image_to_base64(img: Image.Image = None) -> str:
    '''将 PIL Image 编码为 base64 字符串'''
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def image_to_base64_datauri(img: Image.Image = None) -> str:
    '''将 PIL Image 编码为 base64 data URI'''
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def call_llama_vision(image_base64: str = None, prompt: str = None) -> dict:
    '''调用 llama.cpp 视觉模型进行图像分析（自动获取端口）'''
    from modules.llama_port import get_llama_url
    if not image_base64.startswith('data:'):
        image_data = f'data:image/png;base64,{image_base64}'
    else:
        image_data = image_base64
    url = get_llama_url().rstrip('/') + '/v1/chat/completions'
    payload = {
        'model': 'gpt-4-vision-preview',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_data,
                            'detail': 'high'
                        }
                    }
                ]
            }
        ],
        'max_tokens': 4096,
        'temperature': 0.7
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    raw_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
    return {'raw': raw_text}


def parse_llama_proportion_response(raw: str = None) -> dict:
    '''解析人体比例分析返回结果'''
    result = {
        'raw': raw,
        'proportion': '',
        'style': '',
        'negative': ''
    }
    if not raw:
        return result
    import re
    lines = raw.strip().split('\n')
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith('proportion:') or line_lower.startswith('proportion：'):
            if ':' in line:
                result['proportion'] = line.split(':', 1)[1].strip()
            else:
                result['proportion'] = line.split('：', 1)[1].strip()
            continue
        if line_lower.startswith('style:') or line_lower.startswith('style：'):
            if ':' in line:
                result['style'] = line.split(':', 1)[1].strip()
            else:
                result['style'] = line.split('：', 1)[1].strip()
            continue
        if line_lower.startswith('negative:') or line_lower.startswith('negative：'):
            if ':' in line:
                result['negative'] = line.split(':', 1)[1].strip()
            else:
                result['negative'] = line.split('：', 1)[1].strip()
    if not result['proportion'] and 'proportion' in raw.lower():
        match = re.search(r'(?:proportion|比例)[：:\s]*(.*?)(?:\n|$)', raw, re.IGNORECASE)
        if match:
            result['proportion'] = match.group(1).strip()
    if not result['style'] and 'style' in raw.lower():
        match = re.search(r'(?:style|风格)[：:\s]*(.*?)(?:\n|$)', raw, re.IGNORECASE)
        if match:
            result['style'] = match.group(1).strip()
    if not result['negative'] and 'negative' in raw.lower():
        match = re.search(r'(?:negative|负面)[：:\s]*(.*?)(?:\n|$)', raw, re.IGNORECASE)
        if match:
            result['negative'] = match.group(1).strip()
    return result


def parse_llama_scene_response(raw: str = None) -> dict:
    '''解析场景分析返回结果'''
    result = {
        'raw': raw,
        'composition': '',
        'camera': '',
        'lighting': '',
        'positive': '',
        'negative': ''
    }
    if not raw:
        return result
    import re
    fields = {
        'composition': ['composition', '构图'],
        'camera': ['camera', '镜头', '视角'],
        'lighting': ['lighting', '光照', '光线'],
        'positive': ['positive', '正向', 'prompt'],
        'negative': ['negative', '负面']
    }
    for key, keywords in fields.items():
        for kw in keywords:
            pattern = fr'(?:{kw})[：:\s]*(.*?)(?:\n|$)'
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
    return result


def parse_llama_design_response(raw: str = None) -> dict:
    '''解析设计分析返回结果'''
    result = {
        'raw': raw,
        'colors': '',
        'shapes': '',
        'typography': '',
        'style': '',
        'creativity': '',
        'suggestions': '',
        'positive': '',
        'negative': ''
    }
    if not raw:
        return result
    import re
    fields = {
        'colors': ['colors', '颜色', '色彩'],
        'shapes': ['shapes', '形状', '图形'],
        'typography': ['typography', '字体', '排版'],
        'style': ['style', '风格'],
        'creativity': ['creativity', '创意'],
        'suggestions': ['suggestions', '建议', '改进'],
        'positive': ['positive', '正向', 'prompt'],
        'negative': ['negative', '负面']
    }
    for key, keywords in fields.items():
        for kw in keywords:
            pattern = fr'(?:{kw})[：:\s]*(.*?)(?:\n|$)'
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
    return result


def apply_bg_color(img: Image.Image = None, bg_color: str = None) -> Image.Image:
    '''将 RGBA 图像替换为指定背景色，返回 RGB'''
    bg_color = bg_color.lstrip('#')
    if len(bg_color) >= 6:
        r = int(bg_color[0:2], 16)
        g = int(bg_color[2:4], 16)
        b = int(bg_color[4:6], 16)
        background = Image.new('RGBA', img.size, (r, g, b, 255))
        background.paste(img, mask=img.split()[3])
        return background.convert('RGB')
    return img.convert('RGB')


def process_birefnet(img: Image.Image = None, model_type: str = None) -> Image.Image:
    '''使用 BiRefNet 模型处理单张图像，返回 RGBA'''
    import torch
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    from transformers import AutoModelForImageSegmentation
    birefnet_dir = os.path.join(shared.data_path, 'models', 'BiRefNet')
    os.makedirs(birefnet_dir, exist_ok=True)
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'birefnet-matting':
        model_name = 'ZhengPeng7/BiRefNet_HR'
    else:
        model_name = 'ZhengPeng7/BiRefNet'
    birefnet = AutoModelForImageSegmentation.from_pretrained(model_name, cache_dir=birefnet_dir, trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval()
    img_rgb = img.convert('RGB')
    input_tensor = transform_image(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = birefnet(input_tensor)[-1].sigmoid().cpu()
    pred = F.interpolate(pred, size=img.size[::-1], mode='bilinear', align_corners=False)
    mask = pred[0, 0].numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask, mode='L')
    img_rgba = img.convert('RGBA')
    img_rgba.putalpha(mask_pil)
    return img_rgba


def process_inspyrenet(img: Image.Image = None) -> Image.Image:
    '''使用 InSPyReNet 模型处理单张图像，返回 RGBA'''
    import torch
    import numpy as np
    from transparent_background import Remover
    inspyrenet_dir = os.path.join(shared.data_path, 'models', 'InSPyReNet')
    os.makedirs(inspyrenet_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['TRANSPARENT_BACKGROUND_MODELS'] = inspyrenet_dir
    remover = Remover(device=device)
    img_rgba = remover.process(img)
    return img_rgba


def ps_plugin_api(_: gr.Blocks, app: FastAPI):
    '''注册 PS 插件 API 端点'''

    # 修复 Pydantic v2.10+ 兼容性问题
    # 内核 StableDiffusionProcessing 使用了 from __future__ import annotations
    # 导致所有注解变为字符串（如 "ForgeDiffusionEngine"、"torch.Tensor"、"Image"）
    # 需要：1) 提供 _types_namespace 解析前向引用 2) 设置 arbitrary_types_allowed 允许 torch.Tensor 等类型
    import torch
    from backend.diffusion_engine.base import ForgeDiffusionEngine
    from PIL import Image as PIL_Image
    from modules.api import models as api_models

    for model in (api_models.StableDiffusionTxt2ImgProcessingAPI, api_models.StableDiffusionImg2ImgProcessingAPI):
        model.model_config['arbitrary_types_allowed'] = True
        model.model_rebuild(
            _types_namespace={
                'torch': torch,
                'ForgeDiffusionEngine': ForgeDiffusionEngine,
                'Image': PIL_Image,
            }
        )

    @app.get('/sdapi/v1/ps-plugin/loras')
    def list_loras():
        '''
        列出所有 LoRA 文件
        扫描 models/Lora/ 目录下的 .safetensors 文件
        '''
        loras = []
        lora_dir = LORA_DIR
        if not os.path.isdir(lora_dir):
            return []
        try:
            for fname in sorted(os.listdir(lora_dir)):
                if fname.endswith('.safetensors'):
                    name = os.path.splitext(fname)[0]
                    loras.append({'name': name, 'filename': fname})
        except Exception:
            return []
        return loras

    @app.post('/sdapi/v1/ps-plugin/analyze-body-design')
    def analyze_body_design(body: dict):
        '''
        合并：人体比例 + 设计元素分析（一次 llama.cpp 调用）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'raw': '', 'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_base64 = encode_image_to_base64(img)
        except Exception:
            return {'raw': '', 'error': '图片解码失败，请提供有效的 base64 编码图片'}
        prompt = (
            '请详细分析这张图片中的人物身体比例、风格和设计元素。\n'
            '请按以下格式输出，每行以关键词开头：\n'
            'Proportion: 描述身体比例（如：九头身、五五身等）\n'
            'Style: 艺术风格和正向关键词，逗号分隔\n'
            'Negative: 负面提示词，描述不适合的元素\n'
            'Colors: 色彩方案，包括主色调、辅助色、点缀色，以及色彩搭配关系\n'
            'Shapes: 设计形状和图形元素，如几何图形、有机形态、对称等\n'
            'Typography: 排版与布局特点，如字体风格、对齐方式、留白运用\n'
            'Creativity: 创意和独特性，如创新的构图、独特的视角等\n'
            '请确保输出格式准确，每行以英文关键词开头。'
        )
        result = call_llama_vision(image_base64, prompt)
        if 'error' in result:
            return result
        raw = result.get('raw', '')
        # 合并解析：人体比例 + 设计元素
        parsed = parse_llama_proportion_response(raw)
        design = parse_llama_design_response(raw)
        parsed.update(design)
        parsed['raw'] = raw
        return parsed

    @app.post('/sdapi/v1/ps-plugin/analyze-scene-lighting')
    def analyze_scene_lighting(body: dict):
        '''
        合并：场景构图 + 打光分析（一次 llama.cpp 调用）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'raw': '', 'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_base64 = encode_image_to_base64(img)
        except Exception:
            return {'raw': '', 'error': '图片解码失败，请提供有效的 base64 编码图片'}
        prompt = (
            '你是一位专业的摄影灯光师和视觉艺术家。请详细分析这张图片的场景构图和打光方式。\n'
            '请按以下格式输出，每行以关键词开头：\n'
            'Composition: 描述构图方式（如：三分法、对称构图、引导线构图等）\n'
            'Camera: 描述镜头角度和距离（如：广角镜头、俯拍、特写等）\n'
            'Lighting: 描述光照效果（如：自然光、侧光、逆光、暖色调等）\n'
            'Positive: 正向提示词，适合用于 AI 图像生成的场景描述关键词\n'
            'Negative: 负面提示词，描述不适合的场景元素\n'
            'LightingKeywords: 打光关键词，用逗号分隔，如：正面光，侧向打光，柔和光照，暖光\n'
            '请确保输出格式准确，每行以英文关键词开头。'
        )
        result = call_llama_vision(image_base64, prompt)
        if 'error' in result:
            return result
        raw = result.get('raw', '')
        # 合并解析：场景构图 + 打光分析
        parsed = parse_llama_scene_response(raw)
        # 额外提取打光关键词
        keywords = ''
        import re
        for line in raw.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith('lightingkeywords:') or line_lower.startswith('lightingkeywords：'):
                if ':' in line:
                    keywords = line.split(':', 1)[1].strip()
                else:
                    keywords = line.split('：', 1)[1].strip()
                break
        if not keywords:
            # 尝试从 Lighting 字段中提取关键词
            match = re.search(r'(?:lighting|光照|打光)[：:\s]*(.*?)(?:\n|$)', raw, re.IGNORECASE)
            if match:
                keywords = match.group(1).strip()
        parsed['lighting_keywords'] = keywords
        parsed['raw'] = raw
        return parsed

    @app.post('/sdapi/v1/ps-plugin/analyze-llama')
    def analyze_llama(body: dict):
        '''
        使用 llama.cpp 视觉模型分析人体比例（端口跟随启动器配置）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'raw': '', 'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_base64 = encode_image_to_base64(img)
        except Exception:
            return {'raw': '', 'error': '图片解码失败，请提供有效的 base64 编码图片'}
        prompt = '请详细分析这张图片中的人物身体比例、姿势和风格。\n请按以下格式输出：\nProportion: 描述身体比例（如：九头身、五五身等）\nStyle: 描述风格关键词，适合用于 AI 图像生成的提示词（如：anime style, detailed, masterpiece, best quality）\nNegative: 负面提示词，描述不适合的元素（如：bad anatomy, extra limbs, distorted）\n请确保输出格式准确，每行以关键词开头。'
        result = call_llama_vision(image_base64, prompt)
        if 'error' in result:
            return result
        return parse_llama_proportion_response(result.get('raw', ''))

    @app.post('/sdapi/v1/ps-plugin/analyze-scene')
    def analyze_scene(body: dict):
        '''
        使用 llama.cpp 视觉模型分析场景构图（端口跟随启动器配置）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'raw': '', 'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_base64 = encode_image_to_base64(img)
        except Exception:
            return {'raw': '', 'error': '图片解码失败，请提供有效的 base64 编码图片'}
        prompt = '请详细分析这张图片的场景构图。\n请按以下格式输出：\nComposition: 描述构图方式（如：三分法、对称构图、引导线构图等）\nCamera: 描述镜头角度和距离（如：广角镜头、俯拍、特写等）\nLighting: 描述光照效果（如：自然光、侧光、逆光、暖色调等）\nPositive: 正向提示词，适合用于 AI 图像生成的场景描述关键词\nNegative: 负面提示词，描述不适合的场景元素\n请确保输出格式准确，每行以关键词开头。'
        result = call_llama_vision(image_base64, prompt)
        if 'error' in result:
            return result
        return parse_llama_scene_response(result.get('raw', ''))

    @app.post('/sdapi/v1/ps-plugin/analyze-lighting')
    def analyze_lighting(body: dict):
        '''
        使用 llama.cpp 视觉模型分析图片打光方式（端口跟随启动器配置）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'raw': '', 'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_base64 = encode_image_to_base64(img)
        except Exception:
            return {'raw': '', 'error': '图片解码失败，请提供有效的 base64 编码图片'}
        prompt = '你是一位专业的摄影灯光师和视觉艺术家。请分析这张图片的打光方式，并提供详细的打光关键词。\n\n请按以下格式输出：\n\n## 打光分析\n\n### 1. 主光方向\n分析主光源的方向（正面光/侧光/逆光/顶光/底光/伦勃朗光等）\n\n### 2. 光线质量\n分析光线是硬光还是柔光，是否有漫射\n\n### 3. 光影效果\n分析阴影的位置、形状、软硬度\n\n### 4. 光线颜色\n分析光线的色温（暖色/冷色/中性色）\n\n### 5. 特殊光效\n是否有丁达尔光、光晕、轮廓光等特殊效果\n\n### 6. 氛围感\n光线营造的整体氛围（戏剧性/柔和/神秘/明亮等）\n\n## 打光关键词（Prompt）\n\n请生成适合 Stable Diffusion 使用的纯中文打光关键词，用逗号分隔，格式如下：\n\n```\n正面光，侧向打光，柔和光照，暖光，体积光，电影光，高对比，明暗对照\n```\n\n关键词应包含：\n- 光源方向：如 "正面光，侧向打光，逆光，侧逆光"\n- 光线质量：如 "柔和光照，硬光，柔光，聚光"\n- 光影效果：如 "戏剧阴影，高对比，低对比，明暗对照"\n- 光线颜色：如 "暖光，冷光，霓虹光，烛光，月光"\n- 特殊效果：如 "丁达尔光，体积光，光晕，轮廓光，眼神光"\n- 氛围：如 "电影光，舞台光，赛博朋克光，戏剧性"\n\n请确保关键词简洁、专业、适合 AI 绘画使用，使用纯中文关键词。'
        result = call_llama_vision(image_base64, prompt)
        if 'error' in result:
            return result
        raw_text = result.get('raw', '')
        # 提取关键词部分
        keywords = ''
        if '```' in raw_text:
            parts = raw_text.split('```')
            if len(parts) >= 2:
                for i, part in enumerate(parts):
                    part = part.strip()
                    lines = part.split('\n')
                    if lines and not any(x in lines[0].lower() for x in ('', 'text', 'plaintext', 'python')):
                        keywords = part
                        break
                if not keywords:
                    keywords = parts[1].strip()
        else:
            # 尝试从文本中提取关键词行
            import re
            for line in raw_text.split('\n'):
                line = line.strip()
                if re.search(r'(关键词|Keywords|Prompt|prompt)', line):
                    continue
                if line and len(line) > 5:
                    keywords = line
                    break
        return {'raw': raw_text, 'analysis': raw_text, 'keywords': keywords}

    @app.post('/sdapi/v1/ps-plugin/analyze-design')
    def analyze_design(body: dict):
        '''
        使用 llama.cpp 视觉模型分析设计元素（端口跟随启动器配置）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'raw': '', 'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image_base64 = encode_image_to_base64(img)
        except Exception:
            return {'raw': '', 'error': '图片解码失败，请提供有效的 base64 编码图片'}
        prompt = '请详细分析这张图片的设计元素和视觉构成。\n请按以下格式输出：\nColors: 描述主要颜色和配色方案（如：互补色、类似色、冷暖对比等）\nShapes: 描述主要形状和图形元素（如：几何图形、有机形态、对称等）\nTypography: 描述字体和排版风格（如：衬线字体、无衬线字体、手写体等）\nStyle: 描述整体设计风格（如：极简主义、扁平设计、赛博朋克等）\nCreativity: 描述创意和独特性（如：创新的构图、独特的视角等）\nSuggestions: 改进建议和优化方向\nPositive: 正向提示词，适合用于 AI 图像生成的设计关键词\nNegative: 负面提示词，描述不适合的设计元素\n请确保输出格式准确，每行以关键词开头。'
        result = call_llama_vision(image_base64, prompt)
        if 'error' in result:
            return result
        return parse_llama_design_response(result.get('raw', ''))

    @app.post('/sdapi/v1/ps-plugin/matting')
    def matting(body: dict):
        '''
        去除图片背景
        请求体: { "image_base64": "...", "model": "birefnet-general", "bg_color": "#FFFFFF" }
        - model: birefnet-general / birefnet-matting / inspyrenet-base / u2net / u2netp / u2net_human_seg / silueta
        - bg_color: 背景颜色十六进制，不传则透明背景
        '''
        image_base64 = body.get('image_base64', '')
        model = body.get('model', 'birefnet-general')
        bg_color = body.get('bg_color', '')
        if not image_base64:
            return {'error': '缺少 image_base64 参数'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
        except Exception:
            return {'error': '图片解码失败，请提供有效的 base64 编码图片'}
        try:
            model_lower = model.lower()
            if 'birefnet' in model_lower:
                result_img = process_birefnet(img, model_lower)
            elif 'inspyrenet' in model_lower:
                result_img = process_inspyrenet(img)
            else:
                # rembg 模型
                import rembg
                os.environ['U2NET_HOME'] = os.path.join(shared.data_path, 'models', 'rembg')
                session = rembg.new_session(model)
                result_img = rembg.remove(img, session=session)
        except Exception as e:
            return {'error': f'抠图处理失败: {str(e)}'}
        if bg_color:
            result_img = apply_bg_color(result_img, bg_color)
        return {'image_base64': encode_image_to_base64(result_img)}

    @app.post('/sdapi/v1/ps-plugin/see-through')
    def see_through(body: dict):
        '''
        调用 See-Through 图层分离（使用 sd-webui-see-through 插件）
        请求体: { "image_base64": "...", "mode": "人物分割", "resolution": 1024,
                  "use_nf4": true, "num_inference_steps": 50 }
        场景分割模式额外参数: { "max_masks": 10, "min_area": 1000, "model_type": "vit_b" }
        '''
        image_base64 = body.get('image_base64', '')
        mode = body.get('mode', '人物分割')
        resolution = body.get('resolution', 1024)
        use_nf4 = body.get('use_nf4', True)
        num_inference_steps = body.get('num_inference_steps', 50)
        tblr_split = body.get('tblr_split', False)
        max_masks = body.get('max_masks', 10)
        min_area = body.get('min_area', 1000)
        model_type = body.get('model_type', 'vit_b')
        if not image_base64:
            return {'status': 'error', 'message': '缺少 image_base64 参数', 'output_dir': ''}
        try:
            img = decode_base64_to_image(image_base64)
        except Exception:
            return {'status': 'error', 'message': '图片解码失败', 'output_dir': ''}
        temp_dir = os.path.join(shared.data_path, 'temp', 'see-through')
        os.makedirs(temp_dir, exist_ok=True)
        temp_input = os.path.join(temp_dir, 'input.png')
        img.save(temp_input, format='PNG')
        see_through_ext = os.path.join(shared.extensions_dir, 'sd-webui-see-through')
        see_through_scripts_dir = os.path.join(see_through_ext, 'see-through', 'inference', 'scripts')
        if not os.path.isdir(see_through_ext):
            return {'status': 'error', 'message': 'sd-webui-see-through 插件未安装，请先安装该插件', 'output_dir': ''}
        if mode == '人物分割':
            script_path = os.path.join(see_through_scripts_dir, 'inference_psd_optimized.py')
            if not os.path.exists(script_path):
                return {'status': 'error', 'message': '推理脚本不存在', 'output_dir': ''}
            diffusers_dir = os.path.join(shared.data_path, 'models', 'diffusers')
            nf4_layerdiff_snapshot = os.path.join(diffusers_dir, 'models--24yearsold--seethroughv0.0.2_layerdiff3d_nf4', 'snapshots', '39b9881340189810bebabe5462756fb2e8fbd5fa')
            nf4_depth_snapshot = os.path.join(diffusers_dir, 'models--24yearsold--seethroughv0.0.1_marigold_nf4', 'snapshots', 'aad13aafa9f3c72defb40a9d9225cec70b0eab16')
            none_layerdiff_snapshot = os.path.join(diffusers_dir, 'models--layerdifforg--seethroughv0.0.2_layerdiff3d', 'snapshots', '4e30246c06819af6c151dd07ca17dbefb741f42a')
            none_depth_snapshot = os.path.join(diffusers_dir, 'models--24yearsold--seethroughv0.0.1_marigold', 'snapshots', 'aa7a892f83ff68d7b09186a405ba08d5d33f770f')
            cmd = [
                sys.executable, script_path,
                '--srcp', temp_input,
                '--save_dir', temp_dir,
                '--resolution', str(resolution),
                '--num_inference_steps', str(num_inference_steps),
                '--save_to_psd'
            ]
            if use_nf4:
                cmd.extend(['--quant_mode', 'nf4'])
                cmd.extend(['--repo_id_layerdiff', nf4_layerdiff_snapshot])
                cmd.extend(['--repo_id_depth', nf4_depth_snapshot])
            else:
                cmd.extend(['--quant_mode', 'none'])
                cmd.extend(['--repo_id_layerdiff', none_layerdiff_snapshot])
                cmd.extend(['--repo_id_depth', none_depth_snapshot])
            if tblr_split:
                cmd.append('--tblr_split')
            env = os.environ.copy()
            env['HF_HOME'] = diffusers_dir
            env['HF_HUB_CACHE'] = diffusers_dir
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            if result.returncode == 0:
                psd_b64 = ''
                if os.path.isdir(temp_dir):
                    psd_files = [f for f in os.listdir(temp_dir) if f.endswith('.psd')]
                    if psd_files:
                        psd_path = os.path.join(temp_dir, psd_files[0])
                        with open(psd_path, 'rb') as pf:
                            psd_b64 = base64.b64encode(pf.read()).decode('utf-8')
                return {'status': 'success', 'message': '图层分离完成', 'output_dir': temp_dir, 'psd_base64': psd_b64}
            else:
                return {'status': 'error', 'message': '图层分离失败: ' + result.stderr[-500:], 'output_dir': ''}
        else:
            # 场景分割模式
            script_path = os.path.join(see_through_scripts_dir, 'scene_segmenter.py')
            if not os.path.exists(script_path):
                return {'status': 'error', 'message': '场景分割脚本不存在', 'output_dir': ''}
            cmd = [
                sys.executable, script_path,
                '--srcp', temp_input,
                '--save_dir', temp_dir,
                '--resolution', str(resolution),
                '--model_type', model_type,
                '--min_area', str(min_area),
                '--max_masks', str(max_masks),
                '--models_dir', shared.models_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                psd_b64 = ''
                if os.path.isdir(temp_dir):
                    psd_files = [f for f in os.listdir(temp_dir) if f.endswith('.psd')]
                    if psd_files:
                        psd_path = os.path.join(temp_dir, psd_files[0])
                        with open(psd_path, 'rb') as pf:
                            psd_b64 = base64.b64encode(pf.read()).decode('utf-8')
                return {'status': 'success', 'message': '场景分割完成', 'output_dir': temp_dir, 'psd_base64': psd_b64}
            else:
                return {'status': 'error', 'message': '场景分割失败: ' + result.stderr[-500:], 'output_dir': ''}

    @app.post('/sdapi/v1/ps-plugin/point-segmentation')
    def point_segmentation(body: dict):
        '''
        点选分割 - 基于 SAM 的标记点分割
        请求体: { "image_base64": "...", "points": [[x1,y1], [x2,y2], ...],
                  "model_type": "vit_h" }
        '''
        image_base64 = body.get('image_base64', '')
        points = body.get('points', [])
        model_type = body.get('model_type', 'vit_h')
        if not image_base64:
            return {'error': '缺少 image_base64 参数'}
        if not points or len(points) == 0:
            return {'error': '请至少添加一个标记点'}
        try:
            img = decode_base64_to_image(image_base64)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
        except Exception:
            return {'error': '图片解码失败'}
        try:
            result_images = process_point_segmentation(img, points, model_type)
            if not result_images:
                return {'error': '未能生成有效的分割结果'}
            result_b64 = [encode_image_to_base64(rimg) for rimg in result_images]
            return {'masks': result_b64, 'count': len(result_b64)}
        except Exception as e:
            return {'error': f'点选分割失败: {str(e)}'}

    @app.post('/sdapi/v1/ps-plugin/crop-and-mask')
    def crop_and_mask(body: dict):
        '''
        裁剪图像到选区范围并生成蒙版
        请求体: { "image_base64": "...", "left": 100, "top": 100, "width": 200, "height": 200,
                  "target_width": 512, "target_height": 512,
                  "mask_base64": "...",       # 可选，自定义蒙版
                  "mask_expand": 0,            # 可选，蒙版扩展像素数
                  "mask_blur": 4 }             # 可选，蒙版模糊半径
        - 将图像裁剪到选区范围
        - 缩放到目标尺寸
        - 如果提供了 mask_base64，使用该蒙版；否则生成全白蒙版
        - 应用蒙版扩展（膨胀）和蒙版模糊
        '''
        image_base64 = body.get('image_base64', '')
        left = int(body.get('left', 0))
        top = int(body.get('top', 0))
        width = int(body.get('width', 0))
        height = int(body.get('height', 0))
        target_width = int(body.get('target_width', 512))
        target_height = int(body.get('target_height', 512))
        mask_base64 = body.get('mask_base64', '')
        mask_expand = int(body.get('mask_expand', 0))
        mask_blur = int(body.get('mask_blur', 0))
        if not image_base64:
            return {'error': '缺少 image_base64 参数'}
        if width <= 0 or height <= 0:
            return {'error': '选区尺寸无效'}
        img = decode_base64_to_image(image_base64)
        img_w, img_h = img.size
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(img_w, left + width)
        crop_bottom = min(img_h, top + height)
        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        resized = cropped.resize((target_width, target_height), Image.LANCZOS)
        if mask_base64:
            mask_img = decode_base64_to_image(mask_base64)
            mask_cropped = mask_img.crop((crop_left, crop_top, crop_right, crop_bottom))
            mask = mask_cropped.resize((target_width, target_height), Image.LANCZOS)
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask = mask.point(lambda x: 255 if x > 128 else 0)
        else:
            mask = Image.new('L', (target_width, target_height), 255)
        if mask_expand > 0:
            for _ in range(mask_expand):
                mask = mask.filter(ImageFilter.MaxFilter(3))
        if mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
        out_img_b64 = encode_image_to_base64(resized.convert('RGB'))
        out_mask_b64 = encode_image_to_base64(mask)
        return {
            'image_base64': out_img_b64,
            'mask_base64': out_mask_b64,
            'cropped_width': target_width,
            'cropped_height': target_height
        }

    # ===== Bridge 端点 =====

    @app.post('/sdapi/v1/ps-plugin/bridge/send-blender-to-ps')
    def send_blender_to_ps(body: dict):
        '''
        从 Blender 发送图像到 PS（通过 WebUI 中转）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'status': 'error', 'message': '缺少 image_base64 参数'}
        try:
            ps_dir = os.path.join(shared.data_path, 'temp', 'ps-bridge')
            os.makedirs(ps_dir, exist_ok=True)
            # 清除旧文件
            for f in os.listdir(ps_dir):
                fp = os.path.join(ps_dir, f)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            ts = int(time.time() * 1000)
            img_path = os.path.join(ps_dir, f'blender_to_ps_{ts}.png')
            img_data = base64.b64decode(image_base64)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            return {'status': 'success', 'message': '图像已保存，PS 可获取'}
        except Exception as e:
            return {'status': 'error', 'message': f'保存图像失败: {str(e)}'}

    @app.get('/sdapi/v1/ps-plugin/bridge/receive-blender-to-ps')
    def receive_blender_to_ps():
        '''
        PS 获取 Blender 发送的图像
        返回: { "image_base64": "...", "filename": "..." }
        '''
        try:
            ps_dir = os.path.join(shared.data_path, 'temp', 'ps-bridge')
            if not os.path.isdir(ps_dir):
                return {'status': 'error', 'message': '没有可用的图像'}
            img_files = [f for f in os.listdir(ps_dir) if f.endswith('.png')]
            if not img_files:
                return {'status': 'error', 'message': '没有可用的图像'}
            img_files.sort(key=lambda f: os.path.getmtime(os.path.join(ps_dir, f)), reverse=True)
            latest = img_files[0]
            img_path = os.path.join(ps_dir, latest)
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            return {'status': 'success', 'image_base64': img_base64, 'filename': latest}
        except Exception as e:
            return {'status': 'error', 'message': f'读取图像失败: {str(e)}'}

    @app.post('/sdapi/v1/ps-plugin/bridge/send-ps-to-blender')
    def send_ps_to_blender(body: dict):
        '''
        从 PS 发送图像到 Blender（通过 WebUI 中转）
        请求体: { "image_base64": "...", "metadata": {} }
        '''
        image_base64 = body.get('image_base64', '')
        metadata = body.get('metadata', {})
        if not image_base64:
            return {'status': 'error', 'message': '缺少 image_base64 参数'}
        try:
            ps_dir = os.path.join(shared.data_path, 'temp', 'ps-bridge')
            os.makedirs(ps_dir, exist_ok=True)
            # 清除旧文件
            for f in os.listdir(ps_dir):
                fp = os.path.join(ps_dir, f)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            ts = int(time.time() * 1000)
            img_path = os.path.join(ps_dir, f'ps_to_blender_{ts}.png')
            meta_path = os.path.join(ps_dir, f'ps_to_blender_{ts}.json')
            img_data = base64.b64decode(image_base64)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            return {'status': 'success', 'message': '图像已保存，Blender 可获取'}
        except Exception as e:
            return {'status': 'error', 'message': f'保存图像失败: {str(e)}'}

    @app.get('/sdapi/v1/ps-plugin/bridge/receive-ps-to-blender')
    def receive_ps_to_blender():
        '''
        Blender 获取 PS 发送的图像
        返回: { "image_base64": "...", "metadata": {}, "filename": "..." }
        '''
        try:
            ps_dir = os.path.join(shared.data_path, 'temp', 'ps-bridge')
            if not os.path.isdir(ps_dir):
                return {'status': 'error', 'message': '没有可用的图像'}
            # 查找 ps_to_blender 开头的文件
            img_files = [f for f in os.listdir(ps_dir) if f.startswith('ps_to_blender') and f.endswith('.png')]
            if not img_files:
                return {'status': 'error', 'message': '没有可用的图像'}
            img_files.sort(key=lambda f: os.path.getmtime(os.path.join(ps_dir, f)), reverse=True)
            latest = img_files[0]
            img_path = os.path.join(ps_dir, latest)
            # 读取对应的 metadata
            meta_path = img_path.replace('.png', '.json')
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            return {'status': 'success', 'image_base64': img_base64, 'metadata': metadata, 'filename': latest}
        except Exception as e:
            return {'status': 'error', 'message': f'读取图像失败: {str(e)}'}

    @app.post('/sdapi/v1/ps-plugin/bridge/send-webui-glb-to-blender')
    def send_glb_to_blender(body: dict):
        '''
        从 WebUI 发送 GLB 模型数据到 Blender（通过桥接）
        请求体: { "glb_base64": "...", "filename": "model.glb" }
        '''
        glb_base64 = body.get('glb_base64', '')
        filename = body.get('filename', 'model.glb')
        if not glb_base64:
            return {'status': 'error', 'message': '缺少 glb_base64 参数'}
        try:
            # 保存 GLB 文件到临时目录
            glb_dir = os.path.join(shared.data_path, 'temp', 'glb-bridge')
            os.makedirs(glb_dir, exist_ok=True)
            glb_path = os.path.join(glb_dir, filename)
            glb_data = base64.b64decode(glb_base64)
            with open(glb_path, 'wb') as f:
                f.write(glb_data)
            return {'status': 'success', 'message': 'GLB 文件已保存', 'path': glb_path}
        except Exception as e:
            return {'status': 'error', 'message': f'保存 GLB 文件失败: {str(e)}'}

    @app.post('/sdapi/v1/ps-plugin/bridge/send-blender-to-webui')
    def send_blender_to_webui(body: dict):
        '''
        从 Blender 发送图像到 WebUI（供图生图使用）
        请求体: { "image_base64": "..." }
        '''
        image_base64 = body.get('image_base64', '')
        if not image_base64:
            return {'status': 'error', 'message': '缺少 image_base64 参数'}
        try:
            blender_dir = os.path.join(shared.data_path, 'temp', 'blender-bridge')
            os.makedirs(blender_dir, exist_ok=True)
            # 清除旧文件
            for f in os.listdir(blender_dir):
                fp = os.path.join(blender_dir, f)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            ts = int(time.time() * 1000)
            img_path = os.path.join(blender_dir, f'blender_to_webui_{ts}.png')
            img_data = base64.b64decode(image_base64)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            return {'status': 'success', 'message': '图像已保存，可在 WebUI 图生图中使用', 'path': img_path}
        except Exception as e:
            return {'status': 'error', 'message': f'保存图像失败: {str(e)}'}

    @app.get('/sdapi/v1/ps-plugin/bridge/receive-blender-to-webui')
    def receive_blender_to_webui():
        '''
        从 WebUI 获取 Blender 发送的图像（供图生图使用）
        返回: { "image_base64": "...", "filename": "..." }
        '''
        try:
            blender_dir = os.path.join(shared.data_path, 'temp', 'blender-bridge')
            if not os.path.isdir(blender_dir):
                return {'status': 'error', 'message': '没有可用的图像'}
            img_files = [f for f in os.listdir(blender_dir) if f.endswith('.png')]
            if not img_files:
                return {'status': 'error', 'message': '没有可用的图像'}
            img_files.sort(key=lambda f: os.path.getmtime(os.path.join(blender_dir, f)), reverse=True)
            latest = img_files[0]
            img_path = os.path.join(blender_dir, latest)
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            return {'status': 'success', 'image_base64': img_base64, 'filename': latest}
        except Exception as e:
            return {'status': 'error', 'message': f'读取图像失败: {str(e)}'}

    @app.get('/sdapi/v1/ps-plugin/bridge/receive-webui-glb-to-blender')
    def receive_glb_from_webui():
        '''
        从 WebUI 获取 GLB 模型数据（供 Blender 拉取）
        返回: { "glb_base64": "...", "filename": "model.glb" }
        '''
        try:
            glb_dir = os.path.join(shared.data_path, 'temp', 'glb-bridge')
            if not os.path.isdir(glb_dir):
                return {'status': 'error', 'message': '没有可用的 GLB 文件'}
            glb_files = [f for f in os.listdir(glb_dir) if f.endswith('.glb')]
            if not glb_files:
                return {'status': 'error', 'message': '没有可用的 GLB 文件'}
            # 取最新的 GLB 文件
            glb_files.sort(key=lambda f: os.path.getmtime(os.path.join(glb_dir, f)), reverse=True)
            latest = glb_files[0]
            glb_path = os.path.join(glb_dir, latest)
            with open(glb_path, 'rb') as f:
                glb_base64 = base64.b64encode(f.read()).decode('utf-8')
            return {'status': 'success', 'glb_base64': glb_base64, 'filename': latest}
        except Exception as e:
            return {'status': 'error', 'message': f'读取 GLB 文件失败: {str(e)}'}


def process_point_segmentation(img: Image.Image, points: list, model_type: str = 'vit_h') -> list:
    '''
    使用 SAM 进行点选分割

    Args:
        img: PIL Image (RGB)
        points: 标记点坐标列表 [[x1,y1], [x2,y2], ...]
        model_type: SAM 模型类型 (vit_h, vit_l, vit_b)

    Returns:
        list of PIL Image (RGBA) with transparent background
    '''
    from segment_anything import sam_model_registry, SamPredictor
    # 确定 SAM 模型路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 优先使用 shared.data_path/models/sams
    sams_dir = os.path.join(shared.data_path, 'models', 'sams')
    if not os.path.isdir(sams_dir):
        # fallback 到脚本相对路径
        webui_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        sams_dir = os.path.join(webui_root, 'models', 'sams')
    checkpoint_names = {
        'vit_h': 'sam_vit_h_4b8939.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_b': 'sam_vit_b_01ec64.pth'
    }
    # 构建要尝试的模型类型列表
    model_types_to_try = [model_type]
    for mt in ['vit_h', 'vit_l', 'vit_b']:
        if mt not in model_types_to_try:
            model_types_to_try.append(mt)
    checkpoint_path = None
    actual_model_type = model_type
    for mt in model_types_to_try:
        ckpt_name = checkpoint_names.get(mt)
        if not ckpt_name:
            continue
        ckpt_path = os.path.join(sams_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            checkpoint_path = ckpt_path
            actual_model_type = mt
            break
    if checkpoint_path is None:
        raise FileNotFoundError(
            f'SAM 模型未找到！请将模型放置到: {sams_dir}\n'
            f'需要的模型文件: {checkpoint_names.get(model_type, "sam_vit_h_4b8939.pth")}'
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[actual_model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # 转换图像
    img_np = np.array(img.convert('RGB'))
    predictor.set_image(img_np)
    all_masks_list = []
    for point in points:
        input_points = np.array([point], dtype=np.float32)
        input_labels = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        if masks is not None and len(masks) > 0:
            if isinstance(masks, (list, tuple)):
                masks = np.array(masks)
            # 保留前两个掩码（删除第三个大掩码）
            if masks.shape[0] > 2:
                masks = masks[:2]
            if masks.ndim == 3:
                masks = masks[:, None, ...]
            all_masks_list.append(masks)
    if not all_masks_list:
        print('[PS Plugin API] point_segmentation: all_masks is empty, no masks generated')
        return []
    combined_masks = np.concatenate(all_masks_list, axis=0)
    print(f'[PS Plugin API] point_segmentation: {len(combined_masks)} masks generated')
    h, w = img_np.shape[:2]
    # 删除面积最小的掩码
    if len(combined_masks) > 1:
        mask_areas = []
        for i, mask in enumerate(combined_masks):
            mask_2d = np.any(mask, axis=0)
            area = np.count_nonzero(mask_2d)
            mask_areas.append((area, i))
            print(f'[PS Plugin API] Mask {i} 面积: {area}')
        if mask_areas:
            mask_areas.sort(key=lambda x: x[0])
            smallest_idx = mask_areas[0][1]
            print(f'[PS Plugin API] 删除最小掩码 {smallest_idx}，面积: {mask_areas[0][0]}')
            combined_masks = np.delete(combined_masks, smallest_idx, axis=0)
            print(f'[PS Plugin API] 删除后剩余 {len(combined_masks)} 个掩码')
    results = []
    for i, mask in enumerate(combined_masks):
        if i >= 6:
            print(f'[PS Plugin API] 达到最大掩码数(6)，停止处理')
            break
        mask_2d = np.any(mask, axis=0)
        if np.count_nonzero(mask_2d) == 0:
            print(f'[PS Plugin API] 跳过空mask {i}')
            continue
        if img_np.shape[2] == 4:
            result_np = copy.deepcopy(img_np)
            result_np[~mask_2d, 3] = 0
        else:
            alpha_channel = np.full((h, w), 255, dtype=np.uint8)
            alpha_channel[~mask_2d] = 0
            result_np = np.dstack([img_np, alpha_channel])
        results.append(Image.fromarray(result_np))
    print(f'[PS Plugin API] point_segmentation: returning {len(results)} results')
    del sam
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


script_callbacks.on_app_started(ps_plugin_api)


def blender_bridge_ui():
    """在页面头部注入 JavaScript，为 img2img 添加从 Blender 导入按钮"""
    import gradio as gr

    def on_after_component(component, **kwargs):
        if getattr(component, 'elem_id', None) == 'img2img_image':
            # 在 img2img 图片输入下方添加导入按钮
            gr.HTML("""
            <div style="margin: 8px 0;">
                <button id="btn_blender_import"
                    class="gr-button gr-button-lg gr-button-secondary"
                    style="width:100%;padding:8px 16px;cursor:pointer;border-radius:8px;"
                    onclick="(function(){
                        var btn = this;
                        btn.disabled = true;
                        btn.textContent = '⏳ 导入中...';
                        fetch('/sdapi/v1/ps-plugin/bridge/receive-blender-to-webui')
                            .then(function(r){ return r.json(); })
                            .then(function(data){
                                if (data.status === 'success' && data.image_base64) {
                                    var canvas = document.querySelector('[elem_id=\"img2img_image\"]');
                                    if (!canvas) { alert('未找到画布'); return; }
                                    var container = canvas.querySelector('[id^=\"container_\"]');
                                    if (!container) { alert('未找到容器'); return; }
                                    var uuid = container.id.replace('container_', '');
                                    var ta = document.querySelector('#' + uuid + '.logical_image_background textarea');
                                    if (!ta) { alert('未找到数据域'); return; }
                                    ta.value = 'data:image/png;base64,' + data.image_base64;
                                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                                    var img2tab = document.querySelector('#tab_img2img button');
                                    if (img2tab) img2tab.click();
                                    btn.textContent = '✅ 已导入';
                                } else {
                                    btn.textContent = '❌ 无可用图像';
                                }
                                setTimeout(function(){ btn.textContent = '📥 从 Blender 导入'; btn.disabled = false; }, 2000);
                            })
                            .catch(function(e){
                                console.error(e);
                                btn.textContent = '❌ 导入失败';
                                setTimeout(function(){ btn.textContent = '📥 从 Blender 导入'; btn.disabled = false; }, 2000);
                            });
                    })()">
                    📥 从 Blender 导入
                </button>
            </div>
            """)

    script_callbacks.on_after_component(on_after_component)


script_callbacks.on_before_ui(blender_bridge_ui)