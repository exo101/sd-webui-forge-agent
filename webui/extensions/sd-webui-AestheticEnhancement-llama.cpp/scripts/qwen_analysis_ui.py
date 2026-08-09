import gradio as gr
import requests
import base64
from pathlib import Path
import logging
import os
import json

logger = logging.getLogger(__name__)

# 导入 llama.cpp API
try:
    from llamacpp_api import analyze_with_llamacpp, get_llamacpp_models, test_llamacpp_connection
    LLAMACPP_AVAILABLE = True
    logger.info("✅ llama.cpp API 模块加载成功")
except Exception as e:
    LLAMACPP_AVAILABLE = False
    logger.warning(f"llamacpp_api 模块不可用：{e}")

# 读取启动器共享端口配置
try:
    from modules.llama_port import get_llama_url
    _LLAMA_URL = get_llama_url()
except Exception:
    _LLAMA_URL = "http://localhost:8080"

# Ollama API 配置（本地部署的 Qwen3.5 多模态模型）
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"

# 默认配置
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_QWEN_MODEL = DEFAULT_MODEL  # 兼容别名
DEFAULT_BACKEND = "ollama"


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

def analyze_with_ollama(image_data, prompt, model=DEFAULT_MODEL):
    """
    使用本地 Ollama 多模态模型进行分析

    Args:
        image_data: 图片路径或 PIL Image
        prompt: 分析提示词
        model: 模型名称

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

        # 构造 Ollama API 请求
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False
        }

        logger.info(f"正在调用 Ollama 模型：{model}")
        logger.info(f"分析图片：{filename}")

        # 发送请求到 Ollama API
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        # 解析响应
        result = response.json()

        if "message" in result and "content" in result["message"]:
            analysis_text = result["message"]["content"]
            logger.info(f"分析完成，图片：{filename}")

            return {
                "success": True,
                "analysis": analysis_text,
                "model": model,
                "image_path": filename
            }
        else:
            logger.warning(f"Ollama 响应格式异常：{result}")
            return {
                "success": False,
                "analysis": f"⚠️ Ollama 响应格式异常\n\n原始响应：{json.dumps(result, indent=2, ensure_ascii=False)}",
                "error": "响应格式错误"
            }

    except requests.exceptions.ConnectionError as e:
        error_msg = f"""❌ **无法连接到 Ollama 服务**

可能原因：
1. Ollama 服务未启动
2. Ollama 端口不是 {OLLAMA_PORT}
3. 防火墙阻止了连接

解决方法：
1. 启动 Ollama 服务：在开始菜单搜索并运行 Ollama
2. 检查端口设置：确认 Ollama 使用的是 {OLLAMA_PORT} 端口
3. 关闭防火墙或添加例外规则

技术细节：{str(e)}
"""
        logger.error(f"Ollama 连接失败：{e}")
        return {
            "success": False,
            "analysis": error_msg,
            "error": str(e)
        }

    except requests.exceptions.Timeout:
        error_msg = f"""❌ **请求超时**

分析时间超过 120 秒，可能的原因：
1. 模型正在加载（首次运行需要加载模型到显存）
2. 图片分辨率过高
3. GPU 性能不足

建议：
1. 等待片刻后重试（首次加载模型需要时间）
2. 使用更小的图片尺寸（建议 1024x1024 以内）
3. 确保 GPU 有足够显存（推荐 8GB+）
"""
        logger.error("Ollama 请求超时")
        return {
            "success": False,
            "analysis": error_msg,
            "error": "timeout"
        }

    except Exception as e:
        logger.error(f"Ollama 分析失败：{e}")
        return {
            "success": False,
            "analysis": f"❌ 分析过程出错：{str(e)}\n\n请检查：\n1. Ollama 服务是否正常运行\n2. 模型是否已安装\n3. 图片文件是否正常",
            "error": str(e)
        }


def analyze_with_backend(image_path, prompt, model, backend="ollama", llamacpp_host=None):
    """
    统一分析接口，根据后端选择调用方式

    Args:
        image_path: 图片路径
        prompt: 分析提示词
        model: 模型名称
        backend: 后端类型 ("ollama" 或 "llamacpp")
        llamacpp_host: llama.cpp 服务器地址

    Returns:
        dict: 分析结果
    """
    if backend == "ollama":
        return analyze_with_ollama(image_path, prompt, model)
    elif backend == "llamacpp" and LLAMACPP_AVAILABLE:
        return analyze_with_llamacpp(image_path, prompt, model, llamacpp_host or _LLAMA_URL)
    else:
        return {"success": False, "analysis": f"❌ 不支持的后端：{backend}"}


def get_comprehensive_analysis_prompt():
    """获取综合分析提示词"""
    return """请作为专业影视摄影师和视觉艺术家，全面分析这个画面：

## 【分镜与景别】
1. **镜头类型**：远景/全景/中景/近景/特写（具体是哪一种？）
2. **拍摄角度**：平视/俯视/仰视/倾斜
3. **空间关系**：主体与背景的距离感
4. **镜头语言**：想要表达什么情感或信息？

## 【构图技巧】
1. **主要构图方式**：九宫格/三角形/S 型/对角线/引导线/对称/框架式等
2. **视觉焦点**：主体位置和视觉引导路径
3. **画面平衡**：左右/上下的视觉重量分布
4. **层次结构**：前景/中景/背景的排布
5. **负空间运用**：留白和呼吸感处理

## 【灯光设计】
1. **主光位置**：光源方向和强度
2. **光位类型**：顺光/侧光/逆光/顶光/底光
3. **光影对比**：明暗反差和过渡
4. **辅助光**：补光和轮廓光的使用
5. **特殊光效**：丁达尔光/光晕/反射等
6. **氛围营造**：光线如何影响情绪表达

## 【色彩情绪】
1. **主色调**：冷色/暖色/中性色
2. **色彩搭配**：互补色/类似色/三角配色
3. **饱和度**：鲜艳/低饱和/黑白
4. **情感表达**：色彩传递的心理感受

## 【优点总结】
- 这个画面最成功的地方是什么？
- 哪些技巧运用得特别好？

## 【改进建议】
- 如果要优化，可以从哪些方面入手？
- 给出具体的调整建议（构图/灯光/色彩等）

请用专业的视角、详细的分析、易懂的语言进行解读。每个部分都要有具体的观察和解释。"""

def get_composition_only_prompt():
    """获取仅构图分析的提示词"""
    return """请专注于分析这个画面的构图技巧：

## 【构图分析要点】

1. **主要构图方式**
   - 识别使用的经典构图法则（九宫格、三角形、S 型、对角线等）
   - 是否有多种构图的组合使用？

2. **视觉引导**
   - 视线如何被引导？
   - 有哪些隐含的线条和形状？

3. **画面平衡**
   - 左右是否平衡？
   - 上下重量分布如何？

4. **主体位置**
   - 主体放在哪里？为什么这样安排？
   - 是否符合视觉兴趣点？

5. **层次结构**
   - 前景、中景、背景如何分布？
   - 空间深度感如何营造？

6. **负空间运用**
   - 留白处理是否得当？
   - 呼吸感如何？

7. **元素关系**
   - 各元素之间的位置关系
   - 是否有重复、对比、呼应？

请结合具体画面元素详细说明，指出优点和可改进之处。"""

def get_lighting_only_prompt():
    """获取仅灯光分析的提示词"""
    return """请专注于分析这个画面的灯光设计：

## 【灯光分析要点】

1. **光源识别**
   - 主光从哪里来？（方向、角度）
   - 是自然光还是人造光？
   - 光的强度如何？

2. **光位类型**
   - 顺光（正面光）：均匀但平淡
   - 侧光（45 度）：立体感强
   - 逆光（背面光）：轮廓光效果
   - 顶光：从上往下
   - 底光：从下往上（特殊效果）

3. **光影效果**
   - 阴影的形状和位置
   - 明暗对比强烈还是柔和？
   - 过渡是否自然？

4. **光线质量**
   - 硬光（清晰阴影）vs 柔光（模糊阴影）
   - 是否有丁达尔效应（光束）？
   - 有无特殊光效（光晕、耀斑）？

5. **布光技巧**
   - 是否使用了三点布光？
   - 有无补光？从哪里来？
   - 轮廓光的使用？

6. **氛围营造**
   - 光线如何影响画面情绪？
   - 明亮欢快 vs 阴暗神秘？
   - 时间感的暗示（早晨/黄昏/夜晚）

7. **改进建议**
   - 灯光可以如何优化？
   - 如果是你来布光，会怎么做？

请详细分析每个方面，并说明为什么这样的灯光设计有效（或无效）。"""

def get_shot_only_prompt():
    """获取仅分镜分析的提示词"""
    return """请专注于分析这个画面的分镜和镜头语言：

## 【分镜分析要点】

1. **景别识别**
   - **远景**（Extreme Long Shot）：展示大环境
   - **全景**（Full Shot）：完整展现主体和环境
   - **中景**（Medium Shot）：膝盖/腰部以上
   - **近景**（Close-up）：胸部以上/面部
   - **特写**（Extreme Close-up）：局部细节（眼睛、嘴巴）

2. **拍摄角度**
   - **平视**：客观中立
   - **俯视**：从上往下，主体显得弱小
   - **仰视**：从下往上，主体显得高大
   - **倾斜**：不稳定感、动感

3. **镜头焦距**
   - **广角**：视野宽，变形明显，空间感强
   - **标准**：接近人眼视角
   - **长焦**：视野窄，压缩空间，背景虚化

4. **镜头运动暗示**
   - 固定镜头：稳定、平静
   - 推镜头：逐渐靠近主体
   - 拉镜头：远离主体，展示环境
   - 摇镜头：水平转动
   - 移镜头：相机位置移动

5. **空间关系**
   - 主体与观众的距离感
   - 主体与环境的关系
   - 前后景的空间层次

6. **情感表达**
   - 这个镜头想传达什么情绪？
   - 为什么要选择这样的景别和角度？
   - 对叙事有什么作用？

7. **镜头语言解读**
   - 权力关系（谁占主导）
   - 亲密程度（距离远近）
   - 紧张感（角度是否极端）

请详细分析每个方面，并解释这样的镜头选择背后的意图和效果。"""

def get_ollama_models():
    """获取 Ollama 已安装的模型列表"""
    try:
        response = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception as e:
        logger.warning(f"获取 Ollama 模型列表失败：{e}")
    return []


def analyze_single_image(image_data, analysis_type="comprehensive", model=DEFAULT_MODEL, backend="ollama", llamacpp_host=None):
    """
    分析单张图片

    Args:
        image_data: 图片路径或 PIL Image
        analysis_type: 分析类型
        model: 模型名称
        backend: 后端类型
        llamacpp_host: llama.cpp 服务器地址

    Returns:
        dict: 分析结果
    """
    if not image_data:
        return {"error": "图片数据为空", "success": False}

    # 获取文件名（用于日志）
    if isinstance(image_data, str):
        if not Path(image_data).exists():
            return {"error": "图片文件不存在或路径无效", "success": False}
        filename = Path(image_data).name
    else:
        filename = "uploaded_image.png"

    # 根据分析类型选择提示词
    prompts = {
        "comprehensive": get_comprehensive_analysis_prompt(),
        "composition": get_composition_only_prompt(),
        "lighting": get_lighting_only_prompt(),
        "shot": get_shot_only_prompt()
    }

    prompt = prompts.get(analysis_type, prompts["comprehensive"])

    # 添加图片信息
    full_prompt = f"""请分析这张图片：

图片文件名：{filename}

{prompt}"""

    # 调用后端进行分析
    result = analyze_with_backend(image_data, full_prompt, model, backend, llamacpp_host)

    return result


def batch_analyze_images(image_paths, analysis_type="comprehensive", model=DEFAULT_MODEL, backend="ollama", llamacpp_host=None, progress=gr.Progress()):
    """
    批量分析多张图片（按镜头分组）

    Args:
        image_paths: 图片路径列表（已按镜头排序）
        analysis_type: 分析类型
        model: 模型名称
        backend: 后端类型
        llamacpp_host: llama.cpp 服务器地址
        progress: Gradio 进度条

    Returns:
        str: 汇总的分析报告（每个镜头带对应图片）
    """
    if not image_paths:
        return "❌ 未选择图片"

    results = []
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        progress(i / total, desc=f"分析中 ({i+1}/{total})")

        result = analyze_single_image(img_path, analysis_type, model, backend, llamacpp_host)

        # 从文件名提取镜头信息
        filename = Path(img_path).name
        shot_info = ""
        if "shot" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                shot_num = parts[0].replace("shot", "").zfill(2)
                frame_num = parts[1].replace("frame", "").replace(".jpg", "")
                shot_info = f"**镜头 {shot_num}** | 帧位置：{frame_num}"
        else:
            frame_num = filename.replace("frame_", "").replace(".jpg", "")
            shot_info = f"**帧 {frame_num}** (固定间隔抽帧)"

        if result.get("success"):
            analysis_text = result['analysis']
            results.append(f"""## 📊 {shot_info}

![关键帧](file://{img_path})

{analysis_text}

---
""")
        else:
            results.append(f"""## {shot_info}

![关键帧](file://{img_path})

⚠️ {result.get('error', '分析失败')}

{result.get('analysis', '')}

---
""")

    report = "\n".join(results)
    summary = f"""# 🎬 视频画面分析报告

**总计分析**: {total} 个关键帧

---

"""
    return summary + report


def extract_video_frames(video_path, output_dir, frame_interval=30):
    """
    从视频中提取关键帧（基于镜头检测）
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_interval: 帧间隔（当设置为 0 时使用智能镜头检测，否则使用固定间隔）
    
    Returns:
        list: 提取的帧图片路径列表
    
    核心逻辑：
    1. 使用场景变化检测识别镜头切换
    2. 每个镜头只抽取 1-2 个代表性帧
    3. 跳过相似画面，避免重复分析
    4. 长镜头自动多抽几帧，短镜头快速跳过
    """
    try:
        import cv2
        import numpy as np
        
        if not Path(video_path).exists():
            return []
        
        # 创建输出目录
        output_path = Path(output_dir) / f"frames_{Path(video_path).stem}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"视频信息：{total_frames}帧，{fps}fps，时长{duration:.2f}秒")
        
        # ========== 判断使用哪种模式 ==========
        use_smart_detection = (frame_interval == 0 or frame_interval is None)
        
        if use_smart_detection:
            logger.info("使用智能镜头检测模式")
        else:
            logger.info(f"使用固定间隔抽帧模式（间隔：{frame_interval}帧）")
            # ========== 固定间隔抽帧模式 ==========
            frame_paths = []
            count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_interval == 0:
                    frame_path = output_path / f"frame_{saved_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    saved_count += 1
                
                count += 1
            
            cap.release()
            logger.info(f"固定间隔模式：提取 {saved_count} 帧")
            return frame_paths
        
        # ========== 智能镜头检测模式 ==========
        logger.info("正在读取视频帧并计算特征...")
        
        frames_data = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 计算 HSV 直方图作为帧特征
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            frames_data.append({
                'frame': frame,
                'frame_num': count,
                'hist': hist
            })
            count += 1
        
        cap.release()
        
        if len(frames_data) == 0:
            logger.error("视频中没有帧")
            return []
        
        logger.info(f"读取了 {len(frames_data)} 帧")
        
        # ========== 第二步：检测镜头切换 ==========
        logger.info("正在检测镜头切换...")
        
        shot_boundaries = []  # 存储镜头边界帧索引
        similarity_threshold = 0.7  # 相似度阈值，低于此值认为是新镜头
        
        # 第一个帧总是镜头开始
        shot_boundaries.append(0)
        
        for i in range(1, len(frames_data)):
            prev_hist = frames_data[i-1]['hist']
            curr_hist = frames_data[i]['hist']
            
            # 计算相邻帧的相似度（相关系数）
            similarity = np.corrcoef(prev_hist.flatten(), curr_hist.flatten())[0, 1]
            
            # 如果相似度突然下降，说明可能是镜头切换
            if np.isnan(similarity) or similarity < similarity_threshold:
                shot_boundaries.append(i)
                logger.debug(f"在帧 {i} 检测到镜头切换 (相似度：{similarity:.3f})")
        
        # 添加最后一帧作为最后一个镜头的结束
        shot_boundaries.append(len(frames_data) - 1)
        
        logger.info(f"检测到 {len(shot_boundaries)-1} 个镜头切换")
        
        # ========== 第三步：从每个镜头中提取关键帧 ==========
        logger.info("正在从每个镜头中提取关键帧...")
        
        frame_paths = []
        min_shot_duration = 30  # 最小镜头时长（帧数），约 1 秒
        max_keyframes_per_shot = 3  # 每个镜头最多提取的关键帧数
        
        for i in range(len(shot_boundaries) - 1):
            start_idx = shot_boundaries[i]
            end_idx = shot_boundaries[i + 1]
            shot_length = end_idx - start_idx
            
            logger.debug(f"处理镜头 {i+1}: 帧 {start_idx}-{end_idx} (长度：{shot_length})")
            
            # 根据镜头长度决定提取多少帧
            if shot_length < min_shot_duration:
                # 短镜头：只提取 1 帧（中间帧）
                keyframe_indices = [(start_idx + end_idx) // 2]
            elif shot_length < min_shot_duration * 3:
                # 中等镜头：提取 2 帧（1/3 和 2/3 位置）
                keyframe_indices = [
                    start_idx + shot_length // 3,
                    start_idx + 2 * shot_length // 3
                ]
            else:
                # 长镜头：提取 3 帧（1/4, 1/2, 3/4 位置）
                keyframe_indices = [
                    start_idx + shot_length // 4,
                    start_idx + shot_length // 2,
                    start_idx + 3 * shot_length // 4
                ]
            
            # 限制最大关键帧数
            keyframe_indices = keyframe_indices[:max_keyframes_per_shot]
            
            # 保存关键帧
            for frame_idx in keyframe_indices:
                if frame_idx < len(frames_data):
                    frame_data = frames_data[frame_idx]
                    frame_path = output_path / f"shot{i+1:02d}_frame{frame_idx:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame_data['frame'])
                    frame_paths.append(str(frame_path))
                    logger.debug(f"保存关键帧：{frame_path.name} (镜头{i+1}, 帧{frame_idx})")
        
        logger.info(f"成功提取 {len(frame_paths)} 个关键帧（来自 {len(shot_boundaries)-1} 个镜头）")
        
        # 清理临时数据
        del frames_data
        
        return frame_paths
        
    except ImportError as e:
        logger.error(f"缺少依赖库：{e}。请运行：pip install opencv-python numpy")
        return []
    except Exception as e:
        logger.error(f"视频处理失败：{e}", exc_info=True)
        
        # Fallback: 使用简单的固定间隔抽帧
        logger.warning(f"回退到固定间隔抽帧模式")
        try:
            cap = cv2.VideoCapture(video_path)
            output_path = Path(output_dir) / f"frames_{Path(video_path).stem}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            frame_paths = []
            count = 0
            saved_count = 0
            interval = frame_interval if frame_interval > 0 else 30
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % interval == 0:
                    frame_path = output_path / f"frame_{saved_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    saved_count += 1
                
                count += 1
            
            cap.release()
            logger.info(f"回退模式：提取 {saved_count} 帧")
            return frame_paths
        except Exception as e2:
            logger.error(f"回退模式也失败：{e2}")
            return []

def test_ollama_connection():
    """测试 Ollama 服务连接状态"""
    try:
        response = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            qwen_models = [m["name"] for m in models if "qwen" in m["name"].lower()]
            
            if qwen_models:
                return True, f"✅ Ollama 连接成功\n\n可用的 Qwen模型:\n" + "\n".join(qwen_models)
            else:
                return False, "⚠️ Ollama 连接成功，但未找到 Qwen模型\n\n请运行以下命令安装:\n```bash\nollama run qwen3.5:4b\n```"
        else:
            return False, f"❌ Ollama 响应异常：{response.status_code}"
    except Exception as e:
        return False, f"❌ 无法连接到 Ollama 服务\n\n错误信息：{str(e)}\n\n请确保:\n1. Ollama 服务已启动\n2. 端口设置为{OLLAMA_PORT}"

def create_qwen_analysis_ui():
    """创建 AI 智能画面分析 UI，支持 Ollama 和 llama.cpp"""

    custom_css = """
    .analysis-result {
        background: #f9f9f9;
        border-left: 4px solid #4CAF50;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    .analysis-section {
        margin-bottom: 24px;
    }
    .analysis-title {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 8px;
    }
    .frame-preview {
        display: inline-block;
        margin: 4px;
        border: 2px solid #ddd;
        border-radius: 4px;
    }
    """

    # 默认模型列表
    default_ollama_vision_models = [
        "qwen3.5:9b",
        "qwen3.5:4b",
        "qwen3.5:2b",
        "huihui_ai/qwen3.5-abliterated:9B",
        "huihui_ai/qwen3.5-abliterated:4B",
        "huihui_ai/qwen3.5-abliterated:2B",
        "qwen2.5-vl:2b",
        "qwen2.5-vl:7b",
        "qwen2-vl:2b",
        "qwen2-vl:7b",
        "llava:latest",
        "llava:7b",
        "llava:13b",
        "minicpm-v:latest",
    ]

    default_llamacpp_vision_models = [
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "qwen2.5-vl-32b",
        "llava-v1.5-7b",
        "llava-v1.5-13b",
        "llava-v1.6-7b",
        "llava-v1.6-13b",
        "deepseek-vl-2-7b",
        "cogvlm2-18b",
        "minicpm-v-v2.5",
    ]

    with gr.Column():
        gr.Markdown("""
        # 🎬 AI 智能画面分析

        基于 **本地部署的多模态大模型**，对图片和视频进行专业的画面分析

        ### 分析维度
        - **📐 构图分析**: 九宫格、三角形、S 型等经典构图识别
        - **💡 灯光分析**: 光位、光比、氛围营造技巧
        - **🎥 分镜分析**: 景别、角度、镜头语言解读
        - **🎨 色彩情绪**: 色调、配色方案、情感表达
        - **💬 改进建议**: 专业摄影师视角的优化建议
        """)

        # 后端选择
        backend_selector = gr.Radio(
            choices=[
                ("🦙 Ollama", "ollama"),
                ("🦄 llama.cpp", "llamacpp"),
            ],
            value=DEFAULT_BACKEND,
            label="选择模型后端",
            info="Ollama 简单易用，llama.cpp 性能更好"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ 系统配置")

                # Ollama 配置
                with gr.Group(visible=True) as ollama_config:
                    ollama_host = gr.Textbox(
                        label="Ollama 地址",
                        value="localhost",
                        placeholder="localhost"
                    )
                    ollama_port = gr.Number(
                        label="Ollama 端口",
                        value=11434,
                        placeholder="11434"
                    )
                    test_ollama_btn = gr.Button("🔌 测试 Ollama 连接", variant="secondary")

                test_llamacpp_btn = gr.Button("🔌 测试 llama.cpp 连接", variant="secondary")

                connection_info = gr.Textbox(
                    label="连接状态",
                    lines=3,
                    value="点击上方按钮测试连接...",
                    interactive=False
                )

                # 模型选择和刷新
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=default_ollama_vision_models,
                        value=DEFAULT_MODEL,
                        label="选择模型",
                        scale=3,
                        info="推荐使用平衡速度和质量的版本"
                    )
                    refresh_models_btn = gr.Button("🔄 刷新模型列表", scale=1)

                gr.Markdown("### 2️⃣ 选择分析模式")

                analysis_mode = gr.Radio(
                    choices=[
                        ("🖼️ 图片分析", "image"),
                        ("🎬 视频分析", "video")
                    ],
                    value="image",
                    label="分析模式"
                )

                analysis_type = gr.Radio(
                    choices=[
                        ("🔍 综合分析", "comprehensive"),
                        ("📐 仅构图分析", "composition"),
                        ("💡 仅灯光分析", "lighting"),
                        ("🎥 仅分镜分析", "shot")
                    ],
                    value="comprehensive",
                    label="分析类型"
                )

                # 图片输入
                image_input = gr.Image(
                    type="pil",
                    label="上传图片",
                    height=300
                )

                # 视频输入
                video_input = gr.Video(
                    label="上传视频",
                    height=300,
                    visible=False
                )

                frame_interval = gr.Slider(
                    minimum=1,
                    maximum=120,
                    value=30,
                    step=1,
                    label="视频抽帧间隔（帧数）",
                    info="每隔多少帧抽取一帧进行分析",
                    visible=False
                )

                analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 3️⃣ 分析结果")

                result_output = gr.Textbox(
                    label="分析报告",
                    lines=25,
                    show_copy_button=True,
                    placeholder="分析结果将显示在这里..."
                )

                # 帧预览（视频分析时使用）
                frame_gallery = gr.Gallery(
                    label="提取的帧预览",
                    columns=4,
                    height="auto",
                    visible=False
                )

        # 事件处理函数
        def update_backend_ui(backend):
            """根据后端显示不同的配置界面"""
            if backend == "ollama":
                return (
                    gr.update(visible=True),
                    gr.update(choices=default_ollama_vision_models, value=DEFAULT_MODEL)
                )
            else:
                # 尝试刷新 llama.cpp 模型列表
                if LLAMACPP_AVAILABLE:
                    try:
                        logger.info(f"🔄 切换到 llama.cpp 后端，尝试获取模型 (host: {_LLAMA_URL})")
                        models = get_llamacpp_models(_LLAMA_URL)
                        if models:
                            logger.info(f"✅ 获取到 llama.cpp 模型: {models}")
                            return (
                                gr.update(visible=False),
                                gr.update(choices=models + default_llamacpp_vision_models, value=models[0])
                            )
                        else:
                            logger.warning("⚠️ llama.cpp 未返回模型，使用默认列表")
                    except Exception as e:
                        logger.error(f"❌ 获取 llama.cpp 模型时出错: {e}")
                # 回退到默认
                return (
                    gr.update(visible=False),
                    gr.update(choices=default_llamacpp_vision_models, value="qwen3-vl-4b")
                )

        def refresh_ollama_models():
            """刷新 Ollama 模型列表"""
            models = get_ollama_models()
            if models:
                # 优先显示视觉模型
                vision_models = [m for m in models if "qwen" in m.lower() or "vl" in m.lower() or "llava" in m.lower()]
                if vision_models:
                    return gr.update(choices=vision_models + models, value=vision_models[0] if vision_models else models[0])
                else:
                    return gr.update(choices=models, value=models[0] if models else "")
            else:
                return gr.update(choices=default_ollama_vision_models, value=DEFAULT_MODEL)

        def refresh_llamacpp_models():
            """刷新 llama.cpp 模型列表"""
            if not LLAMACPP_AVAILABLE:
                return gr.update(choices=default_llamacpp_vision_models, value="qwen3-vl-4b")
            try:
                logger.info(f"🔄 尝试从 {_LLAMA_URL} 获取 llama.cpp 模型列表")
                models = get_llamacpp_models(_LLAMA_URL)
                logger.info(f"📋 获取到的模型列表: {models}")
                if models:
                    return gr.update(choices=models + default_llamacpp_vision_models, value=models[0])
                else:
                    logger.warning("⚠️ 未获取到模型列表，使用默认列表")
                    return gr.update(choices=default_llamacpp_vision_models, value="qwen3-vl-4b")
            except Exception as e:
                logger.error(f"❌ 刷新 llama.cpp 模型列表时出错: {e}")
                return gr.update(choices=default_llamacpp_vision_models, value="qwen3-vl-4b")

        def refresh_models(backend):
            """根据后端刷新模型列表"""
            if backend == "ollama":
                return refresh_ollama_models()
            else:
                return refresh_llamacpp_models()

        def test_current_connection(backend, ollama_host, ollama_port):
            """测试当前选择的后端连接"""
            if backend == "ollama":
                return test_ollama_connection()
            else:
                if LLAMACPP_AVAILABLE:
                    return test_llamacpp_connection(_LLAMA_URL)
                else:
                    return False, "❌ llama.cpp 模块不可用"

        def update_analysis_mode_ui(mode):
            """更新分析模式界面"""
            return (
                gr.update(visible=(mode == "image")),
                gr.update(visible=(mode == "video")),
                gr.update(visible=(mode == "video"))
            )

        def run_analysis(backend, model, mode, analysis_type, image, video, interval):
            """执行分析"""
            if mode == "image":
                if not image:
                    return "❌ 请先上传图片", gr.update(visible=False)

                result = analyze_single_image(image, analysis_type, model, backend, _LLAMA_URL)

                if result.get("success"):
                    return result.get("analysis", "分析完成"), gr.update(visible=False)
                else:
                    return f"❌ 分析失败\n\n{result.get('analysis', '')}", gr.update(visible=False)

            elif mode == "video":
                if not video:
                    return "❌ 请先上传视频", gr.update(visible=False)

                # 提取视频帧
                frame_paths = extract_video_frames(video, "temp/video_frames", interval)

                if not frame_paths:
                    return "❌ 视频处理失败", gr.update(visible=False, value=[])

                # 批量分析
                report = batch_analyze_images(frame_paths, analysis_type, model, backend, _LLAMA_URL)

                # 返回结果和帧预览
                return report, gr.update(visible=True, value=frame_paths[:16])

        # 绑定事件
        backend_selector.change(
            fn=update_backend_ui,
            inputs=[backend_selector],
            outputs=[ollama_config, model_selector]
        )

        test_ollama_btn.click(
            fn=lambda: test_ollama_connection()[1],
            inputs=[],
            outputs=[connection_info]
        )

        test_llamacpp_btn.click(
            fn=lambda: test_llamacpp_connection(_LLAMA_URL)[1] if LLAMACPP_AVAILABLE else "❌ llama.cpp 模块不可用",
            inputs=[],
            outputs=[connection_info]
        )

        refresh_models_btn.click(
            fn=refresh_models,
            inputs=[backend_selector],
            outputs=[model_selector]
        )

        analysis_mode.change(
            fn=update_analysis_mode_ui,
            inputs=[analysis_mode],
            outputs=[image_input, video_input, frame_interval]
        )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[backend_selector, model_selector, analysis_mode, analysis_type, image_input, video_input, frame_interval],
            outputs=[result_output, frame_gallery]
        )

        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### Ollama 使用方法

            1. **安装 Ollama**: 从 https://ollama.com 下载并安装
            2. **下载模型**: 运行命令 `ollama run qwen3.5:4b`
            3. **选择 Ollama 后端**: 在界面选择 Ollama
            4. **测试连接**: 点击"测试 Ollama 连接"按钮
            5. **开始分析**: 上传图片或视频并分析

            ### llama.cpp 使用方法

            1. **下载 llama.cpp**: 从 https://github.com/ggerganov/llama.cpp/releases 下载
            2. **下载模型**: 从 ModelScope 或 HuggingFace 下载 GGUF 格式的视觉模型
               - 推荐下载 Qwen3-VL 或 Qwen2.5-VL 系列
            3. **启动服务器**:
               ```bash
               llama-server --model path/to/model.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers -1
               ```
            4. **选择 llama.cpp 后端**: 在界面选择 llama.cpp
            5. **配置服务器地址**: 填入你的 llama.cpp 服务器地址
            6. **开始分析**: 上传图片或视频并分析

            ### 模型选择建议

            **Ollama 推荐**:
            - **qwen3.5:4b**: 平衡速度和质量，推荐（8GB+ 显存）
            - **qwen3.5:2b**: 速度最快，适合低显存显卡（4-6GB）
            - **qwen3.5:9b**: 质量最高，速度较慢（12GB+ 显存）

            **llama.cpp 推荐**:
            - **qwen3-vl-4b**: 平衡速度和质量
            - **qwen2.5-vl-7b**: 更好的视觉理解能力
            - **llava-v1.6-7b**: 经典视觉模型

            ### 常见问题

            **Q: 提示"无法连接到服务"**
            A: 请确保 Ollama 或 llama.cpp 服务正在运行

            **Q: 分析时间很长**
            A: 首次运行需要加载模型到显存，可能需要 1-2 分钟。后续分析会快很多

            **Q: 显存不足**
            A: 尝试使用更小的模型，或降低图片分辨率

            **Q: 如何同时使用多个模型？**
            A: 启动多个 llama-server 实例，每个使用不同端口和模型
            """)



if __name__ == "__main__":
    # 测试运行
    print("=" * 60)
    print("Qwen3.5 智能分析模块测试")
    print("=" * 60)
    
    # 测试连接
    print("\n📡 测试 Ollama 连接...")
    success, message = test_ollama_connection()
    print(message)
    
    if success:
        # 测试图片分析
        test_img = "test.jpg"
        if Path(test_img).exists():
            print(f"\n🔍 测试分析图片：{test_img}")
            result = analyze_single_image(test_img, "comprehensive")
            print("\n分析结果:")
            print(result.get("analysis", "分析失败"))
        else:
            print(f"\n⚠️ 测试图片不存在：{test_img}")
            print("请将一张测试图片命名为 test.jpg 放在当前目录后重试")
    else:
        print("\n❌ 无法进行测试，请先启动 Ollama 服务并安装Qwen3.5 模型")
    
    print("\n" + "=" * 60)
