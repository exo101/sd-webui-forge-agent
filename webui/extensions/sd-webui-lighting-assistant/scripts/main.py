"""
打光辅助插件 - Lighting Assistant
用于生成打光关键词和反推图片打光方式
"""

import os
import sys
import gradio as gr
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from modules import script_callbacks, scripts
from modules.shared import opts
from modules.llama_port import get_llama_url

logger = logging.getLogger(__name__)

# 添加 scripts 目录到 Python 路径
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# 默认配置
DEFAULT_LLAMACPP_URL = "http://localhost:8080"
LLAMACPP_AVAILABLE = False

# 导入 llama.cpp API
try:
    import llamacpp_api as llamacpp_module
    DEFAULT_LLAMACPP_URL = llamacpp_module.DEFAULT_LLAMACPP_URL
    LLAMACPP_AVAILABLE = True
    logger.info("llama.cpp API 模块加载成功")
except Exception as e:
    LLAMACPP_AVAILABLE = False
    logger.warning(f"llamacpp_api 模块不可用：{e}")

# 默认的分析提示词
DEFAULT_LIGHTING_ANALYSIS_PROMPT = """你是一位专业的摄影打光分析师。请分析这张图片的打光方式，并生成适合 Stable Diffusion 使用的纯中文打光关键词。

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
正面光，侧向打光，柔和光照，暖光，体积光，电影光
```

关键词应包含：
- 光源方向：如 "正面光，侧向打光，逆光，侧逆光"
- 光线质量：如 "柔和光照，硬光，柔光，聚光"
- 光影效果：如 "戏剧阴影，高对比，低对比，明暗对照"
- 光线颜色：如 "暖光，冷光，霓虹光，烛光，月光"
- 特殊效果：如 "丁达尔光，体积光，光晕，轮廓光"
- 氛围：如 "电影光，舞台光，戏剧性"

请确保关键词简洁、专业、适合 AI 绘画使用，使用纯中文关键词。"""

# ==================== 参考图片映射 ====================
import os
EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(EXTENSION_DIR, "images")

LIGHTING_IMAGE_MAP = {
    # 光源方向
    "正面光": "正面光.png",
    "侧面光": "侧面光.png",
    "侧逆光": "侧逆光.png",
    "正逆光": "正逆光.png",
    "顶光": "顶光.png",
    "底光": "底光.png",
    "伦勃朗光": "伦勃朗光.png",
    "蝴蝶光": "蝴蝶光.png",
    "分割光": "分割光.png",
    "环形光": "环形光.png",
    # 光线质量
    "硬光": "硬光.png",
    "柔光": "柔光.png",
    "漫射光": "漫反射光.png",
    "聚光": "聚光.png",
    # 光线颜色
    "暖光": "暖光.png",
    "冷光": "冷光.png",
    "中性光": "中性光.png",
    "黄金时刻": "黄金时刻.png",
    "蓝色时刻": "蓝色时刻.png",
    "霓虹光": "霓虹光.png",
    "烛光": "烛光.png",
    "月光": "月光.png",
    # 特殊光效
    "丁达尔光": "丁达尔光.png",
    "光晕": "光晕.png",
    "体积光": "体积光.png",
    "轮廓光": "轮毂光.png",
    "电影光": "电影光照.png",
    "舞台光": "舞台光.png",
    # 氛围风格
    "神秘": "神秘.png",
    "浪漫": "浪漫.png",
    "恐怖": "恐怖.png",
    "科幻": "科技.png",
    "梦幻": "梦幻.png",
    "复古": "复古.png",
}

def get_image_path(image_name):
    """获取图片路径"""
    if image_name:
        path = os.path.join(IMAGES_DIR, image_name)
        if os.path.exists(path):
            return path
    return None

# ==================== 打光关键词数据库 ====================

# 光源方向
LIGHT_DIRECTIONS = {
    "正面光": {
        "keywords": "正面光，平光",
        "description": "光源在相机正后方，直接照射主体，阴影最少，适合展现细节但缺乏立体感",
        "effect": "均匀明亮，缺乏层次"
    },
    "侧面光": {
        "keywords": "侧面光，侧向打光",
        "description": "光源在主体侧面，产生明显的明暗分界，增强立体感和质感",
        "effect": "立体感强，戏剧效果"
    },
    "侧逆光": {
        "keywords": "侧逆光，轮廓光",
        "description": "光源在主体侧后方，勾勒出主体轮廓边缘，分离主体与背景",
        "effect": "轮廓清晰，层次分明"
    },
    "正逆光": {
        "keywords": "逆光，剪影光",
        "description": "光源在主体正后方，形成剪影或轮廓光效果",
        "effect": "剪影效果，神秘感"
    },
    "顶光": {
        "keywords": "顶光，蝴蝶光",
        "description": "光源从正上方照射，在眼窝和下巴下方产生阴影",
        "effect": "戏剧性，有时显得阴森"
    },
    "底光": {
        "keywords": "底光",
        "description": "光源从下方照射，常用于恐怖或神秘场景",
        "effect": "诡异，恐怖氛围"
    },
    "伦勃朗光": {
        "keywords": "伦勃朗光，三角光",
        "description": "经典人像布光，在阴影侧脸颊形成三角形光斑",
        "effect": "戏剧性，立体感强"
    },
    "蝴蝶光": {
        "keywords": "蝴蝶光",
        "description": "顶光变体，在鼻子下方形成蝴蝶形阴影",
        "effect": "优雅，适合美女人像"
    },
    "分割光": {
        "keywords": "分割光",
        "description": "光源在侧面90度，将脸部分成明暗两半",
        "effect": "强烈对比，戏剧性"
    },
    "环形光": {
        "keywords": "环形光",
        "description": "侧光变体，在鼻子阴影侧形成小环状阴影",
        "effect": "自然立体，适合人像"
    }
}

# 光线质量
LIGHT_QUALITY = {
    "硬光": {
        "keywords": "硬光，锐利阴影",
        "description": "直射光，阴影边缘清晰锐利，对比强烈",
        "effect": "强烈对比，质感突出"
    },
    "柔光": {
        "keywords": "柔光，柔和光照",
        "description": "经过漫射的光，阴影边缘柔和模糊",
        "effect": "柔和自然，适合人像"
    },
    "漫射光": {
        "keywords": "漫射光，阴天光",
        "description": "经过大面积漫射的光，几乎没有阴影",
        "effect": "均匀柔和，无阴影"
    },
    "聚光": {
        "keywords": "聚光，聚光灯",
        "description": "集中照射的光束，强调特定区域",
        "effect": "聚焦强调，戏剧性"
    }
}

# 光线颜色/色温
LIGHT_COLOR = {
    "暖光": {
        "keywords": "暖光，金色光",
        "description": "偏黄/橙色的光，营造温暖氛围",
        "effect": "温暖舒适，怀旧感"
    },
    "冷光": {
        "keywords": "冷光，蓝色调",
        "description": "偏蓝/青色的光，营造冷静氛围",
        "effect": "冷静科技，神秘感"
    },
    "中性光": {
        "keywords": "中性光，白色光",
        "description": "接近白色的平衡光",
        "effect": "自然真实"
    },
    "黄金时刻": {
        "keywords": "黄金时刻，日落光",
        "description": "日落前后的温暖光线",
        "effect": "浪漫温暖，电影感"
    },
    "蓝色时刻": {
        "keywords": "蓝色时刻，暮光",
        "description": "日落后天空的蓝色光线",
        "effect": "神秘浪漫"
    },
    "霓虹光": {
        "keywords": "霓虹光，赛博朋克光",
        "description": "霓虹灯的多彩光线",
        "effect": "赛博朋克，都市感"
    },
    "烛光": {
        "keywords": "烛光",
        "description": "烛火的温暖摇曳光线",
        "effect": "温馨浪漫，古典感"
    },
    "月光": {
        "keywords": "月光，银白月光",
        "description": "月亮的冷色光线",
        "effect": "神秘梦幻"
    }
}

# 特殊光效
SPECIAL_EFFECTS = {
    "丁达尔光": {
        "keywords": "丁达尔光，光束",
        "description": "光线穿过介质形成的光束效果",
        "effect": "神圣梦幻，空间感"
    },
    "光晕": {
        "keywords": "光晕，镜头光晕",
        "description": "强光进入镜头产生的光晕效果",
        "effect": "电影感，梦幻"
    },
    "体积光": {
        "keywords": "体积光，雾气光",
        "description": "在雾气或灰尘中可见的光束",
        "effect": "氛围感，空间深度"
    },
    "轮廓光": {
        "keywords": "轮廓光，边缘光",
        "description": "勾勒主体轮廓的光线",
        "effect": "分离主体，层次感"
    },
    "电影光": {
        "keywords": "电影光",
        "description": "电影级别的专业布光",
        "effect": "专业感，戏剧性"
    },
    "舞台光": {
        "keywords": "舞台光，聚光灯",
        "description": "舞台表演用的聚光效果",
        "effect": "表演感，聚焦"
    }
}



# 氛围风格
ATMOSPHERE = {
    "神秘": {
        "keywords": "神秘光，忧郁光，诡异光，神秘氛围",
        "description": "神秘的氛围"
    },
    "浪漫": {
        "keywords": "浪漫光，梦幻柔光，温馨光，甜蜜氛围",
        "description": "浪漫温馨"
    },
    "恐怖": {
        "keywords": "恐怖光照，阴森光，诡异光，恐怖氛围",
        "description": "恐怖阴森"
    },
    "科幻": {
        "keywords": "科幻光照，未来光，科技光，未来感",
        "description": "科幻未来感"
    },
    "复古": {
        "keywords": "复古光，老电影感，怀旧光，经典光效",
        "description": "复古怀旧"
    },
    "梦幻": {
        "keywords": "梦幻光，仙境光，童话光，魔法光，飘渺光",
        "description": "梦幻仙境"
    }
}


# ==================== HTML卡片生成函数 ====================

def generate_lighting_cards_html(lighting_items, lighting_data, image_map, category_name):
    """生成打光选项卡片的HTML代码 - 参考美学提升插件风格"""
    import html
    
    # 添加模态框 HTML 和 JavaScript
    modal_html = f"""
    <div id="{category_name}Modal" style="
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.9);
        z-index: 9999;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    " onclick="close{category_name}Modal()">
        <span style="
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        ">&times;</span>
        <img id="{category_name}ModalImg" src="" style="
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        ">
    </div>
    <script>
        function open{category_name}Modal(imgSrc) {{
            document.getElementById('{category_name}ModalImg').src = imgSrc;
            document.getElementById('{category_name}Modal').style.display = 'flex';
        }}
        function close{category_name}Modal() {{
            document.getElementById('{category_name}Modal').style.display = 'none';
        }}
    </script>
    """
    
    html_parts = [modal_html, f"<div style='display: flex; flex-wrap: wrap; gap: 16px; padding: 10px;'>"]
    
    for name in lighting_items:
        desc = html.escape(lighting_data[name].get('description', '') or '暂无介绍')
        img_path = get_image_path(image_map.get(name))
        
        if img_path:
            card_html = f"""
            <div style="
                width: 140px;
                border: 1px solid #444;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                background: #1a1a1a;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            " onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.4)';" 
               onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.3)';"
               onclick="open{category_name}Modal('file={img_path}')">
                <div style="height: 120px; overflow: hidden; background: #1a1a1a; display: flex; align-items: center; justify-content: center;">
                    <img src="file={img_path}" alt="{html.escape(name)}" style="
                        max-width: 100%;
                        max-height: 100%;
                        object-fit: contain;
                    ">
                </div>
                <div style="padding: 8px; text-align: center;">
                    <div style="font-size: 14px; font-weight: bold; color: #fff; margin-bottom: 4px;">{html.escape(name)}</div>
                    <div style="font-size: 10px; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{desc}</div>
                </div>
            </div>
            """
        else:
            card_html = f"""
            <div style="
                width: 140px;
                border: 1px solid #444;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                background: #2a2a2a;
                padding: 16px 8px;
                text-align: center;
            ">
                <div style="font-size: 14px; font-weight: bold; color: #fff; margin-bottom: 4px;">{html.escape(name)}</div>
                <div style="font-size: 10px; color: #aaa;">{desc}</div>
            </div>
            """
        html_parts.append(card_html)
    
    html_parts.append("</div>")
    return "".join(html_parts)


# ==================== 关键词生成函数 ====================

def generate_lighting_keywords(
    directions: List[str],
    qualities: List[str],
    colors: List[str],
    effects: List[str],
    atmospheres: List[str],
    custom_keywords: str = ""
) -> Tuple[str, str]:
    """
    根据选择生成打光关键词
    
    Returns:
        Tuple[str, str]: (关键词字符串, 详细说明)
    """
    all_keywords = []
    descriptions = []
    
    # 光源方向
    for d in directions:
        if d in LIGHT_DIRECTIONS:
            all_keywords.append(LIGHT_DIRECTIONS[d]["keywords"])
            descriptions.append(f"**{d}**: {LIGHT_DIRECTIONS[d]['description']}")
    
    # 光线质量
    for q in qualities:
        if q in LIGHT_QUALITY:
            all_keywords.append(LIGHT_QUALITY[q]["keywords"])
            descriptions.append(f"**{q}**: {LIGHT_QUALITY[q]['description']}")
    
    # 光线颜色
    for c in colors:
        if c in LIGHT_COLOR:
            all_keywords.append(LIGHT_COLOR[c]["keywords"])
            descriptions.append(f"**{c}**: {LIGHT_COLOR[c]['description']}")
    
    # 特殊光效
    for e in effects:
        if e in SPECIAL_EFFECTS:
            all_keywords.append(SPECIAL_EFFECTS[e]["keywords"])
            descriptions.append(f"**{e}**: {SPECIAL_EFFECTS[e]['description']}")
    
    # 氛围风格
    for a in atmospheres:
        if a in ATMOSPHERE:
            all_keywords.append(ATMOSPHERE[a]["keywords"])
            descriptions.append(f"**{a}**: {ATMOSPHERE[a]['description']}")
    
    # 自定义关键词
    if custom_keywords.strip():
        all_keywords.append(custom_keywords.strip())
    
    # 合并关键词
    keywords_str = ", ".join(all_keywords)
    description_str = "\n\n".join(descriptions) if descriptions else "未选择任何打光选项"
    
    return keywords_str, description_str


def analyze_image_lighting(
    image_path: str,
    model: str,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    分析图片打光方式
    
    Args:
        image_path: 图片路径或numpy数组
        model: 模型名称
        progress: 进度回调
    
    Returns:
        Tuple[str, str, str]: (分析结果, 提取的关键词, 状态信息)
    """
    _llama_url = get_llama_url()
    if image_path is None:
        return "", "", "请先上传图片"
    
    # 处理numpy数组（Gradio Image组件默认返回numpy数组）
    import numpy as np
    if isinstance(image_path, np.ndarray):
        # 将numpy数组保存为临时文件
        import tempfile
        import cv2
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_file.name, cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR))
        image_path = temp_file.name
    
    if not LLAMACPP_AVAILABLE:
        return "", "", "llama.cpp API 模块不可用"
    
    progress(0.1, desc="正在连接 llama.cpp 服务...")
    
    # 测试连接
    if hasattr(llamacpp_module, 'test_llamacpp_connection'):
        success, msg = llamacpp_module.test_llamacpp_connection(_llama_url)
    else:
        success, msg = True, "连接成功"
    if not success:
        return "", "", f"连接失败：{msg}"
    
    progress(0.3, desc="正在分析图片打光...")
    
    # 获取分析提示词
    if hasattr(llamacpp_module, 'get_lighting_analysis_prompt'):
        prompt = llamacpp_module.get_lighting_analysis_prompt()
    else:
        prompt = DEFAULT_LIGHTING_ANALYSIS_PROMPT
    
    # 调用分析
    if hasattr(llamacpp_module, 'analyze_with_llamacpp'):
        result = llamacpp_module.analyze_with_llamacpp(
            image_path,
            prompt,
            model,
            _llama_url
        )
    else:
        return "", "", "分析函数不可用"
    
    progress(0.9, desc="正在提取关键词...")
    
    if result.get("success"):
        analysis = result["analysis"]
        if hasattr(llamacpp_module, 'extract_keywords_from_analysis'):
            keywords = llamacpp_module.extract_keywords_from_analysis(analysis)
        else:
            keywords = analysis  # 如果函数不存在，直接返回分析结果
        return analysis, keywords, "分析完成"
    else:
        return result.get("analysis", "分析失败"), "", result.get("analysis", "分析失败")


# ==================== UI 创建 ====================

def create_ui():
    """创建打光辅助界面"""
    
    with gr.Blocks() as ui:
        gr.Markdown("""
        # 💡 打光辅助工具 - Lighting Assistant
        
        帮助你生成专业的打光关键词，或通过 AI 分析图片的打光方式。
        
        **功能说明**：
        - **关键词生成器**：选择光效类型，自动生成打光关键词
        - **图片反推**：上传图片，AI 分析打光方式并生成关键词
        - **发送到提示词**：将生成的关键词发送到文生图/图生图的提示词框
        """)
        
        with gr.Tabs():
            # ==================== 标签页1：关键词生成器 ====================
            with gr.TabItem("关键词生成器"):
                gr.Markdown("### 选择打光参数，生成专业关键词")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # 光源方向 - 使用HTML卡片
                        gr.Markdown("**🔦 光源方向**")
                        direction_cards_html = generate_lighting_cards_html(
                            list(LIGHT_DIRECTIONS.keys()),
                            LIGHT_DIRECTIONS,
                            LIGHTING_IMAGE_MAP,
                            "Direction"
                        )
                        gr.HTML(direction_cards_html)
                        
                        direction_checkboxes = gr.CheckboxGroup(
                            choices=list(LIGHT_DIRECTIONS.keys()),
                            label="选择光源方向",
                            value=[]
                        )
                        
                        # 光线质量 - 使用HTML卡片
                        gr.Markdown("**☀️ 光线质量**")
                        quality_cards_html = generate_lighting_cards_html(
                            list(LIGHT_QUALITY.keys()),
                            LIGHT_QUALITY,
                            LIGHTING_IMAGE_MAP,
                            "Quality"
                        )
                        gr.HTML(quality_cards_html)
                        
                        quality_checkboxes = gr.CheckboxGroup(
                            choices=list(LIGHT_QUALITY.keys()),
                            label="选择光线质量",
                            value=[]
                        )
                        
                        # 光线颜色 - 使用HTML卡片
                        gr.Markdown("**🎨 光线颜色**")
                        color_cards_html = generate_lighting_cards_html(
                            list(LIGHT_COLOR.keys()),
                            LIGHT_COLOR,
                            LIGHTING_IMAGE_MAP,
                            "Color"
                        )
                        gr.HTML(color_cards_html)
                        
                        color_checkboxes = gr.CheckboxGroup(
                            choices=list(LIGHT_COLOR.keys()),
                            label="选择光线颜色",
                            value=[]
                        )
                    
                    with gr.Column(scale=1):
                        # 特殊光效 - 使用HTML卡片
                        gr.Markdown("**✨ 特殊光效**")
                        effect_cards_html = generate_lighting_cards_html(
                            list(SPECIAL_EFFECTS.keys()),
                            SPECIAL_EFFECTS,
                            LIGHTING_IMAGE_MAP,
                            "Effect"
                        )
                        gr.HTML(effect_cards_html)
                        
                        effect_checkboxes = gr.CheckboxGroup(
                            choices=list(SPECIAL_EFFECTS.keys()),
                            label="选择特殊光效",
                            value=[]
                        )
                        
                        # 氛围风格 - 使用HTML卡片
                        gr.Markdown("**🎭 氛围风格**")
                        atmosphere_cards_html = generate_lighting_cards_html(
                            list(ATMOSPHERE.keys()),
                            ATMOSPHERE,
                            LIGHTING_IMAGE_MAP,
                            "Atmosphere"
                        )
                        gr.HTML(atmosphere_cards_html)
                        
                        atmosphere_checkboxes = gr.CheckboxGroup(
                            choices=list(ATMOSPHERE.keys()),
                            label="选择氛围风格",
                            value=[]
                        )
                
                # 自定义关键词
                gr.Markdown("**📝 自定义关键词**")
                custom_keywords = gr.Textbox(
                    label="添加自定义打光关键词（可选）",
                    placeholder="例如：工作室光，专业摄影",
                    lines=2
                )
                
                # 生成按钮
                with gr.Row():
                    generate_btn = gr.Button("🎨 生成关键词", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空选择", variant="secondary")
                
                # 输出区域
                gr.Markdown("---")
                gr.Markdown("### 生成的关键词")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        output_keywords = gr.Textbox(
                            label="打光关键词（可直接复制或发送到提示词框）",
                            lines=3,
                            show_copy_button=True
                        )
                    with gr.Column(scale=1):
                        send_to_txt2img_btn = gr.Button("📤 发送到文生图", variant="primary")
                        send_to_img2img_btn = gr.Button("📤 发送到图生图", variant="secondary")
                
                # 详细说明
                output_description = gr.Markdown(label="选择说明")
            
            # ==================== 标签页2：图片反推 ====================
            with gr.TabItem("图片反推分析"):
                gr.Markdown("### 上传图片，AI 分析打光方式")
                
                if not LLAMACPP_AVAILABLE:
                    gr.Markdown("""
                    ⚠️ **llama.cpp API 模块不可用**
                    
                    请确保 `llamacpp_api.py` 文件存在且正确配置。
                    """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # 图片上传 - 使用简单的方式避免兼容性问题
                        input_image = gr.Image(
                            label="上传图片",
                            height=400
                        )
                        
                        # 模型选择
                        model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=[],
                            value="",
                            allow_custom_value=True
                        )
                        
                        # 刷新模型按钮
                        refresh_models_btn = gr.Button("🔄 刷新模型列表")
                        
                        # 测试连接
                        test_connection_btn = gr.Button("🔌 测试连接")
                        connection_status = gr.Textbox(label="连接状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        # 分析按钮
                        analyze_btn = gr.Button("🔍 分析打光", variant="primary", size="lg")
                        
                        # 分析结果
                        analysis_result = gr.Markdown(label="分析结果")
                        
                        # 提取的关键词
                        gr.Markdown("**📋 提取的打光关键词**")
                        extracted_keywords = gr.Textbox(
                            label="关键词",
                            lines=3,
                            show_copy_button=True
                        )
                        
                        # 发送按钮
                        with gr.Row():
                            send_analyzed_to_txt2img_btn = gr.Button("📤 发送到文生图", variant="primary")
                            send_analyzed_to_img2img_btn = gr.Button("📤 发送到图生图", variant="secondary")
                        
                        # 状态信息
                        status_text = gr.Textbox(label="状态", interactive=False)
            
            # ==================== 标签页3：打光参考 ====================
            with gr.TabItem("打光参考手册"):
                gr.Markdown("""
                ## 📚 打光技巧参考
                
                ### 光源方向详解
                
                | 光位 | 特点 | 适用场景 |
                |------|------|----------|
                | 正面光 | 均匀明亮，阴影少 | 证件照、产品摄影 |
                | 侧面光 | 立体感强，质感突出 | 人像、静物 |
                | 逆光 | 轮廓清晰，剪影效果 | 艺术摄影、氛围营造 |
                | 顶光 | 戏剧性，有时阴森 | 特殊效果 |
                | 底光 | 诡异恐怖 | 恐怖片、特殊效果 |
                | 伦勃朗光 | 经典三角光，戏剧性 | 人像摄影 |
                
                ### 光线质量
                
                | 类型 | 特点 | 效果 |
                |------|------|------|
                | 硬光 | 阴影清晰锐利 | 强烈对比，质感突出 |
                | 柔光 | 阴影柔和模糊 | 自然柔和，适合人像 |
                | 漫射光 | 几乎无阴影 | 均匀柔和 |
                
                ### 常见布光方案
                
                **三点布光法**：
                1. **主光 (Key Light)**：主要光源，决定整体效果
                2. **补光 (Fill Light)**：填充阴影，降低对比
                3. **轮廓光 (Rim Light)**：勾勒轮廓，分离背景
                
                **人像布光推荐**：
                - 美女/时尚：蝴蝶光 + 柔光
                - 男性/硬汉：分割光 + 硬光
                - 艺术人像：伦勃朗光
                - 自然人像：环形光
                
                ### 色温参考
                
                | 光源 | 色温 (K) | 感觉 |
                |------|----------|------|
                | 烛光 | 1800-2000 | 温暖浪漫 |
                | 日出/日落 | 2500-3000 | 金黄温暖 |
                | 钨丝灯 | 3000-3500 | 暖黄 |
                | 日光 | 5500-6000 | 中性白 |
                | 阴天 | 6500-7500 | 偏蓝 |
                | 蓝天 | 10000+ | 冷蓝 |
                """)
        
        # ==================== 事件绑定 ====================
        
        # 生成关键词
        def on_generate(directions, qualities, colors, effects, atmospheres, custom):
            keywords, desc = generate_lighting_keywords(
                directions, qualities, colors, effects, atmospheres, custom
            )
            return keywords, desc
        
        generate_btn.click(
            fn=on_generate,
            inputs=[
                direction_checkboxes,
                quality_checkboxes,
                color_checkboxes,
                effect_checkboxes,
                atmosphere_checkboxes,
                custom_keywords
            ],
            outputs=[output_keywords, output_description]
        )
        
        # 清空选择
        def on_clear():
            return [], [], [], [], [], [], "", "", ""
        
        clear_btn.click(
            fn=on_clear,
            outputs=[
                direction_checkboxes,
                quality_checkboxes,
                color_checkboxes,
                effect_checkboxes,
                atmosphere_checkboxes,
                custom_keywords,
                output_keywords,
                output_description
            ]
        )
        
        # 刷新模型列表
        def on_refresh_models():
            _llama_url = get_llama_url()
            if LLAMACPP_AVAILABLE and hasattr(llamacpp_module, 'get_llamacpp_models'):
                models = llamacpp_module.get_llamacpp_models(_llama_url)
            else:
                models = []
            return gr.Dropdown(choices=models, value=models[0] if models else "")
        
        refresh_models_btn.click(
            fn=on_refresh_models,
            inputs=[],
            outputs=[model_dropdown]
        )
        
        # 测试连接
        def on_test_connection():
            _llama_url = get_llama_url()
            if LLAMACPP_AVAILABLE and hasattr(llamacpp_module, 'test_llamacpp_connection'):
                success, msg = llamacpp_module.test_llamacpp_connection(_llama_url)
            else:
                success, msg = False, "API 不可用"
            return msg
        
        test_connection_btn.click(
            fn=on_test_connection,
            inputs=[],
            outputs=[connection_status]
        )
        
        # 分析图片
        def on_analyze(image, model, progress=gr.Progress()):
            if image is None:
                return "", "", "请上传图片"
            return analyze_image_lighting(image, model, progress)
        
        analyze_btn.click(
            fn=on_analyze,
            inputs=[input_image, model_dropdown],
            outputs=[analysis_result, extracted_keywords, status_text]
        )
        
        # 发送到提示词框的 JavaScript
        send_to_txt2img_btn.click(
            fn=None,
            inputs=[output_keywords],
            js="""
            (keywords) => {
                const textarea = document.querySelector('#txt2img_prompt textarea');
                if (textarea) {
                    textarea.value = textarea.value ? textarea.value + ', ' + keywords : keywords;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    return '已发送到文生图提示词框';
                }
                return '未找到提示词框';
            }
            """
        )
        
        send_to_img2img_btn.click(
            fn=None,
            inputs=[output_keywords],
            js="""
            (keywords) => {
                const textarea = document.querySelector('#img2img_prompt textarea');
                if (textarea) {
                    textarea.value = textarea.value ? textarea.value + ', ' + keywords : keywords;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    return '已发送到图生图提示词框';
                }
                return '未找到提示词框';
            }
            """
        )
        
        send_analyzed_to_txt2img_btn.click(
            fn=None,
            inputs=[extracted_keywords],
            js="""
            (keywords) => {
                const textarea = document.querySelector('#txt2img_prompt textarea');
                if (textarea) {
                    textarea.value = textarea.value ? textarea.value + ', ' + keywords : keywords;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    return '已发送到文生图提示词框';
                }
                return '未找到提示词框';
            }
            """
        )
        
        send_analyzed_to_img2img_btn.click(
            fn=None,
            inputs=[extracted_keywords],
            js="""
            (keywords) => {
                const textarea = document.querySelector('#img2img_prompt textarea');
                if (textarea) {
                    textarea.value = textarea.value ? textarea.value + ', ' + keywords : keywords;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    return '已发送到图生图提示词框';
                }
                return '未找到提示词框';
            }
            """
        )
    
    return ui


def on_ui_tabs():
    """注册到 WebUI 标签页"""
    ui = create_ui()
    return [(ui, "💡 打光辅助", "lighting_assistant")]


# 注册扩展
script_callbacks.on_ui_tabs(on_ui_tabs)

logger.info("打光辅助插件加载完成")
