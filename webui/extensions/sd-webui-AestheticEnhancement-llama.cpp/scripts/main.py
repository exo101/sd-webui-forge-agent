import os
import gradio as gr
from pathlib import Path
import logging
import sys
import base64
import io
import json
import random
import glob
import requests
from modules import script_callbacks, shared
from modules.llama_port import get_llama_url, get_llama_port
from modules import ui
from PIL import Image

logger = logging.getLogger(__name__)

# OpenAI API 客户端（用于视觉 API 模式 - ModelScope 等）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("✅ OpenAI 模块加载成功，API 视觉模式可用")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai 模块未安装，API 视觉模式不可用（pip install openai）")

# 默认 API 视觉模型配置
DEFAULT_API_VISION_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_API_BASE_URL = "https://api-inference.modelscope.cn/v1"

# Qwen 模块可用性检查标志
QWEN_MODULE_AVAILABLE = False

# 默认 Llama.cpp 端口（从 llama_port.py 获取）
try:
    _llama_port = get_llama_port()
except Exception as e:
    logger.warning(f"获取 Llama.cpp 端口失败：{e}，使用默认值 8080")
    _llama_port = 8080

# 确保 scripts 目录在 Python 路径中
try:
    scripts_dir = Path(__file__).parent
    scripts_dir_str = str(scripts_dir)
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)
        logger.info(f"已将 {scripts_dir_str} 添加到 Python 路径")
    
    # 检查 qwen_analysis_ui.py 文件是否存在
    qwen_file = scripts_dir / "qwen_analysis_ui.py"
    if qwen_file.exists():
        logger.info(f"✅ 找到 qwen_analysis_ui.py 文件：{qwen_file}")
        # 尝试导入模块
        try:
            import qwen_analysis_ui
            QWEN_MODULE_AVAILABLE = True
            logger.info("✅ Qwen 分析模块加载成功")
        except Exception as e:
            logger.error(f"❌ Qwen 分析模块导入失败：{e}", exc_info=True)
    else:
        logger.warning(f"❌ qwen_analysis_ui.py 文件不存在：{qwen_file}")
except Exception as e:
    logger.error(f"❌ Qwen 分析模块检查失败：{e}", exc_info=True)

# 检查打光辅助模块
LIGHTING_ASSISTANT_AVAILABLE = False
try:
    import lighting_assistant_module
    LIGHTING_ASSISTANT_AVAILABLE = True
    logger.info("✅ 打光辅助模块加载成功")
except Exception as e:
    logger.error(f"❌ 打光辅助模块加载失败：{e}", exc_info=True)

# 检查人物与场景分析模块
PROPORTION_GUIDE_AVAILABLE = False
try:
    import proportion_guide_module
    PROPORTION_GUIDE_AVAILABLE = True
    logger.info("✅ 人物与场景分析模块加载成功")
except Exception as e:
    logger.error(f"❌ 人物与场景分析模块加载失败：{e}", exc_info=True)

# 获取素材目录路径
SCRIPTS_DIR = Path(__file__).parent
AESTHETIC_ENHANCEMENT_DIR = SCRIPTS_DIR.parent / "Aesthetic-Enhancement"

# ==================== 画师百科模块 ====================

# 画风分类映射
STYLE_CATEGORIES = {
    "二次元画风": "二次元画风",
    "厚涂画风": "厚涂画风",
    "柔和水彩治愈": "柔和水彩治愈",
    "科幻机械暗黑": "科幻机械暗黑",
    "街头风格视觉冲击专辑封面": "街头风格视觉冲击专辑封面",
    "风景为主、魔法场景、宏大构图、光影华丽": "风景为主、魔法场景、宏大构图光影、光影华丽"
}


def get_artists_by_style():
    """获取所有画师信息，按画风分类"""
    artists_by_style = {}
    artists_dir = AESTHETIC_ENHANCEMENT_DIR / "画师百科"
    
    if not artists_dir.exists():
        logger.warning(f"画师百科素材目录不存在：{artists_dir}")
        return artists_by_style
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    for style_dir in artists_dir.iterdir():
        if not style_dir.is_dir():
            continue
        
        style_name = style_dir.name
        style_display = STYLE_CATEGORIES.get(style_name, style_name)
        
        # 收集所有文件
        files = list(style_dir.iterdir())
        logger.info(f"处理风格 {style_display}，找到 {len(files)} 个文件")
        
        # 按画师分组
        artist_files = {}
        for file_path in files:
            if file_path.is_file():
                # 去掉扩展名，得到画师名称
                artist_name = file_path.stem
                # 处理特殊情况，如 Fajyobore..jpg
                if artist_name.endswith('.'):
                    artist_name = artist_name.rstrip('.')
                
                if artist_name not in artist_files:
                    artist_files[artist_name] = {}
                
                if file_path.suffix.lower() in image_extensions:
                    artist_files[artist_name]['image'] = str(file_path)
                    logger.info(f"找到画师 {artist_name} 的图片: {file_path.name}")
                elif file_path.suffix.lower() == '.txt':
                    artist_files[artist_name]['txt'] = str(file_path)
                    logger.info(f"找到画师 {artist_name} 的说明: {file_path.name}")
        
        # 构建画师信息
        artists = []
        for artist_name, files in artist_files.items():
            if 'image' in files:
                artist_info = {
                    'name': artist_name,
                    'style': style_display,
                    'image': files['image'],
                    'description': ''
                }
                # 读取文本说明
                if 'txt' in files:
                    try:
                        with open(files['txt'], 'r', encoding='utf-8') as f:
                            artist_info['description'] = f.read().strip()
                        logger.info(f"成功读取画师 {artist_name} 的说明")
                    except Exception as e:
                        logger.error(f"读取画师说明文件失败: {files['txt']}, {e}")
                artists.append(artist_info)
                logger.info(f"添加画师: {artist_name}")
        
        if artists:
            artists_by_style[style_display] = sorted(artists, key=lambda x: x['name'])
            logger.info(f"风格 {style_display} 加载完成，共 {len(artists)} 个画师")
        else:
            logger.warning(f"风格 {style_display} 没有找到有效的画师信息")
    
    total_artists = sum(len(v) for v in artists_by_style.values())
    logger.info(f"画师百科加载完成：{len(artists_by_style)} 个风格，共 {total_artists} 个画师")
    return artists_by_style


def generate_artists_html(artists, style_name):
    """生成画师卡片的 HTML 代码 - 使用纯 HTML 避免 Gradio 状态共享问题"""
    import html
    
    # 添加模态框 HTML 和 JavaScript
    modal_html = """
    <div id="artistModal" style="
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
    " onclick="closeArtistModal()">
        <span style="
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        ">&times;</span>
        <img id="modalImg" src="" style="
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        ">
    </div>
    <script>
        function openArtistModal(imgSrc) {
            document.getElementById('modalImg').src = imgSrc;
            document.getElementById('artistModal').style.display = 'flex';
        }
        function closeArtistModal() {
            document.getElementById('artistModal').style.display = 'none';
        }
    </script>
    """
    
    html_parts = [modal_html, "<div style='display: flex; flex-wrap: wrap; gap: 32px; padding: 20px;'>"]
    
    for i, artist in enumerate(artists):
        name = html.escape(artist['name'])
        style = html.escape(artist['style'])
        desc = html.escape(artist['description'] or '暂无介绍')
        img_path = artist['image']
        
        card_html = f"""
        <div style="
            width: 360px;
            border: 1px solid #444;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            background: #2a2a2a;
        ">
            <div style="height: 480px; overflow: hidden; position: relative;">
                <img src="file={img_path}" alt="{name}" style="
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    cursor: pointer;
                " onclick="event.stopPropagation(); openArtistModal('file={img_path}')">
            </div>
            <div style="padding: 20px; background: #2a2a2a;">
                <input type="text" value="{name}" readonly style="
                    width: 100%;
                    font-weight: bold;
                    border: none;
                    background: transparent;
                    font-size: 28px;
                    margin-bottom: 8px;
                    color: #fff;
                " onclick="this.select(); document.execCommand('copy');">
                <div style="font-size: 24px; color: #aaa; margin-bottom: 8px;">{style}</div>
                <textarea readonly style="
                    width: 100%;
                    height: 120px;
                    border: 1px solid #555;
                    background: #333;
                    font-size: 22px;
                    resize: none;
                    color: #ddd;
                " onclick="this.select(); document.execCommand('copy');">{desc}</textarea>
            </div>
        </div>
        """
        html_parts.append(card_html)
    
    html_parts.append("</div>")
    return "".join(html_parts)


def create_artist_tab():
    """创建画师百科标签页"""
    gr.Markdown("""
    # 🎨 画师百科
    
    探索不同风格的知名画师及其作品特点，为你的创作提供参考和灵感。
    
    **使用建议**：
    - 了解不同画风的特点和表现手法
    - 学习优秀画师的构图和色彩运用
    - 从不同风格中汲取灵感，融合创新
    """)
    
    # 获取所有画师信息
    artists_by_style = get_artists_by_style()
    
    # 显示加载结果
    if not artists_by_style:
        gr.Markdown("⚠️ 未找到画师素材，请检查素材目录是否正确配置。")
        gr.Markdown(f"素材目录路径: {AESTHETIC_ENHANCEMENT_DIR}")
        return
    else:
        total_artists = sum(len(v) for v in artists_by_style.values())
        gr.Markdown(f"✅ 成功加载 {len(artists_by_style)} 个画风分类，共 {total_artists} 个画师")
    
    # 为每个画风创建子标签
    with gr.Tabs():
        for style_name, artists in artists_by_style.items():
            with gr.Tab(style_name):
                gr.Markdown(f"### {style_name} ({len(artists)} 位画师)")
                # 使用 HTML 方式渲染画师卡片，避免 Gradio 组件状态共享问题
                cards_html = generate_artists_html(artists, style_name)
                gr.HTML(cards_html)
    
    # 画师百科详解
    with gr.Accordion("📖 画风特点详解", open=False):
        gr.Markdown("""
        ## 常见画风特点
        
        ### 二次元画风
        - **特点**: 线条简洁明快，色彩鲜艳，角色形象可爱
        - **代表画师**: anmi, arsenixc, cogecha, hiten, reoenl, shiratama
        - **适用**: 动漫、游戏角色设计
        - **效果**: 青春活力，萌系可爱
        
        ### 厚涂画风
        - **特点**: 色彩层次丰富，质感强烈，笔触明显
        - **代表画师**: guweiz, krenz, swav, wlop
        - **适用**: 插画、概念设计
        - **效果**: 立体感强，视觉冲击力大
        
        ### 柔和水彩治愈
        - **特点**: 色彩柔和，透明感强，氛围温馨
        - **代表画师**: Fajyobore, ds_mile
        - **适用**: 治愈系插画、儿童绘本
        - **效果**: 温暖舒适，清新自然
        
        ### 科幻机械暗黑
        - **特点**: 未来感强烈，机械元素丰富，色调偏暗
        - **代表画师**: neco, redjuice
        - **适用**: 科幻题材、赛博朋克
        - **效果**: 酷炫前卫，科技感十足
        
        ### 街头风格视觉冲击
        - **特点**: 色彩对比强烈，构图大胆，元素多样
        - **代表画师**: lam, mika pikazo, tarou2, yoneyama
        - **适用**: 专辑封面、潮流设计
        - **效果**: 时尚前卫，视觉冲击力强
        
        ### 风景为主宏大构图
        - **特点**: 场景宏大，细节丰富，氛围营造出色
        - **代表画师**: fuzichoco, mocha, rella
        - **适用**: 风景画、场景概念设计
        - **效果**: 气势磅礴，沉浸感强
        
        ## 学习建议
        1. **分析技法**: 观察不同画师的线条、色彩和构图技巧
        2. **风格融合**: 尝试将不同风格的元素结合创新
        3. **练习模仿**: 在理解的基础上进行临摹练习
        4. **形成个性**: 在学习他人的基础上发展自己的风格
        """)

# ==================== 构图技巧模块 ====================

# 构图类型映射（文件名到中文标题）
COMPOSITION_TITLES = {
    "S 型构图.png": "S 型构图",
    "U 字型构图.png": "U 字型构图",
    "x 形式构图.png": "X 形式构图",
    "三角形构图.png": "三角形构图",
    "三角形构图2.png": "三角形构图 2",
    "九宫格构图.png": "九宫格构图",
    "口字形构图.png": "口字形构图",
    "向心式构图.png": "向心式构图",
    "对角线构图.png": "对角线构图",
    "对角线构图2.png": "对角线构图2",
    "引导线构图.png": "引导线构图",
    "放射式构图.png": "放射式构图",
    "点景构图.png": "点景构图",
    "环形式构图.png": "环形式构图",
}

# 构图说明
COMPOSITION_DESCRIPTIONS = {
    "S 型构图": "优雅流畅的曲线构图，营造韵律感和动感",
    "U 字型构图": "稳定的 U 形结构，突出中心主体",
    "X 形式构图": "交叉对称的视觉引导，增强画面张力",
    "三角形构图": "稳定均衡的经典构图，适用于多种场景",
    "九宫格构图": "黄金分割变体，将主体置于交点位置",
    "口字形构图": "框架式构图，聚焦内部主体",
    "向心式构图": "所有元素向中心汇聚，强化视觉焦点",
    "对角线构图": "斜线分割画面，创造动态平衡",
    "引导线构图": "利用线条引导视线，增强纵深感",
    "放射式构图": "从中心向外发散，展现扩张感",
    "点景构图": "点睛之笔，小元素提升整体效果",
    "环形式构图": "环形包围结构，营造围合感",
}


def get_composition_images():
    """获取所有构图素材图片路径"""
    composition_dir = AESTHETIC_ENHANCEMENT_DIR / "构图技巧"
    
    if not composition_dir.exists():
        logger.warning(f"构图技巧素材目录不存在：{composition_dir}")
        return []
    
    image_files = []
    for img_path in composition_dir.glob("*.png"):
        filename = img_path.name
        title = COMPOSITION_TITLES.get(filename, filename.replace(".png", ""))
        image_files.append({
            "path": str(img_path),
            "title": title,
            "description": COMPOSITION_DESCRIPTIONS.get(title, "")
        })
    
    logger.info(f"加载了 {len(image_files)} 个构图素材")
    return sorted(image_files, key=lambda x: x["title"])





# ==================== UI 组件创建 ====================

def create_composition_card(image_info, index):
    """创建单个构图卡片"""
    # 将路径转换为 URL 格式以在 HTML 中使用
    img_url = f"file={image_info['path']}"
    
    with gr.Group():
        # 使用 HTML img 标签以便绑定点击事件
        gr.HTML(f"""
        <div class="gallery-card" data-index="{index}" data-title="{image_info['title']}" data-description="{image_info['description']}" data-src="{image_info['path']}">
            <div class="gallery-image-container">
                <img src="{img_url}" alt="{image_info['title']}" class="gallery-image" />
                <div class="gallery-overlay">
                    <span class="gallery-zoom-icon">🔍</span>
                    <span class="gallery-zoom-text">点击放大</span>
                </div>
            </div>
            <div class="gallery-info">
                <div class="gallery-title">{image_info['title']}</div>
                <div class="gallery-description">{image_info['description']}</div>
            </div>
        </div>
        """)





def create_composition_tab():
    """创建构图技巧标签页"""
    gr.Markdown("""
    # 📐 构图技巧
    
    学习经典构图法则，提升作品美学品质。构图是画面的骨架，决定了作品的视觉结构和美感基础。
    
    **使用建议**：
    - 观察每个构图的视觉引导线和元素排布
    - 理解不同构图传达的情感和视觉效果
    - 在实际创作中灵活运用多种构图技巧
    """)
    
    # 获取所有构图图片
    composition_images = get_composition_images()
    
    if not composition_images:
        gr.Markdown("⚠️ 未找到构图素材，请检查素材目录是否正确配置。")
        return
    
    # 生成构图卡片 HTML
    import html
    
    # 添加模态框 HTML 和 JavaScript
    modal_html = """
    <div id="compositionModal" style="
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
    " onclick="closeCompositionModal()">
        <span style="
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        ">&times;</span>
        <img id="compositionModalImg" src="" style="
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        ">
    </div>
    <script>
        function openCompositionModal(imgSrc) {
            document.getElementById('compositionModalImg').src = imgSrc;
            document.getElementById('compositionModal').style.display = 'flex';
        }
        function closeCompositionModal() {
            document.getElementById('compositionModal').style.display = 'none';
        }
    </script>
    """
    
    html_parts = [modal_html, "<div style='display: flex; flex-wrap: wrap; gap: 32px; padding: 20px;'>"]
    
    for i, img_info in enumerate(composition_images):
        title = html.escape(img_info['title'])
        desc = html.escape(img_info['description'] or '暂无介绍')
        img_path = img_info['path']
        
        card_html = f"""
        <div style="
            width: 500px;
            border: 1px solid #444;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            background: #2a2a2a;
        ">
            <div style="height: 280px; overflow: hidden; position: relative;">
                <img src="file={img_path}" alt="{title}" style="
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    cursor: pointer;
                " onclick="event.stopPropagation(); openCompositionModal('file={img_path}')">
            </div>
            <div style="padding: 20px; background: #2a2a2a;">
                <input type="text" value="{title}" readonly style="
                    width: 100%;
                    font-weight: bold;
                    border: none;
                    background: transparent;
                    font-size: 24px;
                    margin-bottom: 8px;
                    color: #fff;
                " onclick="this.select(); document.execCommand('copy'); this.style.backgroundColor='#333'; setTimeout(() => this.style.backgroundColor='transparent', 500);">
                <textarea readonly style="
                    width: 100%;
                    height: 100px;
                    border: 1px solid #555;
                    background: #333;
                    font-size: 18px;
                    resize: none;
                    color: #ddd;
                " onclick="this.select(); document.execCommand('copy'); this.style.backgroundColor='#444'; setTimeout(() => this.style.backgroundColor='#333', 500);">{desc}</textarea>
            </div>
        </div>
        """
        
        html_parts.append(card_html)
    
    html_parts.append("</div>")
    
    # 渲染 HTML
    gr.HTML(''.join(html_parts))
    
    # 构图技巧详解
    with gr.Accordion("📖 构图技巧详解", open=False):
        gr.Markdown("""
        ## 常见构图技巧与应用
        
        ### 1. S 型构图
        - **特点**: 曲线优美，富有韵律感
        - **适用**: 风景、人像、静物
        - **效果**: 优雅、流动、柔美
        
        ### 2. 三角形构图
        - **特点**: 稳定均衡，层次分明
        - **适用**: 群像、建筑、山景
        - **效果**: 稳固、和谐、庄重
        
        ### 3. 九宫格构图
        - **特点**: 符合黄金分割比例
        - **适用**: 通用构图法则
        - **效果**: 自然舒适，视觉平衡
        
        ### 4. 对角线构图
        - **特点**: 动态感强，打破平衡
        - **适用**: 运动、街拍、创意摄影
        - **效果**: 活力、动感、张力
        
        ### 5. 引导线构图
        - **特点**: 强烈的纵深感和方向性
        - **适用**: 道路、河流、走廊
        - **效果**: 延伸、聚焦、沉浸
        
        ### 6. 向心式构图
        - **特点**: 所有元素指向中心
        - **适用**: 团体照、圆形建筑
        - **效果**: 凝聚、聚焦、统一
        
        ### 7. 框架式构图 (口字形)
        - **特点**: 前景形成画框
        - **适用**: 门窗、洞口、树枝
        - **效果**: 聚焦主体、增加层次
        
        ### 8. 放射式构图
        - **特点**: 从中心向外扩散
        - **适用**: 阳光、花朵、爆炸
        - **效果**: 扩张、活力、冲击力
        
        ## 实践建议
        1. **多观察**: 分析优秀作品的构图规律
        2. **勤练习**: 有意识地运用不同构图法则
        3. **善变通**: 根据实际情况灵活组合多种构图
        4. **敢突破**: 在掌握规则后尝试创新
        """)



_LLAMA_URL = get_llama_url()
_DEFAULT_VISION_MODEL = ""

# 全局变量存储聊天历史
_vision_chat_history = []
# 获取 llama.cpp 端口（使用 try-except 确保不会报错）
try:
    _llama_port = get_llama_port()
    _llama_host = _LLAMA_URL.split(':')[0] if _LLAMA_URL else 'localhost'
except Exception as e:
    logger.warning(f"获取 Llama.cpp 端口或 URL 失败：{e}，使用默认值")
    _llama_port = 8080
    _llama_host = 'localhost'


def _call_llamacpp_vision(message, image_path, model_name):
    """调用 llama.cpp 视觉模型"""
    try:
        if image_path:
            with open(image_path, 'rb') as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': message or "请描述这张图片"},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }]
        else:
            messages = [{'role': 'user', 'content': message}]

        api_url = f"{_LLAMA_URL.rstrip('/')}/v1/chat/completions"
        resp = requests.post(api_url, json={
            'model': model_name,
            'messages': messages,
            'stream': False
        }, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as e:
        logger.error(f"LVM API error: {e}")
        return f"请求出错: {str(e)}"


def _get_llamacpp_vision_models():
    """获取可用的视觉模型列表"""
    try:
        api_url = f"{_LLAMA_URL.rstrip('/')}/v1/models"
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m['id'] for m in data.get('data', [])]
        return models
    except:
        return []


def get_response_vision_api(message: str, image_path: str = None,
                             api_key: str = None,
                             model: str = DEFAULT_API_VISION_MODEL,
                             base_url: str = DEFAULT_API_BASE_URL,
                             timeout: int = 600) -> str:
    """使用 OpenAI 兼容 API 进行视觉分析（支持 ModelScope 等）"""
    if not OPENAI_AVAILABLE:
        return "openai 模块未安装，请执行: pip install openai"
    if not api_key:
        return "API Key 未配置，请在 WebUI 设置中配置 API Key"

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )

        content = []
        if message:
            content.append({"type": "text", "text": message})
        else:
            content.append({"type": "text", "text": "请详细描述这张图片"})

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{img_b64}"
                }
            })

        messages = [{"role": "user", "content": content}]

        logger.info(f"Vision API: 调用模型 {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=4096
        )
        result = response.choices[0].message.content or ""
        logger.info(f"Vision API: 响应完成，长度={len(result)}字符")
        return result
    except Exception as e:
        err_msg = f"视觉 API 请求失败: {str(e)}"
        logger.error(err_msg)
        return err_msg


def create_vision_chat_tab():
    """创建图像识别聊天标签页"""
    gr.Markdown("""
    <div style="margin-bottom:10px;font-size:14px;color:#666;">
        🖼 上传图片并输入文字描述，AI 将分析图片内容并回复
    </div>
    """)

    # ===== llama.cpp 连接配置（紧凑单行：通常只需填端口） =====
    with gr.Row():
        llama_port_input = gr.Textbox(
            label="🔌 llama.cpp 端口",
            placeholder="例如：8090，留空使用默认端口",
            value=str(_llama_port),
            interactive=True,
            max_length=5,
            scale=1,
            min_width=110
        )
        llama_host_input = gr.Textbox(
            label="🖥️ 主机 (默认 localhost)",
            placeholder="localhost",
            value="localhost",
            interactive=True,
            max_length=15,
            scale=1,
            min_width=130
        )
        # 模型选择（动态从 llama.cpp 服务获取模型列表）
        models = _get_llamacpp_vision_models()
        model_dropdown = gr.Dropdown(
            choices=models,
            value=models[0] if models else _DEFAULT_VISION_MODEL,
            label="🤖 视觉模型",
            interactive=True,
            scale=4,
            min_width=240
        )
        refresh_models_btn = gr.Button("🔄 刷新模型列表", scale=1, min_width=100)

    # 模型状态：单行纯文本展示，不再占用独立块
    model_status_text = gr.Textbox(
        label="模型状态",
        interactive=False,
        value="",
        lines=1,
        container=False,
        show_label=False
    )

    def update_llama_config(port, host):
        """端口/主机变化：更新模块级连接地址并自动刷新模型列表"""
        global _llama_port, _llama_host, _LLAMA_URL
        if port and str(port).strip().isdigit():
            _llama_port = int(str(port).strip())
        host = (host or "localhost").strip() or "localhost"
        _llama_host = host
        _LLAMA_URL = f"http://{host}:{_llama_port}"
        logger.info(f"✅ llama.cpp 配置已更新：host={host}, port={_llama_port}, url={_LLAMA_URL}")
        new_models = _get_llamacpp_vision_models()
        if new_models:
            return gr.update(choices=new_models, value=new_models[0]), f"✅ 已连接 {_LLAMA_URL}，发现 {len(new_models)} 个模型: {', '.join(new_models)}"
        return gr.update(choices=[], value=""), f"❌ 未检测到 {_LLAMA_URL} 上的模型，请确认服务已启动"

    def refresh_model_list():
        new_models = _get_llamacpp_vision_models()
        if new_models:
            return gr.update(choices=new_models, value=new_models[0]), f"✅ 发现 {len(new_models)} 个模型: {', '.join(new_models)}"
        else:
            return gr.update(choices=[], value=""), "❌ 未检测到 llama.cpp 服务上的模型，请确认服务已启动"

    refresh_models_btn.click(fn=refresh_model_list, inputs=[], outputs=[model_dropdown, model_status_text])
    # 端口/主机变化时：更新连接地址并自动刷新模型列表（无需再手动点刷新）
    llama_port_input.change(fn=update_llama_config, inputs=[llama_port_input, llama_host_input], outputs=[model_dropdown, model_status_text])
    llama_host_input.change(fn=update_llama_config, inputs=[llama_port_input, llama_host_input], outputs=[model_dropdown, model_status_text])

    # 后端选择
    backend_selector = gr.Radio(
        choices=[
            ("🦙 llama.cpp", "llamacpp"),
            ("☁️ API (ModelScope)", "api"),
        ],
        value="llamacpp",
        label="选择模型后端",
        info="llama.cpp 使用本地模型，API 模式使用 ModelScope 免费 API"
    )

    # API 配置信息（仅 API 模式显示）
    with gr.Group(visible=False) as api_config_info:
        api_vision_model_input = gr.Textbox(
            label="API Vision Model (视觉模型)",
            value=lambda: DEFAULT_API_VISION_MODEL,
            placeholder="例如: Qwen/Qwen3.8-27B",
        )
        gr.Markdown("""
        **API 模式配置说明：**
        1. 在 WebUI 设置中切换到 **API 模型** 模式
        2. 配置 **API Key**（魔搭 Token）
        3. 在上方输入框填写视觉模型名称
        4. 默认模型：`Qwen/Qwen3.8-27B`
        """)

    with gr.Row(equal_height=False):
        # 左侧面板
        with gr.Column(scale=1, min_width=320):
            image_input = gr.Image(
                type="pil",
                label="📤 上传图片（可选）",
                height=200
            )

            # 快捷指令 - 显示文字标签
            with gr.Accordion("⚡ 快捷指令", open=False):
                quick_btns = []
                quick_prompts = [
                    ("📝 自然描述", "自然语言描述这张图片的细节"),
                    ("🎨 MidJourney", "MidJourney 提示词：根据这张图片生成类似风格"),
                    ("📐 构图分析", "分析这张图片的分镜构图"),
                    ("🎬 图生视频", "图生视频描述：根据这张图片生成视频场景"),
                    ("🔖 SEO 关键词", "生成 SEO 关键词和标签"),
                    ("📋 简单描述", "简单描述这张图片的内容")
                ]
                for label, prompt in quick_prompts:
                    btn = gr.Button(label, size="sm")
                    quick_btns.append((btn, prompt))

            # 标签管理
            with gr.Accordion("🏷 标签管理", open=False):
                gr.Markdown("管理文件夹中 .txt 文件的关键词标签")
                tag_folder = gr.Textbox(
                    label="📁 文件夹路径",
                    placeholder="输入文件夹路径...",
                    scale=1
                )
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新文件列表", size="sm", scale=1)
                    tag_files = gr.Dropdown(
                        choices=[],
                        label="选择文件",
                        interactive=True,
                        scale=2
                    )
                tag_keyword = gr.Textbox(
                    label="关键词",
                    placeholder="输入要添加/删除的关键词...",
                    scale=1
                )
                with gr.Row():
                    tag_position = gr.Radio(
                        choices=["开头", "结尾", "随机位置"],
                        value="开头",
                        label="添加位置",
                        interactive=True
                    )
                with gr.Row():
                    tag_add = gr.Button("➕ 添加", size="sm", scale=1)
                    tag_remove = gr.Button("➖ 删除", size="sm", scale=1)
                    tag_add_all = gr.Button("➕ 全部添加", size="sm", scale=1)
                    tag_remove_all = gr.Button("➖ 全部删除", size="sm", scale=1)
                tag_result = gr.Textbox(label="操作结果", interactive=False, lines=2)

            # 批量识别
            with gr.Accordion("🖼 批量识别", open=False):
                gr.Markdown("批量识别目录中的图片")
                batch_dir = gr.Textbox(
                    label="📁 图片目录路径",
                    placeholder="输入图片目录路径...",
                    scale=1
                )
                with gr.Row():
                    load_images_btn = gr.Button("🔄 加载图片", size="sm", scale=1)
                    batch_recognize_btn = gr.Button("🚀 开始批量识别", variant="primary", size="sm", scale=2)
                batch_gallery = gr.Gallery(
                    label="图片列表",
                    columns=4,
                    height=200,
                    object_fit="contain",
                    show_label=False
                )
                batch_progress = gr.Textbox(
                    label="进度",
                    interactive=False,
                    lines=2
                )

        # 右侧：聊天区域
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 对话",
                height=450,
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=(None, None)
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入消息",
                    placeholder="输入消息...",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("清空", scale=1)

    # ========== 事件绑定 ==========

    # 后端选择切换
    def update_backend_ui(backend):
        if backend == "api":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)

    backend_selector.change(
        fn=update_backend_ui,
        inputs=[backend_selector],
        outputs=[api_config_info]
    )

    # 发送消息
    def respond(message, history, pil_image, model_name, backend, api_vision_model):
        if not message and not pil_image:
            return "", history, history
        if not message:
            message = "请描述这张图片"
        history = history or []
        user_msg = message
        if pil_image:
            user_msg = f"[图片] {message}"
        history.append((user_msg, None))

        # 保存 PIL 图片到临时文件
        temp_path = None
        if pil_image is not None:
            try:
                temp_dir = os.path.join(str(SCRIPTS_DIR), "tmp", "vision")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"img_{os.urandom(4).hex()}.png")
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                pil_image.save(temp_path)
            except Exception as e:
                logger.error(f"Save temp image error: {e}")

        if backend == "api":
            # API 模式
            api_key = getattr(shared.opts, 'forge_api_key', '')
            api_model = api_vision_model or DEFAULT_API_VISION_MODEL
            response = get_response_vision_api(
                message=message,
                image_path=temp_path,
                api_key=api_key,
                model=api_model,
                timeout=600
            )
        else:
            # llama.cpp 本地模式
            response = _call_llamacpp_vision(message, temp_path, model_name)

        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

        history[-1] = (user_msg, response)
        return "", history, history

    def clear_history():
        return "", [], []

    send_btn.click(respond, [msg_input, chatbot, image_input, model_dropdown, backend_selector, api_vision_model_input], [msg_input, chatbot, chatbot])
    msg_input.submit(respond, [msg_input, chatbot, image_input, model_dropdown, backend_selector, api_vision_model_input], [msg_input, chatbot, chatbot])
    clear_btn.click(clear_history, None, [msg_input, chatbot, chatbot])

    # 保存 API Vision Model 配置
    def on_api_vision_model_change(model_name: str):
        if model_name:
            try:
                shared.opts.set("forge_api_vision_model", model_name)
            except Exception as e:
                logger.warning(f"保存视觉模型配置失败: {e}")

    api_vision_model_input.change(
        fn=on_api_vision_model_change,
        inputs=[api_vision_model_input],
        queue=False,
        show_progress=False,
    )

    # 快捷指令按钮
    for btn, prompt in quick_btns:
        btn.click(lambda p=prompt: p, None, msg_input)

    # ========== 标签管理事件 ==========

    def refresh_files(folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            return gr.Dropdown(choices=[], value=None), "❌ 无效的文件夹路径"
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        return gr.Dropdown(choices=txt_files, value=txt_files[0] if txt_files else None), \
               f"✅ 找到 {len(txt_files)} 个 .txt 文件"

    def tag_operation(action, folder, file, keyword, position):
        if not folder or not file or not keyword:
            return "❌ 请填写完整信息"
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            return "❌ 文件不存在"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if action == "add":
                if position == "开头":
                    content = keyword + '\n' + content
                elif position == "结尾":
                    content = content + '\n' + keyword
                else:
                    lines = content.split('\n')
                    idx = random.randint(0, len(lines))
                    lines.insert(idx, keyword)
                    content = '\n'.join(lines)
            else:
                content = content.replace(keyword, "")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ 关键词已{'添加' if action == 'add' else '删除'}到 {file}"
        except Exception as e:
            return f"❌ 错误: {str(e)}"

    def tag_operation_all(action, folder, keyword, position):
        if not folder or not keyword or not os.path.isdir(folder):
            return "❌ 请填写完整信息"
        txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        success = 0
        failed = 0
        for fname in txt_files:
            file_path = os.path.join(folder, fname)
            if not os.path.exists(file_path):
                failed += 1
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if action == "add_all":
                    if position == "开头":
                        content = keyword + '\n' + content
                    elif position == "结尾":
                        content = content + '\n' + keyword
                    else:
                        lines = content.split('\n')
                        idx = random.randint(0, len(lines))
                        lines.insert(idx, keyword)
                        content = '\n'.join(lines)
                else:
                    content = content.replace(keyword, "")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                success += 1
            except:
                failed += 1
        return f"操作完成: 成功 {success} 个, 失败 {failed} 个"

    refresh_btn.click(refresh_files, tag_folder, [tag_files, tag_result])
    tag_add.click(lambda f, fi, k, p: tag_operation("add", f, fi, k, p),
                  [tag_folder, tag_files, tag_keyword, tag_position], tag_result)
    tag_remove.click(lambda f, fi, k, p: tag_operation("remove", f, fi, k, p),
                     [tag_folder, tag_files, tag_keyword, tag_position], tag_result)
    tag_add_all.click(lambda f, k, p: tag_operation_all("add_all", f, k, p),
                      [tag_folder, tag_keyword, tag_position], tag_result)
    tag_remove_all.click(lambda f, k, p: tag_operation_all("remove_all", f, k, p),
                         [tag_folder, tag_keyword, tag_position], tag_result)

    # ========== 批量识别事件 ==========

    def load_batch_images(dir_path):
        if not dir_path or not os.path.isdir(dir_path):
            return [], "❌ 无效的目录路径"
        supported = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        images = []
        for ext in supported:
            for f in sorted(glob.glob(os.path.join(dir_path, f"*{ext}"))):
                try:
                    img = Image.open(f)
                    img.verify()
                    images.append(f)
                except:
                    pass
        return images, f"✅ 加载 {len(images)} 张图片"

    def batch_recognize(dir_path, images, model_name, progress):
        if not images:
            return progress, "❌ 没有图片可识别"
        results = []
        for i, item in enumerate(images):
            # Gallery 可能返回 (path, caption) 元组或直接路径
            img_path = item[0] if isinstance(item, (list, tuple)) else item
            progress = f"🔄 正在识别 ({i+1}/{len(images)}): {os.path.basename(img_path)}"
            try:
                resp = _call_llamacpp_vision("简单描述这张图片的内容", img_path, model_name)
                results.append(f"✅ {os.path.basename(img_path)}: {resp[:50]}...")
            except Exception as e:
                results.append(f"❌ {os.path.basename(img_path)}: {str(e)}")
        batch_result = "\n".join(results)
        return f"✅ 识别完成 ({len(images)} 张)", batch_result

    load_images_btn.click(load_batch_images, batch_dir, [batch_gallery, batch_progress])
    batch_recognize_btn.click(
        batch_recognize,
        [batch_dir, batch_gallery, model_dropdown, batch_progress],
        [batch_progress, batch_progress]
    )


def aesthetic_enhancement_ui():
    """创建美学提升模块 UI"""
    
    # 添加自定义 CSS（v2.0 - 网格布局）
    custom_css = """
    /* ===== 画廊网格容器 - 响应式多列布局 v2.0 ===== */
    .gallery-container {
        display: grid !important;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)) !important;
        gap: 20px !important;
        padding: 20px !important;
        background: #f9f9f9 !important;
        border-radius: 8px !important;
        width: 100% !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
        justify-items: center !important;
        flex-direction: initial !important;
    }
    
    /* ===== 画廊卡片 v2.0 ===== */
    .gallery-card {
        background: #fff !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: 320px !important;
        height: auto !important;
        display: block !important;
    }
    
    .gallery-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
    }
    
    /* ===== 图片容器 - 保持固定宽高比 v2.0 ===== */
    .gallery-image-container {
        position: relative !important;
        width: 100% !important;
        padding-top: 75% !important; /* 4:3 宽高比 */
        overflow: hidden !important;
        background: #f0f0f0 !important;
    }
    
    /* ===== 画廊图片样式 v2.0 ===== */
    .gallery-image {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        transition: transform 0.2s ease !important;
    }
    
    .gallery-card:hover .gallery-image {
        transform: scale(1.05) !important;
    }
    
    /* ===== 悬停遮罩层 v2.0 ===== */
    .gallery-overlay {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background: rgba(0, 0, 0, 0.5) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        opacity: 0 !important;
        transition: opacity 0.2s ease !important;
        color: white !important;
        font-size: 14px !important;
    }
    
    .gallery-card:hover .gallery-overlay {
        opacity: 1 !important;
    }
    
    .gallery-zoom-icon {
        font-size: 24px !important;
        margin-bottom: 4px !important;
    }
    
    .gallery-zoom-text {
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    /* ===== 信息区域 v2.0 ===== */
    .gallery-info {
        padding: 15px !important;
        text-align: center !important;
        background: #fff !important;
    }
    
    .gallery-title {
        font-weight: bold !important;
        font-size: 16px !important;
        color: #333 !important;
        margin-bottom: 8px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .gallery-description {
        font-size: 13px !important;
        color: #666 !important;
        line-height: 1.5 !important;
        font-style: italic !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
    }
    
    /* ===== 画师卡片专用样式（参考 Lora 模型卡片）===== */
    .artist-card {
        width: 160px !important;
        margin: 8px !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        transition: all 0.2s ease !important;
        background: #ffffff !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    .artist-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        border-color: #d0d0d0 !important;
    }
    
    .artist-image-container {
        width: 100% !important;
        height: 160px !important;
        position: relative !important;
        overflow: hidden !important;
        background: #f5f5f5 !important;
    }
    
    .artist-image {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        transition: transform 0.2s ease !important;
    }
    
    .artist-card:hover .artist-image {
        transform: scale(1.05) !important;
    }
    
    .artist-overlay {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background: rgba(0, 0, 0, 0.5) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        opacity: 0 !important;
        transition: opacity 0.2s ease !important;
        color: white !important;
    }
    
    .artist-card:hover .artist-overlay {
        opacity: 1 !important;
    }
    
    .artist-zoom-icon {
        font-size: 24px !important;
        margin-bottom: 4px !important;
    }
    
    .artist-zoom-text {
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    .artist-info {
        padding: 12px !important;
        text-align: center !important;
        background: #ffffff !important;
        flex: 1 !important;
    }
    
    .artist-name {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #333 !important;
        margin-bottom: 4px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .artist-style {
        font-size: 12px !important;
        color: #888 !important;
        margin-bottom: 6px !important;
        font-style: normal !important;
    }
    
    .artist-description {
        font-size: 11px !important;
        color: #666 !important;
        line-height: 1.4 !important;
        margin-top: 6px !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
    }
    
    /* 调整画廊容器布局 */
    .gallery-container {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 16px !important;
        padding: 20px !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }
    
    /* 确保 Gradio 不会覆盖样式 */
    .artist-card * {
        box-sizing: border-box !important;
    }
    
    /* 确保卡片在不同容器中都能正确显示 */
    .gr-box .artist-card {
        margin: 8px !important;
    }
    
    /* ===== 模态框样式 v2.0 ===== */
    .modal-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background: rgba(0, 0, 0, 0.85) !important;
        display: none !important;
        justify-content: center !important;
        align-items: center !important;
        z-index: 9999 !important;
        cursor: zoom-out !important;
        animation: fadeIn 0.3s ease !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-image {
        max-width: 90% !important;
        max-height: 90% !important;
        object-fit: contain !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
        animation: zoomIn 0.3s ease !important;
    }
    
    @keyframes zoomIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .modal-close-hint {
        position: absolute !important;
        top: 20px !important;
        right: 30px !important;
        color: white !important;
        font-size: 32px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        z-index: 10000 !important;
        transition: color 0.2s ease !important;
    }
    
    .modal-close-hint:hover {
        color: #ff6b6b !important;
    }
    
    .modal-info {
        position: absolute !important;
        bottom: 30px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        color: white !important;
        font-size: 16px !important;
        text-align: center !important;
        background: rgba(0, 0, 0, 0.8) !important;
        padding: 20px !important;
        border-radius: 8px !important;
        max-width: 90% !important;
        max-height: 40% !important;
        overflow-y: auto !important;
        backdrop-filter: blur(10px) !important;
        line-height: 1.6 !important;
    }
    
    .modal-info strong {
        display: block !important;
        font-size: 22px !important;
        margin-bottom: 12px !important;
        font-weight: bold !important;
    }
    
    /* ===== 响应式适配 v2.0 ===== */
    @media (max-width: 1200px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)) !important;
            gap: 16px !important;
            padding: 16px !important;
        }
    }
    
    @media (max-width: 900px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)) !important;
            gap: 14px !important;
            padding: 14px !important;
        }
        
        .gallery-info {
            padding: 12px !important;
        }
        
        .gallery-title {
            font-size: 15px !important;
        }
        
        .gallery-description {
            font-size: 12px !important;
        }
    }
    
    @media (max-width: 768px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) !important;
            gap: 12px !important;
            padding: 12px !important;
        }
        
        .gallery-info {
            padding: 10px !important;
        }
        
        .gallery-title {
            font-size: 14px !important;
        }
        
        .gallery-description {
            font-size: 11px !important;
            -webkit-line-clamp: 1 !important;
        }
    }
    
    @media (max-width: 480px) {
        .gallery-container {
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)) !important;
            gap: 10px !important;
            padding: 10px !important;
        }
        
        .gallery-image-container {
            padding-top: 100% !important; /* 正方形 */
        }
        
        .gallery-title {
            font-size: 13px !important;
        }
        
        .gallery-description {
            display: none !important;
        }
    }
    """

    # 添加模态框 HTML
    gr.HTML("""
    <div id="imageModal" class="modal-overlay">
        <span class="modal-close-hint" onclick="closeModal()">&times;</span>
        <img id="modalImg" class="modal-image" src="" alt="">
        <div id="modalInfo" class="modal-info"></div>
    </div>
    """)
    
    # 添加自定义 JavaScript
    gr.HTML("""
    <script>
    console.log('🎨 美学提升插件 - CSS 布局 v2.0 已加载');
    
    // 延迟检查画廊容器样式
    setTimeout(() => {
        const galleryContainers = document.querySelectorAll('.gallery-container');
        console.log('📊 找到画廊容器数量:', galleryContainers.length);
        
        galleryContainers.forEach((container, index) => {
            const display = window.getComputedStyle(container).display;
            const gridTemplate = window.getComputedStyle(container).gridTemplateColumns;
            console.log(`📐 画廊容器 ${index + 1}:`, {
                display: display,
                gridTemplate: gridTemplate,
                hasCards: container.querySelectorAll('.gallery-card').length
            });
        });
    }, 2000);
    
    let currentImageInfo = null;
    
    function openModal(imagePath, title, description) {
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImg');
        const modalInfo = document.getElementById('modalInfo');
        
        if (!modal || !modalImg || !modalInfo) {
            console.error('模态框元素未找到');
            return;
        }
        
        // 将文件路径转换为 Gradio 可用的 URL
        const fileUrl = imagePath.startsWith('file=') ? imagePath : `file=${imagePath}`;
        modalImg.src = fileUrl;
        
        // 检查是否是画师卡片
        const isArtist = description && description.length > 50;
        if (isArtist) {
            modalInfo.innerHTML = `<strong>${title}</strong><br><br>${description}`;
        } else {
            modalInfo.innerHTML = `<strong>${title}</strong><br>${description || ''}`;
        }
        
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        
        currentImageInfo = { path: imagePath, title, description };
        console.log('✅ 模态框已打开:', title);
    }
    
    function closeModal() {
        const modal = document.getElementById('imageModal');
        if (!modal) return;
        
        modal.style.display = 'none';
        document.body.style.overflow = '';
        currentImageInfo = null;
        console.log('❌ 模态框已关闭');
    }
    
    // 绑定画廊卡片点击事件
    function bindGalleryClicks() {
        try {
            console.log('🎨 开始绑定画廊卡片点击事件...');
            
            // 查找所有画廊卡片和画师卡片
            const cards = document.querySelectorAll('.gallery-card, .artist-card');
            console.log('📊 找到', cards.length, '个卡片（画廊 + 画师）');
            
            // 检查画师卡片
            const artistCards = document.querySelectorAll('.artist-card');
            console.log('🎨 找到', artistCards.length, '个画师卡片');
            
            artistCards.forEach((card, index) => {
                console.log('🎨 画师卡片', index, ':', {
                    title: card.getAttribute('data-title'),
                    description: card.getAttribute('data-description'),
                    src: card.getAttribute('data-src')
                });
            });
            
            cards.forEach((card, index) => {
                const title = card.getAttribute('data-title') || '';
                const description = card.getAttribute('data-description') || '';
                const src = card.getAttribute('data-src') || '';
                
                if (!src) {
                    console.warn('⚠️ 跳过无效卡片 (无 src):', index);
                    return;
                }
                
                // 为整个卡片添加点击事件
                card.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🖼️ 卡片被点击:', title, '| 索引:', index);
                    openModal(src, title, description);
                });
                
                console.log('✓ 已绑定卡片:', title);
            });
            
            console.log('✅ 画廊点击事件绑定完成，共绑定', cards.length, '个卡片');
        } catch (error) {
            console.error('❌ 绑定画廊点击事件失败:', error);
        }
    }
    
    // 延迟绑定以确保 Gradio 渲染完成
    function initGallery() {
        setTimeout(() => {
            bindGalleryClicks();
        }, 1000);
    }
    
    // 页面加载后绑定事件
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initGallery);
    } else {
        initGallery();
    }
    
    // 支持 Tab 切换后重新绑定
    document.addEventListener('click', function(e) {
        if (e.target.closest('.tab-nav')) {
            setTimeout(() => {
                bindGalleryClicks();
            }, 500);
        }
    });
    
    // ESC 键关闭模态框
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
    
    // 点击模态框背景关闭
    const modalElement = document.getElementById('imageModal');
    if (modalElement) {
        modalElement.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModal();
            }
        });
    }
    </script>
    """)

    # 创建标签页
    with gr.Blocks(css=custom_css) as demo:
        with gr.Tab("🖼 图像识别", elem_id="aesthetic_image_recognition_tab"):
            create_vision_chat_tab()

        with gr.Tab("💡 打光辅助", elem_id="aesthetic_lighting_tab"):
            if LIGHTING_ASSISTANT_AVAILABLE:
                lighting_assistant_module.create_ui()
            else:
                gr.Markdown("⚠️ 打光辅助模块未加载")
        
        with gr.Tab("人物与场景分析", elem_id="aesthetic_character_scene_tab"):
            if PROPORTION_GUIDE_AVAILABLE:
                proportion_guide_module.create_proportion_guide_ui()
            else:
                gr.Markdown("⚠️ 人物与场景分析模块未加载")
        
        with gr.Tab("📐 构图技巧", elem_id="aesthetic_composition_tab"):
            create_composition_tab()
        with gr.Tab("🎨 画师百科", elem_id="aesthetic_artist_tab"):
            create_artist_tab()
    
    return demo


def MultiModal_tab():
    """注册到 WebUI 的标签页"""
    ui = aesthetic_enhancement_ui()
    return [(ui, "图像识别", "aesthetic_enhancement_tab")]


# 注册到 WebUI
try:
    script_callbacks.on_ui_tabs(MultiModal_tab)
except Exception as e:
    pass
