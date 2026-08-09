"""
软件使用教程插件 - SD WebUI Forge 教程中心
提供插件篇、生图模型篇、编辑模型篇、软件知识篇、PS AI插件篇、云端篇、AI辅助设计篇的教程视频
"""

import os
import gradio as gr
import logging
from pathlib import Path
from modules import script_callbacks

logger = logging.getLogger(__name__)

EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(EXTENSION_DIR, "images")

# ============================================================
# 教程数据结构
# 用户可在下方添加/修改教程条目
# 每个条目格式: { "title": "教程标题", "cover": "封面图片文件名", "url": "视频链接" }
# ============================================================

TUTORIAL_DATA = {
    "插件篇": {
        "icon": "🔌",
        "items": [
            { "title": "中英文本地翻译", "cover": "中英文本地翻译.png", "url": "https://www.bilibili.com/video/BV1oUEu6KEWo/" },
            { "title": "打光辅助", "cover": "打光辅助.png", "url": "https://www.bilibili.com/video/BV16JJN6YEBy/" },
            { "title": "图生3D", "cover": "图生3D.png", "url": "https://www.bilibili.com/video/BV1T6GH6zEuv/" },
            { "title": "区域控制", "cover": "区域控制.png", "url": "https://www.bilibili.com/video/BV1ffGG6eEJG/" },
            { "title": "音乐生成", "cover": "音乐生成.png", "url": "https://www.bilibili.com/video/BV1zYVP6VEPk/" },
            { "title": "声音合成", "cover": "声音合成.png", "url": "https://www.bilibili.com/video/BV1m8ATzdEDG/" },
            { "title": "人物图层分离", "cover": "人物图层分离.png", "url": "https://www.bilibili.com/video/BV1AhN167EbH/" },
            { "title": "场景与人物结合分离", "cover": "场景与人物结合分离.png", "url": "https://www.bilibili.com/video/BV1cHogBMEt3/" },
            { "title": "图像识别与数据标注", "cover": "图像识别与数据标注.png", "url": "https://www.bilibili.com/video/BV1xkMHzkE6n/" },
            { "title": "开源社区模型下载器", "cover": "开源社区模型下载器.png", "url": "https://www.bilibili.com/video/BV1ya756bEdx/" },
        ]
    },
    "生图模型篇": {
        "icon": "🫀",
        "items": [
            { "title": "krea2模型", "cover": "krea2模型.png", "url": "https://www.bilibili.com/video/BV19SKQ6HErw/" },
            { "title": "anima模型", "cover": "anima模型.png", "url": "https://www.bilibili.com/video/BV1v6XYBUE1v/" },
            { "title": "开源模型发展史", "cover": "开源模型发展史.jpg", "url": "https://www.bilibili.com/video/BV16STu6LELd/" },
            { "title": "模型知识扫盲", "cover": "模型知识扫盲.png", "url": "https://www.bilibili.com/video/BV1zE1mByEdt/" },
            { "title": "int8加速模型", "cover": "int8加速模型.png", "url": "https://www.bilibili.com/video/BV1dkM86eEr7/" },
        ]
    },
    "编辑模型篇": {
        "icon": "✏️",
        "items": [
            { "title": "klein编辑的十种技巧", "cover": "klein编辑的十种技巧.png", "url": "https://www.bilibili.com/video/BV1zhQ4BqEZ7/" },
            { "title": "3D软件pose编辑+klein", "cover": "3D软件pose编辑+klein.png", "url": "https://www.bilibili.com/video/BV161V26NEQF/" },
        ]
    },
    "软件知识篇": {
        "icon": "📚",
        "items": [
            { "title": "AI软件常用命令教程", "cover": "AI软件常用命令教程.jpg", "url": "https://www.bilibili.com/video/BV1DX7k6UExy/" },
            { "title": "UI界面使用介绍", "cover": "UI界面使用介绍.png", "url": "https://www.bilibili.com/video/BV1WmDyBdEsk/" },
            { "title": "运行加速优化", "cover": "运行加速优化.png", "url": "https://www.bilibili.com/video/BV1S79uBgEnq/" },
            { "title": "webui发展史", "cover": "webui发展史.png", "url": "https://www.bilibili.com/video/BV1AKX1BhEJH/" },
        ]
    },
    "PS AI插件篇": {
        "icon": "🎨",
        "items": [
            { "title": "PS桥接WebUI", "cover": "PS桥接webui.jpg", "url": "https://www.bilibili.com/video/BV1obgi6sEUb/" },
            { "title": "PS-Blender-AI绘画三合一", "cover": "PS-Blender-AI绘画三合一.jpg", "url": "https://www.bilibili.com/video/BV1e73p6JEXR/" },
        ]
    },
    "云端篇": {
        "icon": "☁️",
        "items": [
            { "title": "云服务器使用webui", "cover": "云服务器使用webui.png", "url": "https://www.bilibili.com/video/BV19MdYBHE3K/" },
            { "title": "云端上传与下载", "cover": "云端上传与下载.png", "url": "https://www.bilibili.com/video/BV1Zejo6PEQp/" },
        ]
    },
    "AI辅助设计篇": {
        "icon": "💡",
        "items": [
            { "title": "多人互动插画教程", "cover": "多人互动.png", "url": "https://www.bilibili.com/video/BV1Gk3c6XED5/" },
            { "title": "女精灵射手角色设计辅助", "cover": "精灵射手.png", "url": "https://www.bilibili.com/video/BV1PTKT6VEoe/" },
            { "title": "多视角表情绘制", "cover": "多视角表情.png", "url": "https://www.bilibili.com/video/BV1WE421A7cX/" },
            { "title": "修复手型", "cover": "修复手型.png", "url": "https://www.bilibili.com/video/BV1id7T6fEfm/" },
            { "title": "俯视场景", "cover": "俯视场景.png", "url": "https://www.bilibili.com/video/BV1nu411K7tR/" },
            { "title": "3D辅助+AI二创动画制作全流程", "cover": "3D辅助+AI动画制作全流程.jpg", "url": "https://www.bilibili.com/video/BV1aUMz6HEAi/" },
            { "title": "华强买瓜影视二创全流程", "cover": "华强买瓜影视二创全流程.png", "url": "https://www.bilibili.com/video/BV1HYRBB2Ez2/" },
        ]
    },
    "AI训练篇": {
        "icon": "🏋️",
        "items": [
            { "title": "云端贴图AI训练", "cover": "云端贴图AI训练.png", "url": "https://www.bilibili.com/video/BV1ee3h66EGL/" },
            { "title": "云端训练成本计算", "cover": "云端训练成本计算.png", "url": "https://www.bilibili.com/video/BV1pb4zerEA1/" },
            { "title": "本地训练器界面介绍", "cover": "本地训练器界面介绍.png", "url": "https://www.bilibili.com/video/BV1pzcizxEPL/" },
            { "title": "云端krea2训练龙珠画风", "cover": "云端krea2训练龙珠画风.png", "url": "https://www.bilibili.com/video/BV11ZKo6HEF6/" },
        ]
    }
}


def get_image_path(image_name):
    """获取封面图片路径"""
    if image_name:
        path = os.path.join(IMAGES_DIR, image_name)
        if os.path.exists(path):
            return path
    return None


def generate_tutorial_html(category_name, items, is_active=False):
    """生成单个分类的教程卡片HTML"""
    import html
    display_style = "" if is_active else ' style="display:none"'
    
    cards = []
    for item in items:
        title = html.escape(item.get("title", "教程"))
        cover = item.get("cover", "")
        url = html.escape(item.get("url", ""))
        
        # 封面图片路径
        cover_path = get_image_path(cover) if cover else None
        
        # 卡片
        card_parts = []
        card_parts.append(f'<div class="tutorial-card">')
        
        # 封面区域
        card_parts.append(f'<div class="tutorial-card-cover">')
        if cover_path and os.path.exists(cover_path):
            card_parts.append(f'<img src="file={cover_path}" alt="{title}">')
        else:
            card_parts.append(f'<div class="tutorial-card-placeholder">')
            card_parts.append(f'<span class="placeholder-icon">📖</span>')
            card_parts.append(f'<span class="placeholder-text">{title}</span>')
            card_parts.append(f'</div>')
        card_parts.append(f'</div>')
        
        # 标题
        card_parts.append(f'<div class="tutorial-card-title">{title}</div>')
        
        # 播放按钮
        if url:
            card_parts.append(f'<a href="{url}" target="_blank" class="tutorial-card-btn">▶ 播放</a>')
        else:
            card_parts.append(f'<span class="tutorial-card-btn disabled">即将上线</span>')
        
        card_parts.append(f'</div>')
        cards.append("".join(card_parts))
    
    return f"""
    <div class="tutorial-category-page" data-category="{html.escape(category_name)}"{display_style}>
        <div class="tutorial-category-title">{html.escape(category_name)}</div>
        <div class="tutorial-cards-grid">
            {''.join(cards)}
        </div>
    </div>
    """


def generate_all_html():
    """生成所有分类的完整HTML（纯前端切换，一次性生成）"""
    import html
    
    category_keys = list(TUTORIAL_DATA.keys())
    first_cat = category_keys[0]
    
    # 侧边栏分类按钮（使用 onclick 直接绑定，最可靠的方式）
    sidebar_items = []
    for i, cat_key in enumerate(category_keys):
        cat = TUTORIAL_DATA[cat_key]
        active_class = " active" if cat_key == first_cat else ""
        sidebar_items.append(
            f'<button class="tutorial-cat-btn{active_class}" data-category="{html.escape(cat_key)}">'
            f'{cat.get("icon", "📁")} {html.escape(cat_key)}'
            f'</button>'
        )
    
    # 所有分类的页面内容
    pages_html = []
    for cat_key in category_keys:
        cat = TUTORIAL_DATA[cat_key]
        is_active = (cat_key == first_cat)
        page_html = generate_tutorial_html(cat_key, cat.get("items", []), is_active)
        pages_html.append(page_html)
    
    html_content = f"""
    <div id="tutorial-content-area" class="tutorial-container">
        <div class="tutorial-sidebar">
            <div class="tutorial-sidebar-title">📚 教程目录</div>
            {''.join(sidebar_items)}
        </div>
        <div class="tutorial-main">
            {''.join(pages_html)}
        </div>
    </div>
    """
    
    return html_content


def create_ui():
    """创建教程中心界面"""
    
    html_content = generate_all_html()
    
    with gr.Blocks(css="", elem_id="tutorial-tab") as ui:
        gr.Markdown("""
        # 📚 软件使用教程中心
        
        涵盖插件使用、模型训练、PS AI插件、云端部署等全方位教程，助你快速上手。
        
        ---
        """)
        
        # 教程内容区域（HTML）
        tutorial_content = gr.HTML(value=html_content, elem_id="tutorial-content")
    
    return ui


def on_ui_tabs():
    """注册到 WebUI 标签页"""
    ui = create_ui()
    return [(ui, "📚 教程中心", "tutorial_center")]


# 注册扩展
script_callbacks.on_ui_tabs(on_ui_tabs)

logger.info("教程中心插件加载完成")