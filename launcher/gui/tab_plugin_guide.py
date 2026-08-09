"""插件指南 Tab - 介绍外置插件功能和使用方法"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame
)
from PyQt6.QtCore import Qt
from .theme import COLORS


class PluginGuideTab(QWidget):
    """插件指南标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        content = QWidget()
        content.setStyleSheet(f"background-color:{COLORS['bg_card']};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 32, 32, 32)
        content_layout.setSpacing(24)
        
        # 标题
        title = QLabel("🧩 插件使用指南")
        title.setStyleSheet(
            f"color:{COLORS['text_primary']};font-size:24px;font-weight:bold;"
        )
        content_layout.addWidget(title)
        
        # 说明文字
        intro = QLabel(
            "本页面详细介绍 WebUI Forge Neo v3 集成的所有外置插件功能、所需模型及使用方法。\n"
            "帮助您快速上手各类增强功能。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            f"color:{COLORS['text_secondary']};font-size:13px;line-height:1.6;"
        )
        content_layout.addWidget(intro)
        
        # 分割线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{COLORS['border']};")
        content_layout.addWidget(sep)
        
        # 按字母顺序排列的插件列表
        # 1. ADetailer
        adetailer_section = self._create_adetailer_section()
        content_layout.addWidget(adetailer_section)
        
        # 2. Auto-Photoshop-StableDiffusion-Plugin
        photoshop_section = self._create_photoshop_plugin_section()
        content_layout.addWidget(photoshop_section)
        
        # 3. Civitai Helper
        civitai_section = self._create_civitai_helper_section()
        content_layout.addWidget(civitai_section)
        
        # 4. Infinite Browsing
        infinite_browsing_section = self._create_infinite_browsing_section()
        content_layout.addWidget(infinite_browsing_section)
        
        # 5. LoRA Prompt Tool
        lora_tool_section = self._create_lora_prompt_tool_section()
        content_layout.addWidget(lora_tool_section)
        
        # 6. SD WebUI Aesthetic Enhancement
        aesthetic_section = self._create_aesthetic_enhancement_section()
        content_layout.addWidget(aesthetic_section)
        
        # 7. SD WebUI Aspect Ratio Helper
        aspect_ratio_section = self._create_aspect_ratio_helper_section()
        content_layout.addWidget(aspect_ratio_section)
        
        # 8. SD WebUI Auto Complete
        auto_complete_section = self._create_auto_complete_section()
        content_layout.addWidget(auto_complete_section)
        
        # 9. SD WebUI Camera Angle Selector
        camera_angle_section = self._create_camera_angle_selector_section()
        content_layout.addWidget(camera_angle_section)
        
        # 10. SD WebUI Model Guide
        model_guide_section = self._create_model_guide_section()
        content_layout.addWidget(model_guide_section)
        
        # 11. SD WebUI Multimodal Media
        multimodal_section = self._create_multimodal_media_section()
        content_layout.addWidget(multimodal_section)
        
        # 12. SD WebUI Prompt All-in-One
        prompt_allinone_section = self._create_prompt_allinone_section()
        content_layout.addWidget(prompt_allinone_section)
        
        # 13. SD WebUI Qwen Vision Chat
        qwen_vision_section = self._create_qwen_vision_chat_section()
        content_layout.addWidget(qwen_vision_section)
        
        # 14. SD WebUI SAM Matting
        sam_matting_section = self._create_sam_matting_section()
        content_layout.addWidget(sam_matting_section)
        
        # 15. SD WebUI See-Through
        see_through_section = self._create_see_through_section()
        content_layout.addWidget(see_through_section)
        
        # 16. SD WebUI Storyboard Assistant
        storyboard_section = self._create_storyboard_assistant_section()
        content_layout.addWidget(storyboard_section)
        
        # 17. Stable Diffusion WebUI Localization zh_Hans
        localization_section = self._create_localization_section()
        content_layout.addWidget(localization_section)
        
        # 18. WD14 Tagger
        wd14_tagger_section = self._create_wd14_tagger_section()
        content_layout.addWidget(wd14_tagger_section)
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def _create_section_title(self, title: str) -> QLabel:
        """创建章节标题"""
        label = QLabel(title)
        label.setStyleSheet(
            f"color:{COLORS['accent_light']};font-size:18px;font-weight:bold;margin-top:16px;"
        )
        return label
    
    def _create_subsection_title(self, title: str) -> QLabel:
        """创建子章节标题"""
        label = QLabel(title)
        label.setStyleSheet(
            f"color:{COLORS['text_primary']};font-size:15px;font-weight:bold;margin-top:12px;"
        )
        return label
    
    def _create_description(self, text: str) -> QLabel:
        """创建描述文本"""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(
            f"color:{COLORS['text_secondary']};font-size:12px;line-height:1.6;"
        )
        return label
    
    def _create_code_block(self, code: str) -> QLabel:
        """创建代码块"""
        label = QLabel(code)
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            QLabel {{
                background:{COLORS['bg_dark']};
                color:{COLORS['cyan']};
                border:1px solid {COLORS['border']};
                border-radius:6px;
                padding:12px;
                font-family:'Consolas','Courier New',monospace;
                font-size:11px;
                line-height:1.5;
            }}
        """)
        return label
    
    def _create_note(self, text: str, note_type: str = "info") -> QLabel:
        """创建提示框"""
        colors_map = {
            "info": ("#1e3a5f", COLORS['cyan']),
            "warning": ("#3d2e1f", COLORS['orange']),
            "tip": ("#1a3a2a", COLORS['green']),
        }
        bg, fg = colors_map.get(note_type, colors_map["info"])
        
        label = QLabel(f"💡 {text}")
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            QLabel {{
                background:{bg};
                color:{fg};
                border-left:4px solid {fg};
                border-radius:4px;
                padding:10px 14px;
                font-size:12px;
                line-height:1.5;
            }}
        """)
        return label
    
    def _create_adetailer_section(self) -> QWidget:
        """创建 ADetailer 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🔧 ADetailer - 自动面部修复"))
        layout.addWidget(self._create_description("Stable Diffusion WebUI 的自动遮罩和重绘扩展，类似 Detection Detailer，用于修复生成图像中的面部缺陷。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>自动检测</b>：使用 YOLO 或 MediaPipe 自动检测人脸、手部等<br>
• <b>智能重绘</b>：对检测到的区域进行局部重绘修复<br>
• <b>多对象支持</b>：可同时处理多个人脸、手部或其他对象<br>
• <b>参数调节</b>：支持调整置信度阈值、遮罩扩张等参数"""))
        
        layout.addWidget(self._create_note("""• 适用于修复扭曲的面部、错误的手部等常见问题<br>
• 建议在高分辨率生成时启用<br>
• 可在 Settings → ADetailer 中配置默认参数""", "tip"))
        
        return container
    
    def _create_photoshop_plugin_section(self) -> QWidget:
        """创建 Auto-Photoshop-StableDiffusion-Plugin 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🎨 Auto-Photoshop-StableDiffusion-Plugin - Photoshop 插件"))
        layout.addWidget(self._create_description("将 Stable Diffusion 集成到 Adobe Photoshop 中，实现无缝的 AI 图像编辑工作流。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>文生图</b>：在 Photoshop 中直接使用文本提示生成图像<br>
• <b>图生图</b>：基于选区或图层进行 AI 重绘<br>
• <b>实时预览</b>：在 Photoshop 界面中实时查看生成结果<br>
• <b>图层管理</b>：生成的图像自动创建为新图层<br>
• <b>参数同步</b>：与 WebUI 共享模型和设置"""))
        
        layout.addWidget(self._create_subsection_title("安装要求"))
        layout.addWidget(self._create_description("""• Adobe Photoshop（支持 ExtendScript 的版本）<br>
• WebUI 必须正在运行<br>
• 首次使用需在 Photoshop 中安装插件脚本"""))
        
        layout.addWidget(self._create_note("""• 适合专业设计师和艺术家的工作流程<br>
• 需要保持 WebUI 后台运行<br>
• 详细安装说明请参考插件目录下的 README.md""", "info"))
        
        return container
    
    def _create_civitai_helper_section(self) -> QWidget:
        """创建 Civitai Helper 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🌐 Civitai Helper - Civitai 模型助手"))
        layout.addWidget(self._create_description("帮助您从 Civitai 管理和下载模型的强大工具，支持一键下载、元数据同步和批量操作。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>一键下载</b>：直接从 Civitai 下载模型文件<br>
• <b>预览图下载</b>：自动下载模型的预览图片和示例图<br>
• <b>触发词提取</b>：自动提取并添加模型的推荐触发词<br>
• <b>版本管理</b>：检查并下载模型的最新版本<br>
• <b>元数据同步</b>：将 Civitai 的模型信息写入图片元数据<br>
• <b>批量操作</b>：支持批量下载和管理多个模型"""))
        
        layout.addWidget(self._create_subsection_title("配置要求"))
        layout.addWidget(self._create_description("""• <b>Civitai API Key</b>（可选但推荐）：用于下载受限模型<br>
• <b>代理设置</b>（如需要）：在中国大陆可能需要配置代理"""))
        
        layout.addWidget(self._create_note("""• 配置 API Key 可下载更多模型<br>
• 如遇下载失败，请检查控制台日志<br>
• 详细配置指南请参考插件目录下的 CIVITAI_HELPER_GUIDE.md""", "warning"))
        
        return container
    
    def _create_infinite_browsing_section(self) -> QWidget:
        """创建 Infinite Browsing 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("♾️ Infinite Browsing - 无限浏览"))
        layout.addWidget(self._create_description("提供无限滚动的图像浏览体验，支持大量图像的流畅加载和展示。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>虚拟滚动</b>：高效加载大量图像而不占用过多内存<br>
• <b>懒加载</b>：仅在需要时加载可见区域的图像<br>
• <b>快速导航</b>：支持键盘快捷键和鼠标滚轮快速浏览<br>
• <b>缩略图模式</b>：可切换不同大小的缩略图视图"""))
        
        layout.addWidget(self._create_note("""• 适合浏览大量生成结果<br>
• 显著降低内存占用<br>
• 提升大图库的浏览体验""", "tip"))
        
        return container
    
    def _create_lora_prompt_tool_section(self) -> QWidget:
        """创建 LoRA Prompt Tool 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🎯 LoRA Prompt Tool - LoRA 提示词工具"))
        layout.addWidget(self._create_description("帮助您管理大量 LoRA 模型的提示词、触发词和使用方法，快速调用所需模型。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>模型管理</b>：记录每个 LoRA 模型的专属提示词和触发词<br>
• <b>快速调用</b>：一键插入预设的提示词组合<br>
• <b>分类整理</b>：按风格、角色、场景等分类管理 LoRA<br>
• <b>权重调节</b>：可视化调节 LoRA 权重<br>
• <b>使用说明</b>：为每个 LoRA 保存详细的使用说明和注意事项"""))
        
        layout.addWidget(self._create_subsection_title("适用场景"))
        layout.addWidget(self._create_description("""• 训练了大量 LoRA 模型的用户<br>
• 需要频繁切换不同 LoRA 的工作流程<br>
• 团队协作时的标准化提示词管理"""))
        
        layout.addWidget(self._create_note("""• 建议为每个 LoRA 模型建立详细的档案<br>
• 包含最佳权重范围、适用场景等信息<br>
• 可大幅提升工作效率""", "tip"))
        
        return container
    
    def _create_aesthetic_enhancement_section(self) -> QWidget:
        """创建 Aesthetic Enhancement 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🎨 Aesthetic Enhancement - 美学提升"))
        layout.addWidget(self._create_description("全面的视觉参考工具，提供构图技巧、打光技巧和画师百科，提升 AI 绘画作品的美学品质。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>📐 构图技巧</b>：12 种经典构图示例（S 型、三角形、九宫格、对称式等）<br>
• <b>💡 打光技巧</b>：11 种专业光位演示（伦勃朗光、丁达尔光、逆光、侧光等）<br>
• <b>🎨 画师百科</b>：多风格知名画师作品和风格特点介绍<br>
• <b>🖼️ 点击放大</b>：支持全屏查看高清细节<br>
• <b>📋 一键复制</b>：画师名称和介绍可点击复制<br>
• <b>🎬 AI 智能分析</b>（可选）：Qwen3.5-VL 多模态大模型分析构图、灯光、分镜等元素"""))
        
        layout.addWidget(self._create_subsection_title("AI 分析功能（可选）"))
        layout.addWidget(self._create_description("""如需使用 AI 智能分析功能：<br>
1. 安装 Ollama：https://ollama.com/<br>
2. 下载 Qwen3.5 模型：<code>ollama run qwen3.5:4b</code><br>
3. 重启 WebUI 后即可使用"""))
        
        layout.addWidget(self._create_note("""• 适合 AI 绘画提示词参考<br>
• 摄影构图学习和灯光布置灵感<br>
• 美术基础训练和画师风格研究""", "tip"))
        
        return container
    
    def _create_aspect_ratio_helper_section(self) -> QWidget:
        """创建 Aspect Ratio Helper 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("⚠️ Aspect Ratio Helper - 宽高比助手（已归档）"))
        layout.addWidget(self._create_description("此插件已停止维护并归档。原始作者不再开发，但其功能可能已被其他插件或 WebUI 原生功能替代。"))
        
        layout.addWidget(self._create_note("""• 此仓库已归档，不再积极维护<br>
• 建议寻找更新的替代方案<br>
• WebUI 新版本可能已内置类似功能""", "warning"))
        
        return container
    
    def _create_auto_complete_section(self) -> QWidget:
        """创建 Auto Complete 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("✨ Auto Complete - 自动补全"))
        layout.addWidget(self._create_description("在输入提示词时自动补全和建议 Booru 标签的扩展，提高提示词输入效率。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>智能补全</b>：根据输入内容自动建议相关标签<br>
• <b>Booru 标签库</b>：内置丰富的 Danbooru 风格标签数据库<br>
• <b>实时建议</b>：打字时即时显示匹配的建议<br>
• <b>键盘导航</b>：支持上下键选择建议，Tab 键确认<br>
• <b>兼容 Forge</b>：完全兼容 Stable Diffusion WebUI Forge"""))
        
        layout.addWidget(self._create_note("""• 大幅提升提示词输入速度<br>
• 减少拼写错误<br>
• 帮助发现新的标签和概念""", "tip"))
        
        return container
    
    def _create_camera_angle_selector_section(self) -> QWidget:
        """创建 Camera Angle Selector 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("📷 Camera Angle Selector - 相机角度选择器"))
        layout.addWidget(self._create_description("专业的 Stable Diffusion WebUI 多角度提示词可视化选择器插件。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>3D 可视化</b>：直观的角度选择和预览<br>
• <b>方位角调节</b>：水平旋转相机视角<br>
• <b>高程角调节</b>：垂直俯仰角度调整<br>
• <b>距离控制</b>：调节相机与被摄物体的距离<br>
• <b>一键插入</b>：点击即可将角度提示词插入到提示词框<br>
• <b>预设角度</b>：提供常用的标准角度预设"""))
        
        layout.addWidget(self._create_note("""• 适合需要精确控制构图的场景<br>
• 帮助理解不同角度的视觉效果<br>
• 提升提示词的准确性和专业性""", "tip"))
        
        return container
    
    def _create_model_guide_section(self) -> QWidget:
        """创建 Model Guide 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("📚 Model Guide - 模型使用说明"))
        layout.addWidget(self._create_description("为 SD-WebUI Forge Neo 整合包提供详细的模型使用说明，帮助用户快速了解各种模型的组件结构、目录位置和使用方法。"))
        
        layout.addWidget(self._create_subsection_title("主要内容"))
        layout.addWidget(self._create_description("""• <b>模型架构说明</b>：详细介绍支持的各类模型架构<br>
• <b>组件结构</b>：每种模型所需的文件组成和目录结构<br>
• <b>配置指南</b>：如何正确放置和配置模型文件<br>
• <b>显存要求</b>：不同模型的显存需求和建议配置<br>
• <b>最佳实践</b>：模型使用的注意事项和优化建议"""))
        
        layout.addWidget(self._create_note("""• 新用户必读的模型入门指南<br>
• 遇到模型加载问题时首先查阅此文档<br>
• 定期更新以支持最新模型""", "info"))
        
        return container
    
    def _create_multimodal_media_section(self) -> QWidget:
        """创建 Multimodal Media 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🎬 Multimodal Media - 多模态媒体处理"))
        layout.addWidget(self._create_description("多功能多媒体处理插件，支持 TTS、数字人、视频生成等多种媒体处理功能。"))
        
        layout.addWidget(self._create_subsection_title("功能模块"))
        layout.addWidget(self._create_description("""<table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr style="background:#1f2937;">
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">功能</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">说明</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">主要依赖</th>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;"><b>Qwen3-TTS</b><br>语音合成</td>
                <td style="padding:8px;border:1px solid #374151;">文本转语音，支持多种音色和风格定制</td>
                <td style="padding:8px;border:1px solid #374151;">qwen-tts、soundfile</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;"><b>LatentSync</b><br>数字人对口型</td>
                <td style="padding:8px;border:1px solid #374151;">根据图片 + 音频生成对口型视频</td>
                <td style="padding:8px;border:1px solid #374151;">insightface、onnxruntime-gpu</td>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;"><b>Qwen Video</b><br>万相视频生成</td>
                <td style="padding:8px;border:1px solid #374151;">图生视频、文生视频、关键帧生视频</td>
                <td style="padding:8px;border:1px solid #374151;">dashscope（阿里云 API）</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;"><b>视频关键帧提取</b></td>
                <td style="padding:8px;border:1px solid #374151;">从视频中提取帧图像，支持批量导出</td>
                <td style="padding:8px;border:1px solid #374151;">ffmpeg-python</td>
            </tr>
        </table>"""))
        
        layout.addWidget(self._create_subsection_title("模型组件结构"))
        layout.addWidget(self._create_code_block("""models/qwen3-tts/  # Qwen3-TTS 语音合成模型
├── Qwen3-TTS-12Hz-1.7B-Base           # 基础版本
├── Qwen3-TTS-12Hz-1.7B-CustomVoice    # 自定义音色版本
└── Qwen3-TTS-12Hz-1.7B-VoiceDesign    # 音色设计版本

models/LatentSync/  # LatentSync 唇形同步模型
└── ...  # 唇形同步模型文件"""))
        
        layout.addWidget(self._create_subsection_title("Python 依赖（自动安装）"))
        layout.addWidget(self._create_description("""插件会在 WebUI 启动时自动检查并安装以下 Python 包：<br>
• <code>insightface</code> - 人脸检测和分析<br>
• <code>onnxruntime-gpu</code> - GPU 加速的 ONNX 运行时<br>
• <code>ffmpeg-python</code> - FFmpeg Python 绑定<br>
• <code>torchaudio</code> - PyTorch 音频处理<br>
• <code>qwen-tts</code> - Qwen3-TTS 语音合成<br>
• <code>soundfile</code>、<code>resampy</code>、<code>librosa</code> - 音频处理<br>
• <code>dashscope</code> - 阿里云百炼 SDK（Qwen Video）<br>
• <code>Pillow</code> - 图像处理"""))
        
        layout.addWidget(self._create_subsection_title("系统工具（可选但推荐）"))
        layout.addWidget(self._create_description("""<b>FFmpeg</b>（推荐安装）⭐<br>
用于视频处理和音频转换，可从 https://ffmpeg.org/ 下载<br>
安装后将 ffmpeg.exe 所在目录添加到系统 PATH 环境变量"""))
        
        layout.addWidget(self._create_note("""• 插件会自动安装所有 Python 依赖<br>
• Qwen Video 需要配置阿里云 DashScope API Key<br>
• LatentSync 需要 NVIDIA GPU 支持 CUDA<br>
• 首次使用可能需要较长时间下载模型和依赖""", "warning"))
        
        return container
    
    def _create_prompt_allinone_section(self) -> QWidget:
        """创建 Prompt All-in-One 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("💬 Prompt All-in-One - 提示词一体化"))
        layout.addWidget(self._create_description("强大的提示词管理工具，支持翻译、历史记录、收藏等功能，针对 Forge Neo 最新版本优化。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>实时翻译</b>：中英文提示词双向翻译<br>
• <b>历史记录</b>：自动保存使用过的提示词<br>
• <b>收藏夹</b>：收藏常用提示词组合<br>
• <b>提示词模板</b>：预设多种风格的提示词模板<br>
• <b>语法高亮</b>：提示词语法高亮显示<br>
• <b>快捷操作</b>：一键清除、复制、粘贴提示词"""))
        
        layout.addWidget(self._create_note("""• 针对 Python 3.13 进行了适配和修复<br>
• 显著提升提示词编写效率<br>
• 适合多语言用户使用""", "tip"))
        
        return container
    
    def _create_qwen_vision_chat_section(self) -> QWidget:
        """创建 Qwen Vision Chat 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🤖 Qwen Vision Chat - 图像识别与语言交互"))
        layout.addWidget(self._create_description("基于 Ollama 的 Qwen 多模态大模型，支持图像识别和智能对话"))
        
        layout.addWidget(self._create_subsection_title("支持的视觉模型（带图片识别）"))
        layout.addWidget(self._create_description("""<table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr style="background:#1f2937;">
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">模型名称</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">显存要求</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">特点</th>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;"><code>qwen3.5:9b</code></td>
                <td style="padding:8px;border:1px solid #374151;">16GB+</td>
                <td style="padding:8px;border:1px solid #374151;">高精度版本，识别最准确</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;"><code>qwen3.5:4b</code></td>
                <td style="padding:8px;border:1px solid #374151;">12GB</td>
                <td style="padding:8px;border:1px solid #374151;">平衡版本，推荐大多数用户</td>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;"><code>qwen3-vl:8b</code></td>
                <td style="padding:8px;border:1px solid #374151;">12GB+</td>
                <td style="padding:8px;border:1px solid #374151;">视觉语言模型</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;"><code>qwen3-vl:4b</code></td>
                <td style="padding:8px;border:1px solid #374151;">10GB</td>
                <td style="padding:8px;border:1px solid #374151;">中等视觉模型</td>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;"><code>qwen3-vl:2b</code></td>
                <td style="padding:8px;border:1px solid #374151;">8GB</td>
                <td style="padding:8px;border:1px solid #374151;">轻量级，低显存首选</td>
            </tr>
        </table>"""))
        
        layout.addWidget(self._create_subsection_title("支持的语言模型（纯文本对话）"))
        layout.addWidget(self._create_description("""• <code>qwen3:latest</code> - 最新版本<br>
• <code>qwen3.5:4b</code> - 平衡版本<br>
• <code>deepseek-r1:8b</code> - DeepSeek 模型"""))
        
        layout.addWidget(self._create_subsection_title("安装步骤"))
        layout.addWidget(self._create_description("""1. 安装 Ollama：访问 https://ollama.com/ 下载安装<br>
2. 下载模型（示例）：<br>
&nbsp;&nbsp;&nbsp;<code>ollama run qwen3.5:4b</code> （推荐）<br>
&nbsp;&nbsp;&nbsp;<code>ollama run qwen3-vl:2b</code> （低显存）<br>
3. 插件已预装在 extensions 目录<br>
4. 重启 WebUI Forge，在顶部标签页找到"图像识别与语言交互" """))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• 🖼️ <b>视觉模型对话</b>：上传图片进行智能识别和对话<br>
• 💬 <b>纯文本对话</b>：无需图片即可进行文本交流<br>
• 🔄 <b>多轮对话</b>：支持上传一次图片后连续提问<br>
• 🎯 <b>快捷描述</b>：自然语言、MJ提示词、分镜构图、视频生成等<br>
• 📁 <b>标签管理</b>：批量处理 txt 文件的关键词添加/删除<br>
• 🖼️ <b>图像管理</b>：图片预览、目录加载、网格展示"""))
        
        layout.addWidget(self._create_note("""• 需要先安装 Ollama 并下载对应模型<br>
• 视觉模型（带 vl 后缀）才能识别图片<br>
• 显存不足时选择更小的模型（如 qwen3-vl:2b）<br>
• Ollama 默认端口：11434""", "tip"))
        
        return container
    
    def _create_sam_matting_section(self) -> QWidget:
        """创建 SAM Matting 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("✂️ SAM Matting - 智能分割与抠图"))
        layout.addWidget(self._create_description("集成图像分割、智能抠图和图像定点清理功能的强大工具，基于 Segment Anything Model (SAM)。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>智能分割</b>：基于 SAM 模型的精准图像分割<br>
• <b>一键抠图</b>：快速分离主体和背景<br>
• <b>定点清理</b>：精确移除图像中的指定元素<br>
• <b>多种模式</b>：支持点选、框选、文本提示等多种分割方式<br>
• <b>批量处理</b>：支持批量分割和处理多张图像"""))
        
        layout.addWidget(self._create_subsection_title("应用场景"))
        layout.addWidget(self._create_description("""• 人物抠图和背景替换<br>
• 产品图去背景<br>
• 图像元素移除和清理<br>
• 素材准备和后期处理"""))
        
        layout.addWidget(self._create_note("""• 基于 Meta 的 SAM 模型，分割精度高<br>
• 支持 GPU 加速，处理速度快<br>
• 适合需要精细抠图的工作流程""", "tip"))
        
        return container
    
    def _create_see_through_section(self) -> QWidget:
        """创建 See-Through 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🎨 See-Through - 动漫图层分解"))
        layout.addWidget(self._create_description("基于 AI 的动漫图像分解工具，将单张动漫图像转换为多层 PSD 文件"))
        
        layout.addWidget(self._create_subsection_title("模型组件结构"))
        layout.addWidget(self._create_code_block("""models/diffusers/
├── models--24yearsold--seethroughv0.0.1_marigold_nf4   # Marigold深度估计模型（NF4量化）
└── models--24yearsold--seethroughv0.0.2_layerdiff3d_nf4  # LayerDiff 3D透明层生成模型（NF4量化）

models/sams/  # SAM场景分割模型
└── ...  # vit_b/vit_l/vit_h 等版本"""))
        
        layout.addWidget(self._create_subsection_title("核心技术"))
        layout.addWidget(self._create_description("""• <b>LayerDiff 3D</b>：基于扩散的透明层生成<br>
• <b>Marigold 深度估计</b>：专为动漫优化的伪深度估计<br>
• <b>SAM 分割</b>：语义身体部分分割<br>
• <b>PSD 输出</b>：导出为可编辑的 Photoshop 文件"""))
        
        layout.addWidget(self._create_subsection_title("分割模式"))
        layout.addWidget(self._create_description("""<table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr style="background:#1f2937;">
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">模式</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">适用场景</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">特点</th>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;"><b>人物分割</b></td>
                <td style="padding:8px;border:1px solid #374151;">动漫人物、写实人物、人物插画</td>
                <td style="padding:8px;border:1px solid #374151;">分层头发、身体部位、饰品、装备等</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;"><b>场景分割 (SAM)</b></td>
                <td style="padding:8px;border:1px solid #374151;">复杂场景、背景元素</td>
                <td style="padding:8px;border:1px solid #374151;">自动分割场景中的多个元素</td>
            </tr>
        </table>"""))
        
        layout.addWidget(self._create_subsection_title("SAM 模型类型"))
        layout.addWidget(self._create_description("""• <code>vit_b</code> - 小模型（~1.2GB），速度快<br>
• <code>vit_l</code> - 中模型（~2.5GB），平衡选择<br>
• <code>vit_h</code> - 大模型（~3.9GB），精度最高"""))
        
        layout.addWidget(self._create_subsection_title("显存配置建议"))
        layout.addWidget(self._create_description("""<table style="width:100%;border-collapse:collapse;margin:10px 0;">
            <tr style="background:#1f2937;">
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">显存</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">分辨率</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">批处理</th>
                <th style="padding:8px;border:1px solid #374151;text-align:left;color:#60a5fa;">优化选项</th>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;">8GB</td>
                <td style="padding:8px;border:1px solid #374151;">512-768</td>
                <td style="padding:8px;border:1px solid #374151;">1-2</td>
                <td style="padding:8px;border:1px solid #374151;">启用 NF4 量化、缓存嵌入</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;">12GB</td>
                <td style="padding:8px;border:1px solid #374151;">768-1024</td>
                <td style="padding:8px;border:1px solid #374151;">2-3</td>
                <td style="padding:8px;border:1px solid #374151;">启用缓存嵌入</td>
            </tr>
            <tr>
                <td style="padding:8px;border:1px solid #374151;">16GB</td>
                <td style="padding:8px;border:1px solid #374151;">1024-1280</td>
                <td style="padding:8px;border:1px solid #374151;">3-4</td>
                <td style="padding:8px;border:1px solid #374151;">默认设置</td>
            </tr>
            <tr style="background:#111827;">
                <td style="padding:8px;border:1px solid #374151;">24GB+</td>
                <td style="padding:8px;border:1px solid #374151;">1280-1536</td>
                <td style="padding:8px;border:1px solid #374151;">4-6</td>
                <td style="padding:8px;border:1px solid #374151;">可使用较大分辨率</td>
            </tr>
        </table>"""))
        
        layout.addWidget(self._create_subsection_title("内存优化选项"))
        layout.addWidget(self._create_description("""• <b>NF4 量化 (8GB GPU)</b>：使用 4 位量化，峰值显存~8GB<br>
• <b>缓存文本嵌入</b>：节省约 2GB 显存且无速度损失<br>
• <b>组卸载</b>：将显存降至~0.2GB，但速度降低 2-3 倍<br>
• <b>CPU 卸载</b>：将模型卸载到 CPU 以节省显存"""))
        
        layout.addWidget(self._create_subsection_title("输出文件"))
        layout.addWidget(self._create_description("""• <b>人物分割</b>：<code>extensions\\sd-webui-see-through\\see-through\\workspace\\layerdiff_output\\</code><br>
• <b>场景分割</b>：<code>extensions\\sd-webui-see-through\\see-through\\workspace\\scene_output\\</code><br>
• 包含：分层 PSD 文件、各图层 PNG、深度图、分割掩码"""))
        
        layout.addWidget(self._create_note("""• 适用于动漫人物、写实人物、人物插画<br>
• 动物图像只能获得粗略分割结果<br>
• 插件会自动安装依赖（psd-tools、pycocotools）<br>
• 推理步数推荐 20-30，较低可加快速度""", "tip"))
        
        return container
    
    def _create_storyboard_assistant_section(self) -> QWidget:
        """创建 Storyboard Assistant 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🎬 Storyboard Assistant - 分镜助手"))
        layout.addWidget(self._create_description("专业的 Stable Diffusion WebUI 剧本与分镜管理插件。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>📖 剧本管理</b>：创建和管理多个独立的故事项目，支持多种题材分类<br>
• <b>👥 角色管理</b>：每个故事独立管理角色，支持角色配图和详细档案<br>
• <b>🎨 分镜墙</b>：3×3 布局的分镜宫格（每页 9 个），支持可视化编排<br>
• <b>🖼️ 图片管理</b>：点击宫格上传/替换图片，支持剪贴板粘贴<br>
• <b>🎵 音频支持</b>：每个分镜可添加音频文件，支持音频移动和删除<br>
• <b>📝 注释功能</b>：为每个分镜添加文字注释<br>
• <b>📊 分镜管理</b>：调整分镜顺序、删除分镜、分页浏览<br>
• <b>📄 导出功能</b>：导出完整剧本（含角色档案）为 TXT 文件，导出分镜数据"""))
        
        layout.addWidget(self._create_subsection_title("快速开始"))
        layout.addWidget(self._create_description("""1. 在顶部导航栏点击"分镜助手"标签<br>
2. 点击"➕ 新建"创建新故事<br>
3. 输入故事标题和题材类型<br>
4. 在剧本编辑器中编写剧本<br>
5. 添加角色及其档案<br>
6. 在分镜墙中上传和管理分镜图片<br>
7. 导出完整的剧本和分镜数据"""))
        
        layout.addWidget(self._create_note("""• 适合漫画创作者、动画制作人和影视策划人员<br>
• 支持从构思到分镜的完整工作流程<br>
• 可与 AI 图像生成结合，快速制作概念分镜""", "tip"))
        
        return container
    
    def _create_localization_section(self) -> QWidget:
        """创建 Localization 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🇨🇳 Localization zh_Hans - 简体中文翻译"))
        layout.addWidget(self._create_description("Stable Diffusion WebUI 的简体中文翻译扩展，提供完整的中文界面支持。"))
        
        layout.addWidget(self._create_subsection_title("功能特性"))
        layout.addWidget(self._create_description("""• <b>完整汉化</b>：WebUI 界面的全面中文翻译<br>
• <b>自动更新</b>：每 1 小时从翻译平台拉取最新快照<br>
• <b>持续维护</b>：社区共同维护和更新翻译内容<br>
• <b>无缝切换</b>：可在设置中轻松切换语言"""))
        
        layout.addWidget(self._create_note("""• 安装后在 Settings → User Interface → Localization 中选择"zh_Hans"<br>
• 点击"Reload UI"应用更改<br>
• 欢迎参与翻译改进""", "info"))
        
        return container
    
    def _create_wd14_tagger_section(self) -> QWidget:
        """创建 WD14 Tagger 部分"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        layout.addWidget(self._create_section_title("🏷️ WD14 Tagger - 标签生成器"))
        layout.addWidget(self._create_description("使用 DeepDanbooru 等模型为单张或多张图像文件生成 Booru 风格标签的 interrogate 工具。"))
        
        layout.addWidget(self._create_subsection_title("核心功能"))
        layout.addWidget(self._create_description("""• <b>自动打标</b>：使用多种模型自动生成图像标签<br>
• <b>批量处理</b>：支持同时处理多张图像<br>
• <b>多模型支持</b>：支持 DeepDanbooru、WD14 等多种打标模型<br>
• <b>置信度调节</b>：可调整标签的置信度阈值<br>
• <b>标签编辑</b>：生成后可手动编辑和调整标签<br>
• <b>反向提示词</b>：可选择将某些标签加入负面提示词"""))
        
        layout.addWidget(self._create_subsection_title("应用场景"))
        layout.addWidget(self._create_description("""• 为训练数据集自动生成标签<br>
• 分析现有图像的标签组成<br>
• 学习高质量提示词的构成<br>
• 批量整理图像库"""))
        
        layout.addWidget(self._create_note("""• 支持多种打标模型，可根据需求选择<br>
• 标签生成质量取决于所选模型<br>
• 建议结合人工审核使用""", "tip"))
        
        return container