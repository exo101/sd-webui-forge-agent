"""插件管理 Tab"""
import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QCheckBox, QProgressBar,
    QDialog, QTextEdit, QMessageBox, QMenu, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QFont, QCursor
from .theme import COLORS
from core.extension_manager import (
    ExtInfo, ScanWorker, CheckUpdateWorker, UpdateWorker, InstallWorker,
    uninstall_extension, switch_branch, EXT_DIR
)
from core.paths import WEBUI_DIR

COL_EN   = 0
COL_NAME = 1
COL_URL  = 2
COL_BR   = 3
COL_HASH = 4
COL_DATE = 5
COL_STAT = 6
COL_ACT  = 7

# 常见插件简介（根据插件名称匹配）
EXTENSION_DESCRIPTIONS = {
    "sd-webui-controlnet": "ControlNet - 通过边缘、深度、姿态等条件精确控制生成图像的构图和结构",
    "controlnet": "ControlNet - 通过边缘、深度、姿态等条件精确控制生成图像的构图和结构",
    
    "adetailer": "ADetailer - 自动检测并修复生成图像中的面部缺陷，提升人脸质量",
    "sd-webui-adetailer": "ADetailer - 自动检测并修复生成图像中的面部缺陷，提升人脸质量",
    
    "sd-webui-reactor": "ReActor - 快速换脸插件，支持单张或多张人脸替换",
    "reactor": "ReActor - 快速换脸插件，支持单张或多张人脸替换",
    
    "sd-webui-roop": "Roop - 一键换脸插件，基于 InsightFace 实现人脸识别和替换",
    "roop": "Roop - 一键换脸插件，基于 InsightFace 实现人脸识别和替换",
    
    "sd-webui-infinite-image-browsing": "无限图片浏览 - 强大的图片管理和浏览工具，支持标签、搜索、批量操作",
    "infinite-image-browsing": "无限图片浏览 - 强大的图片管理和浏览工具，支持标签、搜索、批量操作",
    
    "sd-webui-tagcomplete": "Tag Complete - 自动补全提示词标签，支持 Danbooru 等标签库",
    "tagcomplete": "Tag Complete - 自动补全提示词标签，支持 Danbooru 等标签库",
    
    "sd-webui-civitai-helper": "Civitai Helper - Civitai 模型管理助手，支持一键下载和更新模型",
    "civitai-helper": "Civitai Helper - Civitai 模型管理助手，支持一键下载和更新模型",
    
    "sd-webui-model-converter": "模型转换器 - 在不同格式之间转换模型文件（ckpt/safetensors）",
    "model-converter": "模型转换器 - 在不同格式之间转换模型文件（ckpt/safetensors）",
    
    "sd-webui-lora-block-weight": "LoRA Block Weight - 精细调整 LoRA 各层权重，实现更复杂的风格混合",
    "lora-block-weight": "LoRA Block Weight - 精细调整 LoRA 各层权重，实现更复杂的风格混合",
    
    "sd-webui-prompt-all-in-one": "提示词一体化 - 集成提示词管理、翻译、历史记录等功能",
    "prompt-all-in-one": "提示词一体化 - 集成提示词管理、翻译、历史记录等功能",
    
    "sd-webui-openpose-editor": "OpenPose 编辑器 - 可视化编辑人物姿势，生成 OpenPose 参考图",
    "openpose-editor": "OpenPose 编辑器 - 可视化编辑人物姿势，生成 OpenPose 参考图",
    
    "sd-webui-depth-lib": "Depth Library - 深度图库管理，用于 ControlNet 深度图控制",
    "depth-lib": "Depth Library - 深度图库管理，用于 ControlNet 深度图控制",
    
    "sd-webui-rembg": "RemBG - 背景移除工具，支持多种抠图模型（u2net、isnet 等）",
    "rembg": "RemBG - 背景移除工具，支持多种抠图模型（u2net、isnet 等）",
    
    "sd-webui-segment-anything": "Segment Anything - Meta 的万物分割模型，支持任意对象的精确分割",
    "segment-anything": "Segment Anything - Meta 的万物分割模型，支持任意对象的精确分割",
    
    "sd-webui-extra-networks": "Extra Networks - 增强型网络浏览器，改进 LoRA/Embedding 的浏览体验",
    "extra-networks": "Extra Networks - 增强型网络浏览器，改进 LoRA/Embedding 的浏览体验",
    
    "sd-webui-supermerger": "SuperMerger - 高级模型合并工具，支持多种合并策略和参数调整",
    "supermerger": "SuperMerger - 高级模型合并工具，支持多种合并策略和参数调整",
    
    "sd-webui-regional-prompter": "Regional Prompter - 区域提示词，为图像不同区域指定不同的提示词",
    "regional-prompter": "Regional Prompter - 区域提示词，为图像不同区域指定不同的提示词",
    
    "sd-webui-dynamic-thresholding": "Dynamic Thresholding - 动态阈值调整，改善高 CFG 值下的图像质量",
    "dynamic-thresholding": "Dynamic Thresholding - 动态阈值调整，改善高 CFG 值下的图像质量",
    
    "sd-webui-cafe-artistic": "Cafe Artistic - 艺术风格增强插件，提供多种艺术化滤镜效果",
    "cafe-artistic": "Cafe Artistic - 艺术风格增强插件，提供多种艺术化滤镜效果",
    
    "sd-webui-textual-inversion-images": "Textual Inversion Images - 管理 Textual Inversion 训练图片",
    "textual-inversion-images": "Textual Inversion Images - 管理 Textual Inversion 训练图片",

    # 实际安装的插件
    "aadetailer-neoforge": "ADetailer Neo - 自动检测并修复面部/手部等缺陷，支持多种检测模型",
    "adetailer": "ADetailer Neo - 自动检测并修复面部/手部等缺陷，支持多种检测模型",
    "infinite-browsing": "无限图片浏览 - 强大的图片管理和浏览工具，支持标签搜索、批量操作",
    "sd-forge-regional-prompter-neo": "区域提示词 Neo - 为图像不同区域分别指定提示词，实现精细构图控制",
    "sd-webui-qwen-chat-llamacpp": "图像识别美学 - 图像识别与美学提升综合工具",
    "sd-webui-aspect-ratio-helper": "宽高比助手 - 快速设置常用图像宽高比，一键切换横竖屏",
    "sd-webui-camera-angle-selector": "镜头角度选择器 - 快速选择预设的相机镜头角度和构图",
    "sd-webui-forge-neo-seedvr2": "SeedVR2 - 基于扩散模型的图像超分辨率放大插件",
    "sd-webui-image-comparison-neo": "图像对比 Neo - 在原图上叠加对比生成前后的图像差异",
    "sd-webui-lighting-assistant": "光照助手 - 提供多种预设光照方案（伦勃朗光、蝴蝶光、环形光等）",
    "sd-webui-lobe-theme": "Lobe Theme - 现代化的 WebUI 主题美化，支持多语言和自定义布局",
    "sd-webui-model_downloader": "模型下载器 - 从 Hugging Face 等源一键下载模型文件",
    "sd-webui-multimodal-media": "多模态媒体 - 集成 ACE-Step 文生图/图生视频、LatentSync 视频唇形同步、Qwen3-TTS 语音合成",
    "sd-webui-see-through": "透视插件 - 为图像添加 X 光透视效果，生成透明或半透明材质",
    "sd-webui-trellis2": "TRELLIS 2.0 - 基于扩散模型的 3D 内容生成，从单张图片生成 3D 模型",
    "stable-diffusion-webui-wd14-tagger-master": "WD14 标签器 - 自动识别图像并生成 Danbooru 标签，支持自定义模型",
    "sd-webui-openpose-editor": "OpenPose 编辑器 - 可视化编辑人物骨骼姿势，生成 OpenPose 控制图",

    # 内置插件
    "sd_forge_controlnet": "ControlNet - 通过边缘检测、深度图、姿态等条件精准控制图像生成",
    "sd_forge_ipadapter": "IP-Adapter - 图像提示适配器，用参考图直接控制生成风格和内容",
    "sd_forge_lora": "LoRA - 轻量级模型微调，支持多种 LoRA/LyCORIS/DoRA 网络格式",
    "sd_forge_radial": "Radial - 径向注意力控制，优化图像边缘和中心的生成质量",
    "sd_forge_controlllite": "Control-Lite - 轻量级 ControlNet，减少显存占用的条件控制方案",
    "sd_forge_multidiffusion": "MultiDiffusion - 多区域扩散采样，支持不同区域使用不同提示词生成",
    "sd_forge_neveroom": "Never OOM - 内存优化管理，防止显存溢出导致的程序崩溃",
    "sd_forge_compile": "模型编译 - 使用 torch.compile 加速推理，减少生成时间",
    "sd_forge_pid": "PID 控制器 - 比例积分微分控制，稳定采样过程中的引导强度",
    "sd_forge_image_stitch": "图像拼接 - 将多张生成图像拼接成全景图或长图",
    "soft-inpainting": "软修复 - 柔和的图像修复/重绘，保留原始图像的纹理和细节",
    "forge_legacy_preprocessors": "旧版预处理器 - ControlNet 的经典预处理算法集合（Canny/Depth/OpenPose 等）",
    "forge_preprocessor_inpaint": "修复预处理器 - 基于 LaMa 的智能图像修复预处理算法",
    "forge_preprocessor_reference": "参考预处理器 - 将参考图像转换为 ControlNet 可用的特征图",
    "forge_preprocessor_tile": "分块预处理器 - 将大图分块处理的预处理器，用于高清修复",
    "extra-options-section": "扩展选项 - 在设置页面添加额外的配置选项和参数调节",
}

# WebUI 配置文件路径
WEBUI_CONFIG_FILE = os.path.join(WEBUI_DIR, "config.json")


def _read_webui_disabled_extensions() -> list:
    """读取 WebUI config.json 中的 disabled_extensions 列表"""
    if not os.path.exists(WEBUI_CONFIG_FILE):
        return []
    try:
        with open(WEBUI_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("disabled_extensions", [])
    except Exception:
        return []


def _update_webui_disabled_extensions(ext_name: str, disable: bool):
    """在 WebUI config.json 中添加/移除插件的禁用状态"""
    if not os.path.exists(WEBUI_CONFIG_FILE):
        return
    try:
        with open(WEBUI_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        disabled = set(cfg.get("disabled_extensions", []))
        if disable:
            disabled.add(ext_name)
        else:
            disabled.discard(ext_name)

        cfg["disabled_extensions"] = sorted(disabled)

        with open(WEBUI_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[ERROR] 更新 WebUI 配置失败: {e}")


def _get_extension_description(name: str) -> str:
    """根据插件名称获取中文描述"""
    # 尝试精确匹配
    if name in EXTENSION_DESCRIPTIONS:
        return EXTENSION_DESCRIPTIONS[name]
    # 尝试模糊匹配
    name_lower = name.lower()
    for key, desc in EXTENSION_DESCRIPTIONS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return desc
    return ""


class ExtensionsTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._extensions: list[ExtInfo] = []
        self._scan_worker    = None
        self._check_worker   = None
        self._update_worker  = None
        self._install_worker = None
        self._build_ui()



    # ── 样式辅助方法 ─────────────────────────────────────────

    def _btn_style(self, color: str) -> str:
        """按钮样式"""
        return f"""
            QPushButton {{
                background:{color};
                color:#000;
                border:none;
                border-radius:6px;
                padding:0 16px;
                font-weight:bold;
            }}
            QPushButton:hover {{
                background:{color}dd;
            }}
            QPushButton:disabled {{
                background:{COLORS['bg_card']};
                color:{COLORS['text_dim']};
            }}
        """

    def _input_style(self) -> str:
        """输入框样式"""
        return f"""
            QLineEdit, QComboBox {{
                background:{COLORS['bg_dark']};
                color:{COLORS['text_primary']};
                border:1px solid {COLORS['border']};
                border-radius:5px;
                padding:4px 8px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border:1px solid {COLORS['accent']};
            }}
        """

    def _small_btn(self, color: str) -> str:
        """小按钮样式"""
        return f"""
            QPushButton {{
                background:{color};
                color:#000;
                border:none;
                border-radius:4px;
                padding:2px 8px;
                font-size:11px;
            }}
            QPushButton:hover {{
                background:{color}cc;
            }}
            QPushButton:disabled {{
                background:transparent;
                color:{COLORS['text_dim']};
            }}
        """

    def _table_style(self) -> str:
        """表格样式"""
        return f"""
            QTableWidget {{
                background:{COLORS['bg_dark']};
                color:{COLORS['text_primary']};
                border:1px solid {COLORS['border']};
                border-radius:8px;
                gridline-color:{COLORS['border']};
            }}
            QTableWidget::item {{
                padding:6px;
                border-bottom:1px solid {COLORS['border']};
            }}
            QTableWidget::item:selected {{
                background:{COLORS['bg_hover']};
                color:{COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background:{COLORS['bg_card']};
                color:{COLORS['text_secondary']};
                padding:8px;
                border:none;
                border-bottom:2px solid {COLORS['border']};
                font-weight:bold;
                font-size:12px;
            }}
            QScrollBar:vertical {{
                background:{COLORS['bg_dark']};
                width:8px;
                border-radius:4px;
            }}
            QScrollBar::handle:vertical {{
                background:{COLORS['border']};
                border-radius:4px;
                min-height:20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background:{COLORS['text_dim']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height:0px;
            }}
        """

    def _log(self, msg: str):
        """向日志框追加消息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{timestamp}] {msg}")
        # 自动滚动到底部
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(10)

        # ── 标题行 ────────────────────────────────────────────
        title_row = QHBoxLayout()
        lbl = QLabel("🧩  插件管理")
        lbl.setStyleSheet(f"color:{COLORS['text_primary']};font-size:18px;font-weight:bold;")
        title_row.addWidget(lbl)
        title_row.addStretch()

        self.btn_refresh = QPushButton("🔄  刷新列表")
        self.btn_refresh.setFixedHeight(34)
        self.btn_refresh.clicked.connect(self._scan)
        title_row.addWidget(self.btn_refresh)

        self.btn_check = QPushButton("☁  检测更新")
        self.btn_check.setFixedHeight(34)
        self.btn_check.setStyleSheet(self._btn_style(COLORS['accent']))
        self.btn_check.clicked.connect(self._check_updates)
        title_row.addWidget(self.btn_check)

        self.btn_update_all = QPushButton("⬆  一键更新")
        self.btn_update_all.setFixedHeight(34)
        self.btn_update_all.setStyleSheet(self._btn_style(COLORS['green']))
        self.btn_update_all.setEnabled(False)
        self.btn_update_all.clicked.connect(self._update_all)
        title_row.addWidget(self.btn_update_all)

        layout.addLayout(title_row)


        install_frame = QFrame()
        install_frame.setStyleSheet(f"""
            QFrame {{ background:{COLORS['bg_dark']}; border-radius:8px;
                border:1px solid {COLORS['border']}; }}
        """)
        install_layout = QHBoxLayout(install_frame)
        install_layout.setContentsMargins(14, 8, 14, 8)

        lbl_install = QLabel("安装插件:")
        lbl_install.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;")
        lbl_install.setFixedWidth(70)
        install_layout.addWidget(lbl_install)

        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("粘贴 GitHub 仓库地址，例如: https://github.com/exo101/sd-webui-camera-angle-selector")
        self.edit_url.setStyleSheet(self._input_style())
        self.edit_url.returnPressed.connect(self._install)
        install_layout.addWidget(self.edit_url, 1)
        
        # 示例按钮
        btn_example = QPushButton("📝 示例")
        btn_example.setFixedSize(60, 30)
        btn_example.setStyleSheet(self._small_btn(COLORS['yellow']))
        btn_example.clicked.connect(self._show_example)
        install_layout.addWidget(btn_example)

        btn_install = QPushButton("📥  安装")
        btn_install.setFixedSize(80, 30)
        btn_install.setStyleSheet(self._btn_style(COLORS['cyan']))
        btn_install.clicked.connect(self._install)
        install_layout.addWidget(btn_install)

        layout.addWidget(install_frame)

        # ── 搜索 + 统计 ───────────────────────────────────────
        search_row = QHBoxLayout()

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔍  搜索已安装插件...")
        self.edit_search.setStyleSheet(self._input_style())
        self.edit_search.textChanged.connect(self._filter)
        search_row.addWidget(self.edit_search, 1)

        self.lbl_count = QLabel("共 0 个插件")
        self.lbl_count.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        search_row.addWidget(self.lbl_count)

        layout.addLayout(search_row)

        # ── 进度条（检测/更新时显示）─────────────────────────
        self.progress = QProgressBar()
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background:{COLORS['border']}; border:none; border-radius:2px; }}
            QProgressBar::chunk {{ background:{COLORS['accent']}; border-radius:2px; }}
        """)
        self.progress.hide()
        layout.addWidget(self.progress)

        # ── 插件表格 ──────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["启用", "插件名（中文说明）", "远程地址", "当前分支", "版本 ID", "更新日期", "状态", "操作"])
        self.table.setStyleSheet(self._table_style())
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        
        # 添加点击事件，显示插件详情
        self.table.cellClicked.connect(self._on_cell_clicked)
        
        # 点击表格空白处关闭详情
        self.table.viewport().installEventFilter(self)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(COL_EN,   QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(COL_URL,  QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(COL_BR,   QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_HASH, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_DATE, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_STAT, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_ACT,  QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_EN,   40)
        self.table.setColumnWidth(COL_NAME, 380)
        self.table.setColumnWidth(COL_BR,   0)
        self.table.setColumnWidth(COL_HASH, 0)
        self.table.setColumnWidth(COL_DATE, 140)
        self.table.setColumnWidth(COL_STAT, 70)
        self.table.setColumnWidth(COL_ACT,  130)
        # 隐藏"当前分支"和"版本 ID"列
        self.table.setColumnHidden(COL_BR, True)
        self.table.setColumnHidden(COL_HASH, True)

        layout.addWidget(self.table, 1)

        # ── 插件详情面板（点击插件后显示）──────────────────────
        self.detail_frame = QFrame()
        self.detail_frame.setStyleSheet(f"""
            QFrame {{
                background:{COLORS['bg_dark']};
                border:1px solid {COLORS['border']};
                border-radius:6px;
            }}
        """)
        detail_layout = QVBoxLayout(self.detail_frame)
        detail_layout.setContentsMargins(16, 12, 16, 12)
        detail_layout.setSpacing(8)
        
        # 标题行
        detail_header = QHBoxLayout()
        self.detail_title = QLabel("📖 插件详情")
        self.detail_title.setStyleSheet(
            f"color:{COLORS['accent_light']};font-size:14px;font-weight:bold;"
        )
        detail_header.addWidget(self.detail_title)
        detail_header.addStretch()
        
        btn_close_detail = QPushButton("✕ 关闭")
        btn_close_detail.setFixedHeight(28)
        btn_close_detail.setToolTip("点击关闭详情面板")
        btn_close_detail.setStyleSheet(f"""
            QPushButton {{
                background:{COLORS['red']}22;
                color:{COLORS['red']};
                border:1px solid {COLORS['red']}44;
                border-radius:4px;
                padding:0 12px;
                font-size:12px;
            }}
            QPushButton:hover {{
                background:{COLORS['red']}44;
                border:1px solid {COLORS['red']};
            }}
        """)
        btn_close_detail.clicked.connect(self._close_detail)
        detail_header.addWidget(btn_close_detail)
        detail_layout.addLayout(detail_header)
        
        # 详情内容（使用 QTextBrowser 支持 HTML）
        from PyQt6.QtWidgets import QTextBrowser
        self.detail_content = QTextBrowser()
        self.detail_content.setReadOnly(True)
        self.detail_content.setOpenExternalLinks(False)
        self.detail_content.setStyleSheet(f"""
            QTextBrowser {{
                background:transparent;
                color:{COLORS['text_secondary']};
                border:none;
                font-size:12px;
                line-height:1.6;
            }}
        """)
        detail_layout.addWidget(self.detail_content)
        
        self.detail_frame.hide()  # 初始隐藏
        layout.addWidget(self.detail_frame)

        # ── 底部日志 ──────────────────────────────────────────
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(80)
        self.log_edit.setStyleSheet(f"""
            QTextEdit {{ background:{COLORS['bg_dark']}; color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']}; border-radius:5px;
                font-family:Consolas,monospace; font-size:11px; }}
        """)
        self.log_edit.setPlaceholderText("操作日志...")
        layout.addWidget(self.log_edit)

        # 初始扫描
        self._scan()

    # ── 扫描 ──────────────────────────────────────────────────

    def _filter(self, text: str):
        """根据搜索文本过滤插件列表（支持搜索中文描述）"""
        keyword = text.lower().strip()
        if not keyword:
            self._populate_table(self._extensions)
        else:
            filtered = [e for e in self._extensions 
                       if keyword in e.name.lower() 
                       or keyword in (e.remote_url or "").lower()
                       or keyword in _get_extension_description(e.name).lower()]
            self._populate_table(filtered)

    def _populate_table(self, exts: list):
        """填充插件表格"""
        self.table.setRowCount(len(exts))
        
        for row, ext in enumerate(exts):
            # 启用状态
            chk = QCheckBox()
            chk.setChecked(ext.enabled)
            chk.setStyleSheet(f"QCheckBox::indicator {{ width:16px; height:16px; }}")
            chk.stateChanged.connect(lambda state, e=ext: self._toggle_enabled(e, state))
            self.table.setCellWidget(row, COL_EN, chk)
            
            # 名称（使用 QLabel 显示多行文本，包含中文描述）
            desc = _get_extension_description(ext.name)
            name_widget = QLabel()
            if desc:
                name_widget.setText(f"<b>{ext.name}</b><br><span style='color:{COLORS['text_dim']};font-size:11px;'>{desc}</span>")
                name_widget.setToolTip(desc)
                self.table.setRowHeight(row, 56)
                name_widget.setMinimumHeight(50)
            else:
                name_widget.setText(ext.name)
            name_widget.setWordWrap(True)
            name_widget.setStyleSheet(f"""
                QLabel {{
                    color: {COLORS['text_primary']};
                    padding: 2px 6px;
                    background: transparent;
                }}
            """)
            name_widget.setContentsMargins(4, 2, 4, 2)
            name_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.table.setCellWidget(row, COL_NAME, name_widget)
            
            # URL
            url_item = QTableWidgetItem(ext.remote_url or "本地插件")
            url_item.setFlags(url_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            url_item.setForeground(QColor(COLORS['text_dim']))
            self.table.setItem(row, COL_URL, url_item)
            
            # 分支
            br_item = QTableWidgetItem(ext.branch or "?")
            br_item.setFlags(br_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, COL_BR, br_item)
            
            # Commit Hash
            hash_item = QTableWidgetItem(ext.commit_hash[:7] if ext.commit_hash else "?")
            hash_item.setFlags(hash_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, COL_HASH, hash_item)
            
            # 日期
            date_item = QTableWidgetItem(ext.commit_date or "?")
            date_item.setFlags(date_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, COL_DATE, date_item)
            
            # 状态
            self._set_status_cell(row, ext)
            
            # 操作按钮
            self._set_action_cell(row, ext)
        
        self.lbl_count.setText(f"共 {len(self._extensions)} 个插件，显示 {len(exts)} 个")

    def _toggle_enabled(self, ext: ExtInfo, state):
        """切换插件启用状态（同时更新 WebUI config.json 和 disabled 文件）"""
        ext.enabled = (state == Qt.CheckState.Checked.value)
        disabled_file = os.path.join(ext.path, "disabled")
        if ext.enabled:
            if os.path.exists(disabled_file):
                os.remove(disabled_file)
            _update_webui_disabled_extensions(ext.name, disable=False)
            self._log(f"✅ 已启用: {ext.name}")
        else:
            with open(disabled_file, 'w') as f:
                f.write("")
            _update_webui_disabled_extensions(ext.name, disable=True)
            self._log(f"⛔ 已禁用: {ext.name}")

    def _set_status_cell(self, row: int, ext: ExtInfo):
        """设置状态单元格"""
        status_widget = QWidget()
        hl = QHBoxLayout(status_widget)
        hl.setContentsMargins(4, 2, 4, 2)
        
        if ext.has_update:
            lbl = QLabel("🆕 有更新")
            lbl.setStyleSheet(f"color:{COLORS['yellow']};font-size:11px;font-weight:bold;")
        elif ext.error:
            lbl = QLabel(f"❌ {ext.error}")
            lbl.setStyleSheet(f"color:{COLORS['red']};font-size:11px;")
        else:
            lbl = QLabel("✅ 最新")
            lbl.setStyleSheet(f"color:{COLORS['green']};font-size:11px;")
        
        hl.addWidget(lbl)
        self.table.setCellWidget(row, COL_STAT, status_widget)

    def _set_action_cell(self, row: int, ext: ExtInfo):
        """设置操作单元格"""
        cell = QWidget()
        hl = QHBoxLayout(cell)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        btn_update = QPushButton("更新")
        btn_update.setFixedSize(52, 26)
        btn_update.setEnabled(ext.has_update)
        btn_update.setStyleSheet(self._small_btn(COLORS['accent'] if ext.has_update else COLORS['text_dim']))
        btn_update.clicked.connect(lambda _, e=ext: self._update_single(e))
        hl.addWidget(btn_update)

        btn_del = QPushButton("卸载")
        btn_del.setFixedSize(52, 26)
        btn_del.setStyleSheet(self._small_btn(COLORS['red']))
        btn_del.clicked.connect(lambda _, e=ext: self._uninstall(e))
        hl.addWidget(btn_del)

        self.table.setCellWidget(row, COL_ACT, cell)

    def _context_menu(self, pos):
        """表格右键菜单"""
        row = self.table.rowAt(pos.y())
        if row < 0 or row >= len(self._extensions):
            return
        
        ext = self._extensions[row]
        if not ext:
            return
        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background:{COLORS['bg_card']};
                color:{COLORS['text_primary']};
                border:1px solid {COLORS['border']};
                border-radius:6px;
                padding:4px;
            }}
            QMenu::item {{
                padding:6px 20px;
                border-radius:4px;
            }}
            QMenu::item:selected {{
                background:{COLORS['bg_hover']};
            }}
        """)
        
        # 复制远程地址
        action_copy_url = menu.addAction("📋 复制远程地址")
        action_copy_url.triggered.connect(lambda: self._copy_to_clipboard(ext.remote_url or ""))
        
        # 打开 GitHub 页面
        if ext.remote_url and ext.remote_url.startswith("http"):
            action_open_github = menu.addAction("🌐 打开 GitHub 页面")
            action_open_github.triggered.connect(lambda: self._open_url(ext.remote_url.replace(".git", "")))
        
        # 查看插件详情
        action_detail = menu.addAction("📖 查看插件详情")
        action_detail.triggered.connect(lambda: self._show_plugin_detail(ext))
        
        menu.addSeparator()
        
        # 切换分支
        action_switch_branch = menu.addAction("🔀 切换分支...")
        action_switch_branch.triggered.connect(lambda: self._switch_branch_dialog(ext))
        
        # 强制更新
        action_force_update = menu.addAction("⚡ 强制更新")
        action_force_update.triggered.connect(lambda: self._force_update(ext))
        
        menu.addSeparator()
        
        # 卸载
        action_uninstall = menu.addAction("🗑️ 卸载插件")
        action_uninstall.setStyleSheet(f"color:{COLORS['red']};")
        action_uninstall.triggered.connect(lambda: self._uninstall(ext))
        
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _copy_to_clipboard(self, text: str):
        """复制文本到剪贴板"""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self._log(f"📋 已复制到剪贴板: {text}")

    def _open_url(self, url: str):
        """在浏览器中打开URL"""
        import webbrowser
        webbrowser.open(url)
        self._log(f"🌐 打开链接: {url}")

    def _show_plugin_detail(self, ext: ExtInfo):
        """显示插件详情"""
        self._on_cell_clicked(0, 0, ext.name)

    def _switch_branch_dialog(self, ext: ExtInfo):
        """切换分支对话框"""
        from PyQt6.QtWidgets import QInputDialog
        branch, ok = QInputDialog.getText(
            self, 
            "切换分支",
            f"请输入要切换到的分支名称（当前: {ext.branch}）:",
            text=ext.branch or "main"
        )
        if ok and branch:
            self._log(f"🔀 切换 {ext.name} 到分支: {branch}")
            success, msg = switch_branch(ext, branch)
            if success:
                ext.branch = branch
                self._log(f"✅ 切换成功")
                self._scan()
            else:
                self._log(f"❌ 切换失败: {msg}")

    def _force_update(self, ext: ExtInfo):
        """强制更新插件"""
        self._log(f"⚡ 强制更新 {ext.name}...")
        # 这里可以调用特殊的更新逻辑，比如先fetch再reset
        self._update_single(ext)

    def _on_cell_clicked(self, row: int, column: int, ext_name: str = None):
        """点击表格单元格显示插件详情"""
        # 如果直接传入了ext_name，使用它；否则从表格行索引获取
        if ext_name is None:
            if 0 <= row < len(self._extensions):
                ext_name = self._extensions[row].name
            else:
                return
        
        # 查找对应的插件信息
        ext = next((e for e in self._extensions if e.name == ext_name), None)
        if not ext:
            return
        
        # 显示详情面板
        self._show_detail_panel(ext)

    def _show_detail_panel(self, ext: ExtInfo):
        """显示插件详情面板"""
        # 构建详情HTML内容
        html_content = self._build_detail_html(ext)
        
        # 设置内容并显示
        self.detail_content.setHtml(html_content)
        self.detail_frame.show()
        
        self._log(f"📖 查看插件详情: {ext.name}")

    def _build_detail_html(self, ext: ExtInfo) -> str:
        """构建插件详情的HTML内容"""
        # 尝试读取README文件
        readme_content = self._read_readme(ext.path)
        
        if readme_content:
            # 将Markdown转换为HTML
            html_content = self._markdown_to_html(readme_content)
            return f"""
            <div style="color:{COLORS['text_primary']};padding:10px;">
                <h2 style="color:{COLORS['accent']};border-bottom:2px solid {COLORS['border']};padding-bottom:8px;">
                    📦 {ext.name}
                </h2>
                {html_content}
            </div>
            """
        else:
            # 没有README时显示基本信息
            return f"""
            <div style="color:{COLORS['text_primary']};padding:10px;">
                <h2 style="color:{COLORS['accent']};border-bottom:2px solid {COLORS['border']};padding-bottom:8px;">
                    📦 {ext.name}
                </h2>
                <p><strong>远程地址:</strong> {ext.remote_url or '本地插件'}</p>
                <p><strong>当前分支:</strong> {ext.branch or '未知'}</p>
                <p><strong>版本ID:</strong> {ext.commit_hash[:7] if ext.commit_hash else '未知'}</p>
                <p><strong>更新日期:</strong> {ext.commit_date or '未知'}</p>
                <p><strong>状态:</strong> {'🆕 有更新' if ext.has_update else '✅ 最新'}</p>
                <hr style="border-color:{COLORS['border']};margin:15px 0;">
                <p style="color:{COLORS['text_dim']};">
                    💡 该插件暂无 README 文档。<br>
                    建议访问 GitHub 页面查看更多信息。
                </p>
            </div>
            """

    def _read_readme(self, plugin_path: str) -> str:
        """读取插件目录下的README文件"""
        readme_names = ['README.md', 'readme.md', 'README.MD', 'Readme.md']
        
        for readme_name in readme_names:
            readme_path = os.path.join(plugin_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    self._log(f"⚠️ 读取README失败: {e}")
                    return None
        
        return None

    def _markdown_to_html(self, markdown_text: str) -> str:
        """简单的Markdown转HTML"""
        import re
        
        html = markdown_text
        
        # 标题
        html = re.sub(r'^#### (.+)$', r'<h4 style="color:{COLORS["text_secondary"]};margin-top:15px;">\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3 style="color:{COLORS["accent"]};margin-top:18px;">\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2 style="color:{COLORS["accent"]};margin-top:20px;border-bottom:1px solid {COLORS["border"]};padding-bottom:5px;">\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1 style="color:{COLORS["accent"]};margin-top:20px;">\1</h1>', html, flags=re.MULTILINE)
        
        # 粗体和斜体
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # 代码块（多行）
        html = re.sub(r'```(\w*)\n(.*?)```', r'<pre style="background:{COLORS["bg_dark"]};padding:10px;border-radius:5px;overflow-x:auto;"><code>\2</code></pre>', html, flags=re.DOTALL)
        
        # 行内代码
        html = re.sub(r'`([^`]+)`', r'<code style="background:{COLORS["bg_dark"]};padding:2px 5px;border-radius:3px;font-size:11px;">\1</code>', html)
        
        # 链接
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" style="color:{COLORS["accent"]};text-decoration:none;">\1</a>', html)
        
        # 图片
        html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%;border-radius:5px;margin:10px 0;">', html)
        
        # 无序列表
        html = re.sub(r'^[\-\*] (.+)$', r'<li style="margin-left:20px;margin-bottom:5px;">\1</li>', html, flags=re.MULTILINE)
        
        # 有序列表
        html = re.sub(r'^\d+\. (.+)$', r'<li style="margin-left:20px;margin-bottom:5px;">\1</li>', html, flags=re.MULTILINE)
        
        # 引用
        html = re.sub(r'^> (.+)$', r'<blockquote style="border-left:3px solid {COLORS["accent"]};padding-left:10px;margin:10px 0;color:{COLORS["text_dim"]};">\1</blockquote>', html, flags=re.MULTILINE)
        
        # 水平线
        html = re.sub(r'^---$', f'<hr style="border-color:{COLORS["border"]};margin:15px 0;">', html, flags=re.MULTILINE)
        
        # 段落（简单的换行处理）
        html = html.replace('\n\n', '</p><p style="margin:10px 0;line-height:1.6;">')
        html = '<p style="margin:10px 0;line-height:1.6;">' + html + '</p>'
        
        return html

    def _close_detail(self):
        """关闭详情面板"""
        self.detail_frame.hide()
        self._log("已关闭详情面板")

    def _scan(self):
        self.btn_refresh.setEnabled(False)
        self.table.setRowCount(0)
        self._log("扫描插件目录...")
        self._scan_worker = ScanWorker()
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, exts: list):
        self._extensions = exts
        # 同步 WebUI config.json 中的禁用状态
        disabled_names = _read_webui_disabled_extensions()
        # 同时检查 legacy disabled 文件，同步到 WebUI 配置
        need_save = False
        for ext in self._extensions:
            disabled_file = os.path.join(ext.path, "disabled")
            has_disabled_file = os.path.exists(disabled_file)
            if has_disabled_file and ext.name not in disabled_names:
                disabled_names.append(ext.name)
                need_save = True
                ext.enabled = False
            elif ext.name in disabled_names:
                ext.enabled = False
                # 如果 disabled 文件不存在但 WebUI 配置中有，也创建文件保持一致性
                if not has_disabled_file:
                    try:
                        with open(disabled_file, 'w') as f:
                            f.write("")
                    except Exception:
                        pass
        if need_save:
            try:
                with open(WEBUI_CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["disabled_extensions"] = sorted(set(disabled_names))
                with open(WEBUI_CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=4)
            except Exception:
                pass
        self.btn_refresh.setEnabled(True)
        self._populate_table(exts)
        self._log(f"扫描完成，共 {len(exts)} 个插件")
        if self._scan_worker:
            self._scan_worker.deleteLater()
            self._scan_worker = None

    # ── 检测更新 ──────────────────────────────────────────────

    def _check_updates(self):
        if not self._extensions:
            return
        self.btn_check.setEnabled(False)
        self.btn_update_all.setEnabled(False)
        self.progress.setRange(0, len(self._extensions))
        self.progress.setValue(0)
        self.progress.show()
        self._log("开始检测更新（需要网络）...")

        self._check_worker = CheckUpdateWorker(self._extensions, "", "")
        self._check_worker.progress.connect(self._on_check_progress)
        self._check_worker.done.connect(self._on_check_done)
        self._check_worker.start()

    def _on_check_progress(self, idx: int, ext: ExtInfo):
        self._extensions[idx] = ext
        self.progress.setValue(idx + 1)
        self._update_row(idx, ext)

    def _on_check_done(self):
        self.btn_check.setEnabled(True)
        self.progress.hide()
        has_updates = sum(1 for e in self._extensions if e.has_update)
        self.btn_update_all.setEnabled(has_updates > 0)
        self._log(f"检测完成，{has_updates} 个插件有更新")
        if self._check_worker:
            self._check_worker.deleteLater()
            self._check_worker = None

    # ── 一键更新 ──────────────────────────────────────────────

    def _update_all(self):
        to_update = [e for e in self._extensions if e.has_update]
        if not to_update:
            return
        self.btn_update_all.setEnabled(False)
        self.progress.setRange(0, len(to_update))
        self.progress.setValue(0)
        self.progress.show()

        self._update_worker = UpdateWorker(to_update, "", "")
        self._update_worker.log.connect(self._log)
        self._update_worker.done.connect(self._on_update_done)
        self._update_worker.start()

    def _on_update_done(self, ok: int, fail: int):
        self.progress.hide()
        self._log(f"更新完成：成功 {ok} 个，失败 {fail} 个")
        if self._update_worker:
            self._update_worker.deleteLater()
            self._update_worker = None
        self._scan()

    # ── 安装 ──────────────────────────────────────────────────

    def _install(self):
        url = self.edit_url.text().strip()
        if not url:
            self._log("❌ 请输入插件的 GitHub 仓库地址")
            return
        if not url.startswith("http"):
            self._log("❌ 请输入有效的 GitHub 地址（以 http:// 或 https:// 开头）")
            return

        self.edit_url.setEnabled(False)
        self._log(f"📥 开始安装: {url}")
        
        # 直接使用系统 Git 克隆，不使用镜像和代理
        self._log("🔧 使用系统 Git 直接克隆...")

        self._install_worker = InstallWorker(url, "", "")
        self._install_worker.log.connect(self._log)
        self._install_worker.done.connect(self._on_install_done)
        self._install_worker.start()
    
    def _show_example(self):
        """显示示例插件 URL"""
        example_url = "https://github.com/exo101/sd-webui-camera-angle-selector"
        self.edit_url.setText(example_url)
        self._log(f"📝 已设置示例插件 URL: {example_url}")
        
    def _on_install_done(self, ok: bool, msg: str):
        """安装完成回调"""
        self.edit_url.setEnabled(True)
        if ok:
            self.edit_url.clear()
            self._log("✅ 安装成功，重新扫描...")
            self._scan()
        else:
            self._log(f"❌ 安装失败: {msg}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "安装失败", 
                f"插件安装失败，错误信息：\n\n{msg}\n\n建议：\n"
                f"1. 检查网络连接\n"
                f"2. 尝试切换上方的镜像源\n"
                f"3. 配置代理服务器")
        if self._install_worker:
            self._install_worker.deleteLater()
            self._install_worker = None
    
    def _uninstall(self, ext: ExtInfo):
        """卸载插件"""
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, 
            "确认卸载",
            f"确定要卸载插件 '{ext.name}' 吗？\n\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        self._log(f"🗑️ 卸载插件: {ext.name}")
        success, msg = uninstall_extension(ext)
        if success:
            self._log(f"✅ 卸载成功")
            self._scan()
        else:
            self._log(f"❌ 卸载失败: {msg}")
            QMessageBox.critical(self, "卸载失败", f"无法卸载插件：\n{msg}")
    
    def _update_single(self, ext: ExtInfo):
        """更新单个插件"""
        self._log(f"⬆️ 开始更新: {ext.name}...")
            
        # 禁用相关按钮防止重复操作
        self.btn_check.setEnabled(False)
        self.btn_update_all.setEnabled(False)
            
        # 创建单任务更新 Worker
        worker = UpdateWorker([ext], self._get_proxy(), self._get_mirror())
        worker.log.connect(self._log)
            
        def on_single_update_done(ok: int, fail: int):
            if ok > 0:
                self._log(f"✅ {ext.name} 更新成功")
            else:
                self._log(f"❌ {ext.name} 更新失败")
                
            # 恢复按钮状态
            self.btn_check.setEnabled(True)
            # 重新检测是否有其他待更新插件以决定一键更新按钮状态
            has_updates = sum(1 for e in self._extensions if e.has_update and e.name != ext.name)
            # 注意：这里简单处理，实际可能需要重新扫描或更复杂的状态管理
            # 为了保持界面一致性，通常更新后建议刷新列表
            self._scan() 
                
        worker.done.connect(on_single_update_done)
        worker.start()
    
    def _update_row(self, idx: int, ext: ExtInfo):
        """更新单行状态（检测更新后调用）"""
        # 找到表格中对应行（通过 _extensions 列表索引或名称匹配）
        for row in range(self.table.rowCount()):
            # 名称列现在是 QLabel 控件，用索引匹配更可靠
            if row < len(self._extensions) and self._extensions[row].name == ext.name:
                self._set_status_cell(row, ext)
                self._set_action_cell(row, ext)
                break