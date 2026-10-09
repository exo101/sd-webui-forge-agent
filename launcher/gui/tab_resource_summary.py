from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QFrame
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from .theme import COLORS

class ResourceSummaryTab(QWidget):
    """资源汇总标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 12)
        layout.setSpacing(10)

        # 标题
        title = QLabel("📦  资源汇总")
        title.setStyleSheet(f"color:{COLORS['text_primary']};font-size:18px;font-weight:bold;")
        layout.addWidget(title)

        subtitle = QLabel("各大开源网址的汇总，点击链接跳转到相应网站")
        subtitle.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        layout.addWidget(subtitle)

        # 资源网格
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(0, 10, 0, 0)

        # 资源列表
        resources = [
            {
                "name": "GitHub",
                "url": "https://github.com/exo101",
                "description": "sd-webui-forge-agent 项目主页"
            },
            {
                "name": "魔搭社区",
                "url": "https://www.modelscope.cn/home",
                "description": "ModelScope 开源模型社区，汇集业界最新最热的模型、数据集"
            },
            {
                "name": "B站教程",
                "url": "https://space.bilibili.com/403361177?spm_id_from=333.1007.0.0",
                "description": "哔哩哔哩（鸡肉爱土豆）教程主页"
            },
            {
                "name": "Hugging Face",
                "url": "https://huggingface.co/",
                "description": "机器学习模型和数据集的开源平台"
            },
            {
                "name": "哩布哩布",
                "url": "https://www.liblib.art/inspiration",
                "description": "中国领先的AI创作平台，提供图片和视频生成服务"
            },
            {
                "name": "C站",
                "url": "https://civitai.com/",
                "description": "AI艺术模型分享平台，提供各种模型下载和社区交流"
            }
        ]

        # 添加资源卡片
        for i, resource in enumerate(resources):
            row = i // 2
            col = i % 2
            card = self._create_resource_card(resource)
            grid_layout.addWidget(card, row, col)

        layout.addLayout(grid_layout)
        layout.addStretch()

    def _create_resource_card(self, resource):
        """创建资源卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
            }}
            QFrame:hover {{
                border-color: {COLORS['accent_light']};
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # 名称
        name = QLabel(f"{resource['name']}")
        name.setStyleSheet(f"color:{COLORS['text_primary']};font-size:14px;font-weight:bold;")
        layout.addWidget(name)

        # 描述
        description = QLabel(resource['description'])
        description.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;")
        description.setWordWrap(True)
        layout.addWidget(description)

        # 链接按钮
        link_btn = QPushButton(f"🌐  访问 {resource['name']}")
        link_btn.setFixedHeight(32)
        link_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['accent_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']}22;
                border-color: {COLORS['accent_light']};
            }}
        """)
        link_btn.clicked.connect(lambda: self._open_url(resource['url']))
        layout.addWidget(link_btn)

        return card

    def _open_url(self, url: str):
        try:
            QDesktopServices.openUrl(QUrl(url))
        except Exception as ex:
            print(f"[资源汇总] 打开链接失败: {ex}")