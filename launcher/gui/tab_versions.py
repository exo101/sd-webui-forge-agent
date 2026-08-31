"""版本管理标签页 - 已废弃，请使用主控台的更新按钮"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt


class VersionsTab(QWidget):
    """版本管理标签页（已简化，功能移至主控台）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # 提示信息
        lbl_title = QLabel("📦 版本管理")
        lbl_title.setStyleSheet("font-size:18px; font-weight:bold; color:#e0e0e0;")
        layout.addWidget(lbl_title)
        
        lbl_info = QLabel(
            "💡 提示：\n\n"
            "• WebUI 更新请在WebUI界面中进行\n\n"
            "当前版本信息将在主控台中显示"
        )
        lbl_info.setStyleSheet("font-size:13px; color:#a0a0a0; line-height:1.6;")
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)
        
        layout.addStretch()
