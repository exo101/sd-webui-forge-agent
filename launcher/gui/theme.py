# 主题样式 - 深色现代风格（参考秋叶启动器）

COLORS = {
    "bg_dark":      "#0f0f17",
    "bg_main":      "#16161f",
    "bg_card":      "#1e1e2e",
    "bg_hover":     "#2a2a3e",
    "bg_input":     "#252535",
    "accent":       "#7c6af7",       # 主紫色
    "accent_light": "#9d8fff",
    "accent_hover": "#6a58e0",
    "green":        "#50fa7b",
    "red":          "#ff5555",
    "yellow":       "#f1fa8c",
    "orange":       "#ffb86c",
    "cyan":         "#8be9fd",
    "text_primary": "#f8f8f2",
    "text_secondary":"#a0a0b8",
    "text_dim":     "#6272a4",
    "border":       "#2e2e42",
    "border_focus": "#7c6af7",
    "scrollbar":    "#2e2e42",
    "tab_active":   "#7c6af7",
    "tab_inactive": "#1e1e2e",
}

MAIN_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg_main']};
    color: {COLORS['text_primary']};
    font-family: "Microsoft YaHei UI", "微软雅黑", sans-serif;
    font-size: 13px;
}}

/* 标签页 */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['bg_card']};
    border-radius: 0 8px 8px 8px;
}}
QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_secondary']};
    padding: 10px 22px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
    min-width: 80px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border-bottom: 2px solid {COLORS['accent']};
    font-weight: bold;
}}
QTabBar::tab:hover:!selected {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['text_primary']};
}}

/* 按钮 */
QPushButton {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent_hover']};
}}
QPushButton:disabled {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_dim']};
    border-color: {COLORS['border']};
}}
QPushButton#btn_launch {{
    background-color: {COLORS['accent']};
    color: white;
    font-size: 15px;
    font-weight: bold;
    padding: 10px 32px;
    border-radius: 8px;
    border: none;
}}
QPushButton#btn_launch:hover {{
    background-color: {COLORS['accent_light']};
}}
QPushButton#btn_launch:disabled {{
    background-color: #3a3a5c;
    color: {COLORS['text_dim']};
}}
QPushButton#btn_stop {{
    background-color: #3d1f1f;
    color: {COLORS['red']};
    font-size: 13px;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 8px;
    border: 1px solid #5a2020;
}}
QPushButton#btn_stop:hover {{
    background-color: #5a2020;
}}
QPushButton#btn_browse {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    min-width: 50px;
}}
QPushButton#btn_browse:hover {{
    background-color: {COLORS['accent']};
    color: white;
}}

/* 输入框 */
QLineEdit, QSpinBox, QComboBox {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS['border_focus']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {COLORS['text_secondary']};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_focus']};
    selection-background-color: {COLORS['accent']};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {COLORS['bg_hover']};
    border: none;
    width: 18px;
}}

/* 复选框 */
QCheckBox {{
    color: {COLORS['text_primary']};
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['bg_input']};
}}
QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
    image: none;
}}
QCheckBox::indicator:hover {{
    border-color: {COLORS['accent']};
}}

/* 滚动条 */
QScrollBar:vertical {{
    background: {COLORS['bg_dark']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['scrollbar']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {COLORS['bg_dark']};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['scrollbar']};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['accent']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* 文本框 */
QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['green']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: {COLORS['accent']};
}}

/* 分组框 */
QGroupBox {{
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-size: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {COLORS['accent_light']};
    font-weight: bold;
}}

/* 标签 */
QLabel {{
    color: {COLORS['text_primary']};
    background: transparent;
}}
QLabel#label_dim {{
    color: {COLORS['text_dim']};
    font-size: 11px;
}}
QLabel#label_section {{
    color: {COLORS['accent_light']};
    font-weight: bold;
    font-size: 13px;
}}
QLabel#status_running {{
    color: {COLORS['green']};
    font-weight: bold;
}}
QLabel#status_stopped {{
    color: {COLORS['text_dim']};
}}

/* 分割线 */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {COLORS['border']};
}}

/* 滑块 */
QSlider::groove:horizontal {{
    height: 4px;
    background: {COLORS['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {COLORS['accent']};
    border-radius: 2px;
}}

/* 工具提示 */
QToolTip {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_focus']};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}
"""


# ── 公共样式工具函数 ──────────────────────────────────────────

def btn_style(bg: str, fg: str, hover: str = "") -> str:
    hover = hover or f"{fg}33"
    return f"""
        QPushButton {{
            background:{bg}; color:{fg};
            border:1px solid {fg}44; border-radius:5px;
            padding:4px 12px; font-size:12px;
        }}
        QPushButton:hover {{ background:{hover}; }}
        QPushButton:disabled {{
            background:{COLORS['bg_card']}; color:{COLORS['text_dim']};
            border-color:{COLORS['border']};
        }}
    """


def input_style() -> str:
    return f"""
        QLineEdit, QComboBox {{
            background:{COLORS['bg_input']}; color:{COLORS['text_primary']};
            border:1px solid {COLORS['border']}; border-radius:5px;
            padding:5px 10px; font-size:12px;
        }}
        QLineEdit:focus, QComboBox:focus {{ border-color:{COLORS['border_focus']}; }}
        QComboBox::drop-down {{ border:none; width:20px; }}
        QComboBox QAbstractItemView {{
            background:{COLORS['bg_input']}; color:{COLORS['text_primary']};
            border:1px solid {COLORS['border_focus']};
            selection-background-color:{COLORS['accent']};
        }}
    """


def group_style() -> str:
    return f"""
        QGroupBox {{
            color:{COLORS['accent_light']}; border:1px solid {COLORS['border']};
            border-radius:8px; margin-top:12px; padding-top:8px; font-weight:bold;
        }}
        QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; }}
    """
