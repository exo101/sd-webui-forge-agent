"""常用命令标签页"""
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import subprocess
from core.paths import BASE_DIR


class CommandCard(QFrame):
    """命令卡片组件"""
    def __init__(self, title, command, description, parent=None):
        super().__init__(parent)
        self.command = command
        self._setup_ui(title, command, description)
    
    def _setup_ui(self, title, command, description):
        from .theme import COLORS
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            font-weight: bold;
        """)
        layout.addWidget(title_label)
        
        # 命令代码框
        cmd_frame = QFrame()
        cmd_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        cmd_layout = QHBoxLayout(cmd_frame)
        cmd_layout.setContentsMargins(8, 8, 8, 8)
        
        cmd_label = QLabel(command)
        cmd_label.setFont(QFont("Consolas", 10))
        cmd_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-family: Consolas, monospace;
            word-wrap: break-word;
        """)
        cmd_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        cmd_layout.addWidget(cmd_label, 1)
        
        # 复制按钮
        copy_btn = QPushButton("📋 复制")
        copy_btn.setFixedHeight(32)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        copy_btn.clicked.connect(self._copy_command)
        cmd_layout.addWidget(copy_btn)
        
        layout.addWidget(cmd_frame)
        
        # 描述
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            line-height: 1.4;
        """)
        layout.addWidget(desc_label)
    
    def _copy_command(self):
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.command)
        
        # 显示复制成功提示
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "复制成功", "命令已复制到剪贴板！")


class CommandsTab(QWidget):
    """常用命令标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        from .theme import COLORS
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # 标题区域
        header = QFrame()
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(8)
        
        title_label = QLabel("🛠️ 常用命令速查")
        title_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 20px;
            font-weight: bold;
        """)
        header_layout.addWidget(title_label)
        
        subtitle_label = QLabel("为不懂代码的用户准备的实用命令，点击复制即可使用")
        subtitle_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 14px;
        """)
        header_layout.addWidget(subtitle_label)
        
        layout.addWidget(header)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
        """)
        
        # 内容容器
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)
        
        # Python 包管理
        content_layout.addWidget(self._create_section(
            "📦 Python 包管理",
            "使用 pip 管理 Python 依赖包",
            [
                (
                    "安装依赖包",
                    "python -m pip install <包名>",
                    "示例：python -m pip install numpy\n安装指定的 Python 包"
                ),
                (
                    "卸载依赖包",
                    "python -m pip uninstall <包名>",
                    "示例：python -m pip uninstall numpy\n卸载不需要的 Python 包"
                ),
                (
                    "列出所有依赖",
                    "python -m pip list",
                    "查看当前环境已安装的所有 Python 包和版本"
                ),
                (
                    "升级包到最新版",
                    "python -m pip install --upgrade <包名>",
                    "示例：python -m pip install --upgrade pip\n将指定包更新到最新版本"
                ),
                (
                    "查看包信息",
                    "python -m pip show <包名>",
                    "示例：python -m pip show torch\n查看已安装包的详细信息"
                )
            ]
        ))
        
        # NVIDIA 相关
        content_layout.addWidget(self._create_section(
            "🖥️ NVIDIA 显卡信息",
            "查看显卡和 CUDA 相关信息",
            [
                (
                    "查看显卡信息",
                    "nvidia-smi",
                    "显示显卡型号、驱动版本、显存使用、CUDA 支持版本等信息\n非常实用的命令，用于诊断显卡相关问题"
                ),
                (
                    "查看 CUDA Toolkit 版本",
                    "nvcc --version",
                    "查看已安装的 CUDA Toolkit 编译器版本\n注意：这与 nvidia-smi 显示的 CUDA 版本不同！"
                ),
                (
                    "查看 GPU 持续监控",
                    "nvidia-smi -l 2",
                    "每 2 秒刷新一次显卡状态\n按 Ctrl + C 退出"
                )
            ]
        ))
        
        # 进程与端口管理
        content_layout.addWidget(self._create_section(
            "🔌 进程与端口管理",
            "查看和管理系统进程与端口占用",
            [
                (
                    "查看所有运行中的进程",
                    "tasklist",
                    "列出当前所有正在运行的进程\n可用于查看是否有程序在后台占用资源"
                ),
                (
                    "查看端口占用情况",
                    "netstat -ano",
                    "查看所有网络连接和端口占用情况\n可查找哪个程序占用了特定端口"
                ),
                (
                    "查看特定端口占用",
                    "netstat -ano | findstr <端口号>",
                    "示例：netstat -ano | findstr 7860\n查看指定端口被哪个程序占用"
                ),
                (
                    "根据 PID 查看进程名",
                    "tasklist | findstr <PID>",
                    "示例：tasklist | findstr 1234\n根据进程 ID 查找进程名称"
                ),
                (
                    "强制结束进程",
                    "taskkill /F /PID <PID>",
                    "示例：taskkill /F /PID 1234\n强制结束指定进程 ID 的程序\n谨慎使用，可能导致数据丢失"
                ),
                (
                    "按名称结束进程",
                    "taskkill /F /IM <进程名.exe>",
                    "示例：taskkill /F /IM python.exe\n强制结束所有同名进程"
                )
            ]
        ))
        
        # Git 版本控制
        content_layout.addWidget(self._create_section(
            "📝 Git 版本控制",
            "常用的 Git 命令，用于代码管理",
            [
                (
                    "查看当前代码状态",
                    "git status",
                    "查看当前目录的 Git 状态\n显示哪些文件被修改、新增或删除"
                ),
                (
                    "拉取最新代码",
                    "git pull",
                    "从远程仓库拉取最新代码\n常用于更新 WebUI 核心或插件到最新版本"
                ),
                (
                    "强制覆盖本地修改",
                    "git checkout -- .",
                    "撤销当前目录的所有本地修改\n恢复到上一次正常状态\n⚠️ 会丢失所有未保存的修改！"
                ),
                (
                    "查看提交日志",
                    "git log --oneline -10",
                    "查看最近 10 条提交记录\n快速了解代码变更历史"
                ),
                (
                    "查看修改内容",
                    "git diff",
                    "查看当前修改与上一版本的具体差异\n了解改了哪些地方"
                )
            ]
        ))
        
        # 清理缓存
        content_layout.addWidget(self._create_section(
            "🧹 清理命令",
            "清理项目缓存和临时文件",
            [
                (
                    "清理 Python 缓存",
                    r'for /d /r "D:\ai\sd-webui-forge-neo-v3" %d in (__pycache__) do @if exist "%d" rd /s /q "%d"',
                    "递归删除项目中的所有 __pycache__ 文件夹\n用于清理 Python 编译缓存"
                ),
                (
                    "清理当前目录缓存",
                    r'for /d /r "." %d in (__pycache__) do @if exist "%d" rd /s /q "%d"',
                    "删除当前目录及子目录中的所有 __pycache__"
                )
            ]
        ))
        
        # 常用快捷键
        content_layout.addWidget(self._create_section(
            "⌨️ 常用快捷键",
            "终端操作的常用快捷键",
            [
                (
                    "终止程序",
                    "Ctrl + C",
                    "强制终止正在运行的命令或程序\n在终端中按 Ctrl + C"
                ),
                (
                    "清屏",
                    "cls",
                    "清除当前终端窗口的所有内容"
                ),
                (
                    "退出终端",
                    "exit",
                    "关闭当前终端窗口"
                )
            ]
        ))
        
        # 其他实用命令
        content_layout.addWidget(self._create_section(
            "💡 其他实用命令",
            "一些有用的系统和网络命令",
            [
                (
                    "检查 Python 版本",
                    "python --version",
                    "查看当前使用的 Python 版本"
                ),
                (
                    "查看当前路径",
                    "cd",
                    "显示当前所在的目录路径"
                ),
                (
                    "列出当前目录文件",
                    "dir",
                    "查看当前目录下的所有文件和文件夹"
                ),
                (
                    "创建目录",
                    "mkdir <目录名>",
                    "示例：mkdir test\n创建一个新文件夹"
                )
            ]
        ))
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def _create_section(self, title, subtitle, commands):
        from .theme import COLORS
        
        section = QFrame()
        layout = QVBoxLayout(section)
        layout.setSpacing(12)
        
        # 标题
        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        title_layout.setSpacing(4)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 16px;
            font-weight: bold;
        """)
        title_layout.addWidget(title_label)
        
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 13px;
            """)
            title_layout.addWidget(subtitle_label)
        
        layout.addWidget(title_frame)
        
        # 命令卡片网格
        grid = QFrame()
        grid_layout = QGridLayout(grid)
        grid_layout.setSpacing(12)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        
        for i, (cmd_title, cmd_text, cmd_desc) in enumerate(commands):
            card = CommandCard(cmd_title, cmd_text, cmd_desc)
            grid_layout.addWidget(card, i // 1, i % 1)
        
        layout.addWidget(grid)
        
        return section
