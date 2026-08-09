"""主窗口"""
import json
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt

from .theme import MAIN_STYLE, COLORS
from .tab_launch import LaunchTab
from .tab_settings import SettingsTab
from .tab_paths import PathsTab
from .tab_env import EnvTab
from .tab_log import LogTab
from .tab_extensions import ExtensionsTab
from .hw_monitor import HWMonitorBar
from .tab_model_guide import ModelGuideTab
from .tab_plugin_guide import PluginGuideTab
from .tab_commands import CommandsTab
from core.config import load_config, save_config
from core.launcher import LaunchWorker
from core.llama_launcher import LlamaWorker
from core.paths import BASE_DIR


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = None
        self.worker: LaunchWorker | None = None
        self.llama_worker: LlamaWorker | None = None
        
        try:
            self.setWindowTitle("SD WebUI Forge 启动器")
            self.setMinimumSize(960, 660)
            self.resize(1060, 740)
            self.setStyleSheet(MAIN_STYLE)
            
            # 安全加载配置
            self._safe_load_config()
            
            # 启动时检测NVIDIA驱动版本（带错误保护）
            self._check_nvidia_driver_on_startup()
            
            # 构建UI
            self._build_ui()
            self._setup_shortcuts()
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            
            # 打印错误到控制台
            print("=" * 80)
            print("启动器初始化错误")
            print("=" * 80)
            print(tb_str)
            print("=" * 80)
            
            # 尝试构建一个简单的错误显示界面，让用户能看到错误信息
            try:
                from PyQt6.QtWidgets import QLabel, QVBoxLayout, QTextEdit, QWidget, QPushButton
                from PyQt6.QtCore import Qt
                
                central = QWidget()
                self.setCentralWidget(central)
                layout = QVBoxLayout(central)
                layout.setContentsMargins(40, 40, 40, 40)
                layout.setSpacing(16)
                
                # 错误图标
                lbl_icon = QLabel("⚠️")
                lbl_icon.setStyleSheet("font-size: 48px;")
                lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl_icon)
                
                # 错误标题
                lbl_title = QLabel("启动器初始化失败")
                lbl_title.setStyleSheet("color: #dc2626; font-size: 20px; font-weight: bold;")
                lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl_title)
                
                # 错误简述
                lbl_msg = QLabel(
                    f"启动器在初始化过程中遇到了错误，请按照以下步骤排查：\n\n"
                    f"1. 检查 NVIDIA 显卡驱动是否已安装\n"
                    f"2. 尝试重新安装依赖\n"
                    f"3. 如果问题持续，请将下方详细信息发送给开发者\n\n"
                    f"错误信息：{str(e)}"
                )
                lbl_msg.setStyleSheet("color: #6b7280; font-size: 13px;")
                lbl_msg.setWordWrap(True)
                lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl_msg)
                
                # 详情
                lbl_details = QLabel("详细信息：")
                lbl_details.setStyleSheet("color: #374151; font-size: 12px; margin-top: 8px;")
                layout.addWidget(lbl_details)
                
                txt_details = QTextEdit()
                txt_details.setReadOnly(True)
                txt_details.setPlainText(tb_str)
                txt_details.setStyleSheet("""
                    QTextEdit {
                        background-color: #1e1e2e;
                        border: 1px solid #374151;
                        border-radius: 6px;
                        padding: 8px;
                        color: #e5e7eb;
                        font-family: Consolas, monospace;
                        font-size: 11px;
                    }
                """)
                layout.addWidget(txt_details, 1)
                
                # 仍显示，不让用户卡住
                self.setWindowTitle("SD WebUI Forge 启动器 - 初始化失败")
            except Exception:
                pass
            
            # 不再 raise e，让窗口保持显示，用户能看到错误信息
    
    def _safe_load_config(self):
        """安全加载配置，防止配置文件损坏导致闪退"""
        try:
            self.config = load_config()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "配置加载警告",
                f"配置文件加载失败，将使用默认配置：\n{str(e)}"
            )
            # 使用默认配置
            self.config = {}
    
    def _check_nvidia_driver_on_startup(self):
        """启动时检测NVIDIA驱动版本，如果过低则显示警告"""
        try:
            from core.env_checker import check_nvidia_driver_version
            result = check_nvidia_driver_version()
            if result.get("has_gpu") and not result.get("ok"):
                # 驱动版本过低，显示警告
                from PyQt6.QtWidgets import QMessageBox
                
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Icon.Warning)
                msg_box.setWindowTitle("NVIDIA 驱动版本警告")
                msg_box.setText(
                    f"检测到您的 NVIDIA 驱动版本过低！\n\n"
                    f"当前版本: {result.get('driver', '未知')}\n"
                    f"最低要求: 596.21\n\n"
                    f"驱动版本过低可能导致 WebUI 无法正常启动或运行不稳定。\n"
                    f"建议前往 NVIDIA 官网下载并安装最新驱动。"
                )
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
        except Exception:
            # 如果检测失败（比如没有NVIDIA GPU），跳过即可
            pass

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_content(), 1)

        root.addWidget(self._build_statusbar())

    def _setup_shortcuts(self):
        from PyQt6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self._switch_tab(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self._switch_tab(1))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self._switch_tab(2))
        QShortcut(QKeySequence("Ctrl+4"), self, lambda: self._switch_tab(3))
        QShortcut(QKeySequence("Ctrl+5"), self, lambda: self._switch_tab(4))
        QShortcut(QKeySequence("Ctrl+6"), self, lambda: self._switch_tab(5))
        QShortcut(QKeySequence("Ctrl+7"), self, lambda: self._switch_tab(6))
        QShortcut(QKeySequence("Ctrl+8"), self, lambda: self._switch_tab(7))
        QShortcut(QKeySequence("Ctrl+W"), self, self.close)

    def _build_header(self):
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {COLORS['accent']}; font-size: 18px;")
        layout.addWidget(dot)

        title = QLabel("SD WebUI Forge")
        title.setStyleSheet(f"color:{COLORS['text_primary']};font-size:16px;font-weight:bold;margin-left:6px;")
        layout.addWidget(title)

        sub = QLabel("Neo v3")
        sub.setStyleSheet(f"color:{COLORS['accent_light']};font-size:11px;margin-left:4px;margin-top:4px;")
        layout.addWidget(sub)
        layout.addStretch()

        return header

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(160)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(2)

        self._nav_btns = []
        nav_items = [
            ("🏠", "主控台"),
            ("📄", "运行日志"),
            ("⚙️", "参数设置"),
            ("📁", "路径配置"),
            ("🧩", "插件管理"),
            ("🔍", "环境检测"),
            ("📚", "模型指南"),
            ("🛠️", "常用命令"),
        ]
        for i, (icon, text) in enumerate(nav_items):
            btn = QPushButton(f"  {icon}  {text}")
            btn.setCheckable(True)
            btn.setFixedHeight(44)
            btn.setStyleSheet(self._nav_btn_style())
            btn.clicked.connect(lambda checked, idx=i: self._switch_tab(idx))
            layout.addWidget(btn)
            self._nav_btns.append(btn)

        layout.addStretch()

        info = QLabel("Forge Neo v3")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"color:{COLORS['text_dim']};font-size:10px;")
        layout.addWidget(info)

        return sidebar

    def _nav_btn_style(self):
        return f"""
            QPushButton {{
                background: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 0;
                text-align: left;
                padding-left: 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['accent']}22;
                color: {COLORS['accent_light']};
                border-left: 4px solid {COLORS['accent']};
                border-radius: 0 8px 8px 0;
                font-weight: bold;
                padding-left: 16px;
            }}
        """

    def _build_content(self):
        # 创建主容器，使用QTabWidget但隐藏标签栏
        self.stack = QTabWidget()
        self.stack.tabBar().hide()
        self.stack.setStyleSheet(f"""
            QTabWidget::pane {{ border:none; background-color:{COLORS['bg_card']}; }}
        """)

        # 安全创建各个标签页
        try:
            self.tab_launch = LaunchTab(self.config, self)
        except Exception as e:
            print(f"Failed to create LaunchTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_launch = QLabel(f"主控台加载失败：\n{str(e)}")
        
        try:
            self.tab_settings = SettingsTab(self.config, self)
        except Exception as e:
            print(f"Failed to create SettingsTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_settings = QLabel(f"参数设置加载失败：\n{str(e)}")
        
        try:
            self.tab_paths = PathsTab(self.config, self)
        except Exception as e:
            print(f"Failed to create PathsTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_paths = QLabel(f"路径配置加载失败：\n{str(e)}")
        
        try:
            self.tab_extensions = ExtensionsTab(self.config, self)
        except Exception as e:
            print(f"Failed to create ExtensionsTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_extensions = QLabel(f"插件管理加载失败：\n{str(e)}")
        
        try:
            self.tab_env = EnvTab(self.config, self)
        except Exception as e:
            print(f"Failed to create EnvTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_env = QLabel(f"环境检测加载失败：\n{str(e)}")
        
        try:
            self.tab_log = LogTab(self)
        except Exception as e:
            print(f"Failed to create LogTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_log = QLabel(f"运行日志加载失败：\n{str(e)}")
        
        try:
            self.tab_model_guide = ModelGuideTab(self)
        except Exception as e:
            print(f"Failed to create ModelGuideTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_model_guide = QLabel(f"模型指南加载失败：\n{str(e)}")
        
        try:
            self.tab_commands = CommandsTab(self)
        except Exception as e:
            print(f"Failed to create CommandsTab: {e}")
            from PyQt6.QtWidgets import QLabel
            self.tab_commands = QLabel(f"常用命令加载失败：\n{str(e)}")
        
        # 添加到stack
        self.stack.addTab(self.tab_launch, "主控台")
        self.stack.addTab(self.tab_log, "运行日志")
        self.stack.addTab(self.tab_settings, "参数设置")
        self.stack.addTab(self.tab_paths, "路径配置")
        self.stack.addTab(self.tab_extensions, "插件管理")
        self.stack.addTab(self.tab_env, "环境检测")
        self.stack.addTab(self.tab_model_guide, "模型指南")
        self.stack.addTab(self.tab_commands, "常用命令")

        # 安全连接信号
        try:
            if hasattr(self.tab_launch, 'sig_launch'):
                self.tab_launch.sig_launch.connect(self._on_launch)
            if hasattr(self.tab_launch, 'sig_stop'):
                self.tab_launch.sig_stop.connect(self._on_stop)
            if hasattr(self.tab_launch, 'sig_stop_all'):
                self.tab_launch.sig_stop_all.connect(self._on_stop_all_confirm)
            if hasattr(self.tab_launch, 'sig_open_browser'):
                self.tab_launch.sig_open_browser.connect(self._on_open_browser)
            if hasattr(self.tab_launch, 'sig_goto'):
                self.tab_launch.sig_goto.connect(self._switch_tab)
            if hasattr(self.tab_launch, 'sig_llama_launch'):
                self.tab_launch.sig_llama_launch.connect(self._on_llama_launch)
            if hasattr(self.tab_launch, 'sig_llama_stop'):
                self.tab_launch.sig_llama_stop.connect(self._on_llama_stop)
            if hasattr(self.tab_launch, 'sig_llama_open'):
                self.tab_launch.sig_llama_open.connect(self._on_llama_open_browser)
        except Exception as e:
            print(f"Failed to connect launch signals: {e}")
        
        try:
            if hasattr(self.tab_log, 'sig_stop'):
                self.tab_log.sig_stop.connect(self._on_stop)
            if hasattr(self.tab_log, 'sig_restart'):
                self.tab_log.sig_restart.connect(self._on_restart)
        except Exception as e:
            print(f"Failed to connect log signals: {e}")

        return self.stack

    def _build_statusbar(self):
        bar = QFrame()
        bar.setFixedHeight(32)
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        self.lbl_status = QLabel("● 未运行")
        self.lbl_status.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        layout.addWidget(self.lbl_status)

        self.lbl_port = QLabel(f"端口: {self.config.get('port', 7869)}")
        self.lbl_port.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        layout.addWidget(self.lbl_port)

        layout.addStretch()

        return bar

    def _build_hw_bar(self):
        self.hw_bar = HWMonitorBar()
        return self.hw_bar

    def _switch_tab(self, index: int):
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
            for i, btn in enumerate(self._nav_btns):
                btn.setChecked(i == index)

    def _on_launch(self):
        try:
            # 防止重复启动：如果已有进程在运行，先提示用户
            if self.worker and self.worker.isRunning():
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line("⚠️  WebUI 已在运行中，请勿重复启动")
                return
            
            try:
                if hasattr(self.tab_settings, 'apply_to_config'):
                    self.tab_settings.apply_to_config(self.config)
                if hasattr(self.tab_paths, 'apply_to_config'):
                    self.tab_paths.apply_to_config(self.config)
                save_config(self.config)
            except Exception as e:
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(f"⚠️  保存配置失败：{str(e)}")
            
            # 启动前检测 Python 是否可用，不可用时自动安装
            try:
                from core.env_checker import ensure_python_installed
                py_result = ensure_python_installed()
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(py_result["message"])
                if not py_result["ok"]:
                    if hasattr(self.tab_log, 'append_line'):
                        self.tab_log.append_line("❌  Python 未就绪，无法启动 WebUI")
                    return
            except Exception as e:
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(f"⚠️  Python 检测失败：{str(e)}")
            
            # 启动前检测 Git 是否可用，不可用时自动安装
            try:
                from core.env_checker import ensure_git_installed
                git_result = ensure_git_installed()
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(git_result["message"])
                if not git_result["ok"]:
                    if hasattr(self.tab_log, 'append_line'):
                        self.tab_log.append_line("⚠️  Git 未就绪，但 WebUI 仍可尝试启动")
            except Exception as e:
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(f"⚠️  Git 检测失败：{str(e)}")
            
            # 启动前检测端口
            try:
                from core.config import is_port_in_use, find_available_port
                from core.launcher import kill_process_on_port
                
                original_port = self.config.get("port", 7869)
                if is_port_in_use(original_port):
                    # 尝试清理占用端口的进程
                    if hasattr(self.tab_log, 'append_line'):
                        self.tab_log.append_line(f"⚠️  检测到端口 {original_port} 被占用")
                    if kill_process_on_port(original_port):
                        if hasattr(self.tab_log, 'append_line'):
                            self.tab_log.append_line(f"✅ 已清理端口 {original_port} 的占用进程")
                        import time
                        time.sleep(0.5)  # 等待端口释放
                    else:
                        # 如果清理失败，自动切换到新端口
                        new_port = find_available_port(original_port)
                        self.config["port"] = new_port
                        save_config(self.config)
                        if hasattr(self.tab_log, 'append_line'):
                            self.tab_log.append_line(f"⚠️  无法清理端口，自动切换到端口 {new_port}")
                
                if hasattr(self, 'lbl_port'):
                    self.lbl_port.setText(f"端口: {self.config.get('port', 7869)}")
            except Exception as e:
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(f"⚠️  端口检测失败：{str(e)}")
            
            # 启动进程
            try:
                self.worker = LaunchWorker(self.config)
                # 将日志输出到LogTab
                if hasattr(self, 'tab_log') and hasattr(self.tab_log, 'append_line'):
                    self.worker.log_line.connect(self.tab_log.append_line)
                self.worker.finished.connect(self._on_finished)
                self.worker.start()
                
                if hasattr(self.tab_launch, 'set_running'):
                    self.tab_launch.set_running(True)
                if hasattr(self, 'tab_log') and hasattr(self.tab_log, 'set_running'):
                    self.tab_log.set_running(True, 1)
                if hasattr(self, 'hw_bar') and hasattr(self.hw_bar, 'set_proc_count'):
                    self.hw_bar.set_proc_count(1)
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.setText("● 运行中")
                    self.lbl_status.setStyleSheet(f"color:{COLORS['green']};font-weight:bold;font-size:11px;")
                
                # 保持在主控台
                self._switch_tab(0)
                
                # 自动启动 llama.cpp（如果启用）
                if self.config.get("llama", {}).get("enabled", True):
                    try:
                        if hasattr(self.tab_log, 'append_line'):
                            self.tab_log.append_line("⏱  WebUI 启动后自动启动 llama.cpp...")
                        from PyQt6.QtCore import QTimer
                        QTimer.singleShot(2000, self._on_llama_launch)
                    except Exception as e:
                        if hasattr(self.tab_log, 'append_line'):
                            self.tab_log.append_line(f"⚠️  llama 自动启动失败：{str(e)}")
            except Exception as e:
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(f"❌  启动失败：{str(e)}")
                import traceback
                if hasattr(self.tab_log, 'append_line'):
                    self.tab_log.append_line(traceback.format_exc())
        except Exception as e:
            try:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "启动错误",
                    f"启动过程发生严重错误：\n{str(e)}"
                )
            except Exception:
                pass

    def _on_stop(self):
        if self.worker:
            try:
                self.worker.log_line.disconnect()
                self.worker.finished.disconnect()
            except Exception:
                pass
            self.worker.stop()
            if not self.worker.wait(3000):
                try:
                    self.worker.force_kill()
                except Exception:
                    pass
                self.worker.wait(1000)
            try:
                self.worker.deleteLater()
            except Exception:
                pass
            self.worker = None

        self.tab_launch.set_running(False)
        if hasattr(self, 'tab_log'):
            self.tab_log.set_running(False)
        if hasattr(self, 'hw_bar'):
            self.hw_bar.set_proc_count(0)
        self.lbl_status.setText("● 未运行")
        self.lbl_status.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")

    def _on_stop_all(self):
        """停止所有相关进程"""
        if self.worker and self.worker.isRunning():
            self.tab_log.append_line("[停止全部进程] 正在终止 WebUI 进程...")
            self._on_stop()
        else:
            self.tab_log.append_line("[停止全部进程] 当前没有运行中的进程")

    def _on_stop_all_confirm(self):
        # sig_stop_all is already emitted after user confirmed in tab_launch._confirm_stop_all
        # Just execute the stop action
        self._on_stop_all()

    def _on_open_browser(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        port = self.config.get("port", 7869)
        url = f"http://127.0.0.1:{port}"
        QDesktopServices.openUrl(QUrl(url))

    def _on_restart(self):
        """重新启动 WebUI 进程"""
        from .theme import COLORS
        self.tab_log.append_line("[重新启动] 开始重新启动 WebUI 进程...")
        
        # 1. 先停止当前运行的进程
        if self.worker and self.worker.isRunning():
            self.tab_log.append_line("[重新启动] 正在停止当前进程...")
            try:
                self.worker.log_line.disconnect()
                self.worker.finished.disconnect()
            except Exception:
                pass
            self.worker.stop()
            if not self.worker.wait(3000):
                try:
                    self.worker.force_kill()
                except Exception:
                    pass
                self.worker.wait(1000)
            try:
                self.worker.deleteLater()
            except Exception:
                pass
            self.worker = None
            
            self.tab_launch.set_running(False)
            if hasattr(self, 'tab_log'):
                self.tab_log.set_running(False)
            if hasattr(self, 'hw_bar'):
                self.hw_bar.set_proc_count(0)
            self.lbl_status.setText("● 未运行")
            self.lbl_status.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
            
            # 等待端口释放
            import time
            time.sleep(1)
        
        # 2. 启动新的进程
        self.tab_log.append_line("[重新启动] 正在启动新进程...")
        self._on_launch()

    def _on_finished(self, code: int):
        self.tab_launch.set_running(False)
        if hasattr(self, 'tab_log'):
            self.tab_log.set_running(False)
        if hasattr(self, 'hw_bar'):
            self.hw_bar.set_proc_count(0)
        self.lbl_status.setText("● 未运行")
        self.lbl_status.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        # 在LaunchTab的日志中显示退出信息
        self.tab_launch.append_log(f"\n[进程结束] 退出码: {code}")
        if hasattr(self, 'tab_log'):
            self.tab_log.append_line(f"\n[进程结束] 退出码: {code}")
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _on_llama_launch(self):
        """启动 llama.cpp"""
        # 先保存配置
        try:
            if hasattr(self.tab_settings, 'apply_to_config'):
                self.tab_settings.apply_to_config(self.config)
            save_config(self.config)
        except Exception as e:
            self.tab_log.append_line(f"⚠️  保存配置失败：{str(e)}")

        llama_cfg = self.config.get("llama", {})
        port = llama_cfg.get("port", 8080)

        # 如果 llama 已在运行中，则只更新端口配置文件
        if self.llama_worker and self.llama_worker.isRunning():
            self.tab_log.append_line("⚠️  llama.cpp 已在运行中，更新端口配置...")
            self._write_llama_port_file(port)
            return

        # 检测端口是否已被占用（手动启动的场景）
        import socket
        _port_in_use = False
        try:
            with socket.create_connection(("localhost", port), timeout=0.3):
                _port_in_use = True
        except (OSError, socket.timeout):
            pass
        if _port_in_use:
            self.tab_log.append_line(f"⚠️  端口 {port} 已被占用，检测到 llama.cpp 可能已手动启动，更新端口配置...")
            self._write_llama_port_file(port)
            return

        from core.llama_launcher import LLAMA_SERVER, scan_llama_models
        if not os.path.exists(LLAMA_SERVER):
            self.tab_log.append_line(f"❌ 未找到 {LLAMA_SERVER}，请确认 llama.cpp 已正确安装")
            return

        self.tab_log.append_line("🔍 正在检测 llama.cpp 模型...")
        models = scan_llama_models()
        if not models:
            self.tab_log.append_line("❌ 未在 llama.cpp/models/ 目录下找到 .gguf 模型文件")
            return

        ngl = llama_cfg.get("ngl", 100)

        self.tab_log.append_line(f"🚀 正在启动 llama.cpp 模型 → 端口 {port}（-ngl {ngl}）")

        self.llama_worker = LlamaWorker(self.config)
        self.llama_worker.log_line.connect(self.tab_log.append_line)
        self.llama_worker.finished.connect(self._on_llama_finished)
        self.llama_worker.start()

        self.tab_log.append_line("🚀 正在启动 llama.cpp...")

    def _write_llama_port_file(self, port: int):
        """写入 llama.cpp 共享端口配置文件"""
        from core.llama_launcher import LLAMA_PORT_FILE
        try:
            os.makedirs(os.path.dirname(LLAMA_PORT_FILE), exist_ok=True)
            with open(LLAMA_PORT_FILE, "w", encoding="utf-8") as f:
                json.dump({"port": port, "host": "0.0.0.0"}, f)
            self.tab_log.append_line(f"✅ 已更新 llama.cpp 端口配置: {port}")
        except Exception as e:
            self.tab_log.append_line(f"⚠️ 写入共享端口配置失败: {e}")

    def _on_llama_stop(self):
        """停止 llama.cpp"""
        if self.llama_worker and self.llama_worker.isRunning():
            try:
                self.llama_worker.log_line.disconnect()
                self.llama_worker.finished.disconnect()
            except Exception:
                pass
            self.llama_worker.stop()
            if not self.llama_worker.wait(3000):
                try:
                    self.llama_worker.force_kill() if hasattr(self.llama_worker, 'force_kill') else None
                except Exception:
                    pass
            try:
                self.llama_worker.deleteLater()
            except Exception:
                pass
            self.llama_worker = None
            self.tab_log.append_line("🛑 llama.cpp 已停止")

    def _on_llama_open_browser(self):
        """打开 llama.cpp 的 Web 页面"""
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        port = self.config.get("llama", {}).get("port", 8080)
        url = f"http://127.0.0.1:{port}"
        QDesktopServices.openUrl(QUrl(url))

    def _on_llama_finished(self, code: int):
        self.tab_log.append_line(f"\n[llama.cpp 进程结束] 退出码: {code}")
        if self.llama_worker:
            self.llama_worker.deleteLater()
            self.llama_worker = None

    def closeEvent(self, event):
        save_config(self.config)
        
        # 清理所有残留的临时bat文件
        from core.launcher import cleanup_all_temp_files
        cleanup_all_temp_files()
        
        # 清理 tab_launch 的后台线程
        if hasattr(self.tab_launch, 'stop_all'):
            # 异步清理，避免阻塞退出
            import threading
            def cleanup_thread():
                try:
                    self.tab_launch.stop_all()
                except Exception:
                    pass
            threading.Thread(target=cleanup_thread, daemon=True).start()
        
        # 停止 WebUI 进程（非阻塞方式）
        if self.worker and self.worker.isRunning():
            try:
                self.worker.log_line.disconnect()
                self.worker.finished.disconnect()
            except Exception:
                pass
            
            # 异步停止进程，避免阻塞退出
            import threading
            def stop_thread():
                try:
                    self.worker.stop()
                    self.worker.deleteLater()
                except Exception:
                    pass
            threading.Thread(target=stop_thread, daemon=True).start()
        
        # 停止 llama 进程
        if self.llama_worker and self.llama_worker.isRunning():
            def stop_llama():
                try:
                    self.llama_worker.stop()
                    self.llama_worker.deleteLater()
                except Exception:
                    pass
            threading.Thread(target=stop_llama, daemon=True).start()
        
        # 立即接受退出事件，避免卡顿
        event.accept()
