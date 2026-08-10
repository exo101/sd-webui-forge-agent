"""主控台 Tab"""
import os
import subprocess
import webbrowser
import psutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QPlainTextEdit, QMessageBox, QProgressDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QPainter, QPixmap
from .theme import COLORS
from core.paths import BASE_DIR, WEBUI_DIR


class WallpaperWidget(QWidget):
    """自定义 QWidget，通过 paintEvent 绘制背景壁纸，不受样式表影响"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._fallback_color = "#16161f"
        # 阻止样式表绘制背景，全部由 paintEvent 负责
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def set_wallpaper(self, path: str):
        if path and os.path.exists(path):
            self._pixmap = QPixmap(path)
            if self._pixmap.isNull():
                self._pixmap = None
        else:
            self._pixmap = None
        self.update()

    def set_fallback_color(self, color: str):
        self._fallback_color = color
        self.update()

    def paintEvent(self, event):
        if self._pixmap and not self._pixmap.isNull():
            painter = QPainter(self)
            # 先填充纯色背景
            from PyQt6.QtGui import QColor
            painter.fillRect(self.rect(), QColor(self._fallback_color))
            # 缩放壁纸保持比例，完整显示（不裁剪）
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
        else:
            # 无壁纸时使用纯色背景
            from PyQt6.QtGui import QColor
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(self._fallback_color))
            painter.end()


WEBUI_MODELS_DIR = os.path.join(WEBUI_DIR, "models")


class GitCheckWorker(QThread):
    """异步检测 WebUI 是否有新版本"""
    result = pyqtSignal(dict)  # {has_update, local, remote, error}

    def run(self):
        git = os.path.join(BASE_DIR, "system", "git", "bin", "git.exe")
        _env = {**os.environ, "GIT_SSL_NO_VERIFY": "1"}
        try:
            r_local = subprocess.run([git, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=WEBUI_DIR,
                creationflags=subprocess.CREATE_NO_WINDOW, env=_env)
            local = r_local.stdout.strip() if r_local.returncode == 0 else "?"

            subprocess.run([git, "fetch", "origin", "--quiet"],
                capture_output=True, timeout=15, cwd=WEBUI_DIR,
                creationflags=subprocess.CREATE_NO_WINDOW, env=_env)

            r_remote = subprocess.run([git, "rev-parse", "--short", "origin/HEAD"],
                capture_output=True, text=True, timeout=5, cwd=WEBUI_DIR,
                creationflags=subprocess.CREATE_NO_WINDOW, env=_env)
            remote = r_remote.stdout.strip() if r_remote.returncode == 0 else "?"

            has_update = bool(local and remote and local != remote and remote != "?")
            self.result.emit({"has_update": has_update, "local": local, "remote": remote, "error": ""})
        except Exception as e:
            self.result.emit({"has_update": False, "local": "?", "remote": "?", "error": str(e)})


class LaunchTab(QWidget):
    sig_launch       = pyqtSignal()
    sig_stop         = pyqtSignal()
    sig_open_browser = pyqtSignal()
    sig_goto         = pyqtSignal(int)   # 跳转到其他 tab
    sig_stop_all     = pyqtSignal()
    sig_llama_launch = pyqtSignal()
    sig_llama_stop   = pyqtSignal()
    sig_llama_open   = pyqtSignal()
    sig_comfy_launch = pyqtSignal()
    sig_comfy_stop   = pyqtSignal()
    sig_comfy_open   = pyqtSignal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._running = False
        self._git_worker = None
        self._build_ui()
        # 已移除GPU检测和内存检测，环境检测页面已有此功能

    # Stop all processes launched by this launcher (internal helper for outer control)
    def stop_all(self):
        """彻底清理所有WebUI相关进程和资源"""
        import signal
        
        # 1. 停止git检查线程
        try:
            if hasattr(self, "_git_worker") and self._git_worker is not None:
                try:
                    self._git_worker.terminate()
                    self._git_worker.wait(2000)  # 等待最多2秒
                except Exception:
                    pass
                self._git_worker = None
        except Exception:
            pass
        
        # 2. 查找并终止所有Python子进程（WebUI相关）
        try:
            current_pid = os.getpid()  # 当前启动器进程ID
            webui_dir_lower = WEBUI_DIR.lower()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
                try:
                    pid = proc.info['pid']
                    cmdline = proc.info['cmdline']
                    cwd = proc.info['cwd']
                    
                    # 跳过自己
                    if pid == current_pid:
                        continue
                    
                    # 检查是否是WebUI相关的Python进程
                    is_webui_process = False
                    
                    if cmdline:
                        cmdline_str = ' '.join(cmdline).lower()
                        # 检查命令行是否包含webui目录
                        if webui_dir_lower in cmdline_str:
                            is_webui_process = True
                        # 检查是否是gradio/uvicorn等WebUI服务进程
                        elif any(keyword in cmdline_str for keyword in ['webui.py', 'launch.py', 'gradio', 'uvicorn']):
                            is_webui_process = True
                    
                    if cwd and webui_dir_lower in cwd.lower():
                        is_webui_process = True
                    
                    # 如果是WebUI进程，强制终止
                    if is_webui_process:
                        parent = psutil.Process(pid)
                        # 先终止所有子进程
                        children = parent.children(recursive=True)
                        for child in children:
                            try:
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # 再终止父进程
                        try:
                            parent.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            pass
        
        # 3. 清理端口占用
        try:
            port = self.config.get("port", 7869)
            from core.launcher import kill_process_on_port
            if kill_process_on_port(port):
                pass
        except Exception as e:
            pass
        
        # 4. 清理临时文件
        try:
            from core.launcher import cleanup_all_temp_files
            cleanup_all_temp_files()
        except Exception as e:
            pass

    def set_running(self, running: bool):
        """设置运行状态"""
        self._running = running
        
        # 更新按钮状态
        if hasattr(self, 'btn_launch'):
            self.btn_launch.setEnabled(not running)
            if running:
                self.btn_launch.setText("⏳  运行中...")
                self.lbl_hint.setText("WebUI 正在运行中...")
            else:
                self.btn_launch.setText("⚡  启动")
                self.lbl_hint.setText("点击「启动」开始运行 WebUI")
        
        if hasattr(self, 'btn_stop'):
            self.btn_stop.setEnabled(running)
        
        if hasattr(self, 'btn_stop_all'):
            self.btn_stop_all.setEnabled(running)
        
        if hasattr(self, 'btn_browser'):
            self.btn_browser.setEnabled(running)

    def append_log(self, text: str):
        """添加日志到主控台（兼容旧版调用）"""
        # 日志输出到运行日志 Tab 中
        try:
            parent = self.parent()
            if parent and hasattr(parent, 'tab_log') and hasattr(parent.tab_log, 'append_line'):
                parent.tab_log.append_line(text)
        except Exception:
            pass

    def _confirm_stop_all(self):
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要停止所有进程吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.sig_stop_all.emit()

    def _build_ui(self):
        # ── 主容器 ──
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 壁纸标题区（固定高度） ──
        wallpaper_path = self.config.get("wallpaper", "")
        if not wallpaper_path or not os.path.exists(wallpaper_path):
            wallpaper_path = os.path.join(BASE_DIR, "launcher", "主题壁纸.png")

        header = WallpaperWidget()
        header.setMinimumHeight(200)  # 防止被内容区挤压
        if os.path.exists(wallpaper_path):
            header.set_wallpaper(wallpaper_path)
        else:
            header.set_fallback_color(COLORS['bg_card'])

        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(40, 0, 40, 0)
        header_layout.setSpacing(6)

        # 将内容垂直居中
        header_layout.addStretch()
        main_layout.addWidget(header, 1)  # 1/3 空间

        # ── 内容区（可滚动，占 2/3 空间） ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {COLORS['bg_card']}; border: none; }}")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setStyleSheet(f"background: {COLORS['bg_card']};")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(28, 16, 28, 16)
        content_layout.setSpacing(12)

        # ── 更新提示横幅（默认隐藏） ──
        self.update_banner = self._build_update_banner()
        self.update_banner.hide()
        content_layout.addWidget(self.update_banner)

        # ── 内存警告横幅（默认隐藏） ──
        self.mem_banner = self._build_mem_banner()
        self.mem_banner.hide()
        content_layout.addWidget(self.mem_banner)

        # ── 底部控制面板 ──
        # 用 QScrollArea 包裹底部容器，防止窗口缩放时内容交错
        bottom_scroll = QScrollArea()
        bottom_scroll.setWidgetResizable(True)
        bottom_scroll.setFrameShape(QFrame.Shape.NoFrame)
        bottom_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bottom_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        bottom_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        bottom_container = QFrame()
        bottom_container.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border-radius: 10px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        bottom_container.setMinimumWidth(600)
        bottom_row = QHBoxLayout(bottom_container)
        bottom_row.setContentsMargins(12, 12, 12, 12)
        bottom_row.setSpacing(12)

        # 左侧：WebUI 启动按钮组（可伸缩）
        buttons_widget = self._build_launch_buttons()
        bottom_row.addWidget(buttons_widget, 1)

        # 右侧：文字说明（固定宽度）
        info_widget = self._build_compact_info()
        info_widget.setFixedWidth(260)
        bottom_row.addWidget(info_widget, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        bottom_scroll.setWidget(bottom_container)
        content_layout.addWidget(bottom_scroll)

        # ── 状态提示 ──
        self.lbl_hint = QLabel("点击「启动」开始运行 WebUI")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        content_layout.addWidget(self.lbl_hint)

        # ── 资源汇总卡片（放在最底部） ──
        resource_widget = self._build_resource_cards()
        content_layout.addWidget(resource_widget)

        content_layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll, 2)  # 2/3 空间

        # 主布局
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(main_widget)

    # ── 更新横幅 ──────────────────────────────────────────────

    def _build_announcement(self) -> QFrame:
        """构建公告卡片"""
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QLabel
        
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_dark']};
                border-radius: 10px;
                border: 2px solid {COLORS['accent']}55;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # 顶部区域：左侧QQ群图片 + 右侧公告信息并排
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # 左侧：QQ群图片（固定宽度）
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.setAlignment(Qt.AlignmentFlag.AlignTop)  # 顶部对齐
        
        img_loaded = False
        try:
            # 尝试多个可能的图片路径
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "qq群ai交流群.jpg"),
                os.path.join(os.path.dirname(__file__), "..", "..", "qq群ai交流群.jpg"),
                os.path.join(BASE_DIR, "launcher", "qq群ai交流群.jpg"),
            ]
            
            img_path = None
            for p in possible_paths:
                p = os.path.normpath(p)
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path:
                lbl_img = QLabel()
                lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_img.setMinimumSize(150, 150)  # 设置最小尺寸
                
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
                    lbl_img.setPixmap(scaled)
                    
                    lbl_img.setStyleSheet(f"""
                        background-color: {COLORS['bg_card']};
                        border-radius: 8px;
                        padding: 8px;
                    """)
                    left_col.addWidget(lbl_img)
                    img_loaded = True
                    
                    lbl_hint = QLabel("扫码加入 AI 交流群")
                    lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    lbl_hint.setStyleSheet(f"""
                        color: {COLORS['text_primary']};
                        font-size: 11px;
                        font-weight: bold;
                        padding-top: 4px;
                    """)
                    left_col.addWidget(lbl_hint)
        except Exception as e:
            print(f"[DEBUG] Exception loading image: {e}")
        
        if not img_loaded:
            lbl_qq = QLabel("💬\n交流群")
            lbl_qq.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_qq.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 12px;
                padding: 40px 20px;
                background-color: {COLORS['bg_card']};
                border-radius: 8px;
            """)
            left_col.addWidget(lbl_qq)
        
        top_row.addLayout(left_col)

        # 右侧：公告信息和开发者信息
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # 硬件兼容性警告和重要提示
        announcements = [
            ("❌", "不支持AMD显卡与苹果系统）", "#ff6b6b"),
            ("✅", "CUDA 13.0 + PyTorch 2.10 最新环境", COLORS['green']),
            ("⚠️", "英伟达驱动不得低于596.49）", "#ffd93d"),
            ("⚠️", "启动 WebUI 时请关闭代理软件，端口可能会被占用，建议启动后再开启", "#ffd93d"),
            ("💡", "旧版 SD WebUI 模型通用，直接迁移 models 目录即可", COLORS['accent']),
            ("⚠️", "不可把放模型与文件到重复目录或特殊符号命名目录下", "#ff6b6b")
        ]

        for icon, text, color in announcements:
            row = QHBoxLayout()
            row.setSpacing(6)
            
            lbl_icon = QLabel(icon)
            lbl_icon.setStyleSheet(f"font-size: 12px;")
            row.addWidget(lbl_icon)
            
            lbl_text = QLabel(text)
            lbl_text.setWordWrap(True)
            lbl_text.setStyleSheet(f"""
                color: {color};
                font-size: 12px;
                font-weight: bold;
            """)
            row.addWidget(lbl_text, 1)
            
            right_col.addLayout(row)

        # 分隔线
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setStyleSheet(f"background-color: {COLORS['border']};max-height: 1px;")
        right_col.addWidget(separator1)

        # 开发者信息
        dev_items = [
            ("B站", "哔哩哔哩（鸡肉爱土豆）", "https://space.bilibili.com/403361177"),
            ("GitHub", "exo101", "https://github.com/exo101"),
        ]
        
        for label, value, url in dev_items:
            row = QHBoxLayout()
            row.setSpacing(6)
            
            lbl_label = QLabel(f"{label}:")
            lbl_label.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:11px;min-width:40px;")
            row.addWidget(lbl_label)
            
            if url:
                lbl_value = QLabel(value)
                lbl_value.setStyleSheet(f"color:{COLORS['accent']};font-size:11px;")
                lbl_value.setCursor(Qt.CursorShape.PointingHandCursor)
                
                # 创建自定义点击事件处理
                def make_click_handler(link_url):
                    def handler(event):
                        QDesktopServices.openUrl(QUrl(link_url))
                    return handler
                
                lbl_value.mousePressEvent = make_click_handler(url)
                row.addWidget(lbl_value, 1)
            else:
                lbl_value = QLabel(value)
                lbl_value.setStyleSheet(f"color:{COLORS['text_primary']};font-size:11px;")
                row.addWidget(lbl_value, 1)
            
            right_col.addLayout(row)

        # 分隔线
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet(f"background-color: {COLORS['border']};max-height: 1px;")
        right_col.addWidget(separator2)

        # 整合包声明
        disclaimer_text = (
            "此版本整合包通过秋叶aaaki、张吕敏、Haoming02等多位大佬技术总结做出的版本，"
            "不属于任何个人、企业，是非盈利性质的开源软件。"
        )
        lbl_disclaimer = QLabel(disclaimer_text)
        lbl_disclaimer.setWordWrap(True)
        lbl_disclaimer.setStyleSheet(f"""
            color: {COLORS['text_dim']};
            font-size: 10px;
            line-height: 1.4;
        """)
        right_col.addWidget(lbl_disclaimer)

        right_col.addStretch()
        top_row.addLayout(right_col, 1)  #右侧区域占满剩余空间

        layout.addLayout(top_row)

        return frame

    def _build_console_area(self) -> QFrame:
        """构建主控台内容区域"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_dark']};
                border-radius: 10px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 移除内存信息和状态提示

        return frame

    def _build_compact_info(self) -> QFrame:
        """右下角紧凑文字说明"""
        frame = QFrame()
        frame.setMinimumWidth(280)
        frame.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)

        lines = [
            ("❌", "不支持 A卡（AMD显卡）", "#ff6b6b"),
            ("✅", "CUDA 13.0 + PyTorch 2.10 最新环境", COLORS['green']),
            ("⚠️", "使用前务必更新英伟达驱动至最新版（596.49）", "#ffd93d"),
            ("⚠️", "启动时请关闭代理软件，避免端口冲突", "#ffd93d"),
            ("💡", "旧版 SD WebUI 模型通用，直接迁移 models 目录", COLORS['accent']),
        ]

        for icon, text, color in lines:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl_icon = QLabel(icon)
            lbl_icon.setStyleSheet("font-size: 10px; background: transparent;")
            row.addWidget(lbl_icon)
            lbl_text = QLabel(text)
            lbl_text.setWordWrap(True)
            lbl_text.setStyleSheet(f"color: {color}; font-size: 10px; background: transparent;")
            row.addWidget(lbl_text, 1)
            layout.addLayout(row)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background: {COLORS['border']}; max-height: 1px;")
        layout.addWidget(sep)

        # 底部署名
        footnote = QHBoxLayout()
        footnote.setSpacing(8)
        lbl_dev = QLabel("B站: 鸡肉爱土豆 | GitHub: exo101")
        lbl_dev.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 9px; background: transparent;")
        footnote.addWidget(lbl_dev)
        lbl_disclaimer = QLabel("非盈利开源软件")
        lbl_disclaimer.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 9px; background: transparent;")
        footnote.addWidget(lbl_disclaimer)
        footnote.addStretch()
        layout.addLayout(footnote)

        return frame

    def _secondary_btn_style(self) -> str:
        """次要按钮样式"""
        return f"""
            QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']};border-radius:6px;font-size:11px;}}
            QPushButton:hover{{background:{COLORS['bg_hover']};color:{COLORS['text_primary']};}}
            QPushButton:disabled{{background:{COLORS['bg_card']};color:{COLORS['text_dim']};
                border-color:{COLORS['border']};}}
        """

    def _on_stop_click(self):
        """停止按钮点击处理"""
        self.sig_stop.emit()

    def _active_btn_style(self) -> str:
        return f"""
            QPushButton {{
                background: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLORS['accent_light']};
            }}
        """

    def _inactive_btn_style(self) -> str:
        return f"""
            QPushButton {{
                background: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {COLORS['bg_hover']};
                color: {COLORS['text_primary']};
            }}
        """

    def _build_update_banner(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{ background:#1a3a1a; border-radius:10px;
                border:1px solid {COLORS['green']}55; }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(4)

        top = QHBoxLayout()
        icon = QLabel("✅")
        icon.setStyleSheet("font-size:18px;")
        top.addWidget(icon)

        self.lbl_update_title = QLabel("WebUI 有新版本可以更新！")
        self.lbl_update_title.setStyleSheet(
            f"color:{COLORS['green']};font-size:14px;font-weight:bold;")
        top.addWidget(self.lbl_update_title)
        top.addStretch()

        btn_close_banner = QPushButton("关闭")
        btn_close_banner.setFixedSize(50, 24)
        btn_close_banner.setStyleSheet(f"""
            QPushButton{{background:transparent;color:{COLORS['text_dim']};
                border:1px solid {COLORS['border']};border-radius:4px;font-size:11px;}}
            QPushButton:hover{{color:{COLORS['text_primary']};}}
        """)
        btn_close_banner.clicked.connect(lambda: self.update_banner.hide())
        top.addWidget(btn_close_banner)
        layout.addLayout(top)

        self.lbl_update_detail = QLabel("")
        self.lbl_update_detail.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;")
        layout.addWidget(self.lbl_update_detail)

        return frame

    # ── 内存警告横幅 ──────────────────────────────────────────

    def _build_mem_banner(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{ background:#2a2a0a; border-radius:10px;
                border:1px solid {COLORS['yellow']}55; }}
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 12, 20, 12)

        icon = QLabel("⚠️")
        icon.setStyleSheet("font-size:18px;")
        layout.addWidget(icon)

        self.lbl_mem_warn = QLabel("")
        self.lbl_mem_warn.setStyleSheet(f"color:{COLORS['yellow']};font-size:12px;")
        self.lbl_mem_warn.setWordWrap(True)
        layout.addWidget(self.lbl_mem_warn, 1)

        return frame

    # ── 资源汇总卡片 ──────────────────────────────────────────

    def _build_resource_cards(self) -> QFrame:
        """构建资源汇总卡片（2列网格）"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border-radius: 10px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # 标题行
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        lbl_title = QLabel("📦  资源汇总")
        lbl_title.setStyleSheet(f"color:{COLORS['text_primary']};font-size:13px;font-weight:bold;background:transparent;")
        title_row.addWidget(lbl_title)
        lbl_sub = QLabel("点击卡片跳转到相应网站")
        lbl_sub.setStyleSheet(f"color:{COLORS['text_dim']};font-size:10px;background:transparent;")
        title_row.addWidget(lbl_sub)
        title_row.addStretch()
        layout.addLayout(title_row)

        # 资源网格
        resources = [
            {"name": "GitHub",     "url": "https://github.com/exo101",                                "description": "Stable Diffusion WebUI Forge Neo 中文版项目主页"},
            {"name": "魔搭社区",   "url": "https://www.modelscope.cn/home",                            "description": "ModelScope 开源模型社区"},
            {"name": "Ollama",     "url": "https://ollama.com/download/windows",                       "description": "本地运行大语言模型的工具"},
            {"name": "Hugging Face", "url": "https://huggingface.co/",                                 "description": "机器学习模型和数据集的开源平台"},
            {"name": "哩布哩布",   "url": "https://www.liblib.art/inspiration",                        "description": "中国领先的AI创作平台"},
            {"name": "C站",        "url": "https://civitai.com/",                                      "description": "AI艺术模型分享平台"},
        ]

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)

        for i, res in enumerate(resources):
            row = i // 2
            col = i % 2
            card = self._create_resource_card(res)
            grid.addWidget(card, row, col)

        layout.addLayout(grid)
        return frame

    def _create_resource_card(self, resource: dict) -> QFrame:
        """创建单个资源卡片"""
        card = QFrame()
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-radius: 6px;
                border: 1px solid {COLORS['border']};
            }}
            QFrame:hover {{
                border-color: {COLORS['accent_light']};
                background-color: {COLORS['bg_hover']};
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # 名称
        name = QLabel(resource["name"])
        name.setStyleSheet(f"color:{COLORS['text_primary']};font-size:13px;font-weight:bold;background:transparent;")
        layout.addWidget(name)

        # 描述
        desc = QLabel(resource["description"])
        desc.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:11px;background:transparent;")
        desc.setWordWrap(True)
        layout.addWidget(desc, 1)

        # 箭头
        arrow = QLabel("→")
        arrow.setStyleSheet(f"color:{COLORS['accent_light']};font-size:14px;background:transparent;")
        layout.addWidget(arrow)

        # 点击事件
        url = resource["url"]
        def make_handler(u):
            return lambda e: webbrowser.open(u)
        card.mousePressEvent = make_handler(url)

        return card

    # ── 快速启动按钮组 ────────────────────────────────────────

    def _build_launch_buttons(self) -> QFrame:
        frame = QFrame()
        frame.setMinimumWidth(280)
        frame.setStyleSheet("background: transparent;")
        grid_layout = QVBoxLayout(frame)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        grid_layout.setSpacing(6)

        # 第一行：启动按钮
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self.btn_launch = QPushButton("⚡ 启动")
        self.btn_launch.setMinimumHeight(38)
        self.btn_launch.setStyleSheet(f"""
            QPushButton{{background:{COLORS['accent']};color:white;border:none;
                border-radius:6px;font-size:13px;font-weight:bold;}}
            QPushButton:hover{{background:{COLORS['accent_light']};}}
            QPushButton:disabled{{background:#3a3a5c;color:{COLORS['text_dim']};}}
        """)
        self.btn_launch.clicked.connect(self.sig_launch.emit)
        row1.addWidget(self.btn_launch, 2)


        
        grid_layout.addLayout(row1)

        # 第二行：打开界面 + 停止 + 全部停止
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self.btn_browser = QPushButton("🌐 页面")
        self.btn_browser.setMinimumHeight(32)
        self.btn_browser.setEnabled(False)
        self.btn_browser.setStyleSheet(self._secondary_btn_style())
        self.btn_browser.clicked.connect(self.sig_open_browser.emit)
        row2.addWidget(self.btn_browser, 1)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(f"""
            QPushButton{{background:#3d1f1f;color:{COLORS['red']};
                border:1px solid #5a2020;border-radius:6px;font-size:11px;}}
            QPushButton:hover{{background:#5a2020;}}
            QPushButton:disabled{{background:{COLORS['bg_card']};
                color:{COLORS['text_dim']};border-color:{COLORS['border']};}}
        """)
        self.btn_stop.clicked.connect(self._on_stop_click)
        row2.addWidget(self.btn_stop, 1)

        self.btn_stop_all = QPushButton("全部停止")
        self.btn_stop_all.setMinimumHeight(32)
        self.btn_stop_all.setEnabled(False)
        self.btn_stop_all.setStyleSheet(f"""
            QPushButton{{background:#2a2a2a;color:{COLORS['red']};
                border:1px solid {COLORS['border']}; border-radius:6px; font-size:11px;}}
            QPushButton:hover{{background:#3a3a3a;}}
        """)
        self.btn_stop_all.clicked.connect(self._confirm_stop_all)
        row2.addWidget(self.btn_stop_all, 1)
        grid_layout.addLayout(row2)

        # 第三行：打开 llama 端口
        row3 = QHBoxLayout()
        row3.setSpacing(6)

        self.btn_llama_open = QPushButton("🌐 llama 端口")
        self.btn_llama_open.setMinimumHeight(32)
        self.btn_llama_open.setStyleSheet(f"""
            QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']};border-radius:6px;font-size:11px;}}
            QPushButton:hover{{background:{COLORS['bg_hover']};}}
        """)
        self.btn_llama_open.clicked.connect(self.sig_llama_open.emit)
        row3.addWidget(self.btn_llama_open, 1)

        grid_layout.addLayout(row3)

        # 第四行：ComfyUI 按钮
        row4 = QHBoxLayout()
        row4.setSpacing(6)

        self.btn_comfy_launch = QPushButton("🚀 ComfyUI")
        self.btn_comfy_launch.setMinimumHeight(32)
        self.btn_comfy_launch.setStyleSheet(f"""
            QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']};border-radius:6px;font-size:11px;}}
            QPushButton:hover{{background:{COLORS['bg_hover']};}}
        """)
        self.btn_comfy_launch.clicked.connect(self.sig_comfy_launch.emit)
        row4.addWidget(self.btn_comfy_launch, 1)

        self.btn_comfy_stop = QPushButton("⏹ 停止 ComfyUI")
        self.btn_comfy_stop.setMinimumHeight(32)
        self.btn_comfy_stop.setStyleSheet(f"""
            QPushButton{{background:#3d1f1f;color:{COLORS['red']};
                border:1px solid #5a2020;border-radius:6px;font-size:11px;}}
            QPushButton:hover{{background:#5a2020;}}
        """)
        self.btn_comfy_stop.clicked.connect(self.sig_comfy_stop.emit)
        row4.addWidget(self.btn_comfy_stop, 1)

        self.btn_comfy_open = QPushButton("🌐 ComfyUI 页面")
        self.btn_comfy_open.setMinimumHeight(32)
        self.btn_comfy_open.setStyleSheet(f"""
            QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']};border-radius:6px;font-size:11px;}}
            QPushButton:hover{{background:{COLORS['bg_hover']};}}
        """)
        self.btn_comfy_open.clicked.connect(self.sig_comfy_open.emit)
        row4.addWidget(self.btn_comfy_open, 1)

        grid_layout.addLayout(row4)

        return frame

