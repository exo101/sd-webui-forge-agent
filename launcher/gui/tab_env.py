"""环境检测 Tab"""
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGridLayout, QProgressBar,
    QScrollArea, QSizePolicy, QTextEdit, QLineEdit,
    QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from .theme import COLORS
from core.env_checker import (
    check_python, check_git, check_cuda, check_vram,
    check_webui_installed, get_webui_version,
    check_all_system_deps, check_all_gpus,
)


class CheckWorker(QThread):
    result = pyqtSignal(dict)

    def run(self):
        data = {
            "python":   check_python(),
            "git":      check_git(),
            "cuda":     check_cuda(),
            "vram":     check_vram(),
            "webui":    check_webui_installed(),
            "version":  get_webui_version(),
            "sys_deps": check_all_system_deps(),
        }
        self.result.emit(data)


class EnvTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._checked = False
        self._worker: CheckWorker | None = None
        self._build_ui()
    
    def showEvent(self, event):
        """页面显示时触发"""
        super().showEvent(event)
        # 如果还未检测过，延迟后自动检测
        if not self._checked:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(300, self._auto_check)
    
    def _auto_check(self):
        """自动检测（仅在未检测过时执行）"""
        if not self._checked and (not self._worker or not self._worker.isRunning()):
            self._force_check()

    def _build_ui(self):
        from PyQt6.QtWidgets import QTabWidget

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        # ── 顶部标题行 ──
        title_row = QHBoxLayout()
        lbl = QLabel("🔍  环境检测")
        lbl.setStyleSheet(f"color:{COLORS['text_primary']};font-size:18px;font-weight:bold;")
        title_row.addWidget(lbl)
        title_row.addStretch()
        self.btn_refresh = QPushButton("🔄  重新检测")
        self.btn_refresh.clicked.connect(self._force_check)
        title_row.addWidget(self.btn_refresh)
        root.addLayout(title_row)

        # ── 子 Tab：系统环境 / 命令行工具 ──
        self._sub_tabs = QTabWidget()
        self._sub_tabs.setStyleSheet(
            "QTabWidget::pane {"
            "border: 1px solid " + COLORS['border'] + ";"
            "background: " + COLORS['bg_dark'] + ";"
            "border-radius: 0 8px 8px 8px;"
            "}"
            "QTabBar::tab {"
            "background: " + COLORS['bg_dark'] + ";"
            "color: " + COLORS['text_secondary'] + ";"
            "padding: 7px 18px;"
            "border: none;"
            "border-bottom: 2px solid transparent;"
            "font-size: 12px;"
            "}"
            "QTabBar::tab:selected {"
            "color: " + COLORS['text_primary'] + ";"
            "border-bottom: 2px solid " + COLORS['accent'] + ";"
            "font-weight: bold;"
            "}"
            "QTabBar::tab:hover:!selected {"
            "background: " + COLORS['bg_hover'] + ";"
            "color: " + COLORS['text_primary'] + ";"
            "}"
        )
        root.addWidget(self._sub_tabs, 1)

        # ── 子 Tab 1：系统环境 ──
        env_tab = QWidget()
        env_tab.setStyleSheet("background:{bg_dark};".format(bg_dark=COLORS['bg_dark']))
        env_layout = QVBoxLayout(env_tab)
        env_layout.setContentsMargins(12, 12, 12, 12)
        env_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")

        self.cards_widget = QWidget()
        self.cards_widget.setStyleSheet("background:transparent;")
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self._build_placeholder()
        self.cards_layout.addStretch()

        scroll.setWidget(self.cards_widget)
        env_layout.addWidget(scroll, 1)
        self._sub_tabs.addTab(env_tab, "  🖥  系统环境  ")

        # ── 子 Tab 2：命令行工具 ──
        cmd_tab = QWidget()
        cmd_tab.setStyleSheet(f"background:{COLORS['bg_dark']};")
        cmd_layout = QVBoxLayout(cmd_tab)
        cmd_layout.setContentsMargins(12, 12, 12, 12)
        cmd_layout.setSpacing(8)

        # 目录选择
        dir_row = QHBoxLayout()
        dir_lbl = QLabel("📁  切换目录：")
        dir_lbl.setStyleSheet(f"color:{COLORS['text_primary']};font-size:12px;")
        dir_row.addWidget(dir_lbl)
        self.dir_combo = QComboBox()
        self.dir_combo.addItems([
            "Python 目录"
        ])
        self.dir_combo.setStyleSheet(f"color:{COLORS['text_primary']};background:{COLORS['bg_input']};border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-size:12px;")
        dir_row.addWidget(self.dir_combo)
        self.btn_cd = QPushButton("📂  进入目录")
        self.btn_cd.clicked.connect(self._cd_to_directory)
        self.btn_cd.setStyleSheet(f"color:{COLORS['text_primary']};background:{COLORS['bg_hover']};border:1px solid {COLORS['border']};border-radius:4px;padding:4px 12px;font-size:12px;")
        dir_row.addWidget(self.btn_cd)
        dir_row.addStretch()
        cmd_layout.addLayout(dir_row)

        # 常用命令
        cmd_row = QHBoxLayout()
        cmd_lbl = QLabel("⚙️  常用命令：")
        cmd_lbl.setStyleSheet(f"color:{COLORS['text_primary']};font-size:12px;")
        cmd_row.addWidget(cmd_lbl)
        self.cmd_combo = QComboBox()
        self.cmd_combo.addItems([
            "pip 安装包",
            "pip 卸载包",
            "pip 列出依赖",
            "清理缓存垃圾"
        ])
        self.cmd_combo.setStyleSheet(f"color:{COLORS['text_primary']};background:{COLORS['bg_input']};border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-size:12px;")
        cmd_row.addWidget(self.cmd_combo)
        
        # 包名输入框
        pkg_lbl = QLabel("包名：")
        pkg_lbl.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;")
        cmd_row.addWidget(pkg_lbl)
        self.pkg_input = QLineEdit()
        self.pkg_input.setPlaceholderText("输入包名，多个包用空格分隔")
        self.pkg_input.setStyleSheet(f"color:{COLORS['text_primary']};background:{COLORS['bg_input']};border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-size:12px;")
        cmd_row.addWidget(self.pkg_input, 1)
        
        self.btn_run = QPushButton("▶  执行")
        self.btn_run.clicked.connect(self._run_command)
        self.btn_run.setStyleSheet(f"color:{COLORS['text_primary']};background:{COLORS['bg_hover']};border:1px solid {COLORS['border']};border-radius:4px;padding:4px 12px;font-size:12px;")
        cmd_row.addWidget(self.btn_run)
        cmd_row.addStretch()
        cmd_layout.addLayout(cmd_row)

        # 输出区域
        self.cmd_output = QTextEdit()
        self.cmd_output.setReadOnly(True)
        self.cmd_output.setStyleSheet(f"""
            background:{COLORS['bg_card']};
            color:{COLORS['text_primary']};
            border:1px solid {COLORS['border']};
            border-radius:6px;
            padding:8px;
            font-family:'Consolas','Courier New',monospace;
            font-size:12px;
        """)
        cmd_layout.addWidget(self.cmd_output, 1)

        self._sub_tabs.addTab(cmd_tab, "  💻  命令行工具  ")

    def _build_placeholder(self):
        """构建占位符"""
        placeholder = QLabel("点击「重新检测」开始环境检测...")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color:{COLORS['text_dim']};font-size:13px;padding:40px;")
        placeholder.setObjectName("env_placeholder")
        self.cards_layout.addWidget(placeholder)

    def _clear_cards(self):
        """清空卡片"""
        # 先移除所有项（包括stretch）
        while self.cards_layout.count() > 0:
            item = self.cards_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                # 如果是layout项，也需要清理
                sub_layout = item.layout()
                if sub_layout is not None:
                    self._clear_layout(sub_layout)
    
    def _clear_layout(self, layout):
        """递归清空布局"""
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                sub_layout = item.layout()
                if sub_layout is not None:
                    self._clear_layout(sub_layout)

    def _force_check(self):
        """强制重新检测"""
        if self._worker and self._worker.isRunning():
            return
        
        self._clear_cards()
        loading = QLabel("⏳  正在检测环境...")
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:13px;padding:40px;")
        self.cards_layout.insertWidget(0, loading)
        
        self._worker = CheckWorker()
        self._worker.result.connect(self._on_result)
        self._worker.start()

    def _on_result(self, data: dict):
        """检测结果回调"""
        self._clear_cards()
        
        # Python
        python_info = data.get("python", {})
        if isinstance(python_info, bool):
            python_info = {"ok": python_info, "msg": "已安装" if python_info else "未找到"}
        self._add_card("Python", python_info)
        
        # Git
        git_info = data.get("git", {})
        if isinstance(git_info, bool):
            git_info = {"ok": git_info, "msg": "已安装" if git_info else "未找到"}
        self._add_card("Git", git_info)
        
        # CUDA
        cuda_info = data.get("cuda", {})
        if isinstance(cuda_info, bool):
            cuda_info = {"ok": cuda_info, "msg": "已检测到" if cuda_info else "未检测到"}
        elif isinstance(cuda_info, dict):
            # 将CUDA检测结果转换为msg格式
            torch_ver = cuda_info.get("torch", "未知")
            gpu_name = cuda_info.get("gpu", "N/A")
            cuda_available = cuda_info.get("cuda", False)
            if cuda_info.get("ok"):
                cuda_info["msg"] = f"PyTorch {torch_ver} | GPU: {gpu_name}"
            else:
                cuda_info["msg"] = f"CUDA不可用 | PyTorch: {torch_ver}"
        self._add_card("CUDA", cuda_info)
        
        # VRAM
        vram_info = data.get("vram", {})
        if isinstance(vram_info, bool):
            vram_info = {"ok": vram_info, "msg": "正常" if vram_info else "异常"}
        elif isinstance(vram_info, dict):
            # 将VRAM检测结果转换为msg格式
            total_mb = vram_info.get("total_mb", 0)
            gpu_name = vram_info.get("name", "N/A")
            if vram_info.get("ok"):
                vram_info["msg"] = f"{total_mb} MB | {gpu_name}"
            else:
                vram_info["msg"] = "无法检测显存"
        self._add_card("显存", vram_info)
        
        # GPU设备选择（如果有多个GPU）
        all_gpus = check_all_gpus()
        if len(all_gpus) > 1:
            self._add_gpu_selector_card(all_gpus)
        elif len(all_gpus) == 1:
            # 单GPU也显示，但禁用选择
            self._add_gpu_selector_card(all_gpus, single_gpu=True)
        
        # WebUI - 可能是布尔值，需要转换
        webui_info = data.get("webui", False)
        if isinstance(webui_info, bool):
            webui_info = {"ok": webui_info, "msg": "已安装" if webui_info else "未安装"}
        self._add_card("WebUI", webui_info)
        
        # Version
        version = data.get("version")
        if version:
            self._add_card("版本", {"ok": True, "msg": str(version)})
        
        # System Dependencies - 可能是列表或字典
        sys_deps = data.get("sys_deps")
        if sys_deps:
            if isinstance(sys_deps, list):
                # 如果是列表，遍历每个依赖项
                for dep_info in sys_deps:
                    if isinstance(dep_info, dict):
                        dep_name = dep_info.get("name", "未知依赖")
                        self._add_card(dep_name, dep_info)
            elif isinstance(sys_deps, dict):
                # 如果是字典，按原逻辑处理
                for dep_name, dep_info in sys_deps.items():
                    self._add_card(dep_name, dep_info)
        
        self.cards_layout.addStretch()
        self._checked = True
        
        # 强制刷新UI和布局
        self.cards_widget.updateGeometry()
        self.cards_widget.update()
        self.cards_widget.repaint()

    def _add_card(self, title: str, info):
        """添加检测卡片"""
        # 处理不同类型的info参数
        if isinstance(info, bool):
            info = {"ok": info, "msg": "正常" if info else "异常"}
        elif isinstance(info, str):
            info = {"ok": True, "msg": info}
        elif not isinstance(info, dict):
            info = {"ok": False, "msg": str(info)}
        
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background:{COLORS['bg_card']};
                border-radius:8px;
                border:1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # 图标
        icon = QLabel("✅" if info.get("ok") else "❌")
        icon.setStyleSheet("font-size:18px;")
        layout.addWidget(icon)

        # 内容
        content = QVBoxLayout()
        content.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color:{COLORS['text_primary']};font-weight:bold;font-size:13px;")
        content.addWidget(title_lbl)

        # 尝试多种可能的消息字段名
        msg = info.get("msg") or info.get("detail") or info.get("message") or info.get("version") or ""
        if msg:
            msg_lbl = QLabel(str(msg))
            msg_lbl.setWordWrap(True)
            msg_lbl.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:11px;")
            content.addWidget(msg_lbl)

        content.addStretch()
        layout.addLayout(content, 1)

        self.cards_layout.addWidget(card)

    def _add_gpu_selector_card(self, gpus: list[dict], single_gpu: bool = False):
        """添加GPU选择卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background:{COLORS['bg_card']};
                border-radius:8px;
                border:1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # 图标
        icon = QLabel("🎮")
        icon.setStyleSheet("font-size:18px;")
        layout.addWidget(icon)

        # 内容
        content = QVBoxLayout()
        content.setSpacing(4)

        title_lbl = QLabel("GPU 设备选择")
        title_lbl.setStyleSheet(f"color:{COLORS['text_primary']};font-weight:bold;font-size:13px;")
        content.addWidget(title_lbl)

        if single_gpu:
            msg_lbl = QLabel(f"检测到 1 个 GPU: {gpus[0]['name']} ({gpus[0]['vram_gb']} GB)")
            msg_lbl.setWordWrap(True)
            msg_lbl.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:11px;")
            content.addWidget(msg_lbl)
        else:
            # GPU选择下拉框
            selector_layout = QHBoxLayout()
            selector_layout.setSpacing(8)
            
            label = QLabel("选择使用的GPU:")
            label.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:11px;")
            selector_layout.addWidget(label)
            
            self.gpu_combo = QComboBox()
            self.gpu_combo.setStyleSheet(f"""
                QComboBox {{
                    color:{COLORS['text_primary']};
                    background:{COLORS['bg_input']};
                    border:1px solid {COLORS['border']};
                    border-radius:4px;
                    padding:4px 8px;
                    font-size:11px;
                    min-width: 200px;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {COLORS['bg_input']};
                    color: {COLORS['text_primary']};
                    selection-background-color: {COLORS['accent']};
                }}
            """)
            
            # 添加选项：全部GPU + 每个单独的GPU
            self.gpu_combo.addItem("🌐 使用所有 GPU", "")
            for gpu in gpus:
                display_text = f"🔹 GPU {gpu['index']}: {gpu['name']} ({gpu['vram_gb']} GB, CC {gpu['compute_capability']})"
                self.gpu_combo.addItem(display_text, str(gpu['index']))
            
            # 设置当前选择
            current_gpu = self.config.get("gpu_device", "")
            for i in range(self.gpu_combo.count()):
                if self.gpu_combo.itemData(i) == current_gpu:
                    self.gpu_combo.setCurrentIndex(i)
                    break
            
            self.gpu_combo.currentIndexChanged.connect(self._on_gpu_selection_changed)
            selector_layout.addWidget(self.gpu_combo)
            selector_layout.addStretch()
            
            content.addLayout(selector_layout)
            
            # 提示信息
            hint_lbl = QLabel("💡 提示: 切换GPU后需要重启WebUI才能生效")
            hint_lbl.setStyleSheet(f"color:{COLORS['yellow']};font-size:10px;font-style:italic;")
            content.addWidget(hint_lbl)

        content.addStretch()
        layout.addLayout(content, 1)

        self.cards_layout.addWidget(card)

    def _on_gpu_selection_changed(self, index: int):
        """GPU选择改变时的回调"""
        gpu_id = self.gpu_combo.itemData(index)
        self.config["gpu_device"] = gpu_id
        
        # 保存配置
        from core.config import save_config
        save_config(self.config)
        
        # 显示提示
        if gpu_id:
            print(f"✅ GPU选择已更新: 使用 GPU {gpu_id}")
        else:
            print(f"✅ GPU选择已更新: 使用所有可用GPU")

    def _cd_to_directory(self):
        """切换到指定目录"""
        from core.paths import BASE_DIR
        import subprocess
        
        index = self.dir_combo.currentIndex()
        if index == 0:  # Python 目录
            python_dir = os.path.join(BASE_DIR, "system", "python")
            if os.path.exists(python_dir):
                subprocess.Popen(f'explorer "{python_dir}"')

    def _run_command(self):
        """执行常用命令"""
        import subprocess
        from core.paths import BASE_DIR
        
        cmd_index = self.cmd_combo.currentIndex()
        python_exe = os.path.join(BASE_DIR, "system", "python", "python.exe")
        
        if cmd_index == 0:
            # pip 安装包
            packages = self.pkg_input.text().strip()
            if not packages:
                self.cmd_output.append("❌ 请输入要安装的包名")
                return
            cmd = [python_exe, "-m", "pip", "install"] + packages.split()
            
            self.cmd_output.append(f"▶ 执行命令：{' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.stdout:
                    self.cmd_output.append(result.stdout)
                if result.stderr:
                    self.cmd_output.append(f"⚠️  {result.stderr}")
                
                if result.returncode == 0:
                    self.cmd_output.append("✅ 命令执行成功")
                else:
                    self.cmd_output.append(f"❌ 命令执行失败，退出码：{result.returncode}")
            except Exception as e:
                self.cmd_output.append(f"❌ 执行命令失败：{str(e)}")
                
        elif cmd_index == 1:
            # pip 卸载包
            packages = self.pkg_input.text().strip()
            if not packages:
                self.cmd_output.append("❌ 请输入要卸载的包名")
                return
            cmd = [python_exe, "-m", "pip", "uninstall", "-y"] + packages.split()
            
            self.cmd_output.append(f"▶ 执行命令：{' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.stdout:
                    self.cmd_output.append(result.stdout)
                if result.stderr:
                    self.cmd_output.append(f"⚠️  {result.stderr}")
                
                if result.returncode == 0:
                    self.cmd_output.append("✅ 命令执行成功")
                else:
                    self.cmd_output.append(f"❌ 命令执行失败，退出码：{result.returncode}")
            except Exception as e:
                self.cmd_output.append(f"❌ 执行命令失败：{str(e)}")
                
        elif cmd_index == 2:
            # pip 列出依赖
            cmd = [python_exe, "-m", "pip", "list"]
            
            self.cmd_output.append(f"▶ 执行命令：{' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.stdout:
                    self.cmd_output.append(result.stdout)
                if result.stderr:
                    self.cmd_output.append(f"⚠️  {result.stderr}")
                
                if result.returncode == 0:
                    self.cmd_output.append("✅ 命令执行成功")
                else:
                    self.cmd_output.append(f"❌ 命令执行失败，退出码：{result.returncode}")
            except Exception as e:
                self.cmd_output.append(f"❌ 执行命令失败：{str(e)}")
                
        else:
            # 清理缓存垃圾
            self.cmd_output.append("▶ 开始清理缓存垃圾...")
            
            # 1. 清理 Python 缓存文件夹
            python_dir = os.path.join(BASE_DIR, "system", "python")
            ps1_command = f"Get-ChildItem -Path '{python_dir}' -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force"
            self.cmd_output.append(f"▶ 执行命令：PowerShell 清理 __pycache__")
            
            try:
                result = subprocess.run(
                    ["powershell.exe", "-Command", ps1_command],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.stdout:
                    self.cmd_output.append(result.stdout)
                if result.stderr:
                    self.cmd_output.append(f"⚠️  {result.stderr}")
                
                if result.returncode == 0:
                    self.cmd_output.append("✅ Python 缓存文件夹清理完成")
                else:
                    self.cmd_output.append(f"❌ Python 缓存文件夹清理失败，退出码：{result.returncode}")
            except Exception as e:
                self.cmd_output.append(f"❌ 执行清理 Python 缓存命令失败：{str(e)}")
            
            # 2. 清理 pip 缓存
            pip_cmd = [python_exe, "-m", "pip", "cache", "purge"]
            self.cmd_output.append(f"▶ 执行命令：{' '.join(pip_cmd)}")
            
            try:
                result = subprocess.run(
                    pip_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.stdout:
                    self.cmd_output.append(result.stdout)
                if result.stderr:
                    self.cmd_output.append(f"⚠️  {result.stderr}")
                
                if result.returncode == 0:
                    self.cmd_output.append("✅ pip 缓存清理完成")
                else:
                    self.cmd_output.append(f"❌ pip 缓存清理失败，退出码：{result.returncode}")
            except Exception as e:
                self.cmd_output.append(f"❌ 执行清理 pip 缓存命令失败：{str(e)}")
            
            self.cmd_output.append("✅ 缓存垃圾清理完成")
        
        self.cmd_output.append("\n" + "-" * 60 + "\n")
