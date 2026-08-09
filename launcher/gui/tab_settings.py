"""参数设置 Tab"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QSpinBox, QComboBox, QLineEdit,
    QGroupBox, QScrollArea, QFrame, QGridLayout,
    QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from .theme import COLORS
from core.paths import BASE_DIR


class SettingsTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._widgets = {}
        self._build_ui()
        self._load_config()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content.setStyleSheet(f"background-color: {COLORS['bg_card']};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # 服务设置
        layout.addWidget(self._group_service())
        # CUDA 优化
        layout.addWidget(self._group_cuda())
        # 注意力机制
        layout.addWidget(self._group_attention())
        # 显存优化
        layout.addWidget(self._group_vram())
        # 精度设置
        layout.addWidget(self._group_precision())
        # 其他
        layout.addWidget(self._group_misc())
        # 额外参数
        layout.addWidget(self._group_extra())
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {COLORS['border']};")
        separator.setFixedHeight(2)
        layout.addWidget(separator)
        
        # 壁纸设置
        layout.addWidget(self._group_wallpaper())
        # llama.cpp 设置
        layout.addWidget(self._group_llama())
        
        layout.addStretch()

        scroll.setWidget(content)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def _group_service(self) -> QGroupBox:
        g = QGroupBox("服务设置")
        layout = QGridLayout(g)
        layout.setSpacing(12)

        layout.addWidget(QLabel("端口:"), 0, 0)
        spin = QSpinBox()
        spin.setRange(1024, 65535)
        spin.setFixedWidth(100)
        layout.addWidget(spin, 0, 1)
        self._widgets["port"] = spin

        layout.addWidget(QLabel("主题:"), 0, 2)
        combo = QComboBox()
        combo.addItems(["dark", "light"])
        combo.setFixedWidth(100)
        layout.addWidget(combo, 0, 3)
        self._widgets["theme"] = combo

        chk_listen = QCheckBox("监听所有IP (--listen)")
        chk_autolaunch = QCheckBox("自动打开浏览器 (--autolaunch)")
        chk_api = QCheckBox("启用 API (--api)")
        chk_share = QCheckBox("Gradio 分享 (--share)")
        chk_show_console = QCheckBox("显示命令行窗口")
        layout.addWidget(chk_listen, 1, 0, 1, 2)
        layout.addWidget(chk_autolaunch, 1, 2, 1, 2)
        layout.addWidget(chk_api, 2, 0, 1, 2)
        layout.addWidget(chk_share, 2, 2, 1, 2)
        layout.addWidget(chk_show_console, 3, 0, 1, 4)
        self._widgets["listen"] = chk_listen
        self._widgets["autolaunch"] = chk_autolaunch
        self._widgets["api"] = chk_api
        self._widgets["share"] = chk_share
        self._widgets["show_console"] = chk_show_console

        return g

    def _group_cuda(self) -> QGroupBox:
        g = QGroupBox("CUDA 优化")
        layout = QHBoxLayout(g)
        layout.setSpacing(20)

        for key, text in [
            ("cuda_malloc",       "--cuda-malloc"),
            ("cuda_stream",       "--cuda-stream"),
            ("pin_shared_memory", "--pin-shared-memory"),
        ]:
            chk = QCheckBox(text)
            layout.addWidget(chk)
            self._widgets[key] = chk
        layout.addStretch()
        return g

    def _group_attention(self) -> QGroupBox:
        g = QGroupBox("注意力机制")
        layout = QVBoxLayout(g)
        layout.setSpacing(8)
        
        # 说明文字
        hint = QLabel("💡 优先级：FlashAttention > xFormers > PyTorch原生")
        hint.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        layout.addWidget(hint)
        
        # 复选框行
        chk_layout = QHBoxLayout()
        chk_layout.setSpacing(20)
        
        for key, text in [
            ("enable_flash",    "启用 FlashAttention"),
            ("enable_xformers", "启用 xFormers"),
        ]:
            chk = QCheckBox(text)
            chk_layout.addWidget(chk)
            self._widgets[key] = chk
        
        # 禁用 SageAttention
        chk_disable_sage = QCheckBox("禁用 SageAttention (--disable-sage)")
        chk_layout.addWidget(chk_disable_sage)
        self._widgets["disable_sage"] = chk_disable_sage
        
        chk_layout.addStretch()
        layout.addLayout(chk_layout)
        
        return g

    def _group_vram(self) -> QGroupBox:
        g = QGroupBox("显存优化")
        layout = QHBoxLayout(g)
        layout.setSpacing(20)

        lbl = QLabel("显存模式:")
        lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(lbl)

        for key, text in [
            ("lowvram",  "--lowvram (低显存)"),
            ("neveroom", "--neveroom (显存防溢出保护)"),
        ]:
            chk = QCheckBox(text)
            layout.addWidget(chk)
            self._widgets[key] = chk

        # 预留显存
        layout.addWidget(QLabel("预留显存:"))
        spin_reserve = QSpinBox()
        spin_reserve.setRange(0, 64)
        spin_reserve.setSuffix(" GB")
        spin_reserve.setFixedWidth(90)
        spin_reserve.setToolTip("--reserve-vram 参数，预留指定大小的显存")
        layout.addWidget(spin_reserve)
        self._widgets["reserve_vram"] = spin_reserve

        layout.addStretch()
        return g

    def _group_precision(self) -> QGroupBox:
        g = QGroupBox("精度设置")
        layout = QHBoxLayout(g)
        layout.setSpacing(20)

        for key, text in [
            ("no_half",       "--no-half"),
            ("no_half_vae",   "--no-half-vae"),
            ("precision_full","--precision full"),
        ]:
            chk = QCheckBox(text)
            layout.addWidget(chk)
            self._widgets[key] = chk
        layout.addStretch()
        return g

    def _group_misc(self) -> QGroupBox:
        g = QGroupBox("其他选项")
        layout = QHBoxLayout(g)
        layout.setSpacing(20)

        for key, text in [
            ("skip_install",   "--skip-install"),
            ("skip_version",   "--skip-version-check"),
            ("skip_torch",     "--skip-torch-cuda-test"),
        ]:
            chk = QCheckBox(text)
            layout.addWidget(chk)
            self._widgets[key] = chk
        layout.addStretch()
        return g

    def _group_extra(self) -> QGroupBox:
        g = QGroupBox("额外命令行参数")
        layout = QVBoxLayout(g)

        lbl = QLabel("在此输入额外的启动参数，多个参数用空格分隔：")
        lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        layout.addWidget(lbl)

        edit = QLineEdit()
        edit.setPlaceholderText("例如: --no-gradio-queue --gradio-auth user:pass")
        layout.addWidget(edit)
        self._widgets["extra_args"] = edit
        return g

    def _group_wallpaper(self) -> QGroupBox:
        g = QGroupBox("🎨 主控台壁纸")
        layout = QHBoxLayout(g)
        layout.setSpacing(12)

        self.lbl_wallpaper_preview = QLabel()
        self.lbl_wallpaper_preview.setFixedSize(80, 50)
        self.lbl_wallpaper_preview.setStyleSheet(f"""
            background: {COLORS['bg_dark']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        """)
        self.lbl_wallpaper_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_wallpaper_preview)

        info_col = QVBoxLayout()
        info_col.setSpacing(6)

        self.lbl_wallpaper_name = QLabel("默认壁纸")
        self.lbl_wallpaper_name.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        info_col.addWidget(self.lbl_wallpaper_name)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_upload = QPushButton("上传壁纸")
        btn_upload.setFixedHeight(30)
        btn_upload.setStyleSheet(f"""
            QPushButton{{background:{COLORS['accent']};color:white;border:none;
                border-radius:4px;font-size:11px;padding:4px 12px;}}
            QPushButton:hover{{background:{COLORS['accent_light']};}}
        """)
        btn_upload.clicked.connect(self._on_upload_wallpaper)
        btn_row.addWidget(btn_upload)

        btn_reset = QPushButton("恢复默认")
        btn_reset.setFixedHeight(30)
        btn_reset.setStyleSheet(f"""
            QPushButton{{background:transparent;color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']};border-radius:4px;font-size:11px;padding:4px 12px;}}
            QPushButton:hover{{background:{COLORS['bg_hover']};color:{COLORS['text_primary']};}}
        """)
        btn_reset.clicked.connect(self._on_reset_wallpaper)
        btn_row.addWidget(btn_reset)

        info_col.addLayout(btn_row)
        layout.addLayout(info_col, 1)

        self._widgets["wallpaper"] = None  # 不直接存储widget，存路径
        return g

    def _group_llama(self) -> QGroupBox:
        g = QGroupBox("🦙 llama.cpp 服务")
        layout = QGridLayout(g)
        layout.setSpacing(10)

        # 启用开关
        chk_enable = QCheckBox("启用 llama.cpp 服务（与 WebUI 同时启动）")
        layout.addWidget(chk_enable, 0, 0, 1, 4)
        self._widgets["llama_enabled"] = chk_enable

        layout.addWidget(QLabel("端口:"), 1, 0)
        spin_port = QSpinBox()
        spin_port.setRange(1024, 65535)
        spin_port.setFixedWidth(100)
        layout.addWidget(spin_port, 1, 1)
        self._widgets["llama_port"] = spin_port

        layout.addWidget(QLabel("GPU层数 (-ngl):"), 1, 2)
        spin_ngl = QSpinBox()
        spin_ngl.setRange(0, 200)
        spin_ngl.setFixedWidth(80)
        spin_ngl.setToolTip("指定加载到GPU的层数，-1为全部")
        layout.addWidget(spin_ngl, 1, 3)
        self._widgets["llama_ngl"] = spin_ngl

        # 提示信息
        lbl_hint = QLabel("💡 根据端口自动匹配模型：8080→4B, 8079→2B，其余端口自动选择第一个模型")
        lbl_hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        layout.addWidget(lbl_hint, 2, 0, 1, 5)
        return g

    def _on_upload_wallpaper(self):
        """上传壁纸"""
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择壁纸图片", "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp)"
            )
            if path:
                self._set_wallpaper(path)
        except Exception as e:
            print(f"上传壁纸失败: {e}")

    def _on_reset_wallpaper(self):
        """恢复默认壁纸"""
        try:
            self._set_wallpaper("")
        except Exception as e:
            print(f"重置壁纸失败: {e}")

    def _set_wallpaper(self, path: str):
        """设置壁纸并更新预览"""
        from PyQt6.QtGui import QPixmap
        try:
            self.config["wallpaper"] = path
            if path and os.path.exists(path):
                self.lbl_wallpaper_name.setText(os.path.basename(path))
                pix = QPixmap(path)
                if not pix.isNull():
                    scaled = pix.scaled(80, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.lbl_wallpaper_preview.setPixmap(scaled)
                    self.lbl_wallpaper_preview.setStyleSheet(f"""
                        background: {COLORS['bg_dark']};
                        border: 1px solid {COLORS['border']};
                        border-radius: 4px;
                    """)
                else:
                    raise ValueError("图片格式不支持或文件损坏")
            else:
                # 路径无效或文件不存在，清空配置
                if path and not os.path.exists(path):
                    self.config["wallpaper"] = ""
                self.lbl_wallpaper_name.setText("默认壁纸")
                self.lbl_wallpaper_preview.setText("默认")
                self.lbl_wallpaper_preview.setStyleSheet(f"""
                    color: {COLORS['text_dim']}; background: {COLORS['bg_dark']};
                    border: 1px solid {COLORS['border']}; border-radius: 4px; font-size: 10px;
                """)
                self.lbl_wallpaper_preview.setPixmap(QPixmap())
        except Exception as e:
            print(f"设置壁纸失败: {e}")
            # 重置为默认状态
            self.config["wallpaper"] = ""
            self.lbl_wallpaper_name.setText("默认壁纸")
            self.lbl_wallpaper_preview.setText("默认")
            self.lbl_wallpaper_preview.setStyleSheet(f"""
                color: {COLORS['text_dim']}; background: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border']}; border-radius: 4px; font-size: 10px;
            """)
            self.lbl_wallpaper_preview.setPixmap(QPixmap())

    def _load_config(self):
        c = self.config
        self._widgets["port"].setValue(c.get("port", 7869))
        idx = self._widgets["theme"].findText(c.get("theme", "dark"))
        self._widgets["theme"].setCurrentIndex(max(0, idx))

        for key in ["listen", "autolaunch", "api", "share", "show_console",
                    "cuda_malloc", "cuda_stream", "pin_shared_memory",
                    "lowvram", "neveroom", "no_half", "no_half_vae", "precision_full",
                    "skip_install", "skip_version", "skip_torch", "disable_sage"]:
            if key in self._widgets:
                self._widgets[key].setChecked(bool(c.get(key, False)))

        # 预留显存
        if "reserve_vram" in self._widgets:
            self._widgets["reserve_vram"].setValue(int(c.get("reserve_vram", 0)))

        # 注意力机制：默认全部启用，除非明确禁用
        # 从旧的 disable_* 配置迁移，或使用新的 enable_* 配置
        for key, default in [
            ("enable_flash", True),    # 默认启用 FlashAttention
            ("enable_xformers", True), # 默认启用 xFormers
        ]:
            # 优先使用新的 enable_* 配置
            if key in c:
                self._widgets[key].setChecked(bool(c.get(key, default)))
            else:
                # 从旧的 disable_* 配置迁移
                old_key = key.replace("enable_", "disable_")
                if old_key in c:
                    # 旧配置：disable_* 表示禁用，所以 enable_* = False
                    self._widgets[key].setChecked(not bool(c.get(old_key, False)))
                else:
                    # 默认值
                    self._widgets[key].setChecked(default)

        self._widgets["extra_args"].setText(c.get("extra_args", ""))

        # 壁纸加载
        wallpaper_path = c.get("wallpaper", "")
        if wallpaper_path:
            self._set_wallpaper(wallpaper_path)
        else:
            self._set_wallpaper("")

        # llama 配置加载
        llama_cfg = c.get("llama", {})
        self._widgets["llama_enabled"].setChecked(llama_cfg.get("enabled", True))
        self._widgets["llama_port"].setValue(llama_cfg.get("port", 8080))
        self._widgets["llama_ngl"].setValue(llama_cfg.get("ngl", 100))

    def apply_to_config(self, config: dict):
        config["port"]  = self._widgets["port"].value()
        config["theme"] = self._widgets["theme"].currentText()

        for key in ["listen", "autolaunch", "api", "share", "show_console",
                    "cuda_malloc", "cuda_stream", "pin_shared_memory",
                    "lowvram", "neveroom", "no_half", "no_half_vae", "precision_full",
                    "skip_install", "skip_version", "skip_torch", "disable_sage"]:
            if key in self._widgets:
                config[key] = self._widgets[key].isChecked()

        # 预留显存
        if "reserve_vram" in self._widgets:
            config["reserve_vram"] = self._widgets["reserve_vram"].value()

        # 注意力机制：保存启用状态
        for key in ["enable_flash", "enable_xformers"]:
            if key in self._widgets:
                config[key] = self._widgets[key].isChecked()

        config["extra_args"] = self._widgets["extra_args"].text().strip()

        # 壁纸
        wallpaper = self.config.get("wallpaper", "")
        config["wallpaper"] = wallpaper

        # llama 配置
        llama_cfg = config.setdefault("llama", {})
        llama_cfg["enabled"] = self._widgets["llama_enabled"].isChecked()
        llama_cfg["port"] = self._widgets["llama_port"].value()
        llama_cfg["ngl"] = self._widgets["llama_ngl"].value()

    
