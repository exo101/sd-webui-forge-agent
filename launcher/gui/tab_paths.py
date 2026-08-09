"""路径配置 Tab - 模型目录详情展示"""
import os
import yaml
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFrame, QFileDialog,
    QComboBox, QMessageBox, QLineEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from .theme import COLORS
from core.paths import BASE_DIR


# 模型目录详细说明
MODEL_DIR_INFO = {
    "stable_diffusion_dir": {
        "name": "Stable-diffusion (主模型)",
        "description": "Stable Diffusion 主模型文件 (.safetensors / .ckpt)，WebUI 默认的主模型目录",
        "subdir_name": "Stable-diffusion",
    },
    "lora_dir": {
        "name": "Loras (微调模型)",
        "description": "轻量级微调模型，用于调整画风、角色或概念，文件小且灵活",
        "subdir_name": "loras",
    },
    "vae_dir": {
        "name": "VAE (变分自编码器)",
        "description": "负责在潜空间和像素空间之间转换，影响图像色彩和细节质量",
        "subdir_name": "VAE",
    },
    "controlnet_dir": {
        "name": "ControlNet (控制网络)",
        "description": "通过边缘、深度、姿态等条件精确控制生成图像的构图和结构",
        "subdir_name": "controlnet",
    },
    "diffusion_models_dir": {
        "name": "Diffusion Models (扩散模型)",
        "description": "独立的扩散模型文件，用于 Flux、SD3 等新架构模型",
        "subdir_name": "diffusion_models",
    },
    "text_encoder_dir": {
        "name": "Text Encoders (文本编码器)",
        "description": "文本编码模型 (CLIP、T5 等)，负责将提示词转换为向量表示",
        "subdir_name": "text_encoder",
    },
    "esrgan_dir": {
        "name": "ESRGAN (放大模型)",
        "description": "图像超分辨率放大模型 (ESRGAN、Real-ESRGAN 等)，用于提升图像分辨率",
        "subdir_name": "ESRGAN",
    },
    "adetailer_dir": {
        "name": "ADetailer (面部修复)",
        "description": "自动检测并修复生成图像中的面部缺陷，提升人脸质量",
        "subdir_name": "adetailer",
    },
    "whisper_dir": {
        "name": "Whisper (语音识别)",
        "description": "OpenAI 的语音识别模型，用于音频转文字功能 (whisper-tiny 等)",
        "subdir_name": "whisper",
    },
    "birefnet_dir": {
        "name": "BiRefNet (高精度抠图)",
        "description": "双向参考网络，用于高质量图像分割和背景移除",
        "subdir_name": "BiRefNet",
    },
    "inspyrenet_dir": {
        "name": "InSPyReNet (智能抠图)",
        "description": "基于金字塔重构的抠图网络，提供精确的图像分割",
        "subdir_name": "InSPyReNet",
    },
    "rembg_dir": {
        "name": "RemBG (背景移除)",
        "description": "通用背景移除模型，快速去除图像背景",
        "subdir_name": "rembg",
    },
    "sams_dir": {
        "name": "SAMs (Segment Anything)",
        "description": "Meta 的万物分割模型，支持任意对象的精确分割",
        "subdir_name": "sams",
    },
    "layer_model_dir": {
        "name": "Layer Models (图层模型)",
        "description": "分层处理模型，用于多层合成和编辑工作流",
        "subdir_name": "layer_model",
    },
    "diffusers_dir": {
        "name": "Diffusers (Hugging Face 格式)",
        "description": "Hugging Face Diffusers 库格式的模型，支持更多新架构",
        "subdir_name": "diffusers",
    },
    "qwen_tts_dir": {
        "name": "Qwen TTS (文本转语音)",
        "description": "通义千问文本转语音模型，用于音频生成功能",
        "subdir_name": "qwen3-tts",
    },
}


class ModelDirCard(QWidget):
    """单个模型目录卡片"""
    
    def __init__(self, config_key: str, info: dict, parent=None):
        super().__init__(parent)
        self.config_key = config_key
        self.info = info
        self.current_path = ""
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        
        # 标题行
        title_layout = QHBoxLayout()
        
        # 目录名称
        name_label = QLabel(f"📁 {self.info['name']}")
        name_label.setStyleSheet(
            f"color:{COLORS['accent_light']};font-size:14px;font-weight:bold;"
        )
        title_layout.addWidget(name_label)
        
        title_layout.addStretch()
        
        # 浏览按钮
        btn_browse = QPushButton("📂 浏览")
        btn_browse.setFixedSize(80, 30)
        btn_browse.setStyleSheet(self._btn_style(COLORS['bg_hover'], COLORS['cyan']))
        btn_browse.clicked.connect(self._on_browse)
        title_layout.addWidget(btn_browse)
        
        layout.addLayout(title_layout)
        
        # 描述
        desc_label = QLabel(self.info['description'])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        layout.addWidget(desc_label)
        
        # 路径输入框（隐藏，不显示路径信息）
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(f"models/{self.info['subdir_name']}")
        self.path_input.setReadOnly(True)
        self.path_input.textChanged.connect(self._on_path_changed)
        self.path_input.hide()  # 隐藏路径输入框
        
        # 模型文件下拉列表
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(350)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{ 
                background:{COLORS['bg_input']}; 
                color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']}; 
                border-radius:4px;
                padding:4px 8px;
                font-size:12px;
            }}
            QComboBox::drop-down {{
                border:none;
                width:20px;
            }}
            QComboBox QAbstractItemView {{
                background:{COLORS['bg_card']};
                color:{COLORS['text_primary']};
                selection-background-color:{COLORS['accent']}44;
                border:1px solid {COLORS['border']};
            }}
        """)
        model_layout.addWidget(self.model_combo, 1)
        
        # 刷新按钮
        btn_refresh = QPushButton("🔄")
        btn_refresh.setFixedSize(36, 30)
        btn_refresh.setToolTip("刷新模型列表")
        btn_refresh.setStyleSheet(self._btn_style(COLORS['bg_hover'], COLORS['text_primary']))
        btn_refresh.clicked.connect(self._refresh_models)
        model_layout.addWidget(btn_refresh)
        
        layout.addLayout(model_layout)
        
        # 设置卡片样式
        self.setStyleSheet(f"""
            QWidget {{
                background:{COLORS['bg_dark']};
                border:1px solid {COLORS['border']};
                border-radius:6px;
            }}
            QWidget:hover {{
                border-color:{COLORS['accent']}66;
            }}
        """)
    
    def _on_path_changed(self, path: str):
        """路径改变时刷新模型列表"""
        self.current_path = path
        self._refresh_models()
    
    def _refresh_models(self):
        """刷新模型文件列表"""
        self.model_combo.clear()
        
        if not self.current_path or not os.path.exists(self.current_path):
            self.model_combo.addItem("(目录不存在)")
            return
        
        # 扫描模型文件
        models = []
        try:
            for file in os.listdir(self.current_path):
                if file.endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.bin')):
                    models.append(file)
        except Exception as e:
            self.model_combo.addItem(f"(读取失败: {str(e)})")
            return
        
        if not models:
            self.model_combo.addItem("(空目录)")
        else:
            # 按名称排序
            models.sort()
            for model in models:
                # 显示文件名和大小
                file_path = os.path.join(self.current_path, model)
                try:
                    size = os.path.getsize(file_path)
                    size_str = self._format_size(size)
                    self.model_combo.addItem(f"{model} ({size_str})")
                except:
                    self.model_combo.addItem(model)
    
    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"
    
    def _on_browse(self):
        """浏览目录"""
        default_path = os.path.join(BASE_DIR, "webui", "models", self.info['subdir_name'])
        
        path = QFileDialog.getExistingDirectory(
            self,
            f"选择 {self.info['name']} 目录",
            default_path
        )
        
        if path:
            self.path_input.setText(os.path.normpath(path))
    
    def set_path(self, path: str):
        """设置路径"""
        self.current_path = path
        self.path_input.setText(os.path.normpath(path))
        self._refresh_models()
    
    def get_path(self) -> str:
        """获取路径"""
        return self.path_input.text()
    
    def _btn_style(self, bg, fg):
        return f"""
            QPushButton {{ 
                background:{bg}; 
                color:{fg}; 
                border:1px solid {fg}44;
                border-radius:4px; 
                font-size:12px; 
            }}
            QPushButton:hover {{ 
                background:{fg}33; 
            }}
        """
    
    def _input_style(self):
        return f"""
            QLineEdit {{ 
                background:{COLORS['bg_input']}; 
                color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']}; 
                border-radius:4px;
                padding:4px 8px; 
                font-size:12px; 
            }}
        """


class PathsTab(QWidget):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._cards: dict[str, ModelDirCard] = {}
        self._build_ui()
        self._load_config()
    
    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        content = QWidget()
        content.setStyleSheet(f"background-color:{COLORS['bg_card']};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # 标题
        title_label = QLabel("📦 模型目录配置")
        title_label.setStyleSheet(
            f"color:{COLORS['text_primary']};font-size:18px;font-weight:bold;"
        )
        layout.addWidget(title_label)
        
        # 说明
        info_label = QLabel(
            "以下是 WebUI 使用的各类模型目录。点击「浏览」按钮可自定义路径，"
            "下拉列表显示该目录下的模型文件及其大小。"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        layout.addWidget(info_label)
        
        # 分割线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{COLORS['border']};")
        layout.addWidget(sep)
        
        # ComfyUI 共享目录配置
        comfy_group = QGroupBox("🔄 ComfyUI 共享目录")
        comfy_group.setStyleSheet(f"QGroupBox {{ font-size: 14px; font-weight: bold; color: {COLORS['text_primary']}; }}")
        comfy_layout = QVBoxLayout()
        comfy_layout.setContentsMargins(16, 12, 16, 12)
        comfy_layout.setSpacing(10)
        
        # 说明
        comfy_info = QLabel(
            "通过配置 ComfyUI 基础路径，可以共享 ComfyUI 的模型目录，避免重复下载模型。"
        )
        comfy_info.setWordWrap(True)
        comfy_info.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;")
        comfy_layout.addWidget(comfy_info)
        
        # ComfyUI 基础路径
        comfy_path_layout = QHBoxLayout()
        comfy_path_layout.addWidget(QLabel("ComfyUI 基础路径:"))
        
        self.comfy_base_path = QLineEdit()
        self.comfy_base_path.setPlaceholderText("请选择 ComfyUI 的 models 目录")
        self.comfy_base_path.setStyleSheet(self._input_style())
        comfy_path_layout.addWidget(self.comfy_base_path, 1)
        
        btn_browse_comfy = QPushButton("📂 浏览")
        btn_browse_comfy.setFixedSize(80, 30)
        btn_browse_comfy.setStyleSheet(self._btn_style(COLORS['bg_hover'], COLORS['cyan']))
        btn_browse_comfy.clicked.connect(self._on_browse_comfy)
        comfy_path_layout.addWidget(btn_browse_comfy)
        
        comfy_layout.addLayout(comfy_path_layout)
        
        # 应用和清除按钮
        btn_layout_comfy = QHBoxLayout()
        
        btn_apply_comfy = QPushButton("🚀 应用 ComfyUI 路径到所有模型目录")
        btn_apply_comfy.setMinimumHeight(36)
        btn_apply_comfy.setStyleSheet(f"""
            QPushButton {{
                background: #2a2a5a;
                color: #64b5f6;
                border: 1px solid #64b5f644;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #64b5f633;
            }}
        """)
        btn_apply_comfy.clicked.connect(self._on_apply_comfy)
        btn_layout_comfy.addWidget(btn_apply_comfy)
        
        btn_clear_comfy = QPushButton("🗑️ 清除 ComfyUI 路径")
        btn_clear_comfy.setMinimumHeight(36)
        btn_clear_comfy.setStyleSheet(f"""
            QPushButton {{
                background: #3a1a1a;
                color: #ff6b6b;
                border: 1px solid #ff6b6b44;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #5a2a2a;
            }}
        """)
        btn_clear_comfy.clicked.connect(self._on_clear_comfy)
        btn_layout_comfy.addWidget(btn_clear_comfy)
        
        comfy_layout.addLayout(btn_layout_comfy)
        
        comfy_group.setLayout(comfy_layout)
        layout.addWidget(comfy_group)
        

        
        # 分割线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{COLORS['border']};")
        layout.addWidget(sep2)
        
        # 创建各个模型目录卡片
        for config_key, info in MODEL_DIR_INFO.items():
            card = ModelDirCard(config_key, info)
            self._cards[config_key] = card
            layout.addWidget(card)
        
        layout.addStretch()
        
        # 保存按钮
        btn_save = QPushButton("💾 保存路径配置")
        btn_save.setMinimumHeight(40)
        btn_save.setStyleSheet(f"""
            QPushButton {{
                background:{COLORS['accent']};
                color:white;
                border:none;
                border-radius:6px;
                font-size:13px;
                font-weight:bold;
            }}
            QPushButton:hover {{
                background:{COLORS['accent_light']};
            }}
        """)
        btn_save.clicked.connect(self._on_save)
        layout.addWidget(btn_save)
        
        scroll.setWidget(content)
        
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)
    
    def _load_config(self):
        """加载配置"""
        paths = self.config.get("paths", {})
        
        # 检查是否有共享路径配置
        comfy_paths_file = os.path.join(BASE_DIR, "comfy_paths.yaml")
        has_shared_path = False
        shared_type = None
        
        if os.path.exists(comfy_paths_file):
            try:
                with open(comfy_paths_file, 'r', encoding='utf-8') as f:
                    comfy_config = yaml.safe_load(f)
                if comfy_config and 'comfyui' in comfy_config and 'base_path' in comfy_config['comfyui']:
                    self.comfy_base_path.setText(comfy_config['comfyui']['base_path'])
                    has_shared_path = True
                    shared_type = 'comfyui'
            except Exception as e:
                pass
        
        # 使用本地路径
        for config_key, card in self._cards.items():
            path = paths.get(config_key, "")
            
            if path:
                card.set_path(path)
            else:
                # 使用默认路径
                default_path = os.path.join(
                    BASE_DIR, 
                    "webui", 
                    "models", 
                    MODEL_DIR_INFO[config_key]['subdir_name']
                )
                if os.path.exists(default_path):
                    card.set_path(default_path)
    
    def _on_browse_comfy(self):
        """浏览 ComfyUI 目录"""
        path = QFileDialog.getExistingDirectory(
            self,
            "选择 ComfyUI 的 models 目录",
            "D:\\ai"
        )
        
        if path:
            self.comfy_base_path.setText(os.path.normpath(path))
    

    
    def _on_apply_comfy(self):
        """应用 ComfyUI 路径到所有模型目录"""
        comfy_base_path = self.comfy_base_path.text()
        
        if not comfy_base_path or not os.path.exists(comfy_base_path):
            QMessageBox.warning(
                self,
                "路径错误",
                "请先选择有效的 ComfyUI models 目录！"
            )
            return
        
        # 显示确认对话框
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("应用 ComfyUI 路径")
        msg.setText("确定要应用 ComfyUI 路径到所有模型目录吗？")
        msg.setInformativeText("这将保留本地模型路径，同时添加 ComfyUI 模型路径。")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Ok)
        
        if msg.exec() != QMessageBox.StandardButton.Ok:
            return
        
        # 保存到 comfy_paths.yaml
        comfy_config = {}
        
        # 保留原有的配置
        comfy_paths_file = os.path.join(BASE_DIR, "comfy_paths.yaml")
        if os.path.exists(comfy_paths_file):
            try:
                with open(comfy_paths_file, 'r', encoding='utf-8') as f:
                    comfy_config = yaml.safe_load(f)
            except Exception as e:
                pass
        
        # 添加 ComfyUI 配置
        comfy_config["comfyui"] = {
            "base_path": comfy_base_path,
            "checkpoints": "checkpoints",
            "diffusion_models": "diffusion_models",
            "unet": "unet",
            "clip": "clip",
            "text_encoders": "text_encoders",
            "loras": "loras",
            "vae": "vae",
            "sams": "sams",
            "ControlNet": "ControlNet",
        }
        
        try:
            with open(comfy_paths_file, 'w', encoding='utf-8') as f:
                yaml.dump(comfy_config, f, default_flow_style=False, allow_unicode=True)
            
            QMessageBox.information(
                self,
                "应用成功",
                "ComfyUI 路径已成功配置！\n\nWebUI 将同时搜索本地模型和 ComfyUI 模型目录。"
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "保存失败",
                f"保存 ComfyUI 路径配置失败：{str(e)}"
            )
    

    
    def _on_clear_comfy(self):
        """清除 ComfyUI 共享路径"""
        from PyQt6.QtWidgets import QMessageBox
        
        # 显示确认对话框
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("清除 ComfyUI 路径")
        msg.setText("确定要清除 ComfyUI 共享路径吗？")
        msg.setInformativeText("这将恢复到本地模型路径，并删除保存的 ComfyUI 路径配置。")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Ok)
        
        if msg.exec() != QMessageBox.StandardButton.Ok:
            return
        
        # 清除输入框
        self.comfy_base_path.clear()
        
        # 删除 comfy_paths.yaml 中的 ComfyUI 配置
        comfy_paths_file = os.path.join(BASE_DIR, "comfy_paths.yaml")
        if os.path.exists(comfy_paths_file):
            try:
                with open(comfy_paths_file, 'r', encoding='utf-8') as f:
                    comfy_config = yaml.safe_load(f)
                
                # 删除 ComfyUI 配置
                if comfy_config and 'comfyui' in comfy_config:
                    del comfy_config['comfyui']
                
                # 保存更新后的配置
                with open(comfy_paths_file, 'w', encoding='utf-8') as f:
                    yaml.dump(comfy_config, f, default_flow_style=False, allow_unicode=True)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "清除失败",
                    f"清除 ComfyUI 路径配置失败：{str(e)}"
                )
                return
        
        # 恢复到本地模型路径并更新 launcher_config.json
        if "paths" not in self.config:
            self.config["paths"] = {}
        
        for config_key in MODEL_DIR_INFO.keys():
            if config_key in self._cards:
                default_path = os.path.join(
                    BASE_DIR, 
                    "webui", 
                    "models", 
                    MODEL_DIR_INFO[config_key]['subdir_name']
                )
                if os.path.exists(default_path):
                    self._cards[config_key].set_path(default_path)
                    # 更新 launcher_config.json 中的路径
                    self.config["paths"][config_key] = default_path
        
        # 保存配置到 launcher_config.json
        from core.config import save_config
        save_config(self.config)
        
        QMessageBox.information(
            self,
            "清除成功",
            "ComfyUI 共享路径已清除，已恢复到本地模型路径。\n\n注意：请重新启动 WebUI 以使更改生效。"
        )
    

    
    def _input_style(self):
        """输入框样式"""
        return f"""
            QLineEdit {{ 
                background:{COLORS['bg_input']}; 
                color:{COLORS['text_secondary']};
                border:1px solid {COLORS['border']}; 
                border-radius:4px;
                padding:4px 8px; 
                font-size:12px; 
            }}
        """
    
    def _btn_style(self, bg, fg):
        """按钮样式"""
        return f"""
            QPushButton {{ 
                background:{bg}; 
                color:{fg}; 
                border:1px solid {fg}44;
                border-radius:4px; 
                font-size:12px; 
            }}
            QPushButton:hover {{ 
                background:{fg}33; 
            }}
        """
    
    def _on_save(self):
        """保存配置"""
        if "paths" not in self.config:
            self.config["paths"] = {}
        
        for config_key, card in self._cards.items():
            path = card.get_path()
            if path:
                self.config["paths"][config_key] = path
        
        # 保存路径配置到 comfy_paths.yaml
        comfy_config = {}
        
        # 保留原有的配置
        comfy_paths_file = os.path.join(BASE_DIR, "comfy_paths.yaml")
        if os.path.exists(comfy_paths_file):
            try:
                with open(comfy_paths_file, 'r', encoding='utf-8') as f:
                    comfy_config = yaml.safe_load(f)
            except Exception as e:
                pass
        
        # 保存 ComfyUI 路径配置
        comfy_base_path = self.comfy_base_path.text()
        if comfy_base_path:
            comfy_config["comfyui"] = {
                "base_path": comfy_base_path,
                "checkpoints": "checkpoints",
                "diffusion_models": "diffusion_models",
                "unet": "unet",
                "clip": "clip",
                "text_encoders": "text_encoders",
                "loras": "loras",
                "vae": "vae",
                "sams": "sams",
                "ControlNet": "ControlNet",
            }
        

        
        # 保存配置
        try:
            with open(comfy_paths_file, 'w', encoding='utf-8') as f:
                yaml.dump(comfy_config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            pass
        
        QMessageBox.information(self, "成功", "路径配置已保存！")
    
    def apply_to_config(self, config: dict):
        """应用配置到传入的 config 对象"""
        if "paths" not in config:
            config["paths"] = {}
        
        for config_key, card in self._cards.items():
            path = card.get_path()
            if path:
                config["paths"][config_key] = path
