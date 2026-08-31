"""底部硬件监控状态栏"""
import os
import subprocess
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from .theme import COLORS
from core.paths import BASE_DIR, PYTHON_EXE


class HWWorker(QThread):
    data_ready = pyqtSignal(dict)

    def run(self):
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            mem_pct = mem.percent

            # GPU 显存和温度（用 torch）
            gpu_pct, temp = 0, 0
            vram_pct = 0
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    used = torch.cuda.memory_allocated(0)
                    total = props.total_memory
                    gpu_pct = int(used / total * 100) if total else 0
                    vram_pct = gpu_pct
            except Exception:
                pass

            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(",")
                    temp = int(parts[0].strip()) if parts else 0
                    gpu_util = int(parts[1].strip()) if len(parts) > 1 else 0
                    gpu_pct = gpu_util
                    if len(parts) > 3:
                        mem_used = int(parts[2].strip()) if parts[2].strip().isdigit() else 0
                        mem_total = int(parts[3].strip()) if parts[3].strip().isdigit() else 0
                        vram_pct = int(mem_used / mem_total * 100) if mem_total else 0
            except Exception:
                pass

            self.data_ready.emit({
                "cpu": int(cpu),
                "mem": int(mem_pct),
                "gpu": gpu_pct,
                "vram": vram_pct,
                "temp": temp,
            })
        except ImportError:
            # psutil 未安装，用简单方式
            self.data_ready.emit({"cpu": 0, "mem": 0, "gpu": 0, "vram": 0, "temp": 0})


class HWMonitorBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        self._worker: HWWorker | None = None
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(3000)  # 每3秒刷新

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        self._items = {}
        metrics = [
            ("cpu",  "CPU",  COLORS['cyan'],   "%"),
            ("mem",  "内存", COLORS['yellow'],  "%"),
            ("vram", "显存", COLORS['orange'],  "%"),
            ("gpu",  "GPU",  COLORS['accent'],  "%"),
            ("temp", "温度", COLORS['red'],     "°C"),
        ]

        for i, (key, label, color, unit) in enumerate(metrics):
            if i > 0:
                sep = QLabel("  |  ")
                sep.setStyleSheet(f"color:{COLORS['border']}; font-size:11px;")
                layout.addWidget(sep)

            lbl = QLabel(f"{label}")
            lbl.setStyleSheet(f"color:{COLORS['text_dim']}; font-size:11px;")
            layout.addWidget(lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedSize(60, 6)
            bar.setStyleSheet(f"""
                QProgressBar {{ background:{COLORS['border']}; border-radius:3px; border:none; }}
                QProgressBar::chunk {{ background:{color}; border-radius:3px; }}
            """)
            layout.addWidget(bar)

            val_lbl = QLabel("  0%")
            val_lbl.setFixedWidth(38)
            val_lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold;")
            layout.addWidget(val_lbl)

            self._items[key] = (bar, val_lbl, unit)

        layout.addStretch()

        # 进程数
        self.lbl_proc = QLabel("进程: 0")
        self.lbl_proc.setStyleSheet(f"color:{COLORS['text_dim']}; font-size:11px;")
        layout.addWidget(self.lbl_proc)

    def _refresh(self):
        if self._worker is not None:
            try:
                if self._worker.isRunning():
                    return
            except RuntimeError:
                # C++ object already deleted
                self._worker = None
                return
            # worker finished but not yet cleaned up — skip, let _on_worker_finished handle it
            return
        self._worker = HWWorker()
        self._worker.data_ready.connect(self._on_data)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self):
        if self._worker is not None:
            try:
                self._worker.deleteLater()
            except RuntimeError:
                pass
            self._worker = None

    def _on_data(self, d: dict):
        for key, (bar, lbl, unit) in self._items.items():
            val = d.get(key, 0)
            bar.setValue(min(val, 100))
            lbl.setText(f"  {val}{unit}")

    def set_proc_count(self, n: int):
        self.lbl_proc.setText(f"进程: {n}")
