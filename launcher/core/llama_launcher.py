"""llama.cpp launcher - manage llama-server process"""
import os
import json
import subprocess
from PyQt6.QtCore import QThread, pyqtSignal

from core.paths import BASE_DIR


LLAMA_DIR = os.path.join(BASE_DIR, "llama.cpp")
LLAMA_SERVER = os.path.join(LLAMA_DIR, "llama-server.exe")
MODELS_DIR = os.path.join(LLAMA_DIR, "models")
# Shared port config file for WebUI plugin reading
LLAMA_PORT_FILE = os.path.join(BASE_DIR, "launcher", "llama_port.json")


def scan_llama_models() -> list[dict]:
    """扫描 models 目录下的 .gguf 模型文件，返回模型列表"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.isdir(MODELS_DIR):
        return []

    # 收集所有 .gguf 文件，分离模型和 mmproj
    mmproj_files = {}
    model_files = []
    for f in os.listdir(MODELS_DIR):
        if not f.lower().endswith(".gguf"):
            continue
        if "mmproj" in f.lower():
            # mmproj: Qwen3.5-4B-mmproj-BF16.gguf → 基名 qwen3.5-4b
            base = f.lower().replace(".gguf", "")
            if "-mmproj-" in base:
                base = base.split("-mmproj-")[0]
            mmproj_files[base] = os.path.join(MODELS_DIR, f)
        else:
            model_files.append(f)

    models = []
    for f in model_files:
        model_path = os.path.join(MODELS_DIR, f)
        name_no_ext = f.replace(".gguf", "")
        name_lower = name_no_ext.lower()
        # 提取前缀：Qwen3.5-4B-Q6_K → Qwen3.5-4B
        parts = name_no_ext.split("-")
        prefix = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]

        # 匹配 mmproj
        mmproj = None
        for base_key, mpath in mmproj_files.items():
            if base_key == prefix.lower() or base_key in name_lower:
                mmproj = mpath
                break
        models.append({
            "name": f,
            "path": model_path,
            "mmproj": mmproj,
        })
    return models


def _pick_model_by_port(models: list[dict], port: int) -> dict | None:
    """根据端口号匹配模型：
    - 端口末位数字 + 模型名中的 "Xb" 匹配
    - 例如 8080→4B, 8079→2B, 否则用第一个
    """
    if not models:
        return None

    # 从端口末位推断模型大小
    port_str = str(port)
    # 取端口最后能匹配的字符（如 8080 的 "80" → 取最后一位或两位）
    hint = port_str[-2:] if len(port_str) >= 2 else port_str[-1]

    # 先在模型名中找匹配数字
    for m in models:
        name_lower = m["name"].lower()
        for digit in [hint, hint[-1]]:
            # 匹配 "4b"、"2b" 等模式
            if f"{digit}b" in name_lower or f"{digit}.5b" in name_lower:
                return m

    # 没匹配到，用第一个
    return models[0]


class LlamaWorker(QThread):
    log_line = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._processes = []

    def run(self):
        llama_cfg = self.config.get("llama", {})
        port = llama_cfg.get("port", 8080)
        ngl = llama_cfg.get("ngl", 100)

        # 将端口写入共享配置文件，供 WebUI 插件读取
        try:
            os.makedirs(os.path.dirname(LLAMA_PORT_FILE), exist_ok=True)
            with open(LLAMA_PORT_FILE, "w", encoding="utf-8") as f:
                json.dump({"port": port, "host": "0.0.0.0"}, f)
        except Exception as e:
            self.log_line.emit(f"⚠️ 写入共享端口配置失败: {e}")

        if not os.path.exists(LLAMA_SERVER):
            self.log_line.emit(f"❌ llama-server.exe not found at {LLAMA_SERVER}, please install llama.cpp manually")
            self.finished.emit(1)
            return

        models = scan_llama_models()
        if not models:
            self.log_line.emit("❌ 未在 models/ 目录下找到 .gguf 模型文件")
            self.finished.emit(1)
            return

        # 根据端口自动选择模型
        selected = _pick_model_by_port(models, port)
        self.log_line.emit(f"🚀 正在启动模型: {selected['name']} → 端口 {port}")

        cmd = [
            LLAMA_SERVER,
            "--model", selected["path"],
            "--host", "0.0.0.0",
            "--port", str(port),
            "-ngl", str(ngl),
        ]
        if selected["mmproj"] and os.path.exists(selected["mmproj"]):
            cmd += ["--mmproj", selected["mmproj"]]

        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
                cwd=LLAMA_DIR,
            )
            self._processes = [p]
        except Exception as e:
            self.log_line.emit(f"❌ 启动失败: {e}")
            self.finished.emit(1)
            return

        self.log_line.emit(f"✅ llama.cpp 已启动: {selected['name']} → :{port}")
        # 等待进程结束
        try:
            for line in p.stdout:
                pass  # 丢弃输出，等待进程结束
            p.wait()
        except Exception:
            pass
        self.finished.emit(0)

    def stop(self):
        for p in self._processes:
            if p and p.poll() is None:
                try:
                    import psutil
                    parent = psutil.Process(p.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except Exception:
                            pass
                    parent.kill()
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        self._processes.clear()

    def force_kill(self):
        self.stop()

    def get_models_info(self) -> list[dict]:
        """返回当前正在运行的模型信息"""
        llama_cfg = self.config.get("llama", {})
        base_port = llama_cfg.get("port", 8080)
        models = scan_llama_models()
        result = []
        for i, m in enumerate(models):
            result.append({
                "name": m["name"],
                "port": base_port + i,
            })
        return result