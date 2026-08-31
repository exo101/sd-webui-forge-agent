"""ComfyUI launcher - manage ComfyUI process"""
import json
import os
import subprocess
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from core.paths import BASE_DIR


COMFY_PORT_FILE = os.path.join(BASE_DIR, "launcher", "comfy_port.json")


def _find_comfy_main(comfy_path: str) -> str | None:
    """Find the ComfyUI main.py in the given directory."""
    candidates = [
        os.path.join(comfy_path, "main.py"),
        os.path.join(comfy_path, "ComfyUI", "main.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _find_python(comfy_path: str) -> str | None:
    """Auto-detect Python executable in ComfyUI directory."""
    candidates = [
        os.path.join(comfy_path, "python_embeded", "python.exe"),
        os.path.join(comfy_path, "python", "python.exe"),
        os.path.join(comfy_path, "venv", "Scripts", "python.exe"),
        os.path.join(comfy_path, "ComfyUI", "python_embeded", "python.exe"),
        os.path.join(comfy_path, "ComfyUI", "python", "python.exe"),
        os.path.join(comfy_path, "ComfyUI", "venv", "Scripts", "python.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


class ComfyUIWorker(QThread):
    log_line = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._process = None

    def run(self):
        comfy_cfg = self.config.get("comfyui", {})
        comfy_path = comfy_cfg.get("path", "")
        port = comfy_cfg.get("port", 8188)
        python_exe = comfy_cfg.get("python", "")

        if not comfy_path or not os.path.isdir(comfy_path):
            self.log_line.emit("❌ ComfyUI path not configured or does not exist")
            self.finished.emit(1)
            return

        main_py = _find_comfy_main(comfy_path)
        if not main_py:
            self.log_line.emit(f"❌ Could not find main.py in {comfy_path}")
            self.finished.emit(1)
            return

        if not python_exe or not os.path.isfile(python_exe):
            detected = _find_python(comfy_path)
            if detected:
                python_exe = detected
                self.log_line.emit(f"🔍 Auto-detected Python: {python_exe}")
            else:
                self.log_line.emit("❌ Could not find Python executable in ComfyUI directory")
                self.finished.emit(1)
                return

        # Write shared port config for WebUI plugin to read
        try:
            os.makedirs(os.path.dirname(COMFY_PORT_FILE), exist_ok=True)
            with open(COMFY_PORT_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "port": port,
                    "host": "127.0.0.1",
                    "path": comfy_path,
                }, f)
        except Exception as e:
            self.log_line.emit(f"⚠️ Failed to write comfy port config: {e}")

        cmd = [
            python_exe,
            main_py,
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--disable-auto-launch",
            "--enable-cors-header", "http://127.0.0.1:7869",
        ]

        self.log_line.emit(f"🚀 Starting ComfyUI → :{port}")
        self.log_line.emit(f"   Command: {' '.join(cmd)}")

        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
                cwd=os.path.dirname(main_py),
            )
            self._process = p
        except Exception as e:
            self.log_line.emit(f"❌ Failed to start ComfyUI: {e}")
            self.finished.emit(1)
            return

        self.log_line.emit(f"✅ ComfyUI started: http://127.0.0.1:{port}")
        # Wait for process to end
        try:
            for line in p.stdout:
                self.log_line.emit(line.rstrip())
            p.wait()
        except Exception:
            pass
        self.finished.emit(p.returncode if p.returncode is not None else 0)

    def stop(self):
        if self._process and self._process.poll() is None:
            try:
                import psutil
                parent = psutil.Process(self._process.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                parent.kill()
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._process = None