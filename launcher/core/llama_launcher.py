"""llama.cpp launcher - manage llama-server process with auto-download"""
import os
import json
import zipfile
import subprocess
import urllib.request
import urllib.error
from PyQt6.QtCore import QThread, pyqtSignal

from core.paths import BASE_DIR


LLAMA_DIR = os.path.join(BASE_DIR, "llama.cpp")
LLAMA_SERVER = os.path.join(LLAMA_DIR, "llama-server.exe")
MODELS_DIR = os.path.join(LLAMA_DIR, "models")
# Shared port config file for WebUI plugin reading
LLAMA_PORT_FILE = os.path.join(BASE_DIR, "launcher", "llama_port.json")
LLAMA_GITHUB_REPO = "ggml-org/llama.cpp"


def download_llama_cpp(log_callback=None) -> bool:
    """
    Download the latest llama.cpp Windows CUDA release from GitHub and extract it.
    Uses stdlib only (urllib), no external dependency required.

    Args:
        log_callback: optional callback(str) for progress messages

    Returns:
        bool: True if download and extraction succeeded
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(f"[llama.cpp] {msg}")

    os.makedirs(LLAMA_DIR, exist_ok=True)

    # Step 1: fetch latest release info from GitHub API
    api_url = f"https://api.github.com/repos/{LLAMA_GITHUB_REPO}/releases/latest"
    log(f"Fetching latest release info from {LLAMA_GITHUB_REPO}...")

    try:
        req = urllib.request.Request(api_url, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SD-WebUI-Forge-Neo-Launcher",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            release_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log(f"Failed to fetch release info: {e}")
        return False

    # Step 2: find the Windows CUDA asset
    assets = release_data.get("assets", [])
    cuda_asset = None
    for asset in assets:
        name = asset.get("name", "")
        if "win" in name.lower() and "cuda" in name.lower() and name.endswith(".zip"):
            cuda_asset = asset
            break

    if not cuda_asset:
        # fallback: try any win zip
        for asset in assets:
            name = asset.get("name", "")
            if "win" in name.lower() and name.endswith(".zip"):
                cuda_asset = asset
                break

    if not cuda_asset:
        log("No Windows CUDA release asset found on GitHub")
        return False

    download_url = cuda_asset["browser_download_url"]
    file_name = cuda_asset["name"]
    file_size = cuda_asset.get("size", 0)
    tag = release_data.get("tag_name", "latest")

    log(f"Found release: {tag}")
    log(f"Downloading: {file_name} ({_format_size(file_size)})...")

    # Step 3: download the zip
    zip_path = os.path.join(LLAMA_DIR, file_name)
    try:
        req = urllib.request.Request(download_url, headers={
            "User-Agent": "SD-WebUI-Forge-Neo-Launcher",
        })
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            with open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct % 10 == 0 and downloaded < total:
                            log(f"Download progress: {pct}% ({_format_size(downloaded)}/{_format_size(total)})")
        log(f"Download complete: {_format_size(downloaded)}")
    except Exception as e:
        log(f"Download failed: {e}")
        # Cleanup partial download
        if os.path.exists(zip_path):
            os.unlink(zip_path)
        return False

    # Step 4: extract the zip
    log("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get list of members to extract (only files we need)
            members = []
            for m in zf.infolist():
                # Normalize path: remove top-level dir like "llama-b4464-bin-win-cuda-cu12.4.0-x64/"
                parts = m.filename.split("/", 1)
                if len(parts) > 1:
                    m.filename = parts[1]
                else:
                    m.filename = parts[0]
                if m.filename:
                    members.append(m)

            zf.extractall(LLAMA_DIR, members)

        log(f"Extracted to: {LLAMA_DIR}")
    except Exception as e:
        log(f"Extraction failed: {e}")
        return False
    finally:
        # Cleanup zip
        if os.path.exists(zip_path):
            try:
                os.unlink(zip_path)
                log("Cleaned up downloaded zip")
            except Exception:
                pass

    # Verify
    if os.path.exists(LLAMA_SERVER):
        log(f"llama-server.exe is ready at: {LLAMA_SERVER}")
        return True
    else:
        log("llama-server.exe not found after extraction, something went wrong")
        return False


def _format_size(bytes_val: int) -> str:
    """Format bytes to human-readable string"""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / 1024 / 1024:.1f} MB"
    else:
        return f"{bytes_val / 1024 / 1024 / 1024:.2f} GB"


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
            self.log_line.emit("[llama.cpp] llama-server.exe not found, attempting auto-download...")
            ok = download_llama_cpp(log_callback=lambda msg: self.log_line.emit(f"[llama.cpp] {msg}"))
            if not ok:
                self.log_line.emit("[llama.cpp] Auto-download failed. Please check network or manually download from GitHub releases")
                self.finished.emit(1)
                return
            self.log_line.emit("[llama.cpp] Download complete, starting server...")

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