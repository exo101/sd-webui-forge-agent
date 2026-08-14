"""启动器核心逻辑 - 构建参数、启动/停止进程"""
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from core.paths import get_python_exe, get_webui_dir, BASE_DIR, COMFY_YAML, ENV_BAT, WEBUI_DIR
from core.config import find_available_port, is_port_in_use


def build_args(config: dict) -> list[str]:
    """
    构建WebUI启动参数
    
    支持的配置项:
    - theme: 主题 (dark/light)
    - port: 端口号
    - listen: 监听外部连接
    - autolaunch: 自动打开浏览器
    - cuda_malloc: CUDA内存分配优化
    - cuda_stream: CUDA流优化
    - enable_flash/enable_xformers: 注意力机制开关（默认全部启用）
    - lowvram: 显存优化模式
    - gpu_device: GPU设备选择 (通过CUDA_VISIBLE_DEVICES环境变量设置)
    
    注意力机制说明:
    - 默认全部启用，按优先级自动选择：FlashAttention > xFormers > PyTorch原生
    - 不启用时会添加 --disable-* 参数
    
    GPU切换说明:
    - 单显卡系统: 无需配置，自动使用唯一GPU
    - 多显卡系统: 
      * 在"环境检测"Tab中选择要使用的GPU
      * 选择后会自动设置CUDA_VISIBLE_DEVICES环境变量
      * 例如: 选择GPU 0 → set CUDA_VISIBLE_DEVICES=0
      * WebUI进程只能看到指定的GPU设备
      * 切换后需要重启WebUI生效
    """
    args = []
    args += ["--theme", config.get("theme", "dark")]
    args += ["--port", str(config.get("port", 7869))]
    if config.get("listen"):           args.append("--listen")
    if config.get("autolaunch"):       args.append("--autolaunch")
    if config.get("cuda_malloc"):      args.append("--cuda-malloc")
    if config.get("cuda_stream"):      args.append("--cuda-stream")
    if config.get("pin_shared_memory"):args.append("--pin-shared-memory")
    
    # 注意力机制：默认全部启用，不启用时添加 --disable-* 参数
    if not config.get("enable_flash", True):    args.append("--disable-flash")
    if not config.get("enable_xformers", True): args.append("--disable-xformers")
    
    if config.get("lowvram"):          args.append("--lowvram")
    if config.get("no_half"):          args.append("--no-half")
    if config.get("no_half_vae"):      args.append("--no-half-vae")
    if config.get("precision_full"):   args += ["--precision", "full"]
    if config.get("api"):              args.append("--api")
    if config.get("share"):            args.append("--share")
    if config.get("skip_install"):     args.append("--skip-install")
    if config.get("skip_version"):     args.append("--skip-version-check")
    if config.get("skip_torch"):       args.append("--skip-torch-cuda-test")
    if config.get("disable_sage"):     args.append("--disable-sage")
    if config.get("reserve_vram"):     args += ["--reserve-vram", str(config.get("reserve_vram"))]
    if config.get("neveroom"):         args.append("--neveroom")

    paths = config.get("paths", {})
    if paths.get("ckpt_dir"):
        args += ["--ckpt-dirs", paths["ckpt_dir"]]
    if paths.get("diffusion_models_dir"):
        args += ["--ckpt-dirs", paths["diffusion_models_dir"]]
    if paths.get("text_encoder_dir"):
        args += ["--text-encoder-dirs", paths["text_encoder_dir"]]
    if paths.get("lora_dir"):
        args += ["--lora-dirs", paths["lora_dir"]]
    if paths.get("vae_dir"):
        args += ["--vae-dirs", paths["vae_dir"]]
    if paths.get("esrgan_dir"):
        args += ["--esrgan-models-path", paths["esrgan_dir"]]
    if paths.get("controlnet_dir"):
        args += ["--controlnet-dir", paths["controlnet_dir"]]
    if paths.get("controlnet_preprocessor_dir"):
        args += ["--controlnet-preprocessor-models-dir", paths["controlnet_preprocessor_dir"]]

    if os.path.exists(COMFY_YAML):
        args += ["--forge-ref-comfy-yaml", COMFY_YAML]

    extra = config.get("extra_args", "").strip()
    if extra:
        args += extra.split()

    return args


def build_env_vars(config: dict) -> dict:
    """
    构建环境变量，包括GPU设备选择
    
    Returns:
        dict: 环境变量字典
        - CUDA_VISIBLE_DEVICES: 指定可见的GPU设备ID
          * 空字符串或不设置: 所有GPU可见（默认）
          * "0": 只使用GPU 0
          * "1": 只使用GPU 1
          * "0,1": 使用GPU 0和1（多卡并行）
    """
    env_vars = {}
    
    # GPU设备选择
    gpu_device = config.get("gpu_device", "")
    if gpu_device and gpu_device.strip():
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_device.strip()
    
    return env_vars


def build_bat(args: list[str], proxy: str = "", skip_update: bool = False, gpu_device: str = "") -> str:
    """生成临时 bat 文件内容"""
    # 所有参数都加引号，避免路径中反斜杠或空格导致解析错误
    def quote(a: str) -> str:
        # 如果已经有引号就不重复加
        if a.startswith('"') and a.endswith('"'):
            return a
        # 路径类参数（包含盘符或反斜杠）或含空格的都加引号
        if " " in a or (len(a) > 1 and a[1] == ":") or "\\" in a:
            # cmd.exe 中路径末尾的反斜杠会转义结束引号（"path\"），需要补一个反斜杠
            # 例如 "G:\models\" -> "G:\models\\"
            suffix = "\\" if a.endswith("\\") else ""
            return f'"{a}{suffix}"'
        return a

    cmd_args = " ".join(quote(a) for a in args)
    proxy_lines = ""
    if proxy:
        proxy_lines = (
            f"set http_proxy={proxy}\n"
            f"set https_proxy={proxy}\n"
            f"set HTTP_PROXY={proxy}\n"
            f"set HTTPS_PROXY={proxy}\n"
        )
    
    # GPU设备选择
    gpu_lines = ""
    if gpu_device and gpu_device.strip():
        gpu_lines = f"set CUDA_VISIBLE_DEVICES={gpu_device.strip()}\n"
    
    # 跳过更新：设置 SKIP_VENV_TESTS=1 并在 COMMANDLINE_ARGS 里加 --skip-install
    skip_lines = ""
    if skip_update:
        skip_lines = "set SKIP_VENV_TESTS=1\n"
        if "--skip-install" not in cmd_args:
            cmd_args = "--skip-install " + cmd_args
    
    # 先设置 PYTHON 环境变量，确保 webui.bat 使用启动器管理的 Python
    # 同时设置 SKIP_VENV=1，跳过旧 venv（旧 venv 可能是不同 Python 版本创建的）
    python_lines = f'set PYTHON={get_python_exe()}\nset SKIP_VENV=1\nset PYTHONIOENCODING=utf-8\n'
    # 如果 environment.bat 存在则加载，否则跳过
    env_lines = ""
    if os.path.exists(ENV_BAT):
        env_lines = f'call "{ENV_BAT}"\n'
    
    return (
        "@echo off\n"
        f"{env_lines}"
        f'cd /d "{WEBUI_DIR}"\n'
        f"{python_lines}"
        f"{proxy_lines}"
        f"{gpu_lines}"
        f"{skip_lines}"
        f"set COMMANDLINE_ARGS={cmd_args}\n"
        "call webui.bat\n"
    )


class LaunchWorker(QThread):
    log_line  = pyqtSignal(str)
    finished  = pyqtSignal(int)   # exit code

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._process = None
        self._tmp_bat_path = None  # 保存临时bat文件路径用于清理

    def run(self):
        # 启动前清理所有残留的临时文件（防止之前异常退出导致的残留）
        cleanup_all_temp_files()
        
        # 检测端口是否被占用，如果占用则自动切换
        original_port = self.config.get("port", 7869)
        if is_port_in_use(original_port):
            new_port = find_available_port(original_port)
            self.log_line.emit(f"⚠️  端口 {original_port} 已被占用，自动切换到端口 {new_port}")
            self.config["port"] = new_port
        
        args = build_args(self.config)
        bat_content = build_bat(args, "",
                                skip_update=self.config.get("skip_update", False),
                                gpu_device=self.config.get("gpu_device", ""))

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".bat", delete=False,
            encoding="gbk", dir=BASE_DIR, prefix="_launch_tmp_"
        )
        tmp.write(bat_content)
        tmp.close()
        
        # 保存临时文件路径用于后续清理
        self._tmp_bat_path = tmp.name

        try:
            creation_flags = 0 if self.config.get("show_console", False) else subprocess.CREATE_NO_WINDOW
            self._process = subprocess.Popen(
                ["cmd.exe", "/c", tmp.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
            for line in self._process.stdout:
                self.log_line.emit(line.rstrip())
            self._process.wait()
            self.finished.emit(self._process.returncode)
        finally:
            self._cleanup_temp_file()

    def _cleanup_temp_file(self):
        """清理临时bat文件"""
        if self._tmp_bat_path:
            try:
                if os.path.exists(self._tmp_bat_path):
                    os.unlink(self._tmp_bat_path)
                    self.log_line.emit(f"🗑️  已清理临时文件: {os.path.basename(self._tmp_bat_path)}")
                self._tmp_bat_path = None
            except Exception as e:
                self.log_line.emit(f"⚠️  清理临时文件失败: {e}")

    def stop(self):
        # 先清理临时文件
        self._cleanup_temp_file()
        
        if self._process and self._process.poll() is None:
            try:
                import psutil
                parent = psutil.Process(self._process.pid)
                # 递归杀死所有子进程（WebUI 及其相关进程）
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                # 然后杀死父进程（cmd.exe）
                try:
                    parent.kill()
                except Exception:
                    pass
            except Exception:
                # 备用方案：使用 taskkill 命令
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(self._process.pid), "/F"],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass

    def _kill_process_tree(self, pid: int):
        try:
            parent = psutil.Process(pid)
            # Kill all children first
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            # Then kill the parent
            try:
                parent.kill()
            except Exception:
                pass
        except Exception:
            pass

    def force_kill(self):
        # 先清理临时文件
        self._cleanup_temp_file()
        
        if self._process and self._process.poll() is None:
            try:
                import psutil
                parent = psutil.Process(self._process.pid)
                # 递归杀死所有子进程（WebUI 及其相关进程）
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                # 然后杀死父进程（cmd.exe）
                try:
                    parent.kill()
                except Exception:
                    pass
            except Exception:
                # 备用方案：使用 taskkill 命令
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(self._process.pid), "/F"],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass


class GitPullWorker(QThread):
    """Background worker for git pull to update the project kernel"""
    log_line = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def run(self):
        from core.paths import GIT_EXE, BASE_DIR
        # Try portable git first, then system git
        git_cmd = GIT_EXE if os.path.exists(GIT_EXE) else "git"

        # Use extended env to bypass SSL issues on Windows and use portable Git DLLs
        git_env = os.environ.copy()
        git_env.setdefault("GIT_SSL_NO_VERIFY", "1")
        _git_bin = os.path.join(BASE_DIR, "system", "git", "bin")
        _git_lib = os.path.join(BASE_DIR, "system", "git", "libexec", "git-core")
        git_env["PATH"] = f"{_git_bin};{_git_lib};{git_env.get('PATH', '')}"

        self.log_line.emit("=" * 60)
        self.log_line.emit("[UPDATE] Checking for kernel updates...")
        self.log_line.emit("=" * 60)

        try:
            # Step 1: Check if it's a git repository using Python first
            git_dir = os.path.join(BASE_DIR, ".git")
            if not os.path.isdir(git_dir):
                self.log_line.emit("[WARN] Not a git repository (.git not found)")
                self.log_line.emit("[WARN] This project was likely downloaded as a ZIP, not cloned with git")
                self.log_line.emit("[WARN] To enable updates, run: git clone https://github.com/exo101/sd-webui-forge-neo-v3.git")
                self.finished.emit(False, "Not a git repository")
                return

            # Step 2: Get current branch
            r2 = subprocess.run(
                [git_cmd, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=BASE_DIR, env=git_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            branch = (r2.stdout or "").strip() if r2.returncode == 0 else "unknown"
            self.log_line.emit(f"[INFO] Current branch: {branch}")

            # Step 3: Fetch latest changes
            self.log_line.emit("[UPDATE] Fetching latest changes...")
            r3 = subprocess.run(
                [git_cmd, "fetch", "origin"],
                capture_output=True, text=True, timeout=30,
                cwd=BASE_DIR, env=git_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if r3.returncode != 0:
                error_msg = (r3.stderr or "").strip() or "Fetch failed"
                self.log_line.emit(f"[FAIL] Fetch failed: {error_msg}")
                self.finished.emit(False, error_msg)
                return
            self.log_line.emit("[OK] Fetch completed")

            # Step 4: Check if there are local changes
            r4 = subprocess.run(
                [git_cmd, "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
                cwd=BASE_DIR, env=git_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            has_local_changes = bool((r4.stdout or "").strip())
            if has_local_changes:
                self.log_line.emit("[WARN] Local changes detected, stashing...")
                subprocess.run(
                    [git_cmd, "stash"],
                    capture_output=True, timeout=10,
                    cwd=BASE_DIR, env=git_env,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

            # Step 5: Get the current and remote HEAD
            r5 = subprocess.run(
                [git_cmd, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=BASE_DIR, env=git_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            local_commit = (r5.stdout or "").strip() if r5.returncode == 0 else "?"

            r6 = subprocess.run(
                [git_cmd, "rev-parse", "--short", f"origin/{branch}"],
                capture_output=True, text=True, timeout=10,
                cwd=BASE_DIR, env=git_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            remote_commit = (r6.stdout or "").strip() if r6.returncode == 0 else "?"

            if local_commit == remote_commit:
                self.log_line.emit(f"[OK] Already up to date ({local_commit})")
                self.finished.emit(True, "Already up to date")
                return

            # Step 6: Pull the latest code
            self.log_line.emit(f"[UPDATE] Pulling latest code ({remote_commit})...")
            r7 = subprocess.run(
                [git_cmd, "pull", "origin", branch],
                capture_output=True, text=True, timeout=60,
                cwd=BASE_DIR, env=git_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            if r7.returncode == 0:
                # Show what changed
                r8 = subprocess.run(
                    [git_cmd, "log", f"{local_commit}..HEAD", "--oneline", "--no-decorate"],
                    capture_output=True, text=True, timeout=10,
                    cwd=BASE_DIR, env=git_env,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                log_entries = (r8.stdout or "").strip() if r8.returncode == 0 else ""
                if log_entries:
                    self.log_line.emit(f"[OK] Update completed! New commits:")
                    for entry in log_entries.splitlines():
                        self.log_line.emit(f"      {entry}")
                else:
                    self.log_line.emit("[OK] Update completed!")
                self.finished.emit(True, "Update successful")
            else:
                error_msg = (r7.stderr or "").strip() or "Pull failed"
                self.log_line.emit(f"[FAIL] Pull failed: {error_msg}")
                self.finished.emit(False, error_msg)

        except subprocess.TimeoutExpired as e:
            self.log_line.emit(f"[FAIL] Operation timed out: {str(e)}")
            self.finished.emit(False, f"Timeout: {str(e)}")
        except Exception as e:
            self.log_line.emit(f"[FAIL] Update failed: {str(e)}")
            self.finished.emit(False, str(e))


def cleanup_all_temp_files():
    """清理所有残留的临时bat文件（启动器关闭时调用）"""
    import glob
    try:
        pattern = os.path.join(BASE_DIR, "_launch_tmp_*.bat")
        temp_files = glob.glob(pattern)
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except Exception:
                pass
    except Exception:
        pass


def kill_process_on_port(port: int) -> bool:
    """
    查找并终止占用指定端口的进程
    
    Args:
        port: 要清理的端口号
        
    Returns:
        bool: 是否成功找到并终止了进程
    """
    try:
        # 使用 netstat 查找占用端口的 PID
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        target_pid = None
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    target_pid = parts[-1]
                    break
        
        if not target_pid:
            return False
        
        # 终止进程及其子进程
        subprocess.run(
            ["taskkill", "/PID", str(target_pid), "/T", "/F"],
            capture_output=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        return True
    except Exception:
        return False
