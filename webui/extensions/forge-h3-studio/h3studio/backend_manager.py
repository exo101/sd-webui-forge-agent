from __future__ import annotations

import atexit
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .comfy_client import ComfyClient
from .config import EXTENSION_ROOT, load_config
from .errors import H3StudioError


class BackendManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._command: list[str] = []
        self._started_at: float | None = None
        self._logs: deque[str] = deque(maxlen=500)

    def _append_log(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self._logs.append(line)

    def _read_logs(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._append_log(line)
        except Exception as exc:
            self._append_log(f"[H3 Studio] 读取后端日志失败：{exc}")

    @staticmethod
    def discover() -> list[str]:
        """Return likely ComfyUI roots without recursively scanning the disk."""
        forge_root = EXTENSION_ROOT.parent.parent
        raw_candidates = [
            os.environ.get("H3_STUDIO_COMFY_PATH", ""),
            os.environ.get("COMFYUI_PATH", ""),
            str(forge_root / "ComfyUI"),
            str(forge_root.parent / "ComfyUI"),
            str(forge_root.parent / "ComfyUI_windows_portable"),
            str(Path.cwd() / "ComfyUI"),
            str(Path.cwd() / "ComfyUI_windows_portable"),
        ]
        found: list[str] = []
        seen: set[str] = set()
        for raw in raw_candidates:
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate.absolute())
            if key in seen:
                continue
            seen.add(key)
            if (candidate / "main.py").is_file() or (candidate / "ComfyUI" / "main.py").is_file():
                found.append(str(candidate))
        return found

    @staticmethod
    def _resolve_layout(config: dict[str, Any]) -> tuple[Path, Path, Path]:
        configured_path = str(config.get("comfy_path") or "").strip()
        if configured_path:
            selected = Path(configured_path).expanduser()
        else:
            discovered = BackendManager.discover()
            if not discovered:
                raise H3StudioError("尚未配置有效的 ComfyUI 目录，也未在 Forge 相邻目录自动发现")
            selected = Path(discovered[0])
        if not selected.is_dir():
            raise H3StudioError("尚未配置有效的 ComfyUI 目录")

        if (selected / "main.py").is_file():
            comfy_root = selected
            portable_root = selected.parent
        elif (selected / "ComfyUI" / "main.py").is_file():
            portable_root = selected
            comfy_root = selected / "ComfyUI"
        else:
            raise H3StudioError("所选目录中没有找到 ComfyUI/main.py")

        configured_python = str(config.get("python_executable") or "").strip()
        if configured_python:
            python = Path(configured_python).expanduser()
        else:
            candidates = [
                portable_root / "python_embeded" / "python.exe",
                comfy_root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
                portable_root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
            ]
            python = next((item for item in candidates if item.is_file()), Path(sys.executable))
        if not python.is_file():
            raise H3StudioError("未找到可用的 Python，可在设置中手动指定")
        return portable_root, comfy_root, python

    def _build_command(self, config: dict[str, Any]) -> tuple[list[str], Path]:
        _, comfy_root, python = self._resolve_layout(config)
        port = int(config.get("port", 8189))
        extra = shlex.split(str(config.get("extra_args") or ""), posix=os.name != "nt")
        blocked = {"--listen", "--port", "--tls-keyfile", "--tls-certfile"}
        if any(token.split("=", 1)[0] in blocked for token in extra):
            raise H3StudioError("额外启动参数不能覆盖 listen、port 或 TLS 参数")
        command = [
            str(python),
            str(comfy_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-auto-launch",
            *extra,
        ]
        return command, comfy_root

    def start(self) -> dict[str, Any]:
        config = load_config()
        if config.get("backend_mode") == "external":
            health = ComfyClient().health()
            if not health["ok"]:
                raise H3StudioError("外接 ComfyUI 当前不可用，请先启动它或检查地址")
            return self.status()

        with self._lock:
            health = ComfyClient().health()
            if health["ok"]:
                return self.status()
            if self._process is not None and self._process.poll() is None:
                return self.status()
            command, cwd = self._build_command(config)
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            self._append_log("[H3 Studio] 启动托管 ComfyUI：" + " ".join(command))
            self._process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self._command = command
            self._started_at = time.time()
            threading.Thread(target=self._read_logs, args=(self._process,), daemon=True).start()
            return self.status(skip_health=True)

    def stop(self) -> dict[str, Any]:
        config = load_config()
        if config.get("backend_mode") == "external":
            raise H3StudioError("外接模式不会由 Forge 停止后端")
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._process = None
            self._append_log("[H3 Studio] 托管后端已停止")
        return self.status()

    def shutdown(self) -> None:
        """Best-effort cleanup for a ComfyUI process started by this extension."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            try:
                process.terminate()
                process.wait(timeout=8)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            finally:
                self._process = None

    def status(self, *, skip_health: bool = False) -> dict[str, Any]:
        config = load_config()
        with self._lock:
            process_running = self._process is not None and self._process.poll() is None
            exit_code = None if self._process is None or process_running else self._process.poll()
            health = {"ok": False, "base_url": config.get("comfy_url")}
            if not skip_health:
                try:
                    health = ComfyClient().health()
                except Exception as exc:
                    health = {"ok": False, "base_url": config.get("comfy_url"), "error": str(exc)}
            state = "ready" if health.get("ok") else ("starting" if process_running else "stopped")
            if exit_code not in (None, 0) and not health.get("ok"):
                state = "error"
            return {
                "state": state,
                "ready": bool(health.get("ok")),
                "mode": config.get("backend_mode"),
                "url": config.get("comfy_url"),
                "process_running": process_running,
                "pid": self._process.pid if process_running else None,
                "exit_code": exit_code,
                "started_at": self._started_at,
                "command": self._command,
                "health": health,
                "auto_start_on_tab": bool(config.get("auto_start_on_tab", True)),
                "discovered_paths": self.discover(),
            }

    def logs(self, limit: int = 200) -> list[str]:
        with self._lock:
            return list(self._logs)[-max(1, min(limit, 500)):]


backend_manager = BackendManager()
atexit.register(backend_manager.shutdown)
