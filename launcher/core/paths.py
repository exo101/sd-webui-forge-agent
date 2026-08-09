"""
统一路径管理 - 兼容源码运行和 PyInstaller 打包后运行，以及 U 盘换盘符场景。

规则：
  - 打包 exe：BASE_DIR = exe 所在目录（sys.executable 的父目录）
  - 源码运行：BASE_DIR = launcher/ 的上两级目录（项目根目录）
  - 所有路径都基于 BASE_DIR 动态拼接，不存储绝对路径到配置文件
"""
import os
import sys


def get_base_dir() -> str:
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后
        # sys.argv[0] 可能是 "G:path\to\exe.exe"（缺反斜杠），需要修正
        raw = sys.argv[0]
        # 修正 Windows 盘符后缺反斜杠：G:foo -> G:\foo
        if len(raw) >= 2 and raw[1] == ":" and (len(raw) == 2 or raw[2] not in ("/", "\\")):
            raw = raw[:2] + "\\" + raw[2:]
        exe_path = os.path.abspath(raw)
        return os.path.dirname(exe_path)
    else:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BASE_DIR   = get_base_dir()
WEBUI_DIR  = os.path.join(BASE_DIR, "webui")
EXT_DIR    = os.path.join(BASE_DIR, "webui", "extensions")
ENV_BAT    = os.path.join(BASE_DIR, "system", "environment.bat")
COMFY_YAML = os.path.join(BASE_DIR, "comfy_paths.yaml")
CONFIG_FILE = os.path.join(BASE_DIR, "launcher_config.json")
PYTHON_EXE = os.path.join(BASE_DIR, "system", "python", "python.exe")
GIT_EXE    = os.path.join(BASE_DIR, "system", "git", "bin", "git.exe")


def get_python_exe() -> str:
    """获取 Python 解释器路径"""
    return PYTHON_EXE


def get_webui_dir() -> str:
    """获取 WebUI 目录"""
    return WEBUI_DIR
