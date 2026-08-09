"""环境检测 - Python/Git/CUDA/显存等"""
import os
import subprocess
import sys
import glob
import re
from .paths import BASE_DIR, PYTHON_EXE, GIT_EXE


def _run(cmd, **kwargs) -> tuple[int, str]:
    try:
        # 添加 encoding 和 errors 参数，避免编码问题
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        if 'errors' not in kwargs:
            kwargs['errors'] = 'replace'
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW, **kwargs)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return -1, str(e)


def check_python() -> dict:
    code, out = _run([PYTHON_EXE, "--version"])
    ok = code == 0
    return {"ok": ok, "version": out if ok else "未找到", "path": PYTHON_EXE}


def ensure_python_installed() -> dict:
    """
    检查 Python 是否可用，如果不可用则尝试从 launcher/python-3.13.12-amd64.exe 自动安装。
    
    Returns:
        dict: {"ok": bool, "message": str}
    """
    # 先检查 Python 是否已存在
    code, out = _run([PYTHON_EXE, "--version"])
    if code == 0:
        return {"ok": True, "message": f"Python 已就绪: {out.strip()}"}
    
    # Python 不存在，查找安装包
    installer = os.path.join(BASE_DIR, "launcher", "python-3.13.12-amd64.exe")
    if not os.path.exists(installer):
        return {"ok": False, "message": f"未找到 Python 安装包，请将 python-3.13.12-amd64.exe 放入 launcher 目录"}
    
    # 执行静默安装
    python_dir = os.path.join(BASE_DIR, "system", "python")
    os.makedirs(python_dir, exist_ok=True)
    
    try:
        result = subprocess.run(
            [installer, "/quiet", f"TargetDir={python_dir}", "InstallAllUsers=0", "PrependPath=0",
             "Include_launcher=0", "InstallLauncherAllUsers=0", "AssociateFiles=0", "Shortcuts=0"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return {"ok": False, "message": f"Python 安装失败 (code={result.returncode}): {result.stderr}"}
        
        # 验证安装结果
        code2, out2 = _run([PYTHON_EXE, "--version"])
        if code2 == 0:
            return {"ok": True, "message": f"Python 安装成功: {out2.strip()}"}
        else:
            return {"ok": False, "message": "Python 安装完成但无法运行，请检查 system/python 目录"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Python 安装超时（超过 5 分钟），请手动安装"}
    except Exception as e:
        return {"ok": False, "message": f"Python 安装异常: {e}"}


def check_git() -> dict:
    code, out = _run([GIT_EXE, "--version"])
    ok = code == 0
    return {"ok": ok, "version": out if ok else "未找到", "path": GIT_EXE}


def ensure_git_installed() -> dict:
    """
    检查 Git 是否可用，如果不可用则检测 system/git/ 下的 PortableGit 安装包并自动安装。
    同时也检查系统 PATH 中的 git。
    
    Returns:
        dict: {"ok": bool, "message": str, "path": str}
    """
    # 1. 检查便携版 Git
    code, out = _run([GIT_EXE, "--version"])
    if code == 0:
        return {"ok": True, "message": f"Git 已就绪: {out.strip()}", "path": GIT_EXE}
    
    # 2. 检查系统 PATH 中的 git
    try:
        code2, out2 = _run(["git", "--version"])
        if code2 == 0:
            return {"ok": True, "message": f"Git 已就绪（系统 PATH）: {out2.strip()}", "path": "git"}
    except Exception:
        pass
    
    # 3. 查找 system/git/ 目录下的 PortableGit 安装包
    git_dir = os.path.join(BASE_DIR, "system", "git")
    installers = glob.glob(os.path.join(git_dir, "PortableGit*.exe"))
    if not installers:
        return {"ok": False, "message": "未找到 Git，请将 PortableGit-*.exe 放入 system/git/ 目录"}
    
    installer = installers[0]
    try:
        result = subprocess.run(
            [installer, "-o" + git_dir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return {"ok": False, "message": f"Git 安装失败 (code={result.returncode})"}
        
        # 验证安装结果
        code3, out3 = _run([GIT_EXE, "--version"])
        if code3 == 0:
            return {"ok": True, "message": f"Git 安装成功: {out3.strip()}", "path": GIT_EXE}
        else:
            return {"ok": False, "message": "Git 安装完成但无法运行，请检查 system/git/bin 目录"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Git 安装超时（超过 2 分钟），请手动安装"}
    except Exception as e:
        return {"ok": False, "message": f"Git 安装异常: {e}"}


def check_cuda() -> dict:
    code, out = _run(
        [PYTHON_EXE, "-c",
         "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
         "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"]
    )
    if code != 0:
        return {"ok": False, "torch": "未安装", "cuda": False, "gpu": "N/A"}
    lines = out.splitlines()
    torch_ver = lines[0] if len(lines) > 0 else "?"
    cuda_ok   = lines[1].strip().lower() == "true" if len(lines) > 1 else False
    gpu_name  = lines[2] if len(lines) > 2 else "N/A"
    return {"ok": True, "torch": torch_ver, "cuda": cuda_ok, "gpu": gpu_name}


def check_vram() -> dict:
    code, out = _run(
        [PYTHON_EXE, "-c",
         "import torch; t=torch.cuda.get_device_properties(0); "
         "print(t.total_memory//1024//1024); print(t.name)"]
    )
    if code != 0:
        return {"ok": False, "total_mb": 0, "name": "N/A"}
    lines = out.splitlines()
    try:
        mb = int(lines[0])
        name = lines[1] if len(lines) > 1 else "?"
        return {"ok": True, "total_mb": mb, "name": name}
    except Exception:
        return {"ok": False, "total_mb": 0, "name": "N/A"}


def check_all_gpus() -> list[dict]:
    """检测所有可用的GPU设备"""
    code, out = _run(
        [PYTHON_EXE, "-c",
         "import torch\n"
         "if not torch.cuda.is_available():\n"
         "    print('NO_CUDA')\n"
         "else:\n"
         "    count = torch.cuda.device_count()\n"
         "    for i in range(count):\n"
         "        props = torch.cuda.get_device_properties(i)\n"
         "        name = props.name\n"
         "        vram = props.total_memory // 1024 // 1024\n"
         "        cc = f'{props.major}.{props.minor}'\n"
         "        print(f'{i}|{name}|{vram}|{cc}')"]
    )
    
    if code != 0 or out.strip() == "NO_CUDA":
        return []
    
    gpus = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) == 4:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "vram_mb": int(parts[2]),
                "compute_capability": parts[3],
                "vram_gb": round(int(parts[2]) / 1024, 1)
            })
    
    return gpus


def check_webui_installed() -> bool:
    return os.path.exists(os.path.join(BASE_DIR, "webui", "webui.bat"))


def get_webui_version() -> str:
    version_file = os.path.join(BASE_DIR, "webui", "VERSION")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    # 尝试从 git log 获取
    code, out = _run(
        [GIT_EXE, "log", "--oneline", "-1"],
        cwd=os.path.join(BASE_DIR, "webui")
    )
    return out[:40] if code == 0 else "未知"


# ── 依赖检测 ──────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "xformers",
    "gradio",
    "transformers",
    "accelerate",
    "diffusers",
    "safetensors",
    "omegaconf",
    "einops",
    "open_clip",
    "compel",
    "k_diffusion",
]

# import 名 -> pip 安装包名（不一致时需要映射）
PIP_INSTALL_NAME: dict[str, str] = {
    "open_clip":   "open-clip-torch",
    "k_diffusion": "k-diffusion",
}

# 最低版本要求（可选，不满足则显示 ⚠️）
MIN_VERSIONS: dict[str, tuple] = {
    "torch":        (2, 0),
    "gradio":       (3, 0),
    "transformers": (4, 0),
    "diffusers":    (0, 20),
    "safetensors":  (0, 3),
}


def _parse_version(ver_str: str) -> tuple:
    """将版本字符串解析为整数元组，忽略非数字部分。"""
    parts = re.split(r"[.+\-]", ver_str)
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            break
    return tuple(result)


def check_package(pkg_name: str) -> dict:
    """
    检测单个包是否安装（供安装后单独复查使用）。
    返回 {"installed": bool, "version": str, "low_version": bool}
    """
    import_map = {
        "open_clip": "open_clip",
        "k_diffusion": "k_diffusion",
    }
    import_name = import_map.get(pkg_name, pkg_name)

    code, out = _run([
        PYTHON_EXE, "-c",
        f"import importlib.metadata as m; print(m.version('{pkg_name}'))"
    ])
    if code != 0:
        # 回退：尝试直接 import 获取 __version__
        code2, out2 = _run([
            PYTHON_EXE, "-c",
            f"import {import_name}; print(getattr({import_name}, '__version__', 'unknown'))"
        ])
        if code2 != 0:
            return {"installed": False, "version": "", "low_version": False}
        version = out2.strip()
    else:
        version = out.strip()

    # 检查最低版本
    low_version = False
    if pkg_name in MIN_VERSIONS:
        parsed = _parse_version(version)
        min_ver = MIN_VERSIONS[pkg_name]
        if parsed and parsed < min_ver:
            low_version = True

    return {"installed": True, "version": version, "low_version": low_version}


def check_all_packages() -> list[dict]:
    """一次性批量检测所有必需包（单个子进程），避免串行启动 N 个进程。"""
    import_map = {
        "open_clip": "open_clip",
        "k_diffusion": "k_diffusion",
    }
    # 构建一段 Python 脚本，一次性输出所有包的版本
    # 格式：pkg_name=version 或 pkg_name=__MISSING__
    script_lines = ["import importlib.metadata as _m"]
    for pkg in REQUIRED_PACKAGES:
        imp = import_map.get(pkg, pkg)
        script_lines.append(
            f"try:\n"
            f"    print('{pkg}=' + _m.version('{pkg}'))\n"
            f"except Exception:\n"
            f"    try:\n"
            f"        import {imp} as _mod\n"
            f"        print('{pkg}=' + getattr(_mod, '__version__', 'unknown'))\n"
            f"    except Exception:\n"
            f"        print('{pkg}=__MISSING__')"
        )
    script = "\n".join(script_lines)
    code, out = _run([PYTHON_EXE, "-c", script])

    # 解析输出
    version_map: dict[str, str] = {}
    if code == 0:
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                version_map[k.strip()] = v.strip()

    results = []
    for pkg in REQUIRED_PACKAGES:
        ver = version_map.get(pkg, "__MISSING__")
        if ver == "__MISSING__" or not ver:
            results.append({"name": pkg, "installed": False, "version": "", "low_version": False})
            continue
        low_version = False
        if pkg in MIN_VERSIONS:
            parsed = _parse_version(ver)
            if parsed and parsed < MIN_VERSIONS[pkg]:
                low_version = True
        results.append({"name": pkg, "installed": True, "version": ver, "low_version": low_version})
    return results


# ── 系统环境依赖检测 ──────────────────────────────────────────

def check_vcredist() -> dict:
    """检测 Visual C++ Runtime 是否安装（PyTorch 必须）"""
    try:
        import winreg
        # 检查 VC++ 2015-2022 x64
        keys = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        for key_path in keys:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                installed, _ = winreg.QueryValueEx(key, "Installed")
                version_val, _ = winreg.QueryValueEx(key, "Version")
                winreg.CloseKey(key)
                if installed:
                    return {"ok": True, "version": str(version_val), "detail": "Visual C++ 2015-2022 x64"}
            except Exception:
                continue
        return {"ok": False, "version": "", "detail": "未找到 Visual C++ Runtime x64"}
    except Exception as e:
        return {"ok": False, "version": "", "detail": str(e)}


def check_cuda_toolkit() -> dict:
    """检测系统 CUDA Toolkit 版本（通过 nvidia-smi）"""
    # 一次调用获取驱动版本和 GPU 名称
    code, out = _run(["nvidia-smi", "--query-gpu=driver_version,name",
                      "--format=csv,noheader,nounits"], timeout=5)
    if code == 0 and out.strip():
        parts = out.strip().split(",")
        driver = parts[0].strip() if parts else "?"
        gpu = parts[1].strip() if len(parts) > 1 else "?"
        # 从 nvidia-smi 普通输出获取 CUDA 版本（只需一次额外调用）
        code2, out2 = _run(["nvidia-smi"], timeout=5)
        cuda_ver = "?"
        if code2 == 0:
            for line in out2.splitlines():
                if "CUDA Version" in line:
                    import re as _re
                    m = _re.search(r"CUDA Version:\s*([\d.]+)", line)
                    if m:
                        cuda_ver = m.group(1)
                    break
        return {
            "ok": True,
            "cuda_version": cuda_ver,
            "driver": driver,
            "gpu": gpu,
            "detail": f"GPU: {gpu}  |  驱动: {driver}  |  CUDA: {cuda_ver}"
        }
    return {"ok": False, "cuda_version": "", "driver": "", "gpu": "未检测到 NVIDIA GPU",
            "detail": "未检测到 NVIDIA GPU 或驱动未安装"}


def check_disk_space() -> dict:
    """检测项目所在磁盘剩余空间"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(BASE_DIR)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        ok = free_gb >= 10  # 至少 10GB
        return {
            "ok": ok,
            "free_gb": round(free_gb, 1),
            "total_gb": round(total_gb, 1),
            "detail": f"剩余 {free_gb:.1f} GB / 共 {total_gb:.1f} GB"
                      + ("" if ok else "  ⚠️ 空间不足，建议至少保留 10GB")
        }
    except Exception as e:
        return {"ok": False, "free_gb": 0, "total_gb": 0, "detail": str(e)}


def check_windows_version() -> dict:
    """检测 Windows 版本"""
    try:
        import platform
        ver = platform.version()
        release = platform.release()
        # Win10/11 才能正常运行
        major = int(platform.version().split(".")[0])
        ok = major >= 10
        return {"ok": ok, "version": f"Windows {release} ({ver})",
                "detail": f"Windows {release} {ver}"}
    except Exception as e:
        return {"ok": False, "version": "?", "detail": str(e)}


def check_memory() -> dict:
    """检测物理内存"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024**3
        avail_gb = mem.available / 1024**3
        ok = total_gb >= 8
        warn = total_gb >= 8 and avail_gb < 4
        return {
            "ok": ok and not warn,
            "warn": warn,
            "total_gb": round(total_gb, 1),
            "avail_gb": round(avail_gb, 1),
            "detail": f"总计 {total_gb:.1f} GB，可用 {avail_gb:.1f} GB"
                      + ("  ⚠️ 可用内存不足，建议关闭其他程序" if warn else
                         "  ❌ 内存不足 8GB，可能无法运行" if not ok else ""),
        }
    except Exception as e:
        return {"ok": False, "warn": False, "total_gb": 0, "avail_gb": 0, "detail": str(e)}


def check_system_arch() -> dict:
    """检测系统是否为 64 位"""
    import platform
    is64 = platform.machine().endswith("64")
    return {
        "ok": is64,
        "detail": f"{'64 位' if is64 else '32 位'} 系统  ({platform.machine()})",
    }


def check_nvidia_driver_version() -> dict:
    """检测 NVIDIA GPU 和驱动信息"""
    code, out = _run(["nvidia-smi",
                      "--query-gpu=driver_version,name,memory.total",
                      "--format=csv,noheader,nounits"], timeout=8)
    if code != 0 or not out.strip():
        return {"ok": False, "has_gpu": False, "driver": "",
                "gpu": "", "vram_mb": 0,
                "detail": "未检测到 NVIDIA GPU 或驱动未安装",
                "cuda_ver": ""}

    parts = [p.strip() for p in out.strip().split(",")]
    driver  = parts[0] if len(parts) > 0 else "?"
    gpu     = parts[1] if len(parts) > 1 else "?"
    vram_mb = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0

    cuda_ver = "?"
    code2, out2 = _run(["nvidia-smi"], timeout=5)
    if code2 == 0:
        for line in out2.splitlines():
            if "CUDA Version" in line:
                m = re.search(r"CUDA Version:\s*([\d.]+)", line)
                if m:
                    cuda_ver = m.group(1)
                break

    detail = f"GPU: {gpu}  |  驱动: {driver}  |  CUDA: {cuda_ver}  |  显存: {vram_mb//1024}GB"

    return {
        "ok": True, "has_gpu": True,
        "driver": driver, "gpu": gpu,
        "vram_mb": vram_mb, "cuda_ver": cuda_ver,
        "detail": detail,
    }


def check_antivirus() -> dict:
    """检测常见杀毒软件进程（可能拦截 WebUI）"""
    try:
        import psutil
        av_processes = {
            "360tray.exe":    "360安全卫士",
            "360sd.exe":      "360杀毒",
            "QQPCTray.exe":   "腾讯电脑管家",
            "HipsTray.exe":   "火绒安全",
            "avp.exe":        "卡巴斯基",
            "MsMpEng.exe":    "Windows Defender",
            "bdagent.exe":    "Bitdefender",
            "avgui.exe":      "AVG",
            "avguard.exe":    "Avast",
        }
        found = []
        for proc in psutil.process_iter(["name"]):
            name = proc.info.get("name", "")
            if name in av_processes:
                found.append(av_processes[name])

        if found:
            return {
                "ok": False,
                "found": found,
                "detail": f"检测到: {', '.join(found)}  ⚠️ 可能拦截模型加载或 Python 运行",
            }
        return {"ok": True, "found": [], "detail": "未检测到已知杀毒软件干扰"}
    except Exception as e:
        return {"ok": True, "found": [], "detail": f"检测跳过: {e}"}


def check_webui_path_permission() -> dict:
    """检测 WebUI 目录是否有写入权限"""
    test_file = os.path.join(BASE_DIR, "webui", ".perm_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return {"ok": True, "detail": f"目录可写  |  {os.path.join(BASE_DIR, 'webui')}"}
    except Exception as e:
        return {"ok": False, "detail": f"目录无写入权限: {e}  ⚠️ 请以管理员身份运行或检查目录权限"}


def check_long_path() -> dict:
    """检测 Windows 长路径是否启用（路径过长会导致安装失败）"""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem")
        val, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        winreg.CloseKey(key)
        ok = bool(val)
        return {
            "ok": ok,
            "detail": "长路径已启用" if ok else "长路径未启用  ⚠️ 可能导致依赖安装失败",
        }
    except Exception:
        return {"ok": True, "detail": "无法检测（跳过）"}


def check_all_system_deps() -> list[dict]:
    """防小白系统环境全面检测"""
    results = []

    # 1. 系统架构
    r = check_system_arch()
    results.append({
        "name": "系统架构", "required": True, "ok": r["ok"],
        "detail": r["detail"],
        "fix": "需要 64 位 Windows 系统，32 位不支持运行 AI 模型"
    })

    # 2. Windows 版本
    r = check_windows_version()
    results.append({
        "name": "Windows 版本", "required": True, "ok": r["ok"],
        "detail": r["detail"],
        "fix": "需要 Windows 10 (1903+) 或 Windows 11"
    })

    # 3. 内存
    r = check_memory()
    ok = r["ok"] and not r.get("warn", False)
    results.append({
        "name": "内存", "required": True,
        "ok": ok,
        "detail": r["detail"],
        "fix": "建议至少 16GB 内存，最低 8GB。可增加虚拟内存临时缓解"
    })

    # 4. 磁盘空间
    r = check_disk_space()
    results.append({
        "name": "磁盘空间", "required": True, "ok": r["ok"],
        "detail": r["detail"],
        "fix": "模型文件较大，建议至少保留 20GB 空间"
    })

    # 5. Visual C++ Runtime
    r = check_vcredist()
    results.append({
        "name": "Visual C++ Runtime", "required": True, "ok": r["ok"],
        "detail": r["detail"] + (f"  {r['version']}" if r["version"] else ""),
        "fix": "搜索下载「Microsoft Visual C++ 2015-2022 Redistributable x64」并安装"
    })

    # 6. NVIDIA 驱动版本
    r = check_nvidia_driver_version()
    if r["has_gpu"]:
        results.append({
            "name": "NVIDIA 驱动版本", "required": True, "ok": r["ok"],
            "detail": r["detail"],
            "fix": f"当前驱动 {r['driver']} 版本过低，请前往 nvidia.cn 下载最新驱动（需要 ≥ 596.49）"
        })

    # 7. 长路径支持
    r = check_long_path()
    results.append({
        "name": "Windows 长路径", "required": False, "ok": r["ok"],
        "detail": r["detail"],
        "fix": "以管理员身份运行 PowerShell 执行：New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force"
    })

    # 8. WebUI 目录权限
    if os.path.exists(os.path.join(BASE_DIR, "webui")):
        r = check_webui_path_permission()
        results.append({
            "name": "目录写入权限", "required": True, "ok": r["ok"],
            "detail": r["detail"],
            "fix": "右键启动器选择「以管理员身份运行」，或检查文件夹权限设置"
        })

    return results
