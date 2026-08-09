"""插件管理核心逻辑"""
import os
import subprocess
from dataclasses import dataclass, field
from PyQt6.QtCore import QThread, pyqtSignal
from .paths import BASE_DIR, EXT_DIR, GIT_EXE


@dataclass
class ExtInfo:
    name: str
    path: str
    remote_url: str  = ""
    branch: str      = ""
    commit_hash: str = ""
    commit_date: str = ""
    remote_hash: str = ""   # 远程最新 commit
    has_update: bool = False
    enabled: bool    = True
    error: str       = ""


def _git(args: list[str], cwd: str, timeout=30, proxy: str = "") -> tuple[int, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Windows 下 schannel SSL 验证经常失败，禁用 SSL 验证以确保连接 GitHub 正常
    env["GIT_SSL_NO_VERIFY"] = "1"

    # 通过 git -c 直接注入代理配置，比环境变量更可靠
    proxy_args = []
    if proxy:
        env["http_proxy"]  = proxy
        env["https_proxy"] = proxy
        env["HTTP_PROXY"]  = proxy
        env["HTTPS_PROXY"] = proxy
        proxy_args = [
            "-c", f"http.proxy={proxy}",
            "-c", f"https.proxy={proxy}",
        ]

    # 直接使用系统的 Git 命令
    cmd = ["git"] + proxy_args + args
    try:
        r = subprocess.run(
            cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout, cwd=cwd, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"超时（>{timeout}s），请检查网络或代理是否可用"
    except Exception as e:
        # 如果系统 Git 失败，返回错误信息
        return -1, f"Git 命令执行失败: {str(e)}"


# 将 GitHub URL 转为镜像地址
def _mirror_url(url: str, mirror: str) -> str:
    if not mirror:
        return url
    # ghproxy 系列：完整 URL 拼在镜像域名后面
    # 例如 https://ghproxy.com/https://github.com/xxx/yyy
    ghproxy_mirrors = {"ghproxy.com", "mirror.ghproxy.com", "gh.llkk.cc", "ghfast.top"}
    if mirror in ghproxy_mirrors:
        if url.startswith("https://github.com"):
            return f"https://{mirror}/{url}"
        return url
    # 其他镜像：直接替换域名
    return url.replace("https://github.com", f"https://{mirror}")


def scan_extensions() -> list[ExtInfo]:
    """扫描 extensions 目录，返回插件列表"""
    result = []
    if not os.path.isdir(EXT_DIR):
        return result

    try:
        # 获取目录列表
        dirs = []
        try:
            dirs = os.listdir(EXT_DIR)
        except Exception as e:
            return result

        # 遍历目录列表
        for name in sorted(dirs):
            try:
                ext_path = os.path.join(EXT_DIR, name)
                # 检查是否是目录
                if not os.path.isdir(ext_path):
                    continue
                
                # 创建插件信息对象
                info = ExtInfo(name=name, path=ext_path)
                
                # 检查是否有 .git 目录
                git_dir = os.path.join(ext_path, ".git")
                if not os.path.exists(git_dir):
                    info.error = "非 git 仓库"
                    result.append(info)
                    continue

                # 尝试获取 Git 信息
                try:
                    # 远程地址
                    _, url = _git(["remote", "get-url", "origin"], ext_path)
                    info.remote_url = url if url and not url.startswith("fatal") else ""

                    # 当前分支
                    _, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], ext_path)
                    info.branch = branch if branch else "HEAD"

                    # 本地最新 commit
                    _, log = _git(["log", "-1", "--format=%h|%ci"], ext_path)
                    if "|" in log:
                        parts = log.split("|", 1)
                        info.commit_hash = parts[0].strip()
                        info.commit_date = parts[1].strip()[:16]  # 只取到分钟
                    else:
                        info.commit_hash = log[:7] if log else "?"
                except Exception as e:
                    info.error = f"Git 操作失败: {str(e)}"

                result.append(info)
            except Exception as e:
                # 跳过有问题的目录
                continue
    except Exception as e:
        # 如果整个扫描过程失败，返回空列表
        pass

    return result


def check_update(info: ExtInfo, proxy: str = "", mirror: str = "") -> ExtInfo:
    """检测单个插件是否有更新（需要网络）"""
    if not info.remote_url or info.error:
        return info

    # 如果配置了镜像，临时设置 remote url
    actual_url = _mirror_url(info.remote_url, mirror)
    if actual_url != info.remote_url:
        _git(["remote", "set-url", "origin", actual_url], info.path)

    code, out = _git(["fetch", "origin", "--quiet"], info.path, timeout=25, proxy=proxy)

    # 还原 remote url
    if actual_url != info.remote_url:
        _git(["remote", "set-url", "origin", info.remote_url], info.path)

    if code != 0:
        info.error = "fetch 失败"
        return info

    _, remote_hash = _git(["rev-parse", f"origin/{info.branch}"], info.path)
    info.remote_hash = remote_hash[:7] if remote_hash else "?"
    info.has_update = (
        bool(remote_hash) and
        remote_hash[:7] != info.commit_hash[:7]
    )
    return info


def update_extension(info: ExtInfo, proxy: str = "", mirror: str = "") -> tuple[bool, str]:
    """拉取更新，自动处理本地未提交修改和 rebase 冲突"""
    actual_url = _mirror_url(info.remote_url, mirror)
    if actual_url != info.remote_url:
        _git(["remote", "set-url", "origin", actual_url], info.path)

    # 检查是否有未提交修改，有则先 stash
    _, status = _git(["status", "--porcelain"], info.path)
    has_changes = bool(status.strip())
    stash_ok = False
    if has_changes:
        stash_code, stash_out = _git(["stash"], info.path)
        stash_ok = stash_code == 0
        if not stash_ok:
            out_prefix = f"[警告] stash 失败，继续尝试更新: {stash_out}\n"
        else:
            out_prefix = ""

    # 先尝试 rebase
    code, out = _git(["pull", "origin", info.branch, "--rebase"],
                     info.path, timeout=120, proxy=proxy)

    force_reset = False
    if code != 0:
        # rebase 失败时中止 rebase，改用 merge
        _git(["rebase", "--abort"], info.path)
        code, out2 = _git(["pull", "origin", info.branch, "--no-rebase"],
                          info.path, timeout=120, proxy=proxy)
        if code != 0:
            # merge 也失败，强制 reset 到远程版本
            _git(["fetch", "origin", info.branch], info.path, timeout=60, proxy=proxy)
            code, out3 = _git(["reset", "--hard", f"origin/{info.branch}"], info.path)
            out = out + "\n[已强制同步到远程版本]\n" + out3
            force_reset = True
        else:
            out = out2

    if has_changes:
        out = out_prefix + out

    # 还原 stash（仅在 stash 成功且未执行 reset --hard 时）
    if has_changes and stash_ok and not force_reset:
        pop_code, pop_out = _git(["stash", "pop"], info.path)
        if pop_code != 0:
            out += f"\n[警告] stash pop 失败，本地修改可能需要手动恢复: {pop_out}"

    if actual_url != info.remote_url:
        _git(["remote", "set-url", "origin", info.remote_url], info.path)

    return code == 0, out


def install_extension(url: str, proxy: str = "", mirror: str = "") -> tuple[bool, str]:
    """从 GitHub URL 安装新插件"""
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    dest = os.path.join(EXT_DIR, name)
    if os.path.exists(dest):
        return False, f"插件目录已存在: {dest}"
    clone_url = _mirror_url(url, mirror)
    code, out = _git(["clone", clone_url, dest], EXT_DIR, timeout=180, proxy=proxy)
    return code == 0, out


def uninstall_extension(info: ExtInfo) -> tuple[bool, str]:
    """删除插件目录"""
    import shutil
    import subprocess
    try:
        # 先尝试使用 shutil.rmtree 删除
        shutil.rmtree(info.path)
        return True, "已删除"
    except Exception as e:
        # 如果遇到权限错误，尝试使用 Windows 系统命令删除
        if os.name == 'nt':  # 只在 Windows 系统上执行
            try:
                # 使用 rmdir /s /q 命令强制删除目录
                cmd = ['cmd.exe', '/c', 'rmdir', '/s', '/q', info.path]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                if result.returncode == 0:
                    return True, "已删除"
                else:
                    return False, f"系统命令删除失败: {result.stderr}"
            except Exception as e2:
                return False, f"两次删除都失败: {str(e)} | {str(e2)}"
        return False, str(e)


def switch_branch(info: ExtInfo, branch: str) -> tuple[bool, str]:
    code, out = _git(["checkout", branch], info.path, timeout=30)
    return code == 0, out


# ── 异步 Worker ────────────────────────────────────────────────

class ScanWorker(QThread):
    done = pyqtSignal(list)

    def run(self):
        self.done.emit(scan_extensions())


class CheckUpdateWorker(QThread):
    progress = pyqtSignal(int, object)
    done     = pyqtSignal()

    def __init__(self, extensions: list, proxy: str = "", mirror: str = ""):
        super().__init__()
        self.extensions = extensions
        self.proxy  = proxy
        self.mirror = mirror

    def run(self):
        for i, ext in enumerate(self.extensions):
            updated = check_update(ext, self.proxy, self.mirror)
            self.progress.emit(i, updated)
        self.done.emit()


class UpdateWorker(QThread):
    log  = pyqtSignal(str)
    done = pyqtSignal(int, int)

    def __init__(self, extensions: list, proxy: str = "", mirror: str = ""):
        super().__init__()
        self.extensions = extensions
        self.proxy  = proxy
        self.mirror = mirror

    def run(self):
        ok = fail = 0
        for ext in self.extensions:
            self.log.emit(f"更新 {ext.name} ...")
            success, msg = update_extension(ext, self.proxy, self.mirror)
            if success:
                ok += 1
                self.log.emit(f"  ✅ {ext.name} 更新成功")
            else:
                fail += 1
                # 完整输出错误，不截断
                self.log.emit(f"  ❌ {ext.name} 失败:")
                for line in msg.splitlines():
                    if line.strip():
                        self.log.emit(f"     {line}")
        self.done.emit(ok, fail)


class InstallWorker(QThread):
    log  = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, url: str, proxy: str = "", mirror: str = ""):
        super().__init__()
        self.url    = url
        self.proxy  = proxy
        self.mirror = mirror

    def run(self):
        self.log.emit(f"正在克隆: {self.url}")
        ok, msg = install_extension(self.url, self.proxy, self.mirror)
        self.log.emit(msg)
        self.done.emit(ok, msg)
