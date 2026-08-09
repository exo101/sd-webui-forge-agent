import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[2]  # launcher/../../ (to project root launcher)
VERSIONS_FILE = Path(BASE_DIR / 'launcher' / 'versions.json')
CURRENT_VERSION_FILE = Path(BASE_DIR / 'launcher' / 'CURRENT_VERSION.txt')

# GitHub仓库配置
GITHUB_REPO = "exo101/sd-webui-forge-neo-v3"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}"


def load_versions():
    """Load versions manifest from local versions.json.
    Returns a dict with keys: current_version (str) and versions (list of dict).
    If file not exists, create a minimal default structure.
    """
    if not VERSIONS_FILE.exists():
        # provide a minimal default structure
        default = {
            "current_version": "v1.0.0",
            "versions": [
                {"id": "f9b0ea1", "date": "2026-04-09 15:51:41", "message": "update VAE-Utils, fix qwen CN strength"},
                {"id": "ae4bbd2", "date": "2026-04-09 09:03:49", "message": "change qwen2512 presets sampler"},
            ],
        }
        # ensure directory exists
        VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VERSIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default

    try:
        with open(VERSIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # ensure keys exist
            if 'current_version' not in data:
                data['current_version'] = 'v1.0.0'
            if 'versions' not in data or not isinstance(data['versions'], list):
                data['versions'] = []
            return data
    except Exception:
        # fallback to a minimal default in case of parse errors
        return {
            "current_version": "v1.0.0",
            "versions": [],
        }


def get_current_version():
    data = load_versions()
    return data.get('current_version', 'v1.0.0')


def set_current_version(version_id: str):
    data = load_versions()
    data['current_version'] = version_id
    VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # also persist separately for quick reference
    CURRENT_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURRENT_VERSION_FILE, 'w', encoding='utf-8') as f:
        f.write(version_id)


def check_github_update():
    """
    检查GitHub上的最新版本
    
    Returns:
        dict: 包含更新信息的字典，如果没有更新则返回None
        {
            'has_update': bool,
            'latest_version': str,
            'current_version': str,
            'commit_message': str,
            'commit_date': str,
            'download_url': str
        }
    """
    try:
        import requests
        import urllib3
        
        # 禁用SSL警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 直连GitHub API（禁用SSL验证以解决证书问题）
        url = f"https://api.github.com/repos/{GITHUB_REPO}/commits?per_page=1"
        
        print(f"[Update Check] 正在连接GitHub...")
        response = requests.get(
            url,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=30,
            verify=False  # 禁用SSL验证
        )
        
        if response.status_code != 200:
            print(f"[Update Check] 连接失败: HTTP {response.status_code}")
            return None
        
        print(f"[Update Check] ✅ 连接成功")
        
        commits = response.json()
        if not commits:
            return None
        
        latest_commit = commits[0]
        latest_sha = latest_commit['sha'][:7]  # 取前7位作为版本号
        current_version = get_current_version()
        
        # 如果版本相同，无需更新
        if latest_sha == current_version:
            return {
                'has_update': False,
                'current_version': current_version,
                'latest_version': latest_sha
            }
        
        # 有可用更新
        commit_date = latest_commit['commit']['author']['date']
        # 转换为本地时间格式
        try:
            dt = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            date_str = commit_date
        
        return {
            'has_update': True,
            'current_version': current_version,
            'latest_version': latest_sha,
            'commit_message': latest_commit['commit']['message'],
            'commit_date': date_str,
            'download_url': f"https://github.com/{GITHUB_REPO}/archive/{latest_sha}.zip"
        }
        
    except ImportError:
        print("[Update Check] 缺少requests库，无法检查更新")
        return None
    except Exception as e:
        print(f"[Update Check] 检查更新时出错: {e}")
        return None


def update_from_github(progress_callback=None):
    """
    从GitHub更新启动器
    
    Args:
        progress_callback: 进度回调函数 callback(message: str)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        import subprocess
        import shutil
        
        launcher_dir = Path(__file__).resolve().parents[1]  # launcher目录
        project_root = launcher_dir.parent  # 项目根目录 (sd-webui-forge-neo-v3)
        
        # 重要：Git仓库在webui目录下，不是项目根目录
        webui_dir = project_root / 'webui'
        git_repo_path = str(webui_dir) if (webui_dir / '.git').exists() else str(project_root)
        
        def log(msg):
            print(f"[Update] {msg}")
            if progress_callback:
                progress_callback(msg)
        
        log("开始检查Git状态...")
        
        # 检查是否在Git仓库中
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=git_repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            log("错误: 当前目录不是Git仓库")
            return False, "当前目录不是Git仓库，请先初始化Git或使用ZIP包更新"
        
        log("获取远程更新...")
        # 直连GitHub获取更新
        result = subprocess.run(
            ['git', 'fetch', 'origin'],
            cwd=git_repo_path,
            capture_output=True,
            text=True,
            timeout=120  # 2分钟超时
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip()[-300:] if result.stderr else "未知错误"
            log(f"❌ Git fetch失败")
            log(f"错误详情: {error_msg[:200]}")
            return False, f"无法连接到GitHub，请检查网络连接\n\n错误: {error_msg[:200]}"
        
        log("✅ Git fetch成功")
        
        log("检查是否有可用更新...")
        # 比较本地和远程HEAD
        result = subprocess.run(
            ['git', 'rev-list', 'HEAD..origin/main', '--count'],
            cwd=git_repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            # 尝试master分支
            result = subprocess.run(
                ['git', 'rev-list', 'HEAD..origin/master', '--count'],
                cwd=git_repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
        
        if result.returncode != 0:
            log("无法确定更新状态")
            return False, "无法确定更新状态"
        
        commit_count = int(result.stdout.strip())
        
        if commit_count == 0:
            log("已是最新版本")
            return True, "已是最新版本"
        
        log(f"发现 {commit_count} 个新提交，开始更新...")
        
        # 备份当前versions.json
        if VERSIONS_FILE.exists():
            backup_file = VERSIONS_FILE.with_suffix('.json.bak')
            shutil.copy2(VERSIONS_FILE, backup_file)
            log("已备份版本信息")
        
        # 拉取最新代码
        log("正在拉取最新代码...")
        result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],  # 或 'master'
            cwd=git_repo_path,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            # 尝试master分支
            result = subprocess.run(
                ['git', 'pull', 'origin', 'master'],
                cwd=git_repo_path,
                capture_output=True,
                text=True,
                timeout=120
            )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "未知错误"
            log(f"Git pull失败: {error_msg}")
            
            # 恢复备份
            backup_file = VERSIONS_FILE.with_suffix('.json.bak')
            if backup_file.exists():
                shutil.copy2(backup_file, VERSIONS_FILE)
            
            return False, f"更新失败: {error_msg}"
        
        log("更新成功！")
        
        # 获取新的版本信息
        new_version = get_current_version()
        log(f"当前版本: {new_version}")
        
        # 清理备份文件
        backup_file = VERSIONS_FILE.with_suffix('.json.bak')
        if backup_file.exists():
            backup_file.unlink()
        
        return True, f"更新成功！当前版本: {new_version}"
        
    except subprocess.TimeoutExpired:
        log("更新超时，请检查网络连接")
        return False, "更新超时，请检查网络连接"
    except Exception as e:
        log(f"更新过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False, f"更新失败: {str(e)}"


def refresh_local_versions():
    """
    刷新本地版本列表（从Git历史中读取）
    
    Returns:
        list: 版本列表
    """
    try:
        import subprocess
        
        launcher_dir = Path(__file__).resolve().parents[1]  # launcher目录
        project_root = launcher_dir.parent  # 项目根目录 (sd-webui-forge-neo-v3)
        
        # 重要：Git仓库在webui目录下
        webui_dir = project_root / 'webui'
        git_repo_path = str(webui_dir) if (webui_dir / '.git').exists() else str(project_root)
        
        # 获取最近的10个commit
        result = subprocess.run(
            ['git', 'log', '--pretty=format:%h|%ad|%s', '--date=short', '-n', '10'],
            cwd=git_repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print("[Version Refresh] Git命令执行失败")
            return []
        
        versions = []
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                parts = line.split('|', 2)
                if len(parts) == 3:
                    versions.append({
                        'id': parts[0],
                        'date': parts[1],
                        'message': parts[2]
                    })
        
        # 更新versions.json
        if versions:
            data = load_versions()
            data['versions'] = versions
            data['current_version'] = get_current_version()
            
            VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(VERSIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        return versions
        
    except Exception as e:
        print(f"[Version Refresh] 刷新版本列表失败: {e}")
        return []