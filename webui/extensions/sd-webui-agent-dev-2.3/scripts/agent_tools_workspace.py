# =============================================================================
# Agent Tools — 工作区 / 文档 / 联网 / 插件调研工具
# =============================================================================

import csv
import html
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.error
import urllib.parse
import zipfile
from xml.etree import ElementTree

from modules import scripts


def list_extensions_tool():
    """列出所有已安装的扩展插件。"""
    try:
        from modules import extensions
        exts = extensions.list_extensions()
        result = []
        for ext in exts:
            result.append({
                "name": ext.name,
                "enabled": ext.enabled,
                "is_builtin": ext.is_builtin,
            })
        return {"extensions": result}
    except Exception as e:
        return {"error": str(e)}


def research_extension_tool(name):
    """深入研究某个扩展插件的真实功能。

    读取扩展目录下的 README.md、脚本文件注释等，了解扩展的实际用途。
    当用户问到某个你不确定的扩展/功能时，必须先调用此工具调查，绝对不能编造！

    参数:
        name: 扩展名称或关键词（如 'seedvr2', 'controlnet', 'adetailer'）
    """
    try:
        from modules import paths
        # Forge Neo 路径获取：优先用 paths.extensions_dir，多级回退
        extensions_dir = getattr(paths, "extensions_dir", None)
        if not extensions_dir or not os.path.isdir(extensions_dir):
            # 回退1：基于 script_path 推导 WebUI 根目录
            webui_root = getattr(paths, "script_path", None)
            if webui_root and os.path.isdir(os.path.join(webui_root, "extensions")):
                extensions_dir = os.path.join(webui_root, "extensions")
            else:
                # 回退2：基于 scripts.basedir() 推导（当前扩展脚本目录的上上级）
                # scripts.basedir() = webui/extensions/sd-webui-agent/scripts
                # 上两级 = webui/extensions
                base = scripts.basedir()
                extensions_dir = os.path.dirname(os.path.dirname(base))
        print(f"[Agent] research_extension: extensions_dir={extensions_dir}")

        name_lower = name.lower()
        found_dir = None

        # 先精确匹配
        if os.path.isdir(os.path.join(extensions_dir, name)):
            found_dir = os.path.join(extensions_dir, name)
        else:
            # 模糊匹配
            for d in os.listdir(extensions_dir):
                d_path = os.path.join(extensions_dir, d)
                if os.path.isdir(d_path) and name_lower in d.lower():
                    found_dir = d_path
                    break

        if not found_dir:
            return {"status": "not_found", "query": name, "message": f"未找到包含 '{name}' 的扩展目录", "hint": "可先调用 list_extensions 查看所有扩展名称"}

        result = {
            "status": "success",
            "extension_dir": os.path.basename(found_dir),
            "path": found_dir,
            "readme": "",
            "scripts_summary": [],
            "key_files": [],
        }

        # 读取 README.md（最多前 1200 字，避免小模型上下文溢出）
        readme_path = os.path.join(found_dir, "README.md")
        if not os.path.isfile(readme_path):
            # 尝试 vendor 目录下的 README
            for root, dirs, files in os.walk(found_dir):
                for f in files:
                    if f.lower() == "readme.md":
                        readme_path = os.path.join(root, f)
                        break
                if os.path.isfile(readme_path):
                    break
        if os.path.isfile(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(1200)
                    # 截断到第一个 "## " 标题之后，避免太长
                    lines = content.split("\n")
                    useful_lines = []
                    for line in lines:
                        useful_lines.append(line)
                        if len(useful_lines) > 40:
                            break
                    result["readme"] = "\n".join(useful_lines)
            except Exception as e:
                result["readme"] = f"读取失败: {e}"

        # 扫描 scripts 目录下的 Python 文件，提取模块级 docstring
        scripts_dir = os.path.join(found_dir, "scripts")
        if os.path.isdir(scripts_dir):
            for f in sorted(os.listdir(scripts_dir))[:10]:
                if f.endswith(".py"):
                    fpath = os.path.join(scripts_dir, f)
                    result["key_files"].append(os.path.join("scripts", f))
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as sf:
                            lines = sf.readlines()[:30]
                            # 提取 docstring（第一个 """ 或 ''' 包裹的内容）
                            doc_lines = []
                            in_doc = False
                            quote = None
                            for line in lines:
                                stripped = line.strip()
                                if not in_doc:
                                    for q in ('"""', "'''"):
                                        if stripped.startswith(q):
                                            in_doc = True
                                            quote = q
                                            content = stripped[3:].strip()
                                            if content and not content.endswith(q):
                                                doc_lines.append(content)
                                            break
                                else:
                                    if quote in stripped:
                                        content = stripped.replace(quote, "").strip()
                                        if content:
                                            doc_lines.append(content)
                                        break
                                    doc_lines.append(stripped)
                            if doc_lines:
                                result["scripts_summary"].append({
                                    "file": f,
                                    "docstring": " ".join(doc_lines)[:500]
                                })
                    except Exception:
                        pass

        # 也扫描扩展根目录下的 py 文件
        for f in sorted(os.listdir(found_dir))[:5]:
            if f.endswith(".py"):
                result["key_files"].append(f)

        return result
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


_TEXT_FILE_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".css", ".html", ".htm",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".log", ".bat", ".cmd", ".sh", ".ps1", ".xml",
}


_BLOCKED_FILE_EXTENSIONS = {
    ".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin", ".dll", ".exe",
    ".msi", ".zip", ".7z", ".rar", ".tar", ".gz",
}


def _get_workspace_root():
    """返回 Forge Neo 整合包根目录，不依赖具体版本号。"""
    from modules import paths

    webui_root = getattr(paths, "script_path", None)
    if webui_root and os.path.isdir(webui_root):
        webui_root = os.path.realpath(webui_root)
        package_root = os.path.dirname(webui_root)
        if re.match(r"^sd-webui-forge-neo(?:[-_].*)?$", os.path.basename(package_root).lower()):
            return package_root
        return webui_root

    ext_dir = getattr(paths, "extensions_dir", None)
    if ext_dir and os.path.isdir(os.path.dirname(ext_dir)):
        webui_root = os.path.realpath(os.path.dirname(ext_dir))
        package_root = os.path.dirname(webui_root)
        if re.match(r"^sd-webui-forge-neo(?:[-_].*)?$", os.path.basename(package_root).lower()):
            return package_root
        return webui_root

    webui_root = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(scripts.basedir()))))
    package_root = os.path.dirname(webui_root)
    return package_root if re.match(r"^sd-webui-forge-neo(?:[-_].*)?$", os.path.basename(package_root).lower()) else webui_root


def _get_webui_root():
    """返回 WebUI 子目录，供扩展扫描等内部逻辑使用。"""
    from modules import paths
    root = getattr(paths, "script_path", None)
    if root and os.path.isdir(root):
        return os.path.realpath(root)
    ext_dir = getattr(paths, "extensions_dir", None)
    if ext_dir and os.path.isdir(os.path.dirname(ext_dir)):
        return os.path.realpath(os.path.dirname(ext_dir))
    return os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(scripts.basedir()))))


def _resolve_webui_path(path):
    """将路径安全解析到 Forge Neo 整合包根目录内。"""
    workspace_root = _get_workspace_root()
    target = os.path.realpath(os.path.join(workspace_root, str(path or ".")))

    try:
        if os.path.commonpath([workspace_root, target]) != workspace_root:
            raise ValueError("路径越界：只能访问 Forge Neo 整合包目录内的内容")
    except ValueError:
        raise ValueError("路径越界：只能访问 Forge Neo 整合包目录内的内容")

    return workspace_root, target


def _limit_chars(value, default=12000, maximum=999999):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(500, min(value, maximum))


def _read_document_text(file_path, max_chars):
    """提取受支持文件的文本，并返回文本和格式说明。"""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in _BLOCKED_FILE_EXTENSIONS:
        raise ValueError(f"为避免读取大型二进制资源，不支持读取 {extension} 文件")

    if extension in _TEXT_FILE_EXTENSIONS or not extension:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(max_chars + 1)

        if extension == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except (ValueError, TypeError):
                pass
        elif extension in {".csv", ".tsv"}:
            try:
                delimiter = "\t" if extension == ".tsv" else ","
                rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
                preview = rows[:100]
                text = "\n".join(" | ".join(column.strip() for column in row) for row in preview)
                if len(rows) > len(preview):
                    text += "\n... (仅显示前 100 行)"
            except (csv.Error, TypeError):
                pass
        return text, "text"

    if extension == ".docx":
        try:
            with zipfile.ZipFile(file_path) as archive:
                document_xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(document_xml)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs = []
            for paragraph in root.iter(f"{namespace}p"):
                pieces = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
                if pieces:
                    paragraphs.append("".join(pieces))
            return "\n".join(paragraphs), "docx"
        except (KeyError, OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise ValueError(f"DOCX 文档解析失败: {exc}") from exc

    if extension == ".pdf":
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name)
                reader = module.PdfReader(file_path)
                return "\n".join((page.extract_text() or "") for page in reader.pages), "pdf"
            except ImportError:
                continue
            except Exception as exc:
                raise ValueError(f"PDF 文档解析失败: {exc}") from exc
        raise ValueError("当前环境没有可用的 PDF 解析组件，暂时无法读取 PDF；可先提供 TXT、MD、DOCX 或复制文本内容")

    raise ValueError(f"暂不支持读取 {extension or '无扩展名'} 文件")


def read_workspace_file_tool(path, max_chars=12000):
    """读取 WebUI 目录内的文本、配置或代码文件内容。"""
    try:
        webui_root, target = _resolve_webui_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"文件不存在: {path}"}

        max_chars = _limit_chars(max_chars)
        text, file_type = _read_document_text(target, max_chars)
        return {
            "status": "success",
            "path": os.path.relpath(target, webui_root),
            "file_type": file_type,
            "size_bytes": os.path.getsize(target),
            "truncated": len(text) > max_chars,
            "content": text[:max_chars],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def analyze_document_tool(path, question="", max_chars=16000):
    """提取文档内容，供智能体据此总结、解释、审阅或回答问题。"""
    result = read_workspace_file_tool(path, max_chars=max_chars)
    if result.get("status") != "success":
        return result
    result["question"] = question or "请总结这份文档的关键内容。"
    result["instruction"] = "请仅基于提取出的内容回答问题；内容截断时需说明结论可能不完整。"
    return result


def _workspace_backup_path(webui_root, target):
    """Create a recoverable backup path for an agent edit."""
    backup_dir = os.path.join(webui_root, ".agent_backups")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    relative = os.path.relpath(target, webui_root).replace(os.sep, "__")
    return os.path.join(backup_dir, f"{stamp}_{relative}.bak")


def repair_workspace_file_tool(path, old_text, new_text, reason=""):
    """Safely patch one WebUI text file after inspecting the current contents.

    The exact old_text match prevents the agent from overwriting a file based on
    stale context. A timestamped backup is created before every edit.
    """
    try:
        webui_root, target = _resolve_webui_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"文件不存在: {path}"}
        if os.path.splitext(target)[1].lower() not in _TEXT_FILE_EXTENSIONS:
            return {"status": "error", "error": "只允许修改文本、代码和配置文件"}
        if any(secret in os.path.basename(target).lower() for secret in ("token", "secret", "credential")):
            return {"status": "error", "error": "为保护密钥，不允许直接修改凭据文件"}
        if not isinstance(old_text, str) or not old_text:
            return {"status": "error", "error": "old_text 不能为空；必须基于当前文件内容进行精确修改"}
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            current = handle.read()
        occurrences = current.count(old_text)
        if occurrences != 1:
            return {"status": "error", "error": f"精确匹配失败：找到 {occurrences} 处，要求恰好 1 处；请先重新读取文件"}
        backup = _workspace_backup_path(webui_root, target)
        with open(backup, "w", encoding="utf-8", newline="") as handle:
            handle.write(current)
        updated = current.replace(old_text, new_text, 1)
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        return {"status": "success", "path": os.path.relpath(target, webui_root),
                "backup": os.path.relpath(backup, webui_root), "reason": reason or "未提供"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def diagnose_workspace_tool(path="."):
    """Run lightweight, non-destructive diagnostics for a WebUI path."""
    try:
        webui_root, target = _resolve_webui_path(path)
        if os.path.isfile(target) and target.lower().endswith(".py"):
            command = [sys.executable, "-m", "py_compile", target]
        elif os.path.isdir(target):
            command = [sys.executable, "-m", "compileall", "-q", target]
        else:
            return {"status": "error", "error": "诊断目标必须是 Python 文件或目录"}
        completed = subprocess.run(command, cwd=webui_root, capture_output=True, text=True, timeout=120)
        return {"status": "success" if completed.returncode == 0 else "error",
                "path": os.path.relpath(target, webui_root), "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "诊断超时（超过 120 秒）"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def audit_extensions_tool(include_readme=True):
    """整理所有已安装扩展的状态、关键脚本和 README 摘要。"""
    try:
        from modules import extensions

        webui_root = _get_webui_root()
        registered = {}
        for extension in extensions.list_extensions():
            registered[extension.name] = {
                "enabled": bool(extension.enabled),
                "is_builtin": bool(extension.is_builtin),
                "path": getattr(extension, "path", None),
            }

        candidates = {}
        for directory, is_builtin in (
            (os.path.join(webui_root, "extensions"), False),
            (os.path.join(webui_root, "extensions-builtin"), True),
        ):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                full_path = os.path.join(directory, name)
                if os.path.isdir(full_path):
                    candidates[name] = {"path": full_path, "is_builtin": is_builtin}

        for name, info in registered.items():
            if name not in candidates and info["path"] and os.path.isdir(info["path"]):
                candidates[name] = {"path": info["path"], "is_builtin": info["is_builtin"]}

        results = []
        for name in sorted(candidates)[:50]:
            info = candidates[name]
            extension_path = info["path"]
            registration = registered.get(name, {})
            scripts_dir = os.path.join(extension_path, "scripts")
            key_files = []
            if os.path.isdir(scripts_dir):
                key_files = [
                    os.path.join("scripts", filename)
                    for filename in sorted(os.listdir(scripts_dir))
                    if filename.endswith(".py")
                ][:8]

            readme_preview = ""
            if include_readme:
                for filename in ("README.md", "readme.md", "README.txt"):
                    readme_path = os.path.join(extension_path, filename)
                    if os.path.isfile(readme_path):
                        try:
                            with open(readme_path, "r", encoding="utf-8", errors="replace") as handle:
                                readme_preview = " ".join(handle.read(700).split())[:350]
                        except OSError:
                            pass
                        break

            results.append({
                "name": name,
                "enabled": registration.get("enabled", True),
                "is_builtin": registration.get("is_builtin", info["is_builtin"]),
                "key_scripts": key_files,
                "readme_preview": readme_preview,
            })

        return {
            "status": "success",
            "extensions": results,
            "total": len(candidates),
            "returned": len(results),
            "truncated": len(candidates) > len(results),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def _fetch_web_text(url, timeout=20, max_bytes=2_000_000):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read(max_bytes)
        content_type = response.headers.get("content-type", "")
    charset = "utf-8"
    match = re.search(r"charset=([\w\-.]+)", content_type, re.I)
    if match:
        charset = match.group(1)
    return raw.decode(charset, errors="replace"), content_type


def _strip_html_text(markup, max_chars=6000):
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", markup or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|section|article)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()[:max_chars]


def web_search_tool(query, max_results=6, region="zh-cn"):
    """联网搜索实时新闻和公开网页资料。"""
    try:
        query = str(query or "").strip()
        if not query:
            return {"status": "error", "error": "请提供搜索关键词"}
        max_results = max(1, min(int(max_results or 6), 10))
        params = urllib.parse.urlencode({"q": query, "kl": region or "zh-cn"})
        search_url = f"https://duckduckgo.com/html/?{params}"
        markup, _ = _fetch_web_text(search_url, timeout=25)

        results = []
        blocks = re.findall(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|</body>)', markup)
        for href, title_html, tail in blocks:
            href = html.unescape(href)
            if "uddg=" in href:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                href = qs.get("uddg", [href])[0]
            title = _strip_html_text(title_html, max_chars=300)
            snippet_match = re.search(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', tail)
            snippet = _strip_html_text(snippet_match.group(1), max_chars=700) if snippet_match else ""
            if title and href.startswith(("http://", "https://")):
                results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= max_results:
                break

        if not results:
            return {"status": "error", "error": "没有解析到搜索结果，可能是网络受限或搜索页格式变化", "query": query}
        return {"status": "success", "query": query, "results": results, "source": "DuckDuckGo HTML"}
    except urllib.error.URLError as exc:
        return {"status": "error", "error": f"联网搜索失败，请检查网络或代理设置: {exc}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def web_read_url_tool(url, max_chars=8000):
    """读取指定网页并提取可引用的正文文本。"""
    try:
        url = str(url or "").strip()
        if not url.startswith(("http://", "https://")):
            return {"status": "error", "error": "请提供 http 或 https 网页地址"}
        max_chars = max(1000, min(int(max_chars or 8000), 20000))
        markup, content_type = _fetch_web_text(url, timeout=30)
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", markup)
        title = _strip_html_text(title_match.group(1), max_chars=300) if title_match else ""
        text = _strip_html_text(markup, max_chars=max_chars)
        return {"status": "success", "url": url, "title": title, "content_type": content_type, "text": text}
    except urllib.error.URLError as exc:
        return {"status": "error", "error": f"网页读取失败，请检查网络或代理设置: {exc}", "url": url}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc(), "url": url}


def explore_webui_tool(path="."):
    """探索 WebUI 的目录结构和内置功能。

    当用户问到 WebUI 有什么功能、某个目录是干什么的时，调用此工具调查。

    参数:
        path: 相对路径，默认为 '.' 表示 Forge Neo 整合包根目录
    """
    try:
        webui_root, target = _resolve_webui_path(path)
        print(f"[Agent] explore_webui: webui_root={webui_root}")

        if not os.path.isdir(target):
            # 可能是文件
            if os.path.isfile(target):
                size = os.path.getsize(target)
                return {"status": "file", "path": path, "size_bytes": size, "size_kb": round(size/1024, 1)}
            return {"status": "error", "error": f"路径不存在: {path}"}

        entries = []
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            entry = {"name": name, "type": "dir" if os.path.isdir(full) else "file"}
            if entry["type"] == "file":
                entry["size_kb"] = round(os.path.getsize(full) / 1024, 1)
            entries.append(entry)

        return {
            "status": "success",
            "path": path,
            "absolute_path": target,
            "entries": entries[:100],  # 限制数量
            "total": len(entries),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
