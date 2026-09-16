# =============================================================================
# photoshop-mcp 桥接模块 — 让绘梦助手通过 MCP 驱动本机 Photoshop 2026
#
# 原理:
#   1. 以 stdio 子进程方式启动 photoshop-mcp 服务器 (node dist/index.js)
#   2. 手写 JSON-RPC 2.0 stdio 客户端（无需安装 mcp 依赖）
#   3. 向 Agent 注册 4 个轻量桥接工具（约 500 tokens），
#      130 个 PS 工具通过 ps_list_tools 按需发现，
#      避免把 ~30k tokens 的完整 schema 塞进每次 LLM 请求
#
# 路径（不写死，按优先级自动探测，结果缓存）:
#   Photoshop.exe:  agent_config.json "photoshop_path" > 运行中的进程 > 注册表 > 各盘常见位置
#   photoshop-mcp:  agent_config.json "photoshop_mcp_dir" > <PS安装目录>\Plug-ins\photoshop-mcp
# =============================================================================

import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time

# 工具注册（无 webui 依赖，可独立导入测试）
try:
    from scripts.agent_tools_registry import agent_tool
    _REGISTRY_OK = True
except Exception:
    _REGISTRY_OK = False

    def agent_tool(name, description, parameters=None):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# 路径与配置
# =============================================================================

_IMG_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "ps_mcp")
_MAX_RAW_TEXT = 4000        # 返回给 LLM 的纯文本上限
_MAX_JSON_TEXT = 12000      # 返回给 LLM 的 JSON 文本上限
_CALL_TIMEOUT = 300         # PS 工具调用超时（生成式填充等可能较慢）
_INIT_TIMEOUT = 60          # 服务器启动+握手超时
_detect_cache = {}          # 探测结果缓存（每进程只探测一次）


def _load_cfg():
    """直接读 agent_config.json（不 import agent_config，避免 webui 依赖）。"""
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_config.json")
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _ps_year_key(path):
    """从路径中提取 4 位年份，用于优先取较新版本。"""
    m = re.search(r"(\d{4})", path or "")
    return int(m.group(1)) if m else 0


def _find_running_photoshop():
    """枚举运行中进程，找到 photoshop.exe。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        psapi = ctypes.windll.psapi  # EnumProcesses 在 psapi.dll
        kernel32 = ctypes.windll.kernel32
        handles = (wintypes.DWORD * 4096)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcesses(ctypes.byref(handles), ctypes.sizeof(handles), ctypes.byref(needed)):
            return None
        for i in range(needed.value // ctypes.sizeof(wintypes.DWORD)):
            pid = handles[i]
            if not pid:
                continue
            h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                continue
            try:
                buf = ctypes.create_unicode_buffer(4096)
                size = wintypes.DWORD(4096)
                if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)) \
                        and os.path.basename(buf.value).lower() == "photoshop.exe":
                    return buf.value
            finally:
                kernel32.CloseHandle(h)
    except Exception:
        pass
    return None


def _find_photoshop_registry():
    """注册表探测：App Paths / SOFTWARE\\Adobe 安装键，优先较新版本。"""
    if os.name != "nt":
        return None
    try:
        import winreg
    except Exception:
        return None
    cands = []
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
                     winreg.KEY_READ | winreg.KEY_WOW64_32KEY):
            # 1) App Paths
            try:
                k = winreg.OpenKey(root,
                                   r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe",
                                   0, view)
            except OSError:
                k = None
            if k is not None:
                try:
                    val, _ = winreg.QueryValueEx(k, None)
                    if val:
                        cands.append(str(val))
                except OSError:
                    pass
                finally:
                    winreg.CloseKey(k)
            # 2) Adobe 安装键（含 "photoshop" 的子键，读 Path / ApplicationPath / InstallLocation）
            try:
                k = winreg.OpenKey(root, r"SOFTWARE\Adobe", 0, view)
            except OSError:
                continue
            try:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(k, i)
                    except OSError:
                        break
                    i += 1
                    if "photoshop" not in sub.lower():
                        continue
                    try:
                        sk = winreg.OpenKey(root, r"SOFTWARE\Adobe\\" + sub, 0, view)
                    except OSError:
                        continue
                    try:
                        for name in ("Path", "ApplicationPath", "InstallLocation"):
                            try:
                                val, _ = winreg.QueryValueEx(sk, name)
                            except OSError:
                                continue
                            if val:
                                val = str(val)
                                cands.append(val if val.lower().endswith(".exe")
                                             else os.path.join(val, "Photoshop.exe"))
                    finally:
                        winreg.CloseKey(sk)
            finally:
                winreg.CloseKey(k)
    cands = [c for c in dict.fromkeys(cands) if os.path.isfile(c)]
    cands.sort(key=_ps_year_key, reverse=True)
    return cands[0] if cands else None


def _find_photoshop_scan():
    """扫描各固定盘常见 Adobe 安装位置，优先较新版本。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
    except Exception:
        return None
    masks = kernel32.GetLogicalDrives()
    cands = []
    for i in range(26):
        if not (masks >> i) & 1:
            continue
        letter = chr(ord("A") + i)
        try:
            if kernel32.GetDriveTypeW(wintypes.LPCWSTR(letter + ":\\")) != 3:  # DRIVE_FIXED
                continue
        except Exception:
            continue
        for base in (r"Program Files\Adobe", r"Program Files (x86)\Adobe"):
            adobe = os.path.join(letter + ":", base)
            try:
                names = os.listdir(adobe)
            except OSError:
                continue
            for name in names:
                if "photoshop" not in name.lower():
                    continue
                exe = os.path.join(adobe, name, "Photoshop.exe")
                if os.path.isfile(exe):
                    cands.append(exe)
    cands.sort(key=_ps_year_key, reverse=True)
    return cands[0] if cands else None


def _resolve_ps_exe():
    """Photoshop.exe: agent_config.json > 运行中进程 > 注册表 > 常见位置扫描。"""
    if "ps_exe" in _detect_cache:
        return _detect_cache["ps_exe"]
    found = _load_cfg().get("photoshop_path")
    src = "agent_config.json"
    if not (found and os.path.isfile(found)):
        found, src = None, ""
        for finder, name in ((_find_running_photoshop, "运行中进程"),
                             (_find_photoshop_registry, "注册表"),
                             (_find_photoshop_scan, "常见位置")):
            try:
                found = finder()
            except Exception:
                found = None
            if found:
                src = name
                break
    _detect_cache["ps_exe"] = found
    print(f"[Agent] Photoshop 路径探测 ({src or '未找到'}): {found}")
    return found


def _resolve_mcp_dir():
    """photoshop-mcp 目录: agent_config.json > <PS安装目录>\\Plug-ins\\photoshop-mcp。"""
    if "mcp_dir" in _detect_cache:
        return _detect_cache["mcp_dir"]
    found = _load_cfg().get("photoshop_mcp_dir")
    src = "agent_config.json"
    if not (found and os.path.isfile(os.path.join(found, "dist", "index.js"))):
        found, src = None, ""
        exe = _resolve_ps_exe()
        if exe:
            cand = os.path.join(os.path.dirname(exe), "Plug-ins", "photoshop-mcp")
            if os.path.isfile(os.path.join(cand, "dist", "index.js")):
                found, src = cand, "PS安装目录"
    _detect_cache["mcp_dir"] = found
    print(f"[Agent] photoshop-mcp 目录探测 ({src or '未找到'}): {found}")
    return found


def _find_node():
    node = shutil.which("node")
    if node:
        return node
    for cand in (
        os.path.expandvars(r"%ProgramFiles%\nodejs\node.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\nodejs\node.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\nodejs\node.exe"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


# =============================================================================
# MCP stdio 客户端（JSON-RPC 2.0，单进程 + 读取线程 + 自动重启）
# =============================================================================

_lock = threading.RLock()
_proc = None
_ready = False
_tool_cache = None
_pending = {}
_next_id = [1]


class MCPError(RuntimeError):
    pass


def _reader_loop():
    """读取服务器 stdout 行，按 id 分发给等待中的请求。"""
    try:
        for raw in _proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            rid = msg.get("id")
            if rid is None:
                continue  # 通知/日志，忽略
            holder = _pending.get(rid)
            if holder is not None:
                if "error" in msg:
                    holder["error"] = msg["error"]
                else:
                    holder["result"] = msg.get("result")
                holder["event"].set()
    except Exception:
        pass
    finally:
        # 进程退出：唤醒所有等待者，避免死锁
        with _lock:
            for holder in _pending.values():
                holder.setdefault("error", {"code": -32000, "message": "MCP server process exited"})
                holder["event"].set()


def _start_server():
    global _proc, _ready, _tool_cache, _reader_thread
    node = _find_node()
    mcp_dir = _resolve_mcp_dir()
    entry = os.path.join(mcp_dir, "dist", "index.js") if mcp_dir else None
    if not node:
        raise MCPError("未找到 node.exe，无法启动 photoshop-mcp（photoshop-mcp 需要 Node.js >= 18）")
    if not entry or not os.path.isfile(entry):
        raise MCPError(
            "未能自动探测到 photoshop-mcp（dist/index.js 所在目录）。"
            "请在 agent_config.json 中添加 \"photoshop_mcp_dir\": \"<photoshop-mcp目录完整路径>\""
        )

    env = dict(os.environ)
    ps_exe = _resolve_ps_exe()
    if ps_exe:
        env["PHOTOSHOP_PATH"] = ps_exe
    env["ANALYTICS_DISABLED"] = "1"
    env["POSTHOG_DISABLED"] = "1"

    kwargs = dict(
        cwd=mcp_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    _proc = subprocess.Popen([node, entry], **kwargs)
    _ready = False
    _tool_cache = None
    _reader_thread = threading.Thread(target=_reader_loop, daemon=True)
    _reader_thread.start()

    # 握手
    result = _request_raw("initialize", {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "sd-webui-agent", "version": "1.0"},
    }, timeout=_INIT_TIMEOUT)
    _notify("notifications/initialized", {})
    _ready = True
    info = (result or {}).get("serverInfo", {})
    print(f"[Agent] photoshop-mcp 已连接: {info.get('name')} v{info.get('version')}")


def _request_raw(method, params, timeout):
    """发送请求并等待响应（调用方需持有 _lock）。"""
    rid = _next_id[0]
    _next_id[0] += 1
    holder = {"event": threading.Event()}
    _pending[rid] = holder
    try:
        payload = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        _proc.stdin.write(payload.encode("utf-8") + b"\n")
        _proc.stdin.flush()
        if not holder["event"].wait(timeout):
            raise MCPError(f"MCP 请求超时({timeout}s): {method}")
    finally:
        _pending.pop(rid, None)
    if holder.get("error"):
        err = holder["error"]
        raise MCPError(f"MCP 错误 {method}: {err.get('message', err)}")
    return holder.get("result")


def _notify(method, params):
    try:
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        _proc.stdin.write(payload.encode("utf-8") + b"\n")
        _proc.stdin.flush()
    except Exception:
        pass


def _request(method, params, timeout=_CALL_TIMEOUT):
    with _lock:
        # 需要重启：从未启动 / 进程已死 / 未握手
        if _proc is None or _proc.poll() is not None or not _ready:
            _start_server()
        return _request_raw(method, params, timeout)


def _list_tools():
    global _tool_cache
    with _lock:
        if _proc is None or _proc.poll() is not None or not _ready:
            _start_server()
        if _tool_cache is None:
            result = _request_raw("tools/list", {}, timeout=30)
            _tool_cache = (result or {}).get("tools", [])
        return _tool_cache


def _mcp_call(name, arguments, timeout=_CALL_TIMEOUT):
    """调用 PS 工具，返回 (data_or_text, image_blocks, is_error)。"""
    result = _request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
    result = result or {}
    texts, images = [], []
    for block in result.get("content", []) or []:
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "image":
            try:
                images.append({
                    "mimeType": block.get("mimeType", "image/jpeg"),
                    "data": base64.b64decode(block.get("data", "")),
                })
            except Exception:
                pass
    text = "\n".join(t for t in texts if t)
    data = text
    if text:
        try:
            data = json.loads(text)
        except Exception:
            pass
    return data, images, bool(result.get("isError"))


def _save_images(images):
    """把 MCP 返回的 base64 图片存成临时文件，供 Agent 聊天展示。"""
    if not images:
        return []
    os.makedirs(_IMG_SAVE_DIR, exist_ok=True)
    paths = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(images):
        ext = "png" if "png" in img["mimeType"] else "jpg"
        path = os.path.join(_IMG_SAVE_DIR, f"ps_{ts}_{i}.{ext}")
        try:
            with open(path, "wb") as f:
                f.write(img["data"])
            paths.append(path)
        except Exception as e:
            print(f"[Agent] photoshop-mcp 图片保存失败: {e}")
    return paths


def _truncate_data(data):
    """控制返回给 LLM 的体积。"""
    if isinstance(data, (dict, list)):
        s = json.dumps(data, ensure_ascii=False, default=str)
        if len(s) > _MAX_JSON_TEXT:
            return {"_truncated": True, "_note": "结果过大已截断", "preview": s[:_MAX_JSON_TEXT]}
        return data
    s = str(data)
    if len(s) > _MAX_RAW_TEXT:
        return s[:_MAX_RAW_TEXT] + f"\n...[已截断，共 {len(s)} 字符]"
    return s


# =============================================================================
# 4 个桥接工具
# =============================================================================

@agent_tool(
    name="ps_state",
    description=(
        "获取本机 Photoshop 当前状态（已连接的 PS 版本、打开的文档、当前图层/选区）。"
        "对 Photoshop 做任何操作之前，建议先调用此工具确认 PS 已打开且连接正常。"
        "返回 error 时说明 Photoshop 未启动或 photoshop-mcp 不可用。"
    ),
    parameters={"type": "object", "properties": {}},
)
def ps_state():
    try:
        data, images, is_error = _mcp_call("photoshop_get_state", {}, timeout=60)
    except MCPError as e:
        return {"status": "error", "error": str(e),
                "hint": "请确认：1) Photoshop 2026 已打开  2) photoshop-mcp 目录正确  3) node 可用"}
    if is_error:
        return {"status": "error", "error": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}
    return {"status": "success", "data": _truncate_data(data)}


@agent_tool(
    name="ps_list_tools",
    description=(
        "搜索 Photoshop 2026 的 MCP 工具库（130 个工具：描边、选区、图层、文字、滤镜、"
        "颜色调整、生成式填充/移除/扩展、智能对象、动作脚本、导出、背景移除配方等）。"
        "不确定用哪个工具时，先用关键词搜索，例如：描边/stroke、文字/text、背景/background、"
        "模糊/blur、导出/export、抠图/subject。不带关键词则返回全部工具名列表。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "搜索关键词（中英文均可，可多个用空格分隔）。留空=返回全部工具名。",
            },
        },
    },
)
def ps_list_tools(keyword: str = ""):
    try:
        tools = _list_tools()
    except MCPError as e:
        return {"status": "error", "error": str(e)}
    kw = (keyword or "").strip().lower()
    if not kw:
        return {
            "status": "success",
            "total": len(tools),
            "hint": "请用 keyword 再次调用查看具体工具的描述和参数",
            "tools": [{"name": t["name"]} for t in tools],
        }
    tokens = kw.split()
    matches = []
    for t in tools:
        name = t.get("name", "")
        desc = (t.get("description") or "").lower()
        if all(tok in name.lower() or tok in desc for tok in tokens):
            matches.append(t)
    if not matches:
        return {"status": "success", "total": len(tools), "matches": 0,
                "hint": "没有匹配的工具，换个关键词试试（如 stroke/描边/text/导出）"}
    out = []
    for t in matches[:25]:
        item = {
            "name": t.get("name"),
            "description": (t.get("description") or "")[:400],
            "parameters": t.get("inputSchema", {}),
        }
        out.append(item)
    return {"status": "success", "matches": len(matches), "tools": out,
            "note": "确认参数后调用 ps_call 执行" if len(matches) <= 25 else "匹配过多，请细化关键词"}


@agent_tool(
    name="ps_call",
    description=(
        "调用 Photoshop 2026 的指定 MCP 工具。tool 填工具名（先通过 ps_list_tools 查到），"
        "arguments 填 JSON 字符串形式的参数（严格对照该工具 parameters 中的 schema，"
        "例如 {\"width\": 4, \"color\": {\"r\": 0, \"g\": 0, \"b\": 0}}）。"
        "描边=photoshop_stroke_selection，打开图片=photoshop_open_image，"
        "选主体=photoshop_select_subject，存文件=photoshop_save_document/export_as。"
        "完成视觉类操作后应调用 ps_preview 查看效果再向用户汇报。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "tool": {"type": "string", "description": "工具名，如 photoshop_stroke_selection"},
            "arguments": {
                "type": "string",
                "description": "JSON 字符串参数，如 {\"width\": 4}；无参数传 {}",
            },
        },
        "required": ["tool"],
    },
)
def ps_call(tool: str, arguments: str = "{}"):
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return {"status": "error", "error": f"arguments 不是合法 JSON: {arguments[:200]}"}
    if not isinstance(args, dict):
        return {"status": "error", "error": "arguments 必须是 JSON 对象"}
    try:
        data, images, is_error = _mcp_call(tool, args, timeout=_CALL_TIMEOUT)
    except MCPError as e:
        return {"status": "error", "tool": tool, "error": str(e)}
    img_paths = _save_images(images)
    info = {
        "status": "error" if is_error else "success",
        "tool": tool,
        "data": _truncate_data(data),
    }
    if is_error:
        info["hint"] = "PS 工具执行失败：检查参数是否符合 schema，或先用 ps_state 确认 PS 状态"
    if img_paths:
        return img_paths, info
    return info


@agent_tool(
    name="ps_preview",
    description=(
        "获取当前 Photoshop 文档的预览图（JPEG），用于向用户展示/验证视觉效果。"
        "完成描边、调色、合成等视觉操作后、向用户汇报成功之前，必须先调用本工具拿到结果图。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "max_dimension_px": {"type": "integer", "description": "预览图最长边像素，默认 1024"},
        },
    },
)
def ps_preview(max_dimension_px: int = 1024):
    try:
        data, images, is_error = _mcp_call(
            "photoshop_get_preview", {"max_dimension_px": int(max_dimension_px or 1024)}, timeout=120
        )
    except MCPError as e:
        return {"status": "error", "error": str(e),
                "hint": "确认 Photoshop 已打开且有活动文档（ps_state 可查看）"}
    # 兼容两种返回：直接 image content block，或 JSON 文本里带 base64/path
    if not images and isinstance(data, dict):
        b64 = data.get("data") or data.get("base64") or data.get("image")
        if isinstance(b64, str) and len(b64) > 100:
            try:
                images.append({"mimeType": data.get("mimeType", "image/jpeg"),
                               "data": base64.b64decode(b64)})
            except Exception:
                pass
        p = data.get("path") or data.get("file")
        if not images and isinstance(p, str) and os.path.isfile(p):
            return [p], {"status": "success", "note": "预览图（PS 本地文件）"}
    img_paths = _save_images(images)
    if img_paths:
        return img_paths, {"status": "success", "note": "当前文档预览图"}
    if is_error:
        return {"status": "error", "error": data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}
    return {"status": "success", "data": _truncate_data(data)}


# =============================================================================
# 模块自检（调试用）: python -c "from scripts import photoshop_mcp; photoshop_mcp._self_test()"
# =============================================================================

def _self_test():
    print("== ps_state ==")
    print(json.dumps(ps_state(), ensure_ascii=False)[:800])
    print("== ps_list_tools(stroke) ==")
    r = ps_list_tools("stroke")
    print(json.dumps(r, ensure_ascii=False)[:1200])
    print("== ps_preview ==")
    r = ps_preview(512)
    if isinstance(r, tuple):
        print("images:", r[0], "info:", r[1])
    else:
        print(json.dumps(r, ensure_ascii=False)[:600])
