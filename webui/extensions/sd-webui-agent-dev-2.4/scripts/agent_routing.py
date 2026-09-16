"""根据当前配置和用户意图选择图像、视频工具。"""

import re

from scripts.agent_config import load_config
from scripts.agent_models import IMAGE_GENERATION_MODELS
from scripts.agent_tools import _select_image_api_key

API_IMAGE_EDIT_MODELS = {
    "Qwen/Qwen-Image-Edit-2511", "FireRedTeam/FireRed-Image-Edit-1.1",
    "banana2", "bananapro", "gpt-image-2",
}


def _configured_image_model():
    return str(load_config().get("image_model") or "").strip()


def _configured_video_model():
    return str(load_config().get("video_model") or "").strip()


def _is_configured_api_image_model(model_id):
    model = str(model_id or "").strip()
    if model in IMAGE_GENERATION_MODELS:
        return True
    cfg = load_config(resolve_local=False)
    custom = cfg.get("custom_models", {}).get("image", {}) if isinstance(cfg.get("custom_models"), dict) else {}
    return any(model in (values or []) for values in custom.values() if isinstance(values, list))


def _is_dreamina_video_model(model_id):
    return str(model_id or "").strip().lower() in {"dreamina-seedance-2-0-hc", "dreamina-seedance-2-5-hc"}


def _is_h3_video_model(model_id):
    return str(model_id or "").strip().lower() == "minimax-h3"


def _extract_requested_api_model(messages):
    for msg in messages:
        if msg.get("role") == "system":
            matches = re.findall(r"用户指定 API 模型：([^。\n]+)", str(msg.get("content", "")))
            if matches:
                return matches[-1].strip()
    return ""


def _has_requested_local_model(messages):
    latest = ""
    for msg in messages:
        if msg.get("role") == "system":
            latest = str(msg.get("content", ""))
    if not latest or ("API 模型" in latest and ("已切换" in latest or "已选择" in latest)):
        return False
    return "用户指定了 @" in latest and "已自动切换到" in latest


def _should_protect_api_image_model(requested_api_model):
    return requested_api_model in API_IMAGE_EDIT_MODELS


def _has_requested_layer_separation(user_instruction):
    text = str(user_instruction or "").lower()
    return any(x in text for x in ("图层分离", "分层", "psd 图层", "psd图层", "@图层分离"))


def _has_requested_trellis_local(user_instruction):
    text = str(user_instruction or "").lower()
    return any(x in text for x in ("@trellis图生3d", "[trellis图生3d]", "@trellis 3d", "[trellis 3d]"))


def _has_requested_keyframe_local(user_instruction):
    text = str(user_instruction or "")
    return "@视频关键帧" in text or "[视频关键帧]" in text


def _coerce_api_image_edit_tool(tool_name, tool_args, requested_api_model, user_instruction):
    if not _should_protect_api_image_model(requested_api_model):
        return tool_name, tool_args
    if tool_name == "api_image_edit":
        args = dict(tool_args or {})
        args.setdefault("model", requested_api_model)
        args.setdefault("instruction", user_instruction)
        return tool_name, args
    if tool_name not in ("remove_background", "edit_image", "change_background", "img2img"):
        return tool_name, tool_args
    args = dict(tool_args or {})
    instruction = args.get("instruction") or args.get("prompt") or user_instruction
    if tool_name == "remove_background" and args.get("bg_color"):
        instruction = f"Change the image background to {args['bg_color']}, preserve the subject exactly."
    return "api_image_edit", {"model": requested_api_model, "instruction": instruction}


def _coerce_generation_tool_by_selected_model(tool_name, tool_args, user_instruction):
    args = dict(tool_args or {})
    image_model, video_model = _configured_image_model(), _configured_video_model()
    cfg = load_config()
    can_use_api = bool(_select_image_api_key(cfg).strip())
    is_api = _is_configured_api_image_model(image_model)
    if not can_use_api and is_api:
        if tool_name == "txt2img":
            return "api_image_generate", {"model": image_model, "prompt": args.get("prompt") or user_instruction}
        if tool_name in ("edit_image", "change_background", "img2img", "api_image_edit"):
            return "api_image_edit", {"model": image_model, "instruction": args.get("instruction") or args.get("prompt") or user_instruction}
    if is_api and tool_name in ("api_image_edit", "api_image_generate"):
        args["model"] = image_model
        return tool_name, args
    if is_api and tool_name == "txt2img":
        prompt = args.get("prompt") or user_instruction
        return ("api_image_generate", {"model": image_model, "prompt": prompt})
    if is_api and tool_name in ("edit_image", "change_background", "img2img"):
        return "api_image_edit", {"model": image_model, "instruction": args.get("instruction") or args.get("prompt") or user_instruction}
    if tool_name in ("h3_video_generate", "dreamina_video_generate"):
        args.update({"prompt": args.get("prompt") or user_instruction, "duration": args.get("duration", 5), "aspect_ratio": args.get("aspect_ratio") or args.get("ratio") or "16:9"})
        if _is_dreamina_video_model(video_model):
            args["model"] = video_model
            return "dreamina_video_generate", args
        if _is_h3_video_model(video_model):
            return "h3_video_generate", args
    return tool_name, args


# =============================================================================
# 🔺 读取上限提升补丁（30000 → 999999 字符）
# 原 scripts/agent_tools.py 里的 read_workspace_file / analyze_document 会把
# max_chars 钳制在 30000，读取大文件时只能看到开头一段。本补丁在运行时把这两个
# 工具替换为支持 999999 字符的实现（安全校验保持一致：仅允许整合包根目录内的路径），
# 并同步更新工具 schema 描述，让模型知道新的上限。
# 回退方法：删除下面这整段补丁代码，重启 WebUI 即可恢复原状。
# =============================================================================

import os

MAX_READ_CHARS = 999999

_PLAIN_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".yaml", ".yml", ".css", ".html", ".htm", ".xml", ".csv", ".tsv", ".log", ".ini",
    ".cfg", ".toml", ".ps1", ".bat", ".cmd", ".sh",
}

_orig_read_workspace_file = None
_orig_analyze_document = None


def _agent_workspace_root():
    """整合包根目录：scripts → 扩展目录 → extensions → webui → 根目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def _agent_resolve_inside(path):
    """把相对路径解析到整合包根目录内，越界或不存在则返回错误。"""
    root = _agent_workspace_root()
    value = str(path or "").strip()
    if not value:
        return None, {"status": "error", "error": "path 不能为空"}
    target = os.path.normpath(value if os.path.isabs(value) else os.path.join(root, value))
    try:
        if os.path.commonpath([root, target]) != root:
            return None, {"status": "error", "error": f"路径越出整合包根目录，已拒绝: {path}"}
    except ValueError:
        return None, {"status": "error", "error": "路径校验失败（跨磁盘），已拒绝"}
    if not os.path.isfile(target):
        return None, {"status": "error", "error": f"文件不存在: {path}"}
    return target, None


def _agent_read_deep(path, max_chars):
    """按 999999 字符上限读取整合包内文本文件，返回与原工具一致的结构。"""
    target, err = _agent_resolve_inside(path)
    if err is not None:
        return err
    try:
        limit = int(max_chars or 12000)
    except (TypeError, ValueError):
        limit = 12000
    limit = max(1, min(limit, MAX_READ_CHARS))
    try:
        with open(target, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return {"status": "error", "error": f"读取失败: {exc}"}
    text = raw.decode("utf-8", errors="replace")
    ext = os.path.splitext(target)[1].lower()
    return {
        "status": "success",
        "path": os.path.relpath(target, _agent_workspace_root()).replace("\\", "/"),
        "file_type": (ext.lstrip(".") or "text"),
        "size_bytes": len(raw),
        "max_chars": limit,
        "truncated": len(text) > limit,
        "content": text[:limit],
    }


def read_workspace_file(path, max_chars=12000):
    """读取整合包根目录内的代码/配置/文本文件，上限 999999 字符。"""
    return _agent_read_deep(path, max_chars)


def analyze_document(path, question="", max_chars=16000):
    """提取并分析文档：纯文本类走 999999 上限，其他格式回退原始实现。"""
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in _PLAIN_TEXT_EXTS:
        data = _agent_read_deep(path, max_chars or 16000)
        if isinstance(data, dict) and data.get("status") == "success":
            data["question"] = question
        return data
    if _orig_analyze_document is None:
        return {"status": "error", "error": "原始 analyze_document 工具不可用，无法解析该格式文档"}
    try:
        return _orig_analyze_document(path, question=question, max_chars=max_chars)
    except Exception as exc:
        return {"status": "error", "error": f"文档解析回退调用失败: {exc}"}


def _install_readcap_patch():
    """把新版读取工具注入 agent_tools.TOOL_FUNCTIONS，并放宽 schema 描述。"""
    global _orig_read_workspace_file, _orig_analyze_document
    try:
        from scripts import agent_tools as _agent_tools
    except Exception as exc:
        print(f"[Agent] 读取上限补丁未安装：无法导入 agent_tools ({exc})")
        return
    tool_functions = getattr(_agent_tools, "TOOL_FUNCTIONS", None)
    if not isinstance(tool_functions, dict):
        print("[Agent] 读取上限补丁未安装：TOOL_FUNCTIONS 不可用")
        return
    if tool_functions.get("read_workspace_file") is not read_workspace_file:
        _orig_read_workspace_file = tool_functions.get("read_workspace_file")
        tool_functions["read_workspace_file"] = read_workspace_file
    if tool_functions.get("analyze_document") is not analyze_document:
        _orig_analyze_document = tool_functions.get("analyze_document")
        tool_functions["analyze_document"] = analyze_document
    descriptions = {
        "read_workspace_file": "最多读取的字符数，默认 12000，最大 999999",
        "analyze_document": "最多提取的字符数，默认 16000，最大 999999",
    }
    for tool in getattr(_agent_tools, "TOOLS", None) or []:
        try:
            func = tool.get("function") or {}
            name = func.get("name")
            if name not in descriptions:
                continue
            prop = (func.get("parameters") or {}).get("properties") or {}
            max_chars_prop = prop.get("max_chars")
            if isinstance(max_chars_prop, dict):
                max_chars_prop["description"] = descriptions[name]
                max_chars_prop["maximum"] = MAX_READ_CHARS
        except Exception:
            continue


try:
    _install_readcap_patch()
except Exception as _readcap_exc:
    print(f"[Agent] ⚠️ 读取上限补丁安装失败: {_readcap_exc}")
