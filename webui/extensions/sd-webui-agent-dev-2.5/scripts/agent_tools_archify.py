# =============================================================================
# Agent Tools — Archify 图表工具（generate_diagram / validate_diagram）
#
# 通过扩展内置的自包含运行时 <EXT>/runtime/archify 调用 Node CLI：
#   node bin/archify.mjs validate <type> <spec.json> --quality <q> --json
#   node bin/archify.mjs deliver  <type> <spec.json> <out.html> --quality <q> --json
#
# 用法约定（与 skills/archify/SKILL.md 一致）：
#   - spec 参数：JSON 字符串（内联），或 .json 文件路径（相对路径基于 Forge Neo
#     整合包根目录解析，绝对路径直接使用）
#   - 交付产物写入 <EXT>/outputs/diagrams/<时间戳>-<type>.html
#   - Gradio 界面无法内嵌渲染 HTML，返回绝对路径由用户用浏览器打开
# =============================================================================

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

DIAGRAM_TYPES = ["architecture", "workflow", "sequence", "dataflow", "lifecycle"]
QUALITY_LEVELS = ["standard", "showcase"]


def _get_extension_root():
    """返回 sd-webui-agent-dev-2.4 扩展根目录（scripts/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_archify_root():
    """返回内置 archify 运行时目录 <EXT>/runtime/archify。"""
    return os.path.join(_get_extension_root(), "runtime", "archify")


def _get_node():
    """定位 Node 可执行文件；找不到返回 None。"""
    node = shutil.which("node")
    if node:
        return node
    # 常见安装位置回退（含 Forge Neo 整合包内置 system/node）
    candidates = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, "Programs", "nodejs", "node.exe"))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_get_extension_root()))))
    candidates += [
        os.path.join(base_dir, "system", "node", "node.exe"),
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _parse_json_tail(stdout):
    """从 CLI 输出中截取最后一个可解析的 JSON 对象/数组。"""
    text = (stdout or "").strip()
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        try:
            return json.loads(text[start:])
        except (ValueError, TypeError):
            start = text.find("{", start + 1)
    return None


def _resolve_spec(spec):
    """把 spec 参数解析为规格文件路径。

    支持：
      1. JSON 字符串（以 { 或 [ 开头）→ 写入临时文件
      2. 绝对路径 / 相对路径（相对路径基于 Forge Neo 整合包根目录）→ 直接使用
    返回 (path, error)。
    """
    spec_str = str(spec or "").strip()
    if not spec_str:
        return None, "spec 不能为空：请提供 JSON 规格字符串或 .json 文件路径"
    if spec_str.startswith(("{", "[")):
        try:
            data = json.loads(spec_str)
        except ValueError as exc:
            return None, f"spec 不是合法 JSON: {exc}"
        ext_root = _get_extension_root()
        out_dir = os.path.join(ext_root, "outputs", "diagrams")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{stamp}_inline.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return path, None
    # 文件路径：绝对直接用；相对路径依次尝试整合包根、扩展根两个基准
    if os.path.isabs(spec_str):
        path = os.path.realpath(spec_str)
    else:
        path = None
        for base in (_get_workspace_root_fallback(), _get_extension_root()):
            candidate = os.path.realpath(os.path.join(base, spec_str))
            if os.path.isfile(candidate):
                path = candidate
                break
        if path is None:
            path = os.path.realpath(os.path.join(_get_workspace_root_fallback(), spec_str))
    if not os.path.isfile(path):
        return None, f"规格文件不存在: {spec_str}"
    return path, None


def _get_workspace_root_fallback():
    """Forge Neo 整合包根目录：优先复用 agent_tools_workspace，不可用时向上探测。"""
    try:
        from scripts.agent_tools_workspace import _get_workspace_root
        return _get_workspace_root()
    except Exception:
        import re as _re
        current = _get_extension_root()
        for _ in range(8):
            parent = os.path.dirname(current)
            if parent == current:
                break
            if _re.match(r"^sd-webui-forge-neo(?:[-_].*)?$", os.path.basename(parent).lower()):
                return parent
            if os.path.isdir(os.path.join(parent, "webui")) and os.path.isdir(os.path.join(parent, "webui", "extensions")):
                return parent
            current = parent
        return os.path.dirname(os.path.dirname(os.path.dirname(_get_extension_root())))


def _run_archify(args, timeout=240):
    """在 archify 运行时目录执行 node CLI，返回 (rc, json_or_text, raw)。"""
    archify_root = _get_archify_root()
    entry = os.path.join(archify_root, "bin", "archify.mjs")
    node = _get_node()
    if not node:
        return None, "找不到 node 可执行文件：archify 运行时需要在 PATH 中提供 Node.js (>=18)", ""
    if not os.path.isfile(entry):
        return None, f"archify 运行时缺失: {entry}", ""
    try:
        completed = subprocess.run(
            [node, entry] + list(args),
            cwd=archify_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        raw = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        return completed.returncode, _parse_json_tail(completed.stdout) or raw[-8000:], raw[-8000:]
    except subprocess.TimeoutExpired:
        return None, f"archify 命令超时（超过 {timeout} 秒）", ""
    except Exception as exc:
        return None, f"archify 调用失败: {exc}", ""


def _validate_payload(dg_type, spec, quality):
    """共用：解析参数 + validate。返回 (payload_dict, error_dict)。"""
    dg_type = str(dg_type or "").strip().lower()
    if dg_type not in DIAGRAM_TYPES:
        return None, {"status": "error", "error": f"不支持的图类型: {dg_type}，可选: {DIAGRAM_TYPES}"}
    quality = str(quality or "showcase").strip().lower()
    if quality not in QUALITY_LEVELS:
        quality = "showcase"
    spec_path, err = _resolve_spec(spec)
    if err:
        return None, {"status": "error", "error": err}
    rc, payload, raw = _run_archify(
        ["validate", dg_type, os.path.abspath(spec_path), "--quality", quality, "--json"]
    )
    if rc is None:
        return None, {"status": "error", "error": payload, "detail": raw}
    result = {
        "status": "success" if rc == 0 else "validation_failed",
        "type": dg_type,
        "quality": quality,
        "spec_file": spec_path,
        "returncode": rc,
        "receipt": payload if isinstance(payload, (dict, list)) else str(payload)[:8000],
    }
    if rc != 0:
        result["hint"] = "请按回执中诊断出的 subject/evidence/supportedFixes 逐项修复后重新 validate；连续两轮无法降低错误数时停止并如实报告。"
    return result, None


def validate_diagram_tool(dg_type, spec, quality="showcase"):
    """校验 Archify 图表 JSON 规格（不产出 HTML）。

    用于创作/修复迭代：写规格 → validate → 按诊断修复 → 再 validate，
    通过后调用 generate_diagram 交付。

    参数:
        dg_type: 图类型 architecture | workflow | sequence | dataflow | lifecycle
        spec: JSON 规格字符串，或 .json 文件路径
        quality: standard | showcase（默认 showcase，要求 9 项检查全过）
    """
    try:
        result, error = _validate_payload(dg_type, spec, quality)
        if error:
            return error
        return result
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def generate_diagram_tool(dg_type, spec, quality="showcase", title=""):
    """把 Archify JSON 规格渲染成交互式 HTML 图表（交付级校验）。

    内部执行 deliver（冻结规格 + 渲染 + 检查 + 原子提交）。
    成功时返回 HTML 的绝对路径，请用浏览器打开查看；聊天界面无法内嵌渲染 HTML。

    参数:
        dg_type: 图类型 architecture | workflow | sequence | dataflow | lifecycle
        spec: JSON 规格字符串，或 .json 文件路径
        quality: standard | showcase（默认 showcase）
        title: 可选，用于产物文件名（仅允许字母数字-_ 与点号），默认用类型名
    """
    try:
        dg_type = str(dg_type or "").strip().lower()
        if dg_type not in DIAGRAM_TYPES:
            return {"status": "error", "error": f"不支持的图类型: {dg_type}，可选: {DIAGRAM_TYPES}"}
        quality = str(quality or "showcase").strip().lower()
        if quality not in QUALITY_LEVELS:
            quality = "showcase"

        spec_path, err = _resolve_spec(spec)
        if err:
            return {"status": "error", "error": err}

        safe_title = re.sub(r"[^0-9A-Za-z_\-.]", "", str(title or "").strip())[:60]
        if not safe_title:
            safe_title = dg_type
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_get_extension_root(), "outputs", "diagrams")
        os.makedirs(out_dir, exist_ok=True)
        out_html = os.path.join(out_dir, f"{stamp}-{safe_title}.{dg_type}.html")

        rc, payload, raw = _run_archify(
            ["deliver", dg_type, os.path.abspath(spec_path), os.path.abspath(out_html), "--quality", quality, "--json"]
        )
        if rc is None:
            return {"status": "error", "error": payload, "detail": raw}

        if rc != 0 or not os.path.isfile(out_html):
            return {
                "status": "error",
                "type": dg_type,
                "quality": quality,
                "spec_file": spec_path,
                "returncode": rc,
                "receipt": payload if isinstance(payload, (dict, list)) else str(payload)[:8000],
                "hint": "交付失败（非零退出）绝不能描述为成功；请修复诊断后重试，旧产物保持不变。",
            }

        return {
            "status": "success",
            "type": dg_type,
            "quality": quality,
            "html_path": out_html,
            "spec_file": spec_path,
            "returncode": rc,
            "receipt": payload if isinstance(payload, (dict, list)) else str(payload)[:8000],
            "instruction": "请把 html_path 告诉用户并用浏览器打开；不要把命令非零退出描述为成功，也不要声称未做的视觉审查。",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
