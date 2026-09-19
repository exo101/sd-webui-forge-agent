# =============================================================================
# Agent Skills — 通用 Skill 发现 / SKILL.md 解析 / list_skills / read_skill
#
# 约定（参考 archify 等 Agent Skill 生态的 SKILL.md 格式）：
#   <EXT>/skills/<skill-name>/SKILL.md
#   - 文件头部为 YAML frontmatter（--- 包裹），必须包含 name 与 description；
#   - 正文为 Markdown 指令，告诉智能体"何时用、怎么用"。
#
# 智能体侧：
#   - list_skills()     列出所有可用技能（name / description / 路径）
#   - read_skill(name)  读取某个技能的完整 SKILL.md，按其指令行动
# =============================================================================

import os
import re
import traceback


def _get_extension_root():
    """返回 sd-webui-agent-dev-2.4 扩展根目录（scripts/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_skills_dir():
    """返回技能根目录 <EXT>/skills。"""
    return os.path.join(_get_extension_root(), "skills")


def _parse_frontmatter(text):
    """解析 SKILL.md 头部的 YAML frontmatter。

    优先用 PyYAML；不可用时回退到极简解析（支持一层嵌套的 key: value）。
    返回 (dict, body)。无 frontmatter 时返回 ({}, 原文)。
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.S)
    if not match:
        return {}, text
    raw_fm, body = match.group(1), match.group(2)
    data = {}
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(raw_fm)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = _parse_simple_frontmatter(raw_fm)
    return data, body


def _parse_simple_frontmatter(raw):
    """PyYAML 不可用时的极简 frontmatter 解析：顶层 key: value + 一层缩进嵌套。"""
    data = {}
    current_key = None
    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if value:
                data[key] = value
                current_key = None
            else:
                data[key] = {}
                current_key = key
        elif indent > 0 and current_key is not None and ":" in line:
            key, _, value = line.strip().partition(":")
            if isinstance(data.get(current_key), dict):
                data[current_key][key.strip()] = value.strip().strip('"').strip("'")
    return data


def _limit_chars(value, default=12000, maximum=999999):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(500, min(value, maximum))


def discover_skills():
    """扫描 <EXT>/skills/*/SKILL.md，返回技能元数据列表。"""
    skills_dir = _get_skills_dir()
    if not os.path.isdir(skills_dir):
        return []
    skills = []
    for entry in sorted(os.listdir(skills_dir)):
        skill_md = os.path.join(skills_dir, entry, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        try:
            with open(skill_md, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            meta, _body = _parse_frontmatter(text)
            skills.append({
                "name": str(meta.get("name") or entry),
                "description": str(meta.get("description") or "").strip(),
                "path": os.path.relpath(skill_md, _get_extension_root()),
                "absolute_path": skill_md,
                "version": str(meta.get("metadata", {}).get("version", "")) if isinstance(meta.get("metadata"), dict) else "",
            })
        except Exception:
            continue
    return skills


def list_skills_tool():
    """列出智能体当前可用的所有 Skill。

    返回每个技能的 name、description（用途说明）和 SKILL.md 相对路径。
    决定使用某个技能前，先调用 read_skill 读取其完整指令，再按指令行动。
    """
    try:
        skills = discover_skills()
        return {
            "status": "success",
            "count": len(skills),
            "skills": skills,
            "hint": "选中技能后请调用 read_skill(name) 读取完整 SKILL.md 指令，并严格按其步骤使用相应工具。",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}


def read_skill_tool(name, max_chars=24000):
    """读取指定技能的完整 SKILL.md 内容（frontmatter + 正文指令）。

    参数:
        name: 技能名（见 list_skills 返回的 name 字段）
        max_chars: 返回正文最大字符数，默认 24000
    """
    try:
        name = str(name or "").strip().lower()
        if not name:
            return {"status": "error", "error": "请提供技能名"}
        for skill in discover_skills():
            if skill["name"].lower() == name or os.path.basename(os.path.dirname(skill["absolute_path"])).lower() == name:
                max_chars = _limit_chars(max_chars, default=24000, maximum=999999)
                with open(skill["absolute_path"], "r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
                meta, body = _parse_frontmatter(text)
                return {
                    "status": "success",
                    "name": skill["name"],
                    "description": skill["description"],
                    "path": skill["path"],
                    "metadata": meta,
                    "truncated": len(body) > max_chars,
                    "content": body[:max_chars],
                    "instruction": "请严格按上述 SKILL.md 的指令与步骤行动；其中提到的相对路径（如 runtime/archify/...）请通过 read_workspace_file 以扩展根目录为基准读取。",
                }
        return {
            "status": "not_found",
            "name": name,
            "message": f"未找到名为 '{name}' 的技能",
            "hint": "可先调用 list_skills 查看可用技能",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
