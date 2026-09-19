# =============================================================================
# Agent Memory — 轻量级跨会话记忆系统（v2）
# 参考 TencentDB Agent Memory 的 LLM 抽取 + 检索 + 注入架构
# 单模块：JSON 文件存储 + 关键词检索 + LLM 自动抽取
#
# v2 增强：
#   - recent_turns：滚动保存最近几轮 用户/助手 摘要，WebUI 重启后仍能"接着上次继续"
#   - qa 类记忆：记住"请教过的问题 + 回答要点"，重复提问命中后可直接依记忆作答（省 token）
#   - prepare()：一次检索，同时产出 ①注入 system 的 prompt 文本 ②聊天可见块的原始数据
#   - 管理：list / delete(by id / by keyword) / clear，供工具与 UI 记忆库面板调用
# =============================================================================

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from typing import Optional

# =============================================================================
# 常量
# =============================================================================
EXT_DIR = Path(__file__).parent.parent
DATA_DIR = EXT_DIR / "data"
MEMORY_PATH = DATA_DIR / "agent_memory.json"

MAX_MEMORIES_INJECTED = 10
MAX_MEMORIES_STORED = 200
MAX_RECENT_TURNS = 6          # 滚动保存的最近轮数（跨重启续作上下文）
_FILE_LOCK = threading.Lock()

_CATEGORY_TAG = {
    "preference": "偏好",
    "fact": "事实",
    "decision": "决策",
    "style": "风格",
    "workflow": "工作流",
    "qa": "问答",
}

# 各类型 content 长度上限（qa 的问题要点可略长）
_CATEGORY_CAP = {
    "preference": 50,
    "fact": 50,
    "decision": 50,
    "style": 50,
    "workflow": 50,
    "qa": 120,
}
_ANSWER_CAP = 300             # qa 记忆的回答要点上限

# 停用词
_STOPWORDS = frozenset(
    "的 了 是 在 和 我 你 他 她 它 也 都 就 还 又 不 没 有 这 那 们 "
    "个 么 呢 吧 啊 呀 哈 哦 嗯 嘛 哎 什么 怎么 为什么 哪里 哪个 "
    "a an the is are was were be been being to of in on at by for "
    "with about against between into through during before after "
    "above below from up down out off over under again further then "
    "once here there all any both each few more most other some such "
    "no nor not only own same so than too very can will just don "
    "should now i you he she it we they me him her us them my your "
    "his its our their what which who whom this that these those am "
    "do does did having has had would could should may might must shall"
    .split()
)


# =============================================================================
# MemoryStore — 文件读写与 CRUD
# =============================================================================

class MemoryStore:
    """记忆文件读写，所有写操作加锁。"""

    def __init__(self, path: Path = MEMORY_PATH):
        self._path = path

    def _load(self) -> dict:
        """加载记忆文件，不存在或损坏返回空骨架（含 recent_turns）。"""
        try:
            if not self._path.is_file():
                return {"version": 2, "memories": [], "recent_turns": []}
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 2, "memories": [], "recent_turns": []}
            data.setdefault("memories", [])
            data.setdefault("recent_turns", [])
            data["version"] = 2  # v1 旧文件升级为 v2 结构标记
            return data
        except Exception as e:
            print(f"[Agent][Memory] 加载失败，使用空记忆: {e}")
            return {"version": 2, "memories": [], "recent_turns": []}

    def _save(self, data: dict) -> None:
        """原子写：先写 .tmp 再 rename。"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(self._path))

    def list_all(self) -> list[dict]:
        with _FILE_LOCK:
            return self._load().get("memories", [])

    def upsert_batch(self, operations: list[dict]) -> None:
        """批量 upsert：处理 add/update/skip 操作。qa 类型额外存 answer。"""
        with _FILE_LOCK:
            data = self._load()
            memories: list[dict] = data.get("memories", [])
            index = {m["id"]: i for i, m in enumerate(memories) if "id" in m}

            now = _now_iso()
            for op in operations:
                action = op.get("action", "skip")
                if action == "skip":
                    continue

                category = op.get("category", "fact")
                if category not in _CATEGORY_TAG:
                    category = "fact"

                content = (op.get("content") or "").strip()
                if not content:
                    continue
                # 按类型截断
                cap = _CATEGORY_CAP.get(category, 50)
                content = content[:cap]

                answer = (op.get("answer") or "").strip()[:_ANSWER_CAP] if category == "qa" else ""

                keywords = op.get("keywords") or []
                keywords = [k.strip() for k in keywords if k and k.strip()][:5]

                target_id = op.get("target_id")

                if action == "update" and target_id and target_id in index:
                    m = memories[index[target_id]]
                    m["content"] = content
                    if category == "qa":
                        m["answer"] = answer or m.get("answer", "")
                    old_kw = set(m.get("keywords") or [])
                    old_kw.update(keywords)
                    m["keywords"] = list(old_kw)[:5]
                    m["category"] = category
                    m["updated_at"] = now
                else:
                    # 新增（add 或 update 未命中 target 都走新增）
                    # 兜底去重：Jaccard 相似度 >= 0.6 则转为更新
                    dup_id = _find_duplicate(content, memories)
                    if dup_id:
                        m = next(m for m in memories if m["id"] == dup_id)
                        m["content"] = content
                        if category == "qa":
                            m["answer"] = answer or m.get("answer", "")
                        old_kw = set(m.get("keywords") or [])
                        old_kw.update(keywords)
                        m["keywords"] = list(old_kw)[:5]
                        m["category"] = category
                        m["updated_at"] = now
                    else:
                        new_mem = {
                            "id": f"mem_{uuid.uuid4().hex[:6]}",
                            "content": content,
                            "category": category,
                            "keywords": keywords,
                            "created_at": now,
                            "updated_at": now,
                            "source": op.get("source", "webui"),
                            "hit_count": 0,
                        }
                        if category == "qa" and answer:
                            new_mem["answer"] = answer
                        memories.append(new_mem)
                        index[new_mem["id"]] = len(memories) - 1

            # 淘汰超额记忆
            if len(memories) > MAX_MEMORIES_STORED:
                memories.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
                memories = memories[:MAX_MEMORIES_STORED]

            data["memories"] = memories
            self._save(data)


# =============================================================================
# 检索
# =============================================================================

def _tokenize(text: str) -> set[str]:
    """中文按字、英文按词分 token，去停用词。"""
    text = (text or "").lower()
    tokens = set(re.findall(r"[\u4e00-\u9fff]|[a-z_]{2,}", text))
    return tokens - _STOPWORDS


def _find_duplicate(new_content: str, memories: list[dict]) -> Optional[str]:
    """Jaccard 相似度 >= 0.6 返回已有记忆 id。"""
    new_tokens = _tokenize(new_content)
    if not new_tokens:
        return None
    for m in memories:
        old_tokens = _tokenize(m.get("content", ""))
        if not old_tokens:
            continue
        intersection = new_tokens & old_tokens
        union = new_tokens | old_tokens
        if len(union) > 0 and len(intersection) / len(union) >= 0.6:
            return m["id"]
    return None


def _retrieve(memories: list[dict], user_message: str, top_k: int = MAX_MEMORIES_INJECTED) -> list[dict]:
    """关键词匹配检索（qa 记忆的 content+answer+keywords 都参与匹配）。"""
    query_tokens = _tokenize(user_message)
    if not query_tokens:
        return []

    scored = []
    for m in memories:
        mem_text = m.get("content", "") + " " + (m.get("answer") or "") + " " + " ".join(m.get("keywords") or [])
        mem_tokens = _tokenize(mem_text)
        score = len(query_tokens & mem_tokens)
        if score > 0:
            scored.append((score, m.get("hit_count", 0), m))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [m for _, _, m in scored[:top_k]]


# =============================================================================
# 注入文本（prompt）与可见块数据
# =============================================================================

def _format_recent_context(turns: list[dict]) -> str:
    """最近对话 → 续作上下文（跨重启后让模型知道"上次在做什么"）。"""
    if not turns:
        return ""
    lines = ["=== 📌 最近对话（若与当前问题相关，可接着上次继续，不必让用户从头描述） ==="]
    for t in turns[-3:]:
        u = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if not u and not a:
            continue
        lines.append(f"- 用户（{t.get('ts', '')}）: {u}")
        if a:
            lines.append(f"  助手: {a}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_fact_block(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = ["=== 📝 记忆（来自过往对话，仅供参考） ==="]
    for i, m in enumerate(memories, 1):
        tag = _CATEGORY_TAG.get(m.get("category", "fact"), "事实")
        lines.append(f"{i}. [{tag}] {m.get('content', '')}")
    return "\n".join(lines)


def _format_qa_block(memories: list[dict]) -> str:
    """命中的问答记忆：给出问题 + 记忆回答，提示模型直接依记忆作答（省 token）。"""
    if not memories:
        return ""
    lines = ["=== 🔁 你之前问过类似问题（已有记忆回答，若一致请直接依据记忆简洁作答，不要重新长篇思考） ==="]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. 问: {m.get('content', '')}")
        if m.get("answer"):
            lines.append(f"   记忆回答: {m['answer']}")
    return "\n".join(lines)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# =============================================================================
# LLM 抽取
# =============================================================================

EXTRACT_SYSTEM = """你是记忆抽取器。分析给定的用户消息和助手回复，提取值得长期记住的事实/偏好/决策/风格/工作流约定/问答。

规则：
1. 只提取跨会话有用的记忆，不提取一次性的临时指令、生图参数、闲聊。
2. 普通记忆（preference/fact/decision/style/workflow）：content 一句话不超过 50 字。
3. 问答记忆（qa）：当用户"请教/咨询"了一个有明确答案的问题时，提取该问题要点（content，不超过 120 字）
   并把助手给出的关键结论精炼为 answer（不超过 300 字）。便于以后重复提问时直接依据记忆作答、省 token。
   生图/改图等纯操作类请求不要记成 qa。
4. 对比"已有记忆"，判断每条新记忆的动作：
   - add: 全新记忆
   - update: 与某条已有记忆语义重叠/需要补充修正（填 target_id）
   - skip: 不值得记
5. category 取值: preference/fact/decision/style/workflow/qa
6. keywords 给 2-5 个用于检索的关键词（中文词或英文词）

严格输出 JSON，不要任何解释：
{"operations": [{"action":"add|update|skip","target_id":"mem_xxx 或 null","content":"...","answer":"qa 才填，否则空","category":"...","keywords":["..."]}]}"""


def _call_extract_llm(user_message: str, assistant_reply: str, existing_memories: list[dict], cfg: dict) -> dict | None:
    """调用 LLM 抽取记忆，返回解析后的 dict 或 None。"""
    try:
        from openai import OpenAI

        api_key = cfg.get("api_key") or "empty"
        base_url = cfg.get("base_url", "")
        model = cfg.get("model", "")

        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=2)

        existing_lines = []
        for m in existing_memories:
            existing_lines.append(f"- [{m['id']}] {m.get('content', '')}")
        existing_text = "\n".join(existing_lines) if existing_lines else "（暂无已有记忆）"

        user_content = (
            f"用户消息: {user_message[:1000]}\n\n"
            f"助手回复: {assistant_reply[:2000]}\n\n"
            f"已有记忆:\n{existing_text}"
        )

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        # Qwen3 系关闭 thinking
        extra_body = None
        if "qwen3" in model.lower() or "qwen" in model.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": 800,
            "stream": False,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                return json.loads(match.group())
            return None

    except Exception as e:
        print(f"[Agent][Memory] LLM 抽取调用失败: {e}")
        return None


# =============================================================================
# AgentMemory — 高层门面
# =============================================================================

class AgentMemory:
    """记忆系统门面，供 WebUI Agent 和 PS 插件共用。"""

    def __init__(self):
        self.store = MemoryStore()

    # ---- 检索 + 注入（一次调用，产出 prompt 文本 + 可见块原始数据） ----
    def prepare(self, user_message: str, cfg: dict, source: str = "webui",
                include_recent_context: bool = False) -> dict:
        """检索相关记忆。返回:
        {
            "prompt_block": str   注入 system 的文本（含 最近对话/事实/问答 各段）
            "visible_block": str  （保留字段，聊天可见块由 UI/Chat 侧用 hits 渲染）
            "hits": [ {id, content, category, answer, ...} ]  命中的记忆
        }
        """
        result: dict = {"prompt_block": "", "visible_block": "", "hits": []}
        if not cfg.get("memory_enabled"):
            return result
        try:
            with _FILE_LOCK:
                data = self.store._load()
            memories = data.get("memories", [])
            hits = _retrieve(memories, user_message) if memories else []
            result["hits"] = hits
            if hits:
                self._bump_hits([m["id"] for m in hits if "id" in m])

            sections = []
            if include_recent_context:
                rc_block = _format_recent_context(data.get("recent_turns", []))
                if rc_block:
                    sections.append(rc_block)
            fact_hits = [h for h in hits if h.get("category") != "qa"]
            qa_hits = [h for h in hits if h.get("category") == "qa"]
            fact_block = _format_fact_block(fact_hits)
            if fact_block:
                sections.append(fact_block)
            qa_block = _format_qa_block(qa_hits)
            if qa_block:
                sections.append(qa_block)
            if sections:
                result["prompt_block"] = "\n\n".join(sections)
        except Exception as e:
            print(f"[Agent][Memory] 检索注入失败: {e}")
        return result

    def _bump_hits(self, ids: list[str]) -> None:
        """增量更新 hit_count（不影响主流程）。"""
        try:
            with _FILE_LOCK:
                data = self.store._load()
                for m in data.get("memories", []):
                    if m.get("id") in ids:
                        m["hit_count"] = m.get("hit_count", 0) + 1
                self.store._save(data)
        except Exception:
            pass

    # ---- 记录最近一轮（跨重启续作） ----
    def record_turn(self, user_message: str, assistant_reply: str, cfg: dict, source: str = "webui") -> None:
        """把最近一轮 用户/助手 摘要滚动写入 recent_turns（轻量、不调 LLM）。"""
        if not cfg.get("memory_enabled"):
            return
        try:
            user_msg = re.sub(r"\s+", " ", (user_message or "")).strip()
            reply = re.sub(r"\s+", " ", (assistant_reply or "")).strip()
            if not user_msg and not reply:
                return
            with _FILE_LOCK:
                data = self.store._load()
                turns = data.get("recent_turns", [])
                turns.append({
                    "ts": _now_iso(),
                    "user": user_msg[:120],
                    "assistant": reply[:240],
                    "source": source,
                })
                data["recent_turns"] = turns[-MAX_RECENT_TURNS:]
                self.store._save(data)
        except Exception as e:
            print(f"[Agent][Memory] record_turn 失败: {e}")

    # ---- LLM 抽取存储 ----
    def extract_and_store(self, user_message: str, assistant_reply: str, cfg: dict, source: str = "webui") -> None:
        """调 LLM 抽取记忆并存储。同步调用，由上层包在 daemon 线程里。"""
        if not cfg.get("memory_enabled"):
            return
        if not user_message.strip() or not assistant_reply.strip():
            return

        try:
            existing = self.store.list_all()
            result = _call_extract_llm(user_message, assistant_reply, existing, cfg)
            if not result:
                return

            operations = result.get("operations") or []
            if not operations:
                return

            for op in operations:
                op["source"] = source

            self.store.upsert_batch(operations)
            added = sum(1 for o in operations if o.get("action") in ("add", "update"))
            if added:
                print(f"[Agent][Memory] 抽取完成: {added} 条记忆已存储/更新 (source={source})")
        except Exception as e:
            print(f"[Agent][Memory] 抽取存储失败: {e}")

    # ---- 管理：列表 / 删除 / 清空 ----
    def list_memories(self, keyword: Optional[str] = None) -> list[dict]:
        with _FILE_LOCK:
            mems = list(self.store._load().get("memories", []))
        kw = (keyword or "").strip()
        if kw:
            kw_tokens = _tokenize(kw)

            def _match(m: dict) -> bool:
                text = m.get("content", "") + " " + (m.get("answer") or "") + " " + " ".join(m.get("keywords") or [])
                if kw.lower() in text.lower():
                    return True
                return bool(kw_tokens & _tokenize(text))
            mems = [m for m in mems if _match(m)]
        return mems

    def delete_memory_by_id(self, mem_id: str) -> int:
        mem_id = (mem_id or "").strip()
        if not mem_id:
            return 0
        with _FILE_LOCK:
            data = self.store._load()
            before = len(data.get("memories", []))
            data["memories"] = [m for m in data.get("memories", []) if m.get("id") != mem_id]
            deleted = before - len(data["memories"])
            if deleted:
                self.store._save(data)
            return deleted

    def delete_memory_by_keyword(self, keyword: str) -> int:
        keyword = (keyword or "").strip()
        if not keyword:
            return 0
        kw_tokens = _tokenize(keyword)
        with _FILE_LOCK:
            data = self.store._load()
            mems = data.get("memories", [])

            def _match(m: dict) -> bool:
                text = m.get("content", "") + " " + (m.get("answer") or "") + " " + " ".join(m.get("keywords") or [])
                if keyword.lower() in text.lower():
                    return True
                return bool(kw_tokens & _tokenize(text))
            keep = [m for m in mems if not _match(m)]
            deleted = len(mems) - len(keep)
            if deleted:
                data["memories"] = keep
                self.store._save(data)
            return deleted

    def clear_all_memories(self) -> int:
        with _FILE_LOCK:
            data = self.store._load()
            deleted = len(data.get("memories", []))
            if deleted:
                data["memories"] = []
                self.store._save(data)
            return deleted


# =============================================================================
# 模块级单例 + 便捷函数
# =============================================================================

_default = AgentMemory()


def prepare_memory(user_message: str, cfg: dict, source: str = "webui",
                   include_recent_context: bool = False) -> dict:
    """检索并返回 {prompt_block, visible_block, hits}。供 chat 构建消息时调用。"""
    return _default.prepare(user_message, cfg, source, include_recent_context)


def build_memory_block(user_message: str, cfg: dict, source: str = "webui") -> str:
    """（兼容旧接口）检索并返回记忆注入文本。"""
    return _default.prepare(user_message, cfg, source, include_recent_context=False).get("prompt_block", "")


def record_turn(user_message: str, assistant_reply: str, cfg: dict, source: str = "webui") -> None:
    """持久化最近一轮对话（跨重启续作）。"""
    _default.record_turn(user_message, assistant_reply, cfg, source)


def maybe_extract_and_store(user_message: str, assistant_reply: str, cfg: dict, source: str = "webui") -> None:
    """后台 daemon 线程触发记忆抽取，绝不阻塞聊天流。"""
    if not cfg.get("memory_enabled"):
        return
    if not user_message.strip() or not assistant_reply.strip():
        return

    def _safe_extract():
        try:
            _default.extract_and_store(user_message, assistant_reply, cfg, source)
        except Exception as e:
            print(f"[Agent][Memory] 后台抽取异常: {e}")

    t = threading.Thread(target=_safe_extract, daemon=True)
    t.start()


# =============================================================================
# 工具函数（供 function calling 调用：list_memories / delete_memory）
# =============================================================================

def list_memories_tool(keyword: str = "") -> str:
    """列出记忆库（用户问"你记了我什么/我之前问过啥"或管理记忆时调用）。"""
    try:
        mems = _default.list_memories(keyword or None)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
    out = []
    for m in mems:
        item = {
            "id": m.get("id", ""),
            "type": _CATEGORY_TAG.get(m.get("category", "fact"), m.get("category", "fact")),
            "content": m.get("content", ""),
            "updated_at": m.get("updated_at", ""),
        }
        if m.get("answer"):
            item["answer"] = m.get("answer")
        out.append(item)
    return json.dumps({"status": "success", "count": len(out), "memories": out}, ensure_ascii=False)


def delete_memory_tool(mem_id: str = "", keyword: str = "") -> str:
    """删除记忆。优先按 mem_id 精确删除；否则按 keyword 删除所有匹配。"""
    try:
        if (mem_id or "").strip():
            n = _default.delete_memory_by_id(mem_id)
            return json.dumps({"status": "success", "deleted": n, "mode": "id", "mem_id": (mem_id or "").strip()},
                              ensure_ascii=False)
        if (keyword or "").strip():
            n = _default.delete_memory_by_keyword(keyword)
            return json.dumps({"status": "success", "deleted": n, "mode": "keyword", "keyword": (keyword or "").strip()},
                              ensure_ascii=False)
        return json.dumps({"status": "error", "error": "需要提供 mem_id 或 keyword"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
