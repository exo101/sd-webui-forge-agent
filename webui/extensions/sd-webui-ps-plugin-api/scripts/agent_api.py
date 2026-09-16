# =============================================================================
# SD WebUI PS Plugin Agent API
# 为 Photoshop UXP 智能体插件提供 REST API 端点
# 直接复用 sd-webui-agent-dev 的完整工具系统（本地模型 + 云端 API）
# 用户在 PS 面板输入自然语言，LLM 自动决策调用哪个工具
# 通过 script_callbacks.on_app_started 注册到 FastAPI
# =============================================================================

import sys
import os
import io
import json
import base64
import time
import queue
import threading
import traceback
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image
from modules import script_callbacks, shared
from modules_forge.main_thread import run_and_wait_result

# 避免导入错误（gradio 可能不可用）—— 必须在函数定义前导入，否则类型注解报错
try:
    import gradio as gr_or_blocks
except ImportError:
    gr_or_blocks = None

# =============================================================================
# 延迟导入 sd-webui-agent-dev 的核心模块
# 关键：不能在顶层导入 agent.py，否则如果 agent_api.py 比 agent.py 先被加载，
# 会触发 agent.py 的第一次执行（注册 UI），WebUI 再正常加载 agent.py 时
# 可能因模块名差异导致第二次执行 → 两个"绘梦智能体助手"标签页
# =============================================================================
_extensions_root = Path(__file__).parent.parent.parent
AGENT_DEV_DIR = _extensions_root / "sd-webui-agent-dev" / "scripts"
if not AGENT_DEV_DIR.is_dir():
    # 兼容带版本号后缀的目录名，例如 sd-webui-agent-dev-2.2
    for _candidate in sorted(_extensions_root.glob("sd-webui-agent-dev-*")):
        if (_candidate / "scripts" / "agent.py").is_file():
            AGENT_DEV_DIR = _candidate / "scripts"
            break

# 确保 sd-webui-agent-dev/scripts 在 sys.path 中
if str(AGENT_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DEV_DIR))

# 延迟加载的变量
_modules_loaded = False
_load_error = None
_TOOLS = []
_TOOL_FUNCTIONS = {}
_REGISTRY_AVAILABLE = False
_get_registered_tools_fn = None
_load_config_fn = None
_get_current_checkpoint_fn = None
_get_system_prompt_fn = None
_execute_tool_fn = None


def _ensure_modules_loaded():
    """延迟加载 sd-webui-agent-dev 模块（首次调用时执行）"""
    global _modules_loaded, _load_error, _TOOLS, _TOOL_FUNCTIONS
    global _REGISTRY_AVAILABLE, _get_registered_tools_fn, _load_config_fn
    global _get_current_checkpoint_fn, _get_system_prompt_fn, _execute_tool_fn

    if _modules_loaded:
        return

    # 确保 sd-webui-agent-dev/scripts 在 sys.path 中（WebUI 可能重置 sys.path）
    if str(AGENT_DEV_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DEV_DIR))

    try:
        # 直接导入（此时 agent.py 应该已被 WebUI 正常加载，Python 会用缓存）
        from agent_config import load_config, _get_current_checkpoint
        from agent_tools import TOOLS, TOOL_FUNCTIONS
        from agent_prompts import _get_system_prompt
        from agent import _execute_tool

        _load_config_fn = load_config
        _get_current_checkpoint_fn = _get_current_checkpoint
        _get_system_prompt_fn = _get_system_prompt
        _execute_tool_fn = _execute_tool
        _TOOLS = TOOLS
        _TOOL_FUNCTIONS = TOOL_FUNCTIONS

        try:
            from agent_tools_registry import get_registered_tools
            _get_registered_tools_fn = get_registered_tools
            _REGISTRY_AVAILABLE = True
        except Exception:
            _REGISTRY_AVAILABLE = False

        print(f"[PS Agent] ✅ 已加载 sd-webui-agent-dev 模块，工具数: {len(_TOOLS)}")
    except Exception as e:
        _load_error = str(e)
        print(f"[PS Agent] ⚠️ 导入 sd-webui-agent-dev 失败: {e}")
        import traceback
        traceback.print_exc()
        # 降级实现
        _load_config_fn = lambda: {}
        _get_current_checkpoint_fn = lambda: getattr(shared.opts, 'sd_model_checkpoint', '') or ""
        _get_system_prompt_fn = lambda model="": "你是一个 SD WebUI 智能助手。"
        _execute_tool_fn = lambda *a, **kw: ("", [])
        _TOOLS = []
        _TOOL_FUNCTIONS = {}
        _REGISTRY_AVAILABLE = False

    _modules_loaded = True

# =============================================================================
# 过滤工具：PS 插件不需要视频工具
# =============================================================================
VIDEO_TOOL_NAMES = {
    "h3_video_generate",
    "dreamina_video_generate",
    "video_keyframe_extract",
    "video_to_frames",
}

# 合并内置工具 + 注册工具，过滤视频
def _get_filtered_tools():
    """获取过滤后的工具列表（去掉视频相关工具）"""
    _ensure_modules_loaded()
    all_tools = list(_TOOLS)
    if _REGISTRY_AVAILABLE and _get_registered_tools_fn:
        try:
            all_tools.extend(_get_registered_tools_fn())
        except Exception:
            pass
    return [
        t for t in all_tools
        if t.get("function", {}).get("name", "") not in VIDEO_TOOL_NAMES
    ]


# =============================================================================
# 配置加载与保存
# =============================================================================
AGENT_DEV_CONFIG_PATH = AGENT_DEV_DIR.parent / "agent_config.json"


def _get_cfg():
    """加载 sd-webui-agent-dev 的配置"""
    _ensure_modules_loaded()
    try:
        return _load_config_fn()
    except Exception:
        return {}


def _save_cfg(updates: dict):
    """保存配置到 agent_config.json（合并更新）"""
    try:
        cfg = _get_cfg()
        cfg.update(updates)
        # 只保留 DEFAULT_CONFIG 中定义的字段（避免垃圾数据）
        _ensure_modules_loaded()
        try:
            from agent_config import DEFAULT_CONFIG
            valid_keys = set(DEFAULT_CONFIG.keys())
            cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        except Exception:
            pass
        # 确保目录存在
        AGENT_DEV_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AGENT_DEV_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"[PS Agent] ✅ 配置已保存到 {AGENT_DEV_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[PS Agent] ❌ 配置保存失败: {e}")
        return False


def _get_providers():
    """获取支持的 API 供应商列表"""
    _ensure_modules_loaded()
    try:
        from agent_config import API_PROVIDERS
        return {name: info["base_url"] for name, info in API_PROVIDERS.items()}
    except Exception:
        return {
            "ModelScope": "https://api-inference.modelscope.cn/v1",
            "DashScope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "YoboxAI": "https://api.yoboxai.com/v1",
        }


def _provider_base_url(provider):
    """根据供应商名获取默认 base_url"""
    providers = _get_providers()
    return providers.get(provider, "")


# =============================================================================
# 图片处理：PIL → base64
# =============================================================================
def _pil_to_base64(img):
    """PIL Image → base64 PNG 字符串"""
    if img is None:
        return None
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[PS Agent] PIL→base64 失败: {e}")
        return None


def _base64_to_pil(b64str):
    """base64 → PIL Image"""
    if not b64str:
        return None
    try:
        if "," in b64str:
            b64str = b64str.split(",", 1)[1]
        img_data = base64.b64decode(b64str)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception:
        return None


# =============================================================================
# LLM 非流式调用（OpenAI SDK）
# =============================================================================
def _call_llm(messages, cfg, tools, tool_choice="auto"):
    """调用 LLM API（非流式），返回完整响应"""
    try:
        from openai import OpenAI
    except ImportError:
        # 回退到 requests
        return _call_llm_requests(messages, cfg, tools, tool_choice)

    client = OpenAI(
        api_key=cfg.get("api_key", "local"),
        base_url=cfg.get("base_url", "http://localhost:8080/v1"),
        max_retries=3,
    )
    resp = client.chat.completions.create(
        model=cfg.get("model", "Qwen3.5-4B-Q6_K.gguf"),
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=False,
    )
    return resp.model_dump()


def _call_llm_requests(messages, cfg, tools, tool_choice="auto"):
    """requests 回退方案"""
    url = cfg.get("base_url", "http://localhost:8080/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.get("model", "Qwen3.5-4B-Q6_K.gguf"),
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "stream": False,
        "max_tokens": 4096,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# LLM 流式调用（用于 chat-stream 端点，实时推送思考/回复文本）
# 失败时抛出异常，由调用方回退到非流式
# =============================================================================
def _accumulate_chunk(choice, content_state, tool_acc, on_token):
    """累积一个流式 chunk 的 content / tool_calls 增量，返回 finish_reason"""
    delta = choice.delta
    if delta is not None:
        if getattr(delta, "content", None):
            content_state["text"] += delta.content
            if on_token:
                on_token(delta.content)
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = tc.index if getattr(tc, "index", None) is not None else 0
            acc = tool_acc.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if getattr(tc, "id", None):
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["function"]["name"] += fn.name
                if getattr(fn, "arguments", None):
                    acc["function"]["arguments"] += fn.arguments
    return getattr(choice, "finish_reason", None)


def _chunk_to_response(content_state, tool_acc, finish_reason):
    """把流式累积结果组装成与非流式一致的响应结构"""
    message = {"role": "assistant", "content": content_state["text"] or None}
    tool_calls = [tool_acc[i] for i in sorted(tool_acc.keys())]
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason or "stop"}]}


def _call_llm_stream(messages, cfg, tools, tool_choice="auto", on_token=None):
    """调用 LLM API（流式），逐段回调 on_token(delta_text)，返回完整响应（与非流式同构）"""
    try:
        from openai import OpenAI
    except ImportError:
        return _call_llm_stream_requests(messages, cfg, tools, tool_choice, on_token)

    client = OpenAI(
        api_key=cfg.get("api_key", "local"),
        base_url=cfg.get("base_url", "http://localhost:8080/v1"),
        max_retries=1,
    )
    stream = client.chat.completions.create(
        model=cfg.get("model", "Qwen3.5-4B-Q6_K.gguf"),
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
    )
    content_state = {"text": ""}
    tool_acc = {}
    finish_reason = None
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        fr = _accumulate_chunk(choices[0], content_state, tool_acc, on_token)
        if fr:
            finish_reason = fr
    return _chunk_to_response(content_state, tool_acc, finish_reason)


def _call_llm_stream_requests(messages, cfg, tools, tool_choice="auto", on_token=None):
    """requests 流式回退方案（解析 OpenAI SSE 格式）"""
    url = cfg.get("base_url", "http://localhost:8080/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.get("model", "Qwen3.5-4B-Q6_K.gguf"),
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "stream": True,
        "max_tokens": 4096,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    resp = requests.post(url, json=payload, headers=headers, timeout=300, stream=True)
    resp.raise_for_status()
    content_state = {"text": ""}
    tool_acc = {}
    finish_reason = None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content_state["text"] += delta["content"]
            if on_token:
                on_token(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            acc = tool_acc.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc.get("id"):
                acc["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                acc["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                acc["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
    return _chunk_to_response(content_state, tool_acc, finish_reason)


def _call_llm_with_fallback(messages, cfg, tools, tool_choice="auto", on_event=None):
    """优先流式调用（实时推送 token），失败自动回退非流式"""
    if on_event:
        try:
            return _call_llm_stream(
                messages, cfg, tools, tool_choice=tool_choice,
                on_token=lambda d: on_event({"type": "token", "text": d}),
            )
        except Exception as e:
            print(f"[PS Agent] LLM 流式调用失败，回退非流式: {e}")
    return _call_llm(messages, cfg, tools, tool_choice=tool_choice)


# 检测用户是否有工具调用意图
TOOL_INTENT_KEYWORDS = [
    "生成", "画", "绘制", "创作", "制作", "创建", "做一张", "来一张",
    "修改", "编辑", "改", "调整", "优化", "美化",
    "放大", "超分", "清晰",
    "抠图", "去背", "去背景", "分离", "分层",
    "换背景", "改变背景", "换风格",
    "切换模型", "换模型", "用这个模型",
    "列出模型", "有哪些模型", "查看模型",
    "扩图", "延伸", "局部重绘",
]


def _has_tool_intent(text):
    """检测用户消息是否有调用工具的意图"""
    text_lower = text.lower()
    for kw in TOOL_INTENT_KEYWORDS:
        if kw in text or kw in text_lower:
            return True
    return False


# =============================================================================
# 智能体聊天主逻辑（非流式）
# =============================================================================
def agent_chat(body: dict, on_event=None):
    """智能体聊天主逻辑。on_event 可选：流式端点传入，实时推送进度事件"""
    cfg = _get_cfg()
    user_message = body.get("message", "")
    image_base64 = body.get("image_base64", "")  # PS 传入的当前画布图片（单张，兼容）
    image_base64_list = body.get("image_base64_list", [])  # PS 传入的多张参考图
    sd_model = body.get("sd_model", "")  # 用户在 PS 面板选择的 SD 模型

    if not user_message:
        return {"status": "error", "error": "缺少消息内容"}

    # 确保延迟加载模块已导入
    _ensure_modules_loaded()

    # ===== PS 面板传来的完整设置覆盖 =====
    # LLM 设置
    ps_provider = body.get("llm_provider", "")
    ps_base_url = body.get("llm_base_url", "")
    ps_api_key = body.get("llm_api_key", "")
    ps_model = body.get("llm_model", "")
    if ps_provider:
        cfg["api_provider"] = ps_provider
        # 选了供应商但没传 base_url → 自动填充
        if not ps_base_url:
            ps_base_url = _provider_base_url(ps_provider)
    if ps_base_url:
        cfg["base_url"] = ps_base_url
    if ps_api_key:
        cfg["api_key"] = ps_api_key
    if ps_model:
        cfg["model"] = ps_model

    # 图片生成设置
    # PS 插件总是会传 image_provider 和 image_model（空值表示"默认/走本地"）
    ps_image_provider = body.get("image_provider", "")
    ps_image_api_key = body.get("image_api_key", "")
    ps_image_model = body.get("image_model", "")
    ps_image_base_url = body.get("image_base_url", "")
    # 显式覆盖：PS 选"默认"时传空字符串，清空后端配置避免工具强转到 API
    cfg["image_api_provider"] = ps_image_provider
    cfg["image_model"] = ps_image_model
    if ps_image_provider and not ps_image_base_url:
        ps_image_base_url = _provider_base_url(ps_image_provider)
    if ps_image_base_url:
        cfg["image_base_url"] = ps_image_base_url
    if ps_image_api_key:
        cfg["image_api_key"] = ps_image_api_key

    # 如果用户指定了 SD 模型，先切换
    if sd_model:
        print(f"[PS Agent] 用户指定 SD 模型: {sd_model}，切换中...")
        try:
            _execute_tool_fn("switch_model", {"model_name": sd_model})
        except Exception as e:
            print(f"[PS Agent] SD 模型切换失败: {e}")

    # 构建消息
    tools = _get_filtered_tools()
    system_prompt = _get_system_prompt_fn(cfg.get("model", ""))

    # 附加当前模型信息
    current_model = _get_current_checkpoint_fn() if _get_current_checkpoint_fn else ""
    system_prompt += f"\n\n当前 SD 模型: {current_model or '未加载'}"
    system_prompt += f"\n当前 LLM 模型: {cfg.get('model', '')}"
    system_prompt += "\n注意：用户在 Photoshop 中操作，生成的图片会显示在 PS 面板中。"

    messages = [{"role": "system", "content": system_prompt}]

    # 处理图片（支持多张参考图）
    uploaded_image = None  # 主参考图（第一张），用于工具链
    all_images_b64 = []  # 所有图片 base64 列表

    # 优先使用多张图片列表
    if image_base64_list and isinstance(image_base64_list, list):
        all_images_b64 = [b64 for b64 in image_base64_list if b64]
    # 兼容旧的单张 image_base64
    elif image_base64:
        all_images_b64 = [image_base64]

    # 构建 user_content
    if all_images_b64:
        # 第一张作为主参考图（PIL）
        try:
            uploaded_image = _base64_to_pil(all_images_b64[0])
        except Exception as e:
            print(f"[PS Agent] 主参考图转换失败: {e}")
        # 构建多模态内容：文本 + 多张图片
        user_content = [{"type": "text", "text": user_message}]
        for b64 in all_images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        print(f"[PS Agent] 收到 {len(all_images_b64)} 张参考图")
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})

    max_iter = cfg.get("max_tool_iterations", 6)
    generated_images = []  # 收集 base64 图片
    last_tool_images = []  # PIL Image 列表，用于工具链传递
    reply_text = ""

    # 模型感知的 tool_choice 策略
    model_name = str(cfg.get("model", "")).lower()
    provider_name = str(cfg.get("api_provider", "")).lower()
    # Qwen 系列（ModelScope/DashScope）原生支持 auto 模式 function calling
    # gpt-5.4-mini 等 YoboxAI 模型需要 required 强制调用
    use_required_for_model = not (
        "qwen" in model_name or
        provider_name in ("modelscope", "dashscope")
    )
    print(f"[PS Agent] 模型={cfg.get('model')}, provider={cfg.get('api_provider')}, required={use_required_for_model}")

    for iteration in range(max_iter):
        print(f"[PS Agent] LLM 调用 第 {iteration + 1}/{max_iter} 次")
        if on_event:
            on_event({"type": "status", "text": "正在思考…" if iteration == 0 else f"正在思考…（第 {iteration + 1} 轮）"})

        # 模型感知策略：
        # - Qwen/ModelScope/DashScope 模型：原生支持 auto function calling
        # - gpt-5.4-mini 等 YoboxAI 模型：auto 忽略 tools，必须 required
        # - 已生成图片后切换到 auto 让 LLM 给出最终回复
        has_intent = _has_tool_intent(user_message)
        has_images = bool(generated_images)
        if has_intent and not has_images and use_required_for_model and iteration < max_iter - 1:
            tool_choice = "required"
        else:
            tool_choice = "auto"
        print(f"[PS Agent] tool_choice={tool_choice} (has_intent={has_intent}, has_images={has_images}, iter={iteration})")

        try:
            resp = _call_llm_with_fallback(messages, cfg, tools, tool_choice=tool_choice, on_event=on_event)
        except Exception as e:
            # required 可能不被某些代理支持，回退到 auto 重试
            if tool_choice == "required":
                print(f"[PS Agent] required 模式失败，回退 auto: {e}")
                try:
                    resp = _call_llm_with_fallback(messages, cfg, tools, tool_choice="auto", on_event=on_event)
                except Exception as e2:
                    error_msg = str(e2)
                    print(f"[PS Agent] LLM 调用失败: {error_msg}")
                    return {
                        "status": "error",
                        "error": f"LLM 调用失败: {error_msg}",
                        "hint": f"请确保 LLM 服务运行: {cfg.get('base_url', '')}",
                    }
            else:
                error_msg = str(e)
                print(f"[PS Agent] LLM 调用失败: {error_msg}")
                return {
                    "status": "error",
                    "error": f"LLM 调用失败: {error_msg}",
                    "hint": f"请确保 LLM 服务运行: {cfg.get('base_url', '')}",
                }

        choices = resp.get("choices", [])
        if not choices:
            return {"status": "error", "error": "LLM 返回为空"}

        choice_msg = choices[0].get("message", {})
        tool_calls = choice_msg.get("tool_calls", [])

        # 保存 reasoning + content
        reply_text = choice_msg.get("content", "") or reply_text

        # 如果没有工具调用但有意图，用 required 重试一次
        if not tool_calls and has_intent and tool_choice != "required":
            print(f"[PS Agent] auto 模式无工具调用，用 required 重试...")
            try:
                resp = _call_llm_with_fallback(messages, cfg, tools, tool_choice="required", on_event=on_event)
                choices = resp.get("choices", [])
                if choices:
                    choice_msg = choices[0].get("message", {})
                    tool_calls = choice_msg.get("tool_calls", [])
                    reply_text = choice_msg.get("content", "") or reply_text
            except Exception as e:
                print(f"[PS Agent] required 重试失败: {e}")

        if not tool_calls:
            # 没有工具调用，返回最终回复
            return {
                "status": "success",
                "reply": reply_text or "操作完成",
                "images": generated_images,
                "model": current_model,
            }

        # 把 assistant 消息（含 tool_calls）加入历史
        # OpenAI SDK 返回的格式需要规范化
        assistant_msg = {
            "role": "assistant",
            "content": choice_msg.get("content") or None,
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        # 处理每个工具调用
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name", "")
            try:
                func_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                func_args = {}

            print(f"[PS Agent] 调用工具: {func_name}({func_args})")
            if on_event:
                on_event({"type": "tool", "name": func_name, "args": func_args})

            # 执行工具（复用新版本的 _execute_tool，自动注入图片参数）
            try:
                result_str, tool_images = _execute_tool_fn(
                    func_name,
                    func_args,
                    uploaded_image,
                    None,  # uploaded_video
                    last_tool_images,
                )
            except Exception as e:
                result_str = json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
                tool_images = []

            if on_event:
                on_event({"type": "tool_done", "name": func_name, "images": len(tool_images) if tool_images else 0})

            # 收集图片
            if tool_images:
                for img in tool_images:
                    b64 = _pil_to_base64(img)
                    if b64:
                        generated_images.append({
                            "image_base64": b64,
                            "prompt": func_args.get("prompt", ""),
                            "tool": func_name,
                        })
                last_tool_images = list(tool_images)

            # 工具结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result_str,
            })

    # 达到最大迭代次数
    # 如果已经生成了图片，状态为 success（对 PS 前端更友好）
    final_status = "success" if generated_images else "max_iterations"
    final_reply = reply_text or ("操作完成" if generated_images else "已达到最大工具调用次数")
    return {
        "status": final_status,
        "reply": final_reply,
        "images": generated_images,
    }


# =============================================================================
# FastAPI 端点注册
# =============================================================================
def agent_api_callbacks(_: gr_or_blocks, app: FastAPI):
    """注册智能体 API 端点"""

    # CORS 响应头（手动添加，因为 FastAPI 启动后不能用 add_middleware）
    CORS_HEADERS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }

    # 通用 OPTIONS 预检处理器
    @app.options("/sdapi/v1/ps-plugin/agent/{full_path:path}")
    async def options_handler(full_path: str):
        from fastapi.responses import Response
        return Response(status_code=200, headers=CORS_HEADERS)

    def _cors_response(data=None, status_code=200):
        """包装 JSON 响应并添加 CORS 头"""
        from fastapi.responses import JSONResponse
        return JSONResponse(content=data, status_code=status_code, headers=CORS_HEADERS)

    @app.get("/sdapi/v1/ps-plugin/agent/config")
    def get_agent_config():
        """获取智能体配置（脱敏，PS 面板显示用）"""
        cfg = _get_cfg()
        # 图片模型列表（从已加载的 agent 模块获取，不触发重新导入）
        image_models = []
        try:
            agent_mod = sys.modules.get("agent")
            if agent_mod:
                image_models = getattr(agent_mod, "IMAGE_GENERATION_MODELS", [])
        except Exception:
            pass

        providers = _get_providers()
        return _cors_response({
            # 供应商列表（PS 前端渲染下拉用）
            "providers": providers,
            # LLM 配置
            "llm_model": cfg.get("model", ""),
            "llm_base_url": cfg.get("base_url", ""),
            "llm_provider": cfg.get("api_provider", ""),
            "llm_api_key_set": bool(cfg.get("api_key", "")),
            # 图片生成配置
            "image_model": cfg.get("image_model", ""),
            "image_provider": cfg.get("image_api_provider", ""),
            "image_base_url": cfg.get("image_base_url", ""),
            "image_api_key_set": bool(cfg.get("image_api_key", "")),
            "image_models": image_models,
            # 视频配置（PS 不需要，但返回信息）
            "video_model": cfg.get("video_model", ""),
            "video_provider": cfg.get("video_api_provider", ""),
            # 生图默认参数
            "default_width": cfg.get("default_width", 1024),
            "default_height": cfg.get("default_height", 1024),
            "default_steps": cfg.get("default_steps", 4),
            "default_cfg_scale": cfg.get("default_cfg_scale", 1.0),
            # 当前 SD 模型
            "current_model": _get_current_checkpoint_fn() if _get_current_checkpoint_fn else "",
            # 工具统计
            "available_tools": len(_get_filtered_tools()),
            "video_tools_excluded": True,
            # 加载状态（调试用）
            "agent_dev_loaded": _load_error is None,
            "agent_dev_error": _load_error,
        })

    @app.get("/sdapi/v1/ps-plugin/agent/llm-models")
    def get_llm_models(
        llm_provider: str = "",
        llm_api_key: str = "",
        llm_base_url: str = "",
    ):
        """代理获取 LLM 模型列表（支持临时覆盖供应商/Key，不写回配置）"""
        cfg = _get_cfg()
        # 临时覆盖（不写回文件）
        if llm_provider:
            cfg["api_provider"] = llm_provider
            if not llm_base_url:
                llm_base_url = _provider_base_url(llm_provider)
        if llm_base_url:
            cfg["base_url"] = llm_base_url
        if llm_api_key:
            cfg["api_key"] = llm_api_key

        base_url = cfg.get("base_url", "http://localhost:8080/v1").rstrip("/")
        url = base_url + "/models"
        headers = {}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            current = cfg.get("model", "")
            if current and current not in models:
                models.insert(0, current)
            return _cors_response({"status": "success", "models": models, "current": current})
        except Exception as e:
            return _cors_response({
                "status": "error",
                "error": str(e),
                "models": [cfg.get("model", "")],
                "current": cfg.get("model", ""),
            })

    @app.get("/sdapi/v1/ps-plugin/agent/image-models")
    def get_image_models(
        image_provider: str = "",
    ):
        """获取云端图片生成模型列表（支持临时覆盖供应商）"""
        models = []
        cfg = _get_cfg()
        # 临时覆盖（不写回文件）
        if image_provider:
            cfg["image_api_provider"] = image_provider
        try:
            agent_mod = sys.modules.get("agent")
            if agent_mod:
                by_provider = getattr(agent_mod, "IMAGE_GENERATION_MODELS_BY_PROVIDER", {})
                provider = cfg.get("image_api_provider", "")
                models = by_provider.get(provider, []) or by_provider.get("YoboxAI", [])
        except Exception:
            pass
        return _cors_response({
            "status": "success",
            "models": models,
            "current": cfg.get("image_model", ""),
            "provider": cfg.get("image_api_provider", ""),
        })

    @app.get("/sdapi/v1/ps-plugin/agent/tools")
    def get_tools():
        """获取当前可用的工具列表（PS 面板显示用）"""
        return _cors_response({
            "status": "success",
            "tools": [
                {
                    "name": t.get("function", {}).get("name", ""),
                    "description": t.get("function", {}).get("description", ""),
                }
                for t in _get_filtered_tools()
            ],
            "excluded_video_tools": list(VIDEO_TOOL_NAMES),
        })

    @app.post("/sdapi/v1/ps-plugin/agent/settings")
    def save_settings(body: dict):
        """
        保存智能体配置到 agent_config.json
        请求体: {
            "llm_provider": "ModelScope",
            "llm_base_url": "https://api-inference.modelscope.cn/v1",
            "llm_api_key": "ms-xxx",
            "llm_model": "Qwen/Qwen3.8-27B",
            "image_provider": "YoboxAI",
            "image_base_url": "https://api.yoboxai.com/v1",
            "image_api_key": "sk-xxx",
            "image_model": "nano-banana"
        }
        """
        # 构建更新字典
        updates = {}
        if body.get("llm_provider") is not None:
            updates["api_provider"] = body["llm_provider"]
            # 选了供应商但没传 base_url → 自动填充
            if not body.get("llm_base_url"):
                body["llm_base_url"] = _provider_base_url(body["llm_provider"])
        if body.get("llm_base_url") is not None:
            updates["base_url"] = body["llm_base_url"]
        if body.get("llm_api_key") is not None and body["llm_api_key"]:
            updates["api_key"] = body["llm_api_key"]
        if body.get("llm_model") is not None:
            updates["model"] = body["llm_model"]

        if body.get("image_provider") is not None:
            updates["image_api_provider"] = body["image_provider"]
            if not body.get("image_base_url"):
                body["image_base_url"] = _provider_base_url(body["image_provider"])
        if body.get("image_base_url") is not None:
            updates["image_base_url"] = body["image_base_url"]
        if body.get("image_api_key") is not None and body["image_api_key"]:
            updates["image_api_key"] = body["image_api_key"]
        if body.get("image_model") is not None:
            updates["image_model"] = body["image_model"]

        success = _save_cfg(updates)
        if success:
            # 返回更新后的脱敏配置
            cfg = _get_cfg()
            return _cors_response({
                "status": "success",
                "message": "配置已保存",
                "llm_provider": cfg.get("api_provider", ""),
                "llm_model": cfg.get("model", ""),
                "llm_api_key_set": bool(cfg.get("api_key", "")),
                "image_provider": cfg.get("image_api_provider", ""),
                "image_model": cfg.get("image_model", ""),
                "image_api_key_set": bool(cfg.get("image_api_key", "")),
            })
        else:
            return _cors_response({"status": "error", "error": "配置保存失败"}, status_code=500)

    @app.post("/sdapi/v1/ps-plugin/agent/chat")
    def chat_endpoint(body: dict):
        """
        智能体聊天端点
        请求体: {
            "message": "生成一张猫",
            "image_base64": "...",        // 可选，画布图像
            "sd_model": "model_name",      // 可选，用户选择的 SD 模型
            "llm_provider": "ModelScope",  // 可选，LLM 供应商
            "llm_base_url": "...",         // 可选，LLM base_url
            "llm_api_key": "...",          // 可选，LLM API Key
            "llm_model": "model_id",      // 可选，用户选择的 LLM 模型
            "image_provider": "YoboxAI",   // 可选，图片供应商
            "image_base_url": "...",       // 可选，图片 base_url
            "image_api_key": "...",        // 可选，图片 API Key
            "image_model": "nano-banana"   // 可选，图片模型
        }
        返回: { "status": "success", "reply": "...", "images": [...] }
        """
        return _cors_response(agent_chat(body))

    @app.post("/sdapi/v1/ps-plugin/agent/chat-stream")
    def chat_stream_endpoint(body: dict):
        """
        智能体聊天流式端点（SSE）
        逐条推送事件: data: {json}\\n\\n
        事件类型:
          {"type":"status","text":"正在思考…"}            LLM 开始思考
          {"type":"token","text":"..."}                   LLM 输出文本增量
          {"type":"tool","name":"txt2img","args":{...}}   开始执行工具
          {"type":"tool_done","name":"txt2img","images":1} 工具执行完成
          {"type":"done", "status":"success","reply":"...","images":[...]}  最终结果（与 /chat 返回同构）
          {"type":"done","status":"error","error":"..."}  出错
        """
        q = queue.Queue()
        SENTINEL = object()

        def on_event(ev):
            q.put(ev)

        def worker():
            try:
                result = agent_chat(body, on_event=on_event)
                if not isinstance(result, dict):
                    result = {"status": "error", "error": "LLM 返回格式异常"}
                event = {"type": "done"}
                event.update(result)
                q.put(event)
            except Exception as e:
                print(f"[PS Agent] ❌ chat-stream 异常: {e}")
                traceback.print_exc()
                q.put({"type": "done", "status": "error", "error": str(e)})
            finally:
                q.put(SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        def gen():
            while True:
                ev = q.get()
                if ev is SENTINEL:
                    break
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

        headers = dict(CORS_HEADERS)
        headers["Cache-Control"] = "no-cache"
        headers["X-Accel-Buffering"] = "no"
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    @app.post("/sdapi/v1/ps-plugin/agent/debug-llm")
    def debug_llm(body: dict):
        """调试端点：直接测试 LLM 调用，返回原始响应"""
        cfg = _get_cfg()
        user_message = body.get("test_prompt", body.get("message", "生成一张红色方块图片"))
        tool_choice = body.get("tool_choice", "auto")

        # 参数覆盖（与 chat 端点一致）
        ps_provider = body.get("llm_provider", "")
        ps_base_url = body.get("llm_base_url", "")
        ps_api_key = body.get("llm_api_key", "")
        ps_model = body.get("llm_model", "")
        if ps_provider:
            cfg["api_provider"] = ps_provider
            if not ps_base_url:
                ps_base_url = _provider_base_url(ps_provider)
        if ps_base_url:
            cfg["base_url"] = ps_base_url
        if ps_api_key:
            cfg["api_key"] = ps_api_key
        if ps_model:
            cfg["model"] = ps_model

        # 如果模型支持 auto function calling，用 auto；否则 required
        model_lower = str(cfg.get("model", "")).lower()
        provider_lower = str(cfg.get("api_provider", "")).lower()
        if "qwen" in model_lower or provider_lower in ("modelscope", "dashscope"):
            tool_choice = "auto"

        tools = _get_filtered_tools()
        system_prompt = _get_system_prompt_fn(cfg.get("model", ""))
        messages = [
            {"role": "system", "content": system_prompt[:2000]},  # 截断避免太长
            {"role": "user", "content": user_message},
        ]
        try:
            resp = _call_llm(messages, cfg, tools, tool_choice=tool_choice)
            choice = resp.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            return _cors_response({
                "success": True,
                "status": "success",
                "reply": content,
                "tool_choice_used": tool_choice,
                "finish_reason": choice.get("finish_reason"),
                "has_tool_calls": bool(msg.get("tool_calls")),
                "tool_calls": msg.get("tool_calls", []),
                "content": content,
                "tools_count": len(tools),
                "model_used": cfg.get("model", ""),
                "base_url": cfg.get("base_url", ""),
            })
        except Exception as e:
            return _cors_response({
                "success": False,
                "status": "error",
                "error": str(e),
                "tools_count": len(tools),
                "model_used": cfg.get("model", ""),
            })


script_callbacks.on_app_started(agent_api_callbacks)
