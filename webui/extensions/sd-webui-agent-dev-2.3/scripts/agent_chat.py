# =============================================================================
# Agent Chat 核心 — 流式对话 + Function Calling 循环
# =============================================================================

import html
import json
import os
import re
import time
import traceback

from scripts.agent_config import (
    load_config, _save_pil_to_tempfile,
    _REGISTRY_AVAILABLE, get_registered_tools, get_tool_function,
)
from scripts.agent_tools import TOOLS, TOOL_FUNCTIONS
from scripts.agent_prompts import _get_system_prompt
from scripts.agent_history import (
    _extract_text_from_history_content, _normalize_image_path,
    _lenient_image_path, _normalize_image_paths,
    _latest_image_paths_from_history, _image_to_compressed_base64,
)
from scripts.agent_routing import (
    _configured_image_model, _is_configured_api_image_model,
    _extract_requested_api_model, _has_requested_local_model,
    _has_requested_layer_separation, _has_requested_trellis_local,
    _has_requested_keyframe_local,
    _coerce_api_image_edit_tool, _coerce_generation_tool_by_selected_model,
)


def _execute_tool(tool_name, tool_args, uploaded_image=None, uploaded_video=None, last_tool_images=None):
    """执行工具调用，返回 (result_string, images_list)。
    自动传入上传的图片/视频 + 上次工具生成的图片到对应参数。
    支持工具链：如 提取关键帧 → 拼接 → 放大。"""
    if last_tool_images is None:
        last_tool_images = []

    # 内置工具和动态注册工具都必须经过下面的参数注入流程。
    func = TOOL_FUNCTIONS.get(tool_name)
    registered_func = None
    if func is None and _REGISTRY_AVAILABLE:
        registered_func = get_tool_function(tool_name)
        func = registered_func
    if func is None:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False), []

    try:
        # ===== 上传图来源解析（三层防御，修复"请先上传图片"bug） =====
        # 原实现直接信任 uploaded_image（文件存在即注入），但前端上传偶发
        # 只留下损坏/占位文件，或图片上下文根本没进工具层，导致下游工具拿不到图。
        # 现在：① 宽容校验 LLM 参数图 ② 校验 uploaded_image。
        resolved_upload = _lenient_image_path(uploaded_image) if uploaded_image is not None else None
        if resolved_upload is None and uploaded_image is not None:
            print(f"[Agent] uploaded_image 无效({uploaded_image!r})，无有效上传图片")

        # ===== 自动注入图片参数（优先级：用户上传 > 上次工具生成） =====
        # 单图工具：需要自动注入上传图片或上一步输出图片。
        # layer_separation 虽然不在旧注释列表中，但同样必须接收 image。
        if tool_name in (
            "img2img", "upscale", "apply_adetailer", "remove_background",
            "layer_separation", "trellis2_image_to_3d", "edit_image", "change_background", "api_image_edit",
        ):
            # LLM 可能会生成空字符串或无效占位路径，也视为未提供图片。
            if not _lenient_image_path(tool_args.get("image")):
                if resolved_upload is not None:
                    tool_args["image"] = resolved_upload
                elif last_tool_images:
                    # 用上次生成的最后一张图
                    tool_args["image"] = last_tool_images[-1]
                else:
                    tool_args["image"] = None

        # 导入的 Gradio API 工具统一使用其示例中的图片参数名。
        if tool_name.startswith("gradio_api_") and not _lenient_image_path(tool_args.get("image")):
            if resolved_upload is not None:
                tool_args["image"] = resolved_upload
            elif last_tool_images:
                tool_args["image"] = last_tool_images[-1]

        # H3 视频生成：自动注入参考图（首帧/参考图）
        if tool_name == "h3_video_generate":
            if ("first_frame" not in tool_args or tool_args["first_frame"] is None) and \
               ("reference_image" not in tool_args or tool_args["reference_image"] is None):
                if uploaded_image is not None:
                    tool_args["first_frame"] = uploaded_image
                elif last_tool_images:
                    tool_args["first_frame"] = last_tool_images[-1]

        if tool_name == "dreamina_video_generate":
            if ("first_frame" not in tool_args or tool_args["first_frame"] is None) and \
               ("reference_image" not in tool_args or tool_args["reference_image"] is None):
                if uploaded_image is not None:
                    tool_args["first_frame"] = uploaded_image
                elif last_tool_images:
                    tool_args["first_frame"] = last_tool_images[-1]

        # 多图工具：stitch_images
        if tool_name == "stitch_images":
            if "images" not in tool_args or not tool_args["images"]:
                if last_tool_images:
                    tool_args["images"] = last_tool_images
                elif uploaded_image is not None:
                    tool_args["images"] = [uploaded_image]

        # 自动传入视频参数
        if tool_name in ("video_keyframe_extract", "video_to_frames") and uploaded_video is not None:
            if "video_path" not in tool_args:
                tool_args["video_path"] = uploaded_video

        result = func(**tool_args)

        if isinstance(result, tuple) and len(result) == 2:
            images, info = result
            # 工具可能返回 None 表示失败
            if images is None:
                return json.dumps({"status": "error", "info": info}, ensure_ascii=False), []
            # 部分图像/视频工具用空列表加 error 信息表示失败，不能误报为成功。
            if isinstance(info, dict) and info.get("error"):
                return json.dumps({"status": "error", "info": info}, ensure_ascii=False), []
            # 统一转为列表
            if not isinstance(images, list):
                images = [images]
            result_str = json.dumps({"status": "success", "info": info, "image_count": len(images)}, ensure_ascii=False)
            return result_str, images

        return json.dumps({"status": "success", "data": result}, ensure_ascii=False, default=str), []

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[Agent] 工具执行失败 [{tool_name}]: {error_msg}")
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), []


def _extract_delta_reasoning(delta):
    """从流式 delta 中提取思考内容，兼容不同网关/模型的字段命名。

    - DeepSeek / 多数国内网关：delta.reasoning_content
    - OpenAI o 系列 / 部分网关：delta.reasoning
    - Anthropic 风格网关：delta.thinking
    """
    for field in ("reasoning_content", "reasoning", "reasoning_text", "thinking"):
        val = getattr(delta, field, None)
        if val:
            return val if isinstance(val, str) else str(val)
    return ""


def _thinking_block(reasoning_text, live=False):
    """把思考内容渲染为内嵌的可折叠小窗口（<details>）。

    容器限高、内部滚动（样式见 agent_ui.py 的 .agent-thinking CSS），
    不占据大块聊天空间；live=True 时思考进行中默认展开并实时跟随。
    """
    body = html.escape(reasoning_text or "").rstrip()
    label = "🧠 思考中，请稍候…" if live else "🧠 思考过程（点击展开查看）"
    open_attr = " open" if live else ""
    return (
        f'<details{open_attr} class="agent-thinking">'
        f"<summary>{label}</summary>"
        f'<div class="agent-thinking-body">{body}</div>'
        f"</details>"
    )


def _with_thinking(reasoning_text, answer_text):
    """拼接思考块与回答文本（无思考内容时原样返回回答）。"""
    if not (reasoning_text or "").strip():
        return answer_text or ""
    return _thinking_block(reasoning_text) + "\n\n" + (answer_text or "")


_THINKING_BLOCK_RE = re.compile(r'<details[^>]*class="agent-thinking"[^>]*>.*?</details>\s*', re.DOTALL)


def _strip_thinking_block(text):
    """从历史消息中移除思考块 HTML。

    思考块仅用于 UI 展示；回传给 LLM 时剔除，避免把长思考文本
    作为模型上下文反复发送（浪费 token 且可能干扰模型）。
    """
    if not isinstance(text, str) or "agent-thinking" not in text:
        return text
    return _THINKING_BLOCK_RE.sub("", text).strip()


def chat_stream(history, uploaded_image=None, uploaded_video=None, attachments=None):
    """流式聊天生成器。从 history 中提取最后一条用户消息。

    Yields: (history_update, status_message)
    """
    cfg = load_config()
    _refresh_registered_agent_tools()

    # 从 history 提取最后一条用户文本消息（跳过纯图片消息）
    user_message = ""
    for h in reversed(history):
        if isinstance(h, dict) and h.get("role") == "user":
            text = _extract_text_from_history_content(h.get("content", ""))
            if text:
                user_message = text
                break

    if not user_message:
        yield history, "⚠️ 请输入消息"
        return

    uploaded_image_list = uploaded_image if isinstance(uploaded_image, (list, tuple)) else ([uploaded_image] if uploaded_image is not None else [])
    recent_history_image_paths = _latest_image_paths_from_history(history)
    if not uploaded_image_list and recent_history_image_paths:
        uploaded_image_list = recent_history_image_paths
    primary_uploaded_image = uploaded_image_list[0] if uploaded_image_list else None

    # 下拉框选择的本地工具必须确定性执行，不能交给 LLM 自行猜测是否调用。
    forced_tool = None
    forced_args = {}
    if _has_requested_trellis_local(user_message):
        forced_tool = "trellis2_image_to_3d"
        forced_args = {"image": _normalize_image_path(primary_uploaded_image)}
    elif _has_requested_keyframe_local(user_message):
        forced_tool = "video_keyframe_extract"
        if isinstance(uploaded_video, dict):
            forced_args = {"video_path": uploaded_video.get("path")}
        else:
            forced_args = {"video_path": uploaded_video}
    if forced_tool:
        if forced_tool == "trellis2_image_to_3d" and not _normalize_image_path(primary_uploaded_image):
            final_history = list(history)
            final_history.append({"role": "assistant", "content": "TRELLIS图生3D需要先上传一张图片。"})
            yield final_history, "⚠️ 请先上传图片"
            return
        if forced_tool == "video_keyframe_extract":
            video_path = forced_args.get("video_path")
            if not video_path or not os.path.isfile(video_path):
                final_history = list(history)
                final_history.append({"role": "assistant", "content": "视频关键帧需要先上传一个可读取的视频文件。"})
                yield final_history, "⚠️ 请先上传视频"
                return
        yield history, f"🔧 正在执行本地工具: {forced_tool}..."
        result_str, images = _execute_tool(forced_tool, forced_args, primary_uploaded_image, uploaded_video, [])
        try:
            result_data = json.loads(result_str)
        except Exception:
            result_data = {}
        final_history = list(history)
        if result_data.get("status") == "success":
            info = result_data.get("info") or {}
            detail = info.get("glb_path") or info.get("output_dir") or info.get("extracted") or "完成"
            final_history.append({"role": "assistant", "content": f"本地工具 {forced_tool} 完成：{detail}"})
            glb_path = info.get("glb_path")
            if glb_path and os.path.isfile(glb_path):
                final_history.append({"role": "assistant", "content": {"path": glb_path, "alt_text": "TRELLIS.2 GLB 三维模型"}})
            for i, img in enumerate(images or []):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    final_history.append({"role": "assistant", "content": {"path": img_path, "alt_text": f"本地工具输出 {i + 1}"}})
            yield final_history, f"✅ {forced_tool} 完成"
        else:
            info = result_data.get("info") or result_data
            error = info.get("error") if isinstance(info, dict) else str(info)
            final_history.append({"role": "assistant", "content": f"本地工具 {forced_tool} 失败：{error}"})
            yield final_history, f"❌ {forced_tool} 失败：{error}"
        return

    if _has_requested_layer_separation(user_message) and primary_uploaded_image is not None:
        separation_tool = "layer_separation"
        yield history, f"🔧 正在执行: {separation_tool}..."
        result_str, images = _execute_tool(
            separation_tool,
            {"output_format": "psd"},
            primary_uploaded_image,
            uploaded_video,
            [],
        )
        try:
            result_data = json.loads(result_str)
        except Exception:
            result_data = {}
        if result_data.get("status") == "success":
            info = result_data.get("info") or {}
            final_history = list(history)
            final_history.append({
                "role": "assistant",
                "content": f"图层分离完成。输出目录：{info.get('output_dir', '图层分离输出目录')}",
            })
            for i, img in enumerate(images or []):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    final_history.append({
                        "role": "assistant",
                        "content": {"path": img_path, "alt_text": f"分离图层预览 {i + 1}"},
                    })
            yield final_history, f"✅ {separation_tool} 完成"
        else:
            info = result_data.get("info") or result_data
            error = info.get("error") if isinstance(info, dict) else str(info)
            final_history = list(history)
            final_history.append({"role": "assistant", "content": f"图层分离失败：{error}"})
            yield final_history, f"❌ {separation_tool} 失败：{error}"
        return

    # 构建 OpenAI 消息格式
    # Gradio history 中图片用 {"path": filepath, "alt_text": "..."} 格式
    # 需要转换为 OpenAI 的 image_url (base64) 格式
    messages = [{"role": "system", "content": _get_system_prompt(cfg.get("model", ""))}]
    uploaded_image_paths = _normalize_image_paths(uploaded_image)
    context_image_paths = recent_history_image_paths if not uploaded_image_paths else []
    uploaded_image_path = uploaded_image_paths[0] if uploaded_image_paths else None
    uploaded_video_path = uploaded_video if isinstance(uploaded_video, str) else (uploaded_video.get("path") if isinstance(uploaded_video, dict) else None)
    # 读取上传的文档/代码，作为真正的模型上下文，而不是只显示文件名。
    from scripts.agent_tools import _read_document_text, video_keyframe_extract_tool
    document_exts = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".css", ".html", ".htm", ".xml", ".csv", ".tsv", ".log", ".ini", ".cfg", ".toml", ".ps1", ".bat", ".cmd", ".sh", ".docx", ".pdf"}
    for attachment in attachments or []:
        if not attachment or not os.path.isfile(attachment):
            continue
        if os.path.splitext(attachment)[1].lower() in document_exts:
            try:
                text_content, file_type = _read_document_text(attachment, 120000)
                messages.append({"role": "user", "content": f"附件文件：{os.path.basename(attachment)}（{file_type}）\n请基于以下附件内容回答用户问题：\n{text_content[:120000]}"})
                print(f"[Agent] ✅ 已读取文档/代码附件: {os.path.basename(attachment)} ({file_type})")
            except Exception as e:
                print(f"[Agent] ❌ 读取文本附件失败: {e}")
    # 视频不能直接作为 OpenAI image_url 发送，自动抽取代表性帧供多模态模型理解。
    if uploaded_video_path and os.path.isfile(uploaded_video_path):
        try:
            video_frames, video_info = video_keyframe_extract_tool(uploaded_video_path, num_frames=6, method="even")
            frame_parts = [{"type": "text", "text": f"这是上传视频的代表性画面。视频信息：{video_info.get('duration', '')}，分辨率：{video_info.get('resolution', '')}。请结合这些画面分析视频。"}]
            for frame in video_frames[:6]:
                frame_path = _save_pil_to_tempfile(frame)
                data_url, _, _ = _image_to_compressed_base64(frame_path) if frame_path else (None, 0, 0)
                if data_url:
                    frame_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            if len(frame_parts) > 1:
                messages.append({"role": "user", "content": frame_parts})
                print(f"[Agent] ✅ 已读取视频并注入 {len(frame_parts) - 1} 个关键帧")
        except Exception as e:
            print(f"[Agent] ❌ 读取视频关键帧失败: {e}")
    history_has_image = False
    history_has_video = False
    for item in history:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, dict) and content.get("path"):
            if not history_has_image and content["path"] == uploaded_image_path:
                history_has_image = True
            if not history_has_video and content["path"] == uploaded_video_path:
                history_has_video = True

    # 合并连续的同角色消息（特别是 user 的 text + image）
    pending_user_content = []
    image_count = 0

    def _flush_pending_user():
        nonlocal pending_user_content, image_count
        if pending_user_content:
            if len(pending_user_content) == 1 and isinstance(pending_user_content[0], str):
                # 纯文本：可以直接用字符串
                messages.append({"role": "user", "content": pending_user_content[0]})
            else:
                # 混合内容：必须全部用 typed dict 格式（OpenAI API 要求）
                typed_content = []
                for item in pending_user_content:
                    if isinstance(item, str):
                        typed_content.append({"type": "text", "text": item})
                    else:
                        typed_content.append(item)  # 已经是 image_url dict
                messages.append({"role": "user", "content": typed_content})
            pending_user_content = []

    for h in history:
        if not isinstance(h, dict):
            continue
        role = h.get("role", "user")
        content = h.get("content", "")

        if role == "system":
            # 系统提示消息（如 @mention 注入的自动操作提示）
            # 本地 LLM 的 chat template 只允许开头有一条 system 消息，
            # 有多条会报 "System message must be at the beginning"。
            # 因此合并到第一条 system 消息中，而不是追加新消息。
            _flush_pending_user()
            if isinstance(content, str) and content.strip():
                # 合并到第一条 system 消息中
                if len(messages) > 0 and messages[0]["role"] == "system":
                    messages[0]["content"] += "\n\n" + content
                else:
                    messages.append({"role": "system", "content": content})
        elif role == "user":
            if isinstance(content, str):
                pending_user_content.append(content)
            elif isinstance(content, dict) and "path" in content:
                # 图片消息：读取文件转 base64
                try:
                    img_path = content["path"]
                    if os.path.splitext(img_path)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"):
                        continue
                    if not os.path.isfile(img_path):
                        print(f"[Agent] ⚠️ 图片文件不存在: {img_path}")
                        continue
                    data_url, orig_size, comp_size = _image_to_compressed_base64(img_path)
                    if data_url:
                        pending_user_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                        image_count += 1
                        print(f"[Agent] ✅ 图片已加载: {os.path.basename(img_path)} ({orig_size//1024}KB → {comp_size//1024}KB)")
                except Exception as e:
                    print(f"[Agent] ❌ 读取图片失败: {e}")
        else:
            # assistant 消息
            _flush_pending_user()
            if isinstance(content, str):
                # 剔除 UI 用的思考块，只把正式回答回传给模型
                messages.append({"role": "assistant", "content": _strip_thinking_block(content) or content})
            # 图片文件稍后作为当前用户上下文注入，避免模型把图片当成旧助手文本。

    _flush_pending_user()

    for context_image_path in context_image_paths[-3:]:
        try:
            data_url, orig_size, comp_size = _image_to_compressed_base64(context_image_path)
            if data_url:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是聊天上下文中最近生成/展示的图片，若我要求继续修改图片，请默认使用它作为参考图。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                })
                image_count += 1
                print(f"[Agent] ✅ 上下文图片已注入: {os.path.basename(context_image_path)} ({orig_size//1024}KB → {comp_size//1024}KB)")
        except Exception as e:
            print(f"[Agent] ❌ 注入上下文图片失败: {e}")

    for extra_image_path in uploaded_image_paths:
        if history_has_image and extra_image_path == uploaded_image_path:
            continue
        try:
            data_url, orig_size, comp_size = _image_to_compressed_base64(extra_image_path)
            if data_url:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }]
                })
                image_count += 1
                print(f"[Agent] ✅ 上传图片已注入: {os.path.basename(extra_image_path)} ({orig_size//1024}KB → {comp_size//1024}KB)")
        except Exception as e:
            print(f"[Agent] ❌ 注入上传图片失败: {e}")

    if uploaded_video_path and not history_has_video:
        # 仅保留路径由下游视频工具使用；模型本身不直接消费视频二进制。
        print(f"[Agent] ✅ 参考视频已保留: {os.path.basename(uploaded_video_path)}")

    # 调试日志：确认最终消息结构
    print(f"[Agent] 构建消息完成: {len(messages)} 条消息, 其中图片 {image_count} 张")
    for i, m in enumerate(messages):
        c = m.get("content", "")
        if isinstance(c, str):
            preview = c[:80] + ("..." if len(c) > 80 else "")
            print(f"  msg[{i}] role={m['role']} content='{preview}'")
        elif isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        parts.append(f"[image_url len={len(url)}]")
                    else:
                        parts.append(f"[{item.get('type','?')}]")
                else:
                    parts.append(f"[{type(item).__name__}]")
            print(f"  msg[{i}] role={m['role']} parts={parts}")

    # 添加 assistant 占位消息
    assistant_message = {"role": "assistant", "content": ""}
    new_history = list(history)
    new_history.append(assistant_message)

    reasoning_text = ""
    answer_text = ""
    done_reasoning = False
    pending_images = []  # 最终展示的图片
    pending_videos = []   # 最终展示的视频路径
    pending_files = []    # 最终展示的可下载文件（如 PSD）
    last_tool_images = list(recent_history_image_paths) if not uploaded_image_paths else []  # 上次工具生成的图片，用于工具链传递
    requested_api_model = _extract_requested_api_model(messages)
    requested_local_model = _has_requested_local_model(messages)

    try:
        from openai import OpenAI
        chat_model = cfg["model"]
        chat_base_url = cfg["base_url"]
        # 本地 Llama 服务（llama.cpp/Ollama/LM Studio）通常不需要 Key，
        # 但 OpenAI SDK 要求 api_key 非空，空时兜底为 "empty"（本地服务会忽略鉴权）。
        chat_api_key = cfg.get("api_key") or "empty"
        # 诊断日志：确认 LLM 配置
        _key_preview = cfg.get("api_key") or ""
        if len(_key_preview) > 12:
            _key_mask = _key_preview[:8] + "..." + _key_preview[-4:]
        elif _key_preview:
            _key_mask = "***"
        elif str(chat_base_url).startswith(("http://127.0.0.1", "http://localhost")):
            _key_mask = "(本地免Key)"
        else:
            _key_mask = "(空)"
        print(f"[Agent] LLM 客户端初始化: model={chat_model}, base_url={chat_base_url}, api_key={_key_mask}")
        client = OpenAI(api_key=chat_api_key, base_url=chat_base_url, max_retries=3)

        for iteration in range(cfg["max_tool_iterations"]):
            # 注意: Qwen3 thinking mode 不支持 tool_choice="required"（会报 400），
            # 因此始终使用 "auto"；@minimax-h3 等强制调用场景已通过 system prompt
            # 中的 "🔴🔴🔴 必须调用" 指令约束，效果等价。
            current_tool_choice = "auto"

            # 针对 429 上游限流的显式重试（指数退避：3s → 6s → 12s）
            stream = None
            for retry_attempt in range(3):
                try:
                    stream = client.chat.completions.create(
                        model=chat_model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice=current_tool_choice,
                        max_tokens=int(cfg.get("max_completion_tokens", 32768)),
                        stream=True,
                    )
                    break
                except Exception as api_err:
                    is_rate_limit = (
                        getattr(api_err, "status_code", None) == 429
                        or api_err.__class__.__name__ == "RateLimitError"
                    )
                    if not is_rate_limit or retry_attempt == 2:
                        raise
                    wait_sec = 3 * (2 ** retry_attempt)
                    print(f"[Agent] 对话模型限流(429)，第 {retry_attempt + 1}/3 次重试，等待 {wait_sec}s...")
                    yield new_history, f"⏳ 服务繁忙，{wait_sec}秒后重试...（第 {retry_attempt + 1}/3 次）"
                    time.sleep(wait_sec)

            current_answer = ""
            tool_calls_accumulator = {}
            last_thinking_yield = 0.0

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # reasoning（思考过程）：渲染为内嵌折叠小窗口，实时跟随；
                # 限频 0.2s 一次 yield，避免思考过长时高频刷新造成卡顿感
                reasoning = _extract_delta_reasoning(delta)
                if reasoning:
                    reasoning_text += reasoning
                    _now = time.time()
                    if _now - last_thinking_yield >= 0.2:
                        last_thinking_yield = _now
                        new_history[-1] = {
                            "role": "assistant",
                            "content": _thinking_block(reasoning_text, live=True),
                        }
                        yield new_history, "思考中..."

                # content
                content = delta.content
                if content:
                    if not done_reasoning and reasoning_text:
                        done_reasoning = True
                    current_answer += content
                    answer_text += content
                    new_history[-1] = {
                        "role": "assistant",
                        "content": _with_thinking(reasoning_text, answer_text),
                    }
                    yield new_history, "生成回复中..."

                # tool_calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if tc_delta.index is not None else 0
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                        tc = tool_calls_accumulator[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += tc_delta.function.arguments

            if tool_calls_accumulator:
                assistant_tool_calls = []
                for idx in sorted(tool_calls_accumulator.keys()):
                    tc = tool_calls_accumulator[idx]
                    assistant_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                    })

                messages.append({
                    "role": "assistant",
                    "content": current_answer or None,
                    "tool_calls": assistant_tool_calls,
                })

                for tc in assistant_tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_args = {}
                    original_tool_name = tool_name
                    requested_layer_separation = _has_requested_layer_separation(user_message)
                    # 检查配置中是否有 API 模型（用户在设置区保存的优先于历史本地标记）
                    _cfg_img_model_now = _configured_image_model()
                    _cfg_has_api_model_now = _is_configured_api_image_model(_cfg_img_model_now)
                    if requested_layer_separation and tool_name in ("remove_background", "segment_anything", "sam"):
                        tool_name = "layer_separation"
                        tool_args = {"output_format": tool_args.get("output_format", "psd")}
                        print(f"[Agent] 用户明确要求图层分离，已阻止 {original_tool_name}，强制使用 See-Through layer_separation")
                    elif requested_local_model and not _cfg_has_api_model_now and tool_name == "api_image_edit":
                        tool_name = "edit_image"
                        tool_args = {"instruction": tool_args.get("instruction") or user_message}
                        print("[Agent] 用户指定本地模型，已阻止 api_image_edit，改用本地 edit_image")
                    elif requested_local_model and not _cfg_has_api_model_now and tool_name == "api_image_generate":
                        # 用户选择了本地模型（如 @z_image），不能用 API 生图，强制改回本地 txt2img
                        prompt = tool_args.get("prompt") or user_message
                        tool_name = "txt2img"
                        tool_args = {"prompt": prompt}
                        print("[Agent] 用户指定本地模型，已阻止 api_image_generate，改用本地 txt2img")
                    else:
                        tool_name, tool_args = _coerce_api_image_edit_tool(
                            tool_name, tool_args, requested_api_model, user_message
                        )
                        # 只有当用户明确选择了本地模型 AND 配置中没有 API 模型时，才跳过 API 路由
                        # 如果配置了 API 模型（用户在设置区保存的），优先路由到 API
                        _cfg_img_model = _configured_image_model()
                        _cfg_has_api_model = _is_configured_api_image_model(_cfg_img_model)
                        if not requested_local_model or _cfg_has_api_model:
                            if requested_local_model and _cfg_has_api_model:
                                print(f"[Agent] 配置了 API 模型 {_cfg_img_model}，优先使用 API 路由，忽略历史本地模型标记")
                            tool_name, tool_args = _coerce_generation_tool_by_selected_model(
                                tool_name, tool_args, user_message
                            )
                    if tool_name != original_tool_name:
                        print(f"[Agent] 已按当前生成模型设置修正工具: {original_tool_name} -> {tool_name}")

                    yield new_history, f"🔧 正在执行: {tool_name}..."

                    tool_uploaded_image = uploaded_image_list if tool_name == "api_image_edit" else primary_uploaded_image
                    result_str, images = _execute_tool(
                        tool_name, tool_args, tool_uploaded_image, uploaded_video, last_tool_images
                    )
                    # 更新工具链图片（用于下游工具自动注入）
                    if images:
                        last_tool_images = images

                    # 解析工具结果，提取可下载文件（PSD 等）和视频
                    psd_path = None
                    video_path = None
                    glb_path = None
                    try:
                        result_data = json.loads(result_str)
                        info = result_data.get("info", result_data.get("data", {}))
                        if isinstance(info, dict):
                            psd_path = info.get("psd_path")
                            video_path = info.get("video_path")
                            glb_path = info.get("glb_path")
                    except Exception:
                        pass

                    # 图层分离有 PSD 产物时，只展示 PSD 下载，不展示图层预览图
                    if psd_path and os.path.isfile(psd_path):
                        pending_files.append(psd_path)
                    elif glb_path and os.path.isfile(glb_path):
                        pending_files.append(glb_path)
                    else:
                        pending_images.extend(images)

                    # 提取视频路径（h3_video_generate 等视频工具）
                    if video_path and os.path.isfile(video_path):
                        pending_videos.append(video_path)

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})

                    if psd_path and os.path.isfile(psd_path):
                        layer_count = 0
                        try:
                            result_data = json.loads(result_str)
                            layer_count = result_data.get("info", {}).get("layer_count", 0)
                        except Exception:
                            pass
                        yield new_history, f"✅ {tool_name} 完成，生成 {layer_count} 个图层，PSD 已就绪"
                    elif images:
                        yield new_history, f"✅ {tool_name} 完成，生成 {len(images)} 张图片"
                    elif pending_videos and len(pending_videos) > 0:
                        yield new_history, f"✅ {tool_name} 完成，生成视频"
                    else:
                        yield new_history, f"✅ {tool_name} 完成"

                continue

            break

        # 最终回答 + 图片 + 视频 + 可下载文件（PSD 等）
        # Gradio 5 Chatbot type='messages' 不支持列表混合格式，
        # 所以文本和图片/视频/文件分成多条 assistant 消息
        if pending_images or pending_videos or pending_files:
            # 文本消息
            # ===== 提前停止检测（修复"消息被截断/变空却无任何提示"） =====
            # 模型没写完就输出 EOS 时，llama.cpp 会当作正常完成（truncated=0），
            # 上下文往往还远没满。低量化本地模型（如 IQ2_XXS 2-bit）尤其常见：
            # 只输出思考过程没有正式回答，或思考中途停笔。
            # 这里保留思考过程并给出明确提示，避免用户误以为是"截断"或程序丢了内容。
            if not answer_text.strip():
                if reasoning_text.strip():
                    final_text = (_thinking_block(reasoning_text)
                                  + "\n\n⚠️ 提前停止提示：模型只完成了思考过程，未给出正式回答就停止了。"
                                    "这是模型自身提前结束（EOS），不是上下文上限问题。"
                                    "低量化本地模型（IQ2_XXS 等）较常见，建议重试，或换更高量化/云端模型。")
                else:
                    final_text = "⚠️ 模型返回了空回复（提前停止，不是上下文上限问题）。建议重试，或换更高量化/云端模型。"
            else:
                _thinking_unclosed = ("<think" in answer_text and "</think" not in answer_text) or (
                    bool(reasoning_text.strip()) and not done_reasoning
                )
                if _thinking_unclosed:
                    final_text = (answer_text
                                  + "\n\n⚠️ 提前停止提示：回复似乎被中断了，模型未生成完毕就停止了。"
                                    "低量化本地模型（IQ2_XXS 等）较常见。建议重试，或换更高量化/云端模型。")
                else:
                    final_text = answer_text
                final_text = _with_thinking(reasoning_text, final_text)
            new_history[-1] = {"role": "assistant", "content": final_text}
            # 每条图片单独一条消息，用 path dict 格式
            for i, img in enumerate(pending_images):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    new_history.append({
                        "role": "assistant",
                        "content": {"path": img_path, "alt_text": f"生成的图片 {i+1}"}
                    })
            # 每条视频单独一条消息，用 path dict 格式
            for i, vpath in enumerate(pending_videos):
                new_history.append({
                    "role": "assistant",
                    "content": {"path": vpath, "alt_text": f"生成的视频 {i+1}"}
                })
            # 每个可下载文件单独一条消息（PSD 等），Gradio 会自动渲染为下载附件
            for fpath in pending_files:
                fname = os.path.basename(fpath)
                new_history.append({
                    "role": "assistant",
                    "content": {"path": fpath, "alt_text": fname}
                })
            yield new_history, "✅ 完成"
        else:
            new_history[-1] = {"role": "assistant", "content": _with_thinking(reasoning_text, answer_text)}
            yield new_history, "✅ 完成"

    except Exception as e:
        if getattr(e, "status_code", None) == 429 or e.__class__.__name__ == "RateLimitError":
            error_msg = (
                "⏳ 对话模型服务繁忙（HTTP 429），上游负载已饱和。"
                f"当前对话模型: {chat_model}，端点: {chat_base_url}。"
                "已自动重试 3 次仍未成功，请稍后再试，或切换其他对话模型/供应商。"
            )
        elif getattr(e, "status_code", None) == 401 or e.__class__.__name__ in ("AuthenticationError", "APIConnectionError"):
            _ck = cfg.get("api_key") or ""
            _ck_mask = _ck[:8] + "..." + _ck[-4:] if len(_ck) > 12 else ("(空)" if not _ck else "***")
            error_msg = (
                "❌ Agent 对话模型认证失败（HTTP 401）。\n"
                f"当前对话模型: {chat_model}，端点: {chat_base_url}，API Key: {_ck_mask}\n"
                "这通常是因为：\n"
                "1. API Key 为空或无效 — 请在「Agent 大脑设置」中重新填写 API Key 并点保存\n"
                "2. API Key 与供应商不匹配 — 请确认供应商选择正确（如 ModelScope 的 key 不能用于 YoboxAI）\n"
                "3. API Key 已过期或额度用尽\n"
                "注意：这里的 Key 是 Agent 大脑（LLM 对话）用的，不是图像生成 Key。"
            )
        elif getattr(e, "status_code", None) == 403 or e.__class__.__name__ == "PermissionDeniedError":
            error_msg = (
                "❌ Agent 对话模型被上游拒绝（HTTP 403）。"
                f"当前对话模型: {chat_model}，端点: {chat_base_url}。"
                "这与图片上传或图像模型无关。请确认该供应商的 API Key 有权调用此模型的 chat/completions 接口，"
                "并确认 Base URL 是该供应商的对话接口地址。"
            )
        else:
            error_msg = f"❌ 出错了: {str(e)}"
        print(f"[Agent] 错误: {traceback.format_exc()}")
        new_history[-1] = {"role": "assistant", "content": error_msg}
        yield new_history, "❌ 错误"


# =============================================================================
# 合并注册系统中的外部工具
# =============================================================================

if _REGISTRY_AVAILABLE:
    # 合并已注册的工具到 TOOLS 和 TOOL_FUNCTIONS
    for reg_tool in get_registered_tools():
        name = reg_tool["function"]["name"]
        if name not in TOOL_FUNCTIONS:
            TOOLS.append(reg_tool)
            TOOL_FUNCTIONS[name] = get_tool_function(name)


def _refresh_registered_agent_tools():
    """在每次对话前同步其他插件新注册的 Agent 工具。"""
    if not _REGISTRY_AVAILABLE:
        return
    for reg_tool in get_registered_tools():
        name = reg_tool["function"]["name"]
        if name not in TOOL_FUNCTIONS:
            TOOLS.append(reg_tool)
            TOOL_FUNCTIONS[name] = get_tool_function(name)
