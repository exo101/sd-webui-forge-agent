# =============================================================================
# SD Webui Agent — AI 全能生图智能体（入口模块）
# 子模块: agent_config.py (配置/工具函数), agent_tools.py (工具/TOOLS),
#         agent_prompts.py (系统提示词)
# =============================================================================

import os
import json
import re
import time
import base64
import traceback

import gradio as gr
from PIL import Image

from modules import shared, scripts, sd_models, script_callbacks
from modules.processing import process_images

# =============================================================================
# 子模块导入
# =============================================================================
from scripts.agent_config import (
    load_config, save_config, _save_pil_to_tempfile, _detect_local_llama,
    _local_detection_cache, _local_detection_time,
    _REGISTRY_AVAILABLE, get_registered_tools, get_tool_function, list_registered_tools,
)
from scripts.agent_tools import (
    TOOLS, TOOL_FUNCTIONS, MODEL_GUIDE,
    set_model_components_tool,
)
from scripts.agent_prompts import _get_system_prompt


# =============================================================================
# @mention 系统：用户用 @标签 快速指定模型/功能
# =============================================================================

# @标签 → 动作映射
MENTION_MAP = {
    # 模型标签（type=model，值为 MODEL_GUIDE 的 key）
    "krea2": ("model", "krea2"),
    "klein": ("model", "flux-2-klein"),
    "klein9b": ("model", "flux-2-klein"),
    "anima": ("model", "anima"),
    "z_image": ("model", "z_image"),
    "zimage": ("model", "z_image"),
    "qwen": ("model", "qwen_image_edit"),
    "qwen-edit": ("model", "qwen_image_edit"),
    "XL": ("model", "xl"),
    "sdxl": ("model", "xl"),
    "illustrious": ("model", "illustrious"),

    # 功能标签（type=tool_hint，值为隐藏指令）
    "智能抠图": ("tool_hint", "用户明确要求抠图，请使用 remove_background 工具，mode=auto"),
    "点选分割": ("tool_hint", "用户明确要求点选分割，请使用 remove_background 工具，mode=point_click（需要用户提供坐标点）"),
    "图像清理": ("tool_hint", "用户明确要求图像清理，请使用 remove_background 工具，mode=cleanup"),
    "图层分离": ("tool_hint", "用户明确要求图层分离，请使用 remove_background 工具，mode=layer_separation"),
    "视频关键帧": ("tool_hint", "用户明确要求视频关键帧提取，请使用 video_keyframe_extract 工具"),
    "换背景": ("tool_hint", "用户明确要求换背景，请使用 change_background 工具"),
    "放大": ("tool_hint", "用户明确要求放大，请使用 upscale 工具"),
    "修脸": ("tool_hint", "用户明确要求修复人脸，请使用 apply_adetailer 工具"),
    "拼接": ("tool_hint", "用户明确要求图片拼接，请使用 stitch_images 工具"),

    # 扩展功能标签（stub，尚未实现，给 LLM 明确提示）
    "TTS": ("stub", "Qwen3-TTS 语音合成功能尚未集成到 Agent，建议使用 WebUI 原生 TTS 扩展"),
    "语音合成": ("stub", "Qwen3-TTS 语音合成功能尚未集成到 Agent，建议使用 WebUI 原生 TTS 扩展"),
    "Kling": ("stub", "Kling 可灵视频生成功能需要 API 配置，尚未集成到 Agent"),
    "可灵": ("stub", "Kling 可灵视频生成功能需要 API 配置，尚未集成到 Agent"),
    "ACE-Step": ("stub", "ACE-Step 音乐生成功能尚未集成到 Agent"),
    "音乐生成": ("stub", "ACE-Step 音乐生成功能尚未集成到 Agent"),
}


def _parse_mentions(user_text):
    """解析用户消息中的 @mention 标签。

    返回: (clean_text, actions)
    - clean_text: 去除 @标签后的用户文本（保留上下文）
    - actions: list of (tag, type, value)
    """
    if not user_text:
        return user_text, []

    actions = []
    # 匹配 @标签（支持中英文、数字、下划线、连字符）
    pattern = r"@([A-Za-z0-9_\-\u4e00-\u9fa5]+)"

    def replace_match(m):
        tag = m.group(1)
        # 大小写不敏感匹配
        tag_lower = tag.lower()
        for mention_tag, (atype, value) in MENTION_MAP.items():
            if tag_lower == mention_tag.lower():
                actions.append((tag, atype, value))
                return f"[{tag}]"  # 保留标签标记，让 LLM 知道用户指定了
        # 未知标签也保留
        return f"@{tag}"

    clean_text = re.sub(pattern, replace_match, user_text)
    return clean_text, actions


def _handle_model_mention(actions):
    """处理模型 @mention，自动切换模型。

    返回: (extra_system_note, switch_results)
    - extra_system_note: 注入给 LLM 的隐藏系统提示
    - switch_results: 切换结果列表
    """
    model_actions = [a for a in actions if a[1] == "model"]
    if not model_actions:
        return "", []

    notes = []
    results = []
    for tag, atype, guide_key in model_actions:
        try:
            guide = MODEL_GUIDE.get(guide_key, {})

            # 查找匹配的模型 — 关键词匹配，不写死文件名
            # 例如 @krea2 → 匹配标题含"krea2"的模型，@klein → 含"klein"的
            keyword = guide_key.lower()
            try:
                sd_models.list_models()
            except Exception:
                pass

            # 用关键词直接搜索 checkpoint title（比传文件名更可靠）
            matched_model = None
            if hasattr(sd_models, "get_closet_checkpoint_match"):
                matched_model = sd_models.get_closet_checkpoint_match(keyword)

            if not matched_model and hasattr(sd_models, "checkpoints_list") and sd_models.checkpoints_list:
                # 回退：遍历文件名匹配
                for info in sd_models.checkpoints_list.values():
                    if keyword in info.filename.lower():
                        matched_model = info
                        break

            if not matched_model:
                note = f"⚠️ 用户指定了 @{tag}，但未找到对应模型，请先安装该模型"
                notes.append(note)
                results.append({"tag": tag, "status": "not_found"})
                continue

            # 获取推荐的 TE/VAE
            te = guide.get("recommended_te", [None])[0] if guide.get("recommended_te") else None
            vae = guide.get("recommended_vae", [None])[0] if guide.get("recommended_vae") else None

            # 执行切换 — 传入 title 而非文件名，确保 get_closet_checkpoint_match 能匹配
            model_title = matched_model.title if hasattr(matched_model, "title") else matched_model.filename
            switch_result = set_model_components_tool(model_name=model_title, te_name=te, vae_name=vae)
            if switch_result.get("status") == "success":
                model_name = guide.get("name", model_title)

                # 同步切换 UI preset + 更新 agent 配置默认值
                preset_arch = guide.get("preset_arch")
                if preset_arch and hasattr(shared.opts, "forge_preset"):
                    shared.opts.set("forge_preset", preset_arch)

                    # 从 WebUI 预设系统中读取实际参数值，写入 agent 配置
                    try:
                        from modules_forge.presets import STEPS, SAMPLERS, SCHEDULERS, CFG, PresetArch
                        arch = PresetArch[preset_arch]
                        preset_steps = STEPS.get(arch)
                        preset_sampler = SAMPLERS.get(arch)
                        preset_scheduler = SCHEDULERS.get(arch)
                        preset_cfg = CFG.get(arch)

                        # 更新 agent 配置中的默认值，使 txt2img_tool 等函数使用正确的预设参数
                        _cfg = load_config()
                        if preset_steps:
                            _cfg["default_steps"] = preset_steps
                        if preset_cfg is not None:
                            _cfg["default_cfg_scale"] = preset_cfg
                        save_config(_cfg)
                    except Exception as e:
                        print(f"[Agent] 读取预设参数失败: {e}")

                    try:
                        shared.opts.save(shared.config_filename)
                    except Exception:
                        pass
                    preset_note = f"（已应用 {preset_arch.upper()} 预设: steps={preset_steps}, CFG={preset_cfg}, sampler={preset_sampler}）"
                else:
                    preset_note = ""

                note = f"✅ 用户指定了 @{tag}，已自动切换到 {model_name} 模型，现在可以直接生图，无需再次切换模型。{preset_note}"
                notes.append(note)
                results.append({"tag": tag, "status": "success", "model": model_name, "file": matched_model.filename if hasattr(matched_model, "filename") else model_title, "preset": preset_arch})
            else:
                note = f"⚠️ 用户指定了 @{tag}，但模型切换失败: {switch_result.get('error', '未知错误')}"
                notes.append(note)
                results.append({"tag": tag, "status": "failed", "error": switch_result.get("error")})
        except Exception as e:
            note = f"⚠️ @{tag} 模型切换异常: {e}"
            notes.append(note)
            results.append({"tag": tag, "status": "error", "error": str(e)})

    return "\n".join(notes), results


def _handle_tool_mentions(actions):
    """处理工具 @mention，收集隐藏指令。"""
    hints = []
    for tag, atype, value in actions:
        if atype == "tool_hint":
            hints.append(f"[用户标签 @{tag}] {value}")
        elif atype == "stub":
            hints.append(f"[用户标签 @{tag}] {value}")
    return "\n".join(hints)


# =============================================================================
# Agent 核心：流式对话 + Function Calling 循环
# =============================================================================

def _execute_tool(tool_name, tool_args, uploaded_image=None, uploaded_video=None, last_tool_images=None):
    """执行工具调用，返回 (result_string, images_list)。
    自动传入上传的图片/视频 + 上次工具生成的图片到对应参数。
    支持工具链：如 提取关键帧 → 拼接 → 放大。"""
    if last_tool_images is None:
        last_tool_images = []

    if tool_name not in TOOL_FUNCTIONS:
        # 检查注册系统
        registered_func = get_tool_function(tool_name) if _REGISTRY_AVAILABLE else None
        if registered_func:
            try:
                result = registered_func(**tool_args)
                if isinstance(result, tuple) and len(result) == 2:
                    images, info = result
                    return json.dumps({"status": "success", "info": info}, ensure_ascii=False), images
                return json.dumps({"status": "success", "data": result}, ensure_ascii=False), []
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), []
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False), []

    try:
        func = TOOL_FUNCTIONS[tool_name]

        # ===== 自动注入图片参数（优先级：用户上传 > 上次工具生成） =====
        # 单图工具：img2img, upscale, apply_adetailer, remove_background, edit_image, change_background
        if tool_name in ("img2img", "upscale", "apply_adetailer", "remove_background", "edit_image", "change_background"):
            if "image" not in tool_args or tool_args["image"] is None:
                if uploaded_image is not None:
                    tool_args["image"] = uploaded_image
                elif last_tool_images:
                    # 用上次生成的最后一张图
                    tool_args["image"] = last_tool_images[-1]

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
            # 统一转为列表
            if not isinstance(images, list):
                images = [images]
            result_str = json.dumps({"status": "success", "info": info, "image_count": len(images)}, ensure_ascii=False)
            return result_str, images

        return json.dumps({"status": "success", "data": result}, ensure_ascii=False), []

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[Agent] 工具执行失败 [{tool_name}]: {error_msg}")
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False), []


def _extract_text_from_history_content(content):
    """从 history 的 content 中提取文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    # 图片 dict {"path": ...} 没有文本
    if isinstance(content, dict) and "path" in content:
        return ""
    return str(content)


def chat_stream(history, uploaded_image=None, uploaded_video=None):
    """流式聊天生成器。从 history 中提取最后一条用户消息。

    Yields: (history_update, status_message)
    """
    cfg = load_config()

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

    # 构建 OpenAI 消息格式
    # Gradio history 中图片用 {"path": filepath, "alt_text": "..."} 格式
    # 需要转换为 OpenAI 的 image_url (base64) 格式
    messages = [{"role": "system", "content": _get_system_prompt(cfg.get("model", ""))}]

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
                    if not os.path.isfile(img_path):
                        print(f"[Agent] ⚠️ 图片文件不存在: {img_path}")
                        continue
                    file_size = os.path.getsize(img_path)
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    pending_user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                    image_count += 1
                    print(f"[Agent] ✅ 图片已加载: {os.path.basename(img_path)} ({file_size//1024}KB, base64长度={len(b64)})")
                except Exception as e:
                    print(f"[Agent] ❌ 读取图片失败: {e}")
        else:
            # assistant 消息
            _flush_pending_user()
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            # 忽略 assistant 的图片消息（不发给 API）

    _flush_pending_user()

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
    last_tool_images = []  # 上次工具生成的图片，用于工具链传递

    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

        for iteration in range(cfg["max_tool_iterations"]):
            stream = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            current_answer = ""
            tool_calls_accumulator = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # reasoning_content
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_text += reasoning
                    new_history[-1] = {
                        "role": "assistant",
                        "content": f"**思考过程：**\n{reasoning_text}\n\n---\n\n{answer_text}" if not done_reasoning else answer_text
                    }
                    yield new_history, "思考中..."

                # content
                content = delta.content
                if content:
                    if not done_reasoning and reasoning_text:
                        done_reasoning = True
                        answer_text += "\n\n"
                    current_answer += content
                    answer_text += content
                    new_history[-1] = {"role": "assistant", "content": answer_text}
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

                    yield new_history, f"🔧 正在执行: {tool_name}..."

                    result_str, images = _execute_tool(
                        tool_name, tool_args, uploaded_image, uploaded_video, last_tool_images
                    )
                    pending_images.extend(images)
                    # 更新工具链图片（用于下游工具自动注入）
                    if images:
                        last_tool_images = images

                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})

                    if images:
                        yield new_history, f"✅ {tool_name} 完成，生成 {len(images)} 张图片"
                    else:
                        yield new_history, f"✅ {tool_name} 完成"

                continue

            break

        # 最终回答 + 图片
        # Gradio 5 Chatbot type='messages' 不支持列表混合格式，
        # 所以文本和图片分成多条 assistant 消息
        if pending_images:
            # 文本消息
            final_text = answer_text if answer_text else "已为你生成图片："
            new_history[-1] = {"role": "assistant", "content": final_text}
            # 每条图片单独一条消息，用 path dict 格式
            for i, img in enumerate(pending_images):
                img_path = _save_pil_to_tempfile(img)
                if img_path:
                    new_history.append({
                        "role": "assistant",
                        "content": {"path": img_path, "alt_text": f"生成的图片 {i+1}"}
                    })
            yield new_history, "✅ 完成"
        else:
            new_history[-1] = {"role": "assistant", "content": answer_text}
            yield new_history, "✅ 完成"

    except Exception as e:
        error_msg = f"❌ 出错了: {str(e)}"
        print(f"[Agent] 错误: {traceback.format_exc()}")
        new_history[-1] = {"role": "assistant", "content": error_msg}
        yield new_history, "❌ 错误"


# =============================================================================
# UI
# =============================================================================

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as agent_interface:
        gr.HTML("""
        <div style="text-align:center; margin-bottom: 10px;">
            <h2 style="color: #c084fc;">绘梦智能体助手 — AI 全能生图智能体</h2>
            <p style="color: #9ca3af; font-size: 14px;">
                基于 Qwen3.8 · ModelScope API · 可指挥我生图/改图/换模型/调参数/放大
            </p>
        </div>
        """)

        # 辅助函数：创建标签按钮并绑定点击插入（纯客户端 DOM 操作，不依赖 msg_input 定义顺序）
        def _make_tag_btn(label, tag_text, css_class):
            btn = gr.Button(label, elem_classes=f"agent-tag-btn {css_class}", size="sm")
            if tag_text == "__clear__":
                js_code = """() => { const ta = document.querySelector('#agent_chat_input textarea'); if(!ta) return; ta.value=''; ta.dispatchEvent(new Event('input',{bubbles:true})); ta.dispatchEvent(new Event('change',{bubbles:true})); ta.focus(); }"""
            else:
                safe_tag = tag_text.replace("'", "\\'")
                js_code = f"""() => {{ const ta = document.querySelector('#agent_chat_input textarea'); if(!ta) return; let v=ta.value||''; if(v && !v.endsWith(' ') && !v.endsWith('\\n')) v+=' '; ta.value=v+'{safe_tag} '; ta.dispatchEvent(new Event('input',{{bubbles:true}})); ta.dispatchEvent(new Event('change',{{bubbles:true}})); ta.focus(); }}"""
            # fn=lambda: None 确保事件注册，js 为纯客户端 DOM 操作
            btn.click(fn=lambda: None, inputs=None, outputs=None, js=js_code)
            return btn

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话", height=520, type="messages", show_label=False,
                )

                # 预设标签 chips（一键插入 @标签）— 使用 Gradio 原生按钮，确保交互可靠
                gr.HTML("""
<style>
.agent-tag-label { font-size: 11px; color: var(--body-text-color-subdued, #888); font-weight: 600; min-width: 32px; user-select: none; display: inline-flex; align-items: center; }
#agent_tags_models, #agent_tags_tools, #agent_tags_ext { flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
#agent_tags_models > *, #agent_tags_tools > *, #agent_tags_ext > * { flex: 0 0 auto !important; }
.agent-tag-btn {
  background: transparent; border: none; outline: none; box-shadow: none;
  cursor: pointer; user-select: none;
  padding: 3px 10px; font-size: 12px; line-height: 1.5;
  border-radius: 14px; transition: all 0.18s ease;
  font-weight: 500; display: inline-block; text-align: center;
  margin: 1px;
}
.agent-tag-btn:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(0,0,0,0.12); }
.agent-tag-btn:active { transform: translateY(0); box-shadow: inset 0 1px 3px rgba(0,0,0,0.15); }
.tag-model-btn { background: linear-gradient(135deg, #e6f4ff 0%, #d0e8ff 100%); color: #1677ff; }
.tag-model-btn:hover { background: linear-gradient(135deg, #d0e8ff 0%, #bae0ff 100%); }
.tag-tool-btn { background: linear-gradient(135deg, #f6ffed 0%, #e8f8d5 100%); color: #389e0d; }
.tag-tool-btn:hover { background: linear-gradient(135deg, #e8f8d5 0%, #d9f7be 100%); }
.tag-ext-btn { background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%); color: #595959; }
.tag-ext-btn:hover { background: linear-gradient(135deg, #f0f0f0 0%, #e6e6e6 100%); }
.tag-clear-btn { background: linear-gradient(135deg, #fff2f0 0%, #ffe0de 100%); color: #cf1322; }
.tag-clear-btn:hover { background: linear-gradient(135deg, #ffe0de 0%, #ffccc7 100%); }
</style>
""")

                # 模型/功能/扩展标签按钮 — 全部收进折叠列表减少空间占用
                with gr.Accordion("@ 快捷指令", open=False):
                    # 模型标签行
                    with gr.Row(elem_id="agent_tags_models"):
                        gr.HTML('<span class="agent-tag-label">模型</span>')
                        _make_tag_btn("@krea2", "@krea2", "tag-model-btn")
                        _make_tag_btn("@klein", "@klein", "tag-model-btn")
                        _make_tag_btn("@anima", "@anima", "tag-model-btn")
                        _make_tag_btn("@z_image", "@z_image", "tag-model-btn")
                        _make_tag_btn("@qwen", "@qwen", "tag-model-btn")
                        _make_tag_btn("@XL", "@XL", "tag-model-btn")

                    # 功能标签行
                    with gr.Row(elem_id="agent_tags_tools"):
                        gr.HTML('<span class="agent-tag-label">功能</span>')
                        _make_tag_btn("@智能抠图", "@智能抠图", "tag-tool-btn")
                        _make_tag_btn("@点选分割", "@点选分割", "tag-tool-btn")
                        _make_tag_btn("@图像清理", "@图像清理", "tag-tool-btn")
                        _make_tag_btn("@图层分离", "@图层分离", "tag-tool-btn")
                        _make_tag_btn("@视频关键帧", "@视频关键帧", "tag-tool-btn")
                        _make_tag_btn("@换背景", "@换背景", "tag-tool-btn")
                        _make_tag_btn("@放大", "@放大", "tag-tool-btn")
                        _make_tag_btn("@修脸", "@修脸", "tag-tool-btn")
                        _make_tag_btn("@拼接", "@拼接", "tag-tool-btn")

                    # 扩展标签行
                    with gr.Row(elem_id="agent_tags_ext"):
                        gr.HTML('<span class="agent-tag-label">扩展</span>')
                        _make_tag_btn("@TTS", "@TTS", "tag-ext-btn")
                        _make_tag_btn("@Kling", "@Kling", "tag-ext-btn")
                        _make_tag_btn("@ACE-Step", "@ACE-Step", "tag-ext-btn")
                        _make_tag_btn("✕ 清除", "__clear__", "tag-clear-btn")

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入消息",
                        placeholder="例如：画一只可爱的橘猫，坐在窗台上晒太阳...",
                        scale=4, show_label=False,
                        elem_id="agent_chat_input",
                    )
                    send_btn = gr.Button("📤 发送", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ 清空", scale=1)

                status = gr.Textbox(label="状态", value="就绪", interactive=False, show_label=False)

            # 隐藏的 State 保存上传的图片/视频（避免被 UI 清除影响事件链）
            state_image = gr.State(None)
            state_video = gr.State(None)

            with gr.Column(scale=1):
                upload_image = gr.Image(label="参考图片 (可选)", type="pil", height=180)
                clear_image_btn = gr.Button("清除图片", size="sm")

                upload_video = gr.Video(
                    label="参考视频 (可选, 用于关键帧提取)",
                    height=180,
                    sources=["upload"],
                )
                clear_video_btn = gr.Button("清除视频", size="sm")

                with gr.Accordion("⚙️ API 设置", open=False):
                    cfg_init = load_config()
                    local_mode = gr.Checkbox(
                        label="优先使用本地 llama-server（自动检测，找不到回退云端）",
                        value=cfg_init.get("local_mode", True)
                    )
                    with gr.Row():
                        detect_btn = gr.Button("🔍 重新检测本地模型", size="sm")
                        detect_status = gr.Textbox(show_label=False, interactive=False, scale=3)
                    api_key = gr.Textbox(label="API Key（云端用，本地可留空）", value=cfg_init["api_key"], type="password")
                    base_url = gr.Textbox(label="Base URL", value=cfg_init["base_url"])
                    model_name = gr.Textbox(label="模型 ID", value=cfg_init["model"])
                    save_settings_btn = gr.Button("💾 保存设置", variant="secondary")
                    settings_status = gr.Textbox(show_label=False, interactive=False)

                with gr.Accordion("📖 使用提示", open=False):
                    gr.Markdown("""
                    **🎨 生图：**
                    - "画一只可爱的橘猫坐在窗台上"
                    - "生成赛博朋克风格的城市夜景，4K分辨率"

                    **🖼️ 图生图：** 上传参考图后说
                    - "把这张图变成水彩画风格"
                    - "放大这张图2倍" / "修复人脸"

                    **🎬 视频处理：** 上传视频后说
                    - "提取关键帧" / "截取5帧"
                    - "每秒截取一帧"

                    **🔧 指挥操作：**
                    - "切换到 SDXL 模型"
                    - "把步数改成30，尺寸改成1024x1024"
                    - "把这几张图拼起来"

                    **📋 查询：**
                    - "有哪些模型？" / "当前设置是什么？" / "有哪些插件？"
                    """)

        # ===== 事件绑定 =====

        def on_image_upload(img):
            """图片上传时保存到 State。"""
            return img

        def on_video_upload(vid):
            """视频上传时保存到 State。"""
            return vid

        def prepare_message(message, history, image, video):
            """把用户消息加入 history，清空输入框，返回新 history。
            Gradio 5 Chatbot type='messages' 不支持列表混合格式，
            所以文本和图片分成两条消息。
            同时解析 @mention 标签，自动切换模型/注入工具提示。"""
            new_history = list(history)

            # ===== @mention 处理 =====
            clean_text, actions = _parse_mentions(message)

            # 处理模型 @mention（自动切换）
            model_note, switch_results = _handle_model_mention(actions)
            # 处理工具 @mention（收集隐藏指令）
            tool_hints = _handle_tool_mentions(actions)

            # 如果有隐藏提示，作为 system 消息注入（LLM 能看到但用户 UI 不显示）
            # 注意：不能 append 到 history 末尾——本地 LLM 的 chat template 要求
            # system 消息必须在开头，放在 assistant 后面会报错 "System message must be at the beginning"。
            # 改为合并到已有的第一条 system 消息中，或插入到 history 开头。
            hidden_notes = []
            if model_note:
                hidden_notes.append(model_note)
            if tool_hints:
                hidden_notes.append(tool_hints)
            if hidden_notes:
                hint_text = "[系统提示 - 以下是用户 @标签 触发的自动操作]\n" + "\n".join(hidden_notes) + "\n[请根据以上提示执行任务]"
                # 找到开头连续 system 消息中的最后一条
                merge_idx = None
                for i, msg in enumerate(new_history):
                    if msg["role"] == "system":
                        merge_idx = i
                    else:
                        break
                if merge_idx is not None:
                    new_history[merge_idx]["content"] += "\n\n" + hint_text
                else:
                    new_history.insert(0, {"role": "system", "content": hint_text})

            # 用户消息（保留 [标签] 标记让 LLM 理解上下文）
            new_history.append({"role": "user", "content": clean_text})

            if image is not None:
                img_path = _save_pil_to_tempfile(image)
                if img_path:
                    new_history.append({"role": "user", "content": {"path": img_path, "alt_text": "参考图片"}})
            if video is not None:
                video_path = video if isinstance(video, str) else (video.get("path") if isinstance(video, dict) else None)
                if video_path and os.path.isfile(video_path):
                    new_history.append({"role": "user", "content": {"path": video_path, "alt_text": "参考视频"}})
            return "", new_history

        def send_chat(history, image, video):
            """从 history 提取最后一条用户消息进行回复。
            image/video 从 State 获取，不会被 UI 清除影响。"""
            video_path = video if isinstance(video, str) else (video.get("path") if isinstance(video, dict) else None)
            yield from chat_stream(history, image, video_path)

        def clear_uploads():
            """发送完成后清除上传组件和 State。"""
            return None, None, None, None

        def save_settings(key, url, model, use_local):
            cfg = load_config()
            cfg["api_key"] = key
            cfg["base_url"] = url
            cfg["model"] = model
            cfg["local_mode"] = use_local
            ok = save_config(cfg)
            return "✅ 设置已保存" if ok else "❌ 保存失败"

        def detect_local():
            """手动触发本地模型检测，清除缓存强制重新扫描。"""
            global _local_detection_cache, _local_detection_time
            _local_detection_cache = None
            _local_detection_time = 0
            detected = _detect_local_llama()
            if detected:
                base_url_val, _, model_val = detected
                return f"✅ 检测到本地模型: {model_val} @ {base_url_val}", model_val, base_url_val, "local"
            return "⚠️ 未检测到本地 llama-server（端口 8079-8090），请先启动启动器", "", "", ""

        # 上传事件：同步到 State
        upload_image.change(fn=on_image_upload, inputs=[upload_image], outputs=[state_image])
        upload_video.change(fn=on_video_upload, inputs=[upload_video], outputs=[state_video])

        save_settings_btn.click(fn=save_settings, inputs=[api_key, base_url, model_name, local_mode], outputs=[settings_status])
        detect_btn.click(fn=detect_local, outputs=[detect_status, model_name, base_url, api_key])
        clear_image_btn.click(fn=lambda: (None, None), outputs=[upload_image, state_image])
        clear_video_btn.click(fn=lambda: (None, None), outputs=[upload_video, state_video])

        # 发送按钮：prepare → send_chat → clear_uploads
        send_btn.click(
            fn=prepare_message,
            inputs=[msg_input, chatbot, state_image, state_video],
            outputs=[msg_input, chatbot],
        ).then(
            fn=send_chat,
            inputs=[chatbot, state_image, state_video],
            outputs=[chatbot, status],
        ).then(
            fn=clear_uploads,
            outputs=[upload_image, upload_video, state_image, state_video],
        )

        # 回车发送
        msg_input.submit(
            fn=prepare_message,
            inputs=[msg_input, chatbot, state_image, state_video],
            outputs=[msg_input, chatbot],
        ).then(
            fn=send_chat,
            inputs=[chatbot, state_image, state_video],
            outputs=[chatbot, status],
        ).then(
            fn=clear_uploads,
            outputs=[upload_image, upload_video, state_image, state_video],
        )

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

    return [(agent_interface, "绘梦智能体助手", "sd_webui_agent")]


script_callbacks.on_ui_tabs(on_ui_tabs, name="sd_webui_agent_tab")


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
            print(f"[Agent] 已加载外部注册工具: {name}")

    registered = list_registered_tools()
    if registered:
        print(f"[Agent] 共加载 {len(registered)} 个外部注册工具")

print(f"[Agent] SD Webui Agent 扩展已加载 (全能版, 共 {len(TOOL_FUNCTIONS)} 个工具)")