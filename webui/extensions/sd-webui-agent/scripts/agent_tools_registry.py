# =============================================================================
# Agent Tools Registry — 工具注册系统
# 其他插件可以通过 @agent_tool 装饰器自主注册工具到 Agent
# =============================================================================

from typing import Dict, Callable, Any
import inspect

# 全局工具注册表
_REGISTERED_TOOLS: Dict[str, dict] = {}


def agent_tool(name: str, description: str, parameters: dict = None):
    """装饰器：把一个函数注册为 Agent 可调用的工具。

    参数:
        name: 工具名称 (英文，唯一)
        description: 工具描述 (中文，告诉 AI 何时使用)
        parameters: OpenAI function calling schema 的 parameters 字段

    示例:
        @agent_tool(
            name="my_plugin_action",
            description="执行我的插件的某个功能",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "提示词"}
                },
                "required": ["prompt"]
            }
        )
        def my_action(prompt: str):
            return {"status": "success", "result": prompt}
    """
    def decorator(func: Callable):
        # 自动从函数签名推断参数（如果没提供）
        tool_params = parameters
        if tool_params is None:
            sig = inspect.signature(func)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "image", "uploaded_image"):
                    continue
                ptype = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in (int,):
                        ptype = "integer"
                    elif param.annotation in (float,):
                        ptype = "number"
                    elif param.annotation in (bool,):
                        ptype = "boolean"
                properties[param_name] = {"type": ptype, "description": f"参数 {param_name}"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            tool_params = {"type": "object", "properties": properties}
            if required:
                tool_params["required"] = required

        _REGISTERED_TOOLS[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": tool_params,
            },
            "_func": func,
        }
        print(f"[Agent Registry] 工具已注册: {name}")
        return func
    return decorator


def get_registered_tools():
    """获取所有已注册工具的 OpenAI schema（不含私有 _func）。"""
    return [
        {k: v for k, v in tool.items() if k != "_func"}
        for tool in _REGISTERED_TOOLS.values()
    ]


def get_tool_function(name: str):
    """获取已注册工具的函数。"""
    tool = _REGISTERED_TOOLS.get(name)
    return tool["_func"] if tool else None


def list_registered_tools():
    """列出所有已注册工具名称。"""
    return list(_REGISTERED_TOOLS.keys())


# =============================================================================
# Stub 工具占位：未来可由对应插件替换为真实实现
# =============================================================================

@agent_tool(
    name="qwen_tts",
    description="Qwen3-TTS 语音合成：把文字转为语音。用户说'朗读'/'配音'/'语音'时使用。当前为 stub，需要配置 TTS 扩展。",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要合成语音的文本内容"},
            "voice": {"type": "string", "description": "语音风格/角色（可选）", "default": "default"},
        },
        "required": ["text"],
    },
)
def qwen_tts_stub(text: str, voice: str = "default"):
    return {
        "status": "not_configured",
        "message": "Qwen3-TTS 语音合成尚未集成到 Agent",
        "hint": "请安装 Qwen3-TTS WebUI 扩展，或在 WebUI 原生 TTS 功能中使用",
        "requested_text": text,
        "requested_voice": voice,
    }


@agent_tool(
    name="kling_video",
    description="Kling 可灵视频生成：根据文字或图片生成视频。用户说'生成视频'/'可灵'/'动起来'时使用。当前为 stub，需要配置 Kling API。",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "视频描述提示词"},
            "image_path": {"type": "string", "description": "参考图片路径（可选，图生视频）"},
            "duration": {"type": "integer", "description": "视频时长秒数", "default": 5},
        },
        "required": ["prompt"],
    },
)
def kling_video_stub(prompt: str, image_path: str = "", duration: int = 5):
    return {
        "status": "not_configured",
        "message": "Kling 可灵视频生成尚未集成到 Agent",
        "hint": "需要配置 Kling API Key 后才能使用，请在设置中添加或手动使用可灵平台",
        "requested_prompt": prompt,
        "requested_duration": duration,
    }


@agent_tool(
    name="ace_step_music",
    description="ACE-Step 音乐生成：根据描述生成音乐。用户说'生成音乐'/'配乐'/'BGM'时使用。当前为 stub，需要配置 ACE-Step 扩展。",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "音乐风格/描述提示词"},
            "duration": {"type": "integer", "description": "音乐时长秒数", "default": 30},
            "genre": {"type": "string", "description": "音乐流派（可选）", "default": ""},
        },
        "required": ["prompt"],
    },
)
def ace_step_music_stub(prompt: str, duration: int = 30, genre: str = ""):
    return {
        "status": "not_configured",
        "message": "ACE-Step 音乐生成尚未集成到 Agent",
        "hint": "请安装 ACE-Step WebUI 扩展后即可使用",
        "requested_prompt": prompt,
        "requested_duration": duration,
        "requested_genre": genre,
    }
