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
