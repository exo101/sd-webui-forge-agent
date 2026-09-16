# =============================================================================
# SD Webui Agent — AI 全能生图智能体（入口模块）
# 子模块:
#   agent_chat.py    对话核心（chat_stream / _execute_tool / 注册表合并）
#   agent_ui.py      Gradio 界面 / @mention 系统 / 供应商管理
#   agent_config.py  配置/工具函数
#   agent_tools.py   工具/TOOLS（门面，实现按领域拆分）
#   agent_prompts.py 系统提示词
# =============================================================================

from modules import script_callbacks

from scripts.agent_chat import (
    chat_stream, _execute_tool, _refresh_registered_agent_tools,
)
from scripts.agent_ui import (
    on_ui_tabs, MENTION_MAP,
    _parse_mentions, _handle_model_mention,
    _activate_api_model_mentions, _handle_tool_mentions,
    _upload_path, _classify_uploads,
)
from scripts.agent_tools import TOOLS, TOOL_FUNCTIONS, MODEL_GUIDE

# photoshop-mcp 桥接：注册 ps_state/ps_list_tools/ps_call/ps_preview 四个工具，
# 让 Agent 能驱动本机 Photoshop 2026（懒启动，不调用 PS 工具时零开销）
from scripts import photoshop_mcp  # noqa: F401

script_callbacks.on_ui_tabs(on_ui_tabs, name="sd_webui_agent_tab")
