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

# 首次启动时自动补齐 Agent 自己的第三方依赖。
# 必须在导入 agent_chat / agent_ui 之前执行，否则缺少 openai 时会直接加载失败。
try:
    from scripts.install_dependencies import ensure_dependencies
    ensure_dependencies()
except Exception as exc:
    # 不阻塞 WebUI 启动；聊天时会给出更明确的依赖错误。
    print(f"[Agent] 依赖自动检查失败: {exc}")

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

script_callbacks.on_ui_tabs(on_ui_tabs, name="sd_webui_agent_tab")
