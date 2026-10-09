"""绘梦智能体 REST API

为 MCP 服务器提供 HTTP 接口，直接复用绘梦智能体的完整工具实现。
端点:
  GET  /agent/api/tools          — 列出所有可用工具
  POST /agent/api/execute_tool   — 执行指定工具
"""
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional


class ExecuteToolRequest(BaseModel):
    tool_name: str
    tool_args: dict[str, Any] = {}
    image_path: Optional[str] = None
    video_path: Optional[str] = None


def _safe_json(data: Any) -> str:
    """安全序列化，处理不可序列化的对象。"""
    return json.dumps(data, ensure_ascii=False, default=str)


def register_api(app: FastAPI):
    """注册绘梦智能体的 REST API 路由。"""
    from scripts.agent_tools import TOOLS, TOOL_FUNCTIONS
    from scripts.agent_chat import _execute_tool

    @app.get("/agent/api/tools")
    async def list_tools():
        """返回所有可用工具的 schema。"""
        # 用 JSONResponse 手动序列化，避免 FastAPI jsonable_encoder 出错
        return JSONResponse(content={"tools": TOOLS})

    @app.post("/agent/api/execute_tool")
    async def execute_tool(req: ExecuteToolRequest):
        """执行指定工具，返回结果字符串和图片列表。"""
        tool_name = req.tool_name
        tool_args = req.tool_args or {}
        image_path = req.image_path
        video_path = req.video_path

        # 检查工具是否存在
        func = TOOL_FUNCTIONS.get(tool_name)
        if func is None:
            raise HTTPException(status_code=404, detail=f"未知工具: {tool_name}")

        try:
            result_str, images = _execute_tool(
                tool_name, tool_args,
                uploaded_image=image_path,
                uploaded_video=video_path,
            )
            # 将 PIL Image 保存为临时文件并返回路径，方便 MCP 客户端读取
            safe_images = []
            for img in (images or []):
                if img is None:
                    continue
                try:
                    from scripts.agent_config import _save_pil_to_tempfile
                    img_path = _save_pil_to_tempfile(img)
                    safe_images.append(img_path)
                except Exception:
                    # 如果保存失败，尝试转字符串
                    safe_images.append(str(img))
            return JSONResponse(content={
                "result": str(result_str) if result_str else "",
                "images": safe_images,
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"工具执行失败: {str(e)}\n{traceback.format_exc()[:500]}",
            )
