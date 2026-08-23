from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger("forge_h3_studio.minimax_api")

MINIMAX_API_BASE = "https://api.minimaxi.com"
MINIMAX_VIDEO_ENDPOINT = f"{MINIMAX_API_BASE}/v2/video_generation"
MINIMAX_QUERY_ENDPOINT = f"{MINIMAX_API_BASE}/v2/video_generation/query"

# 支持的参数映射
RESOLUTION_MAP = {
    "768P": "768P",
    "2K": "2K",
}
DURATION_OPTIONS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
RATIO_OPTIONS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

# 生成模式 -> API content role 映射
MODE_ROLE_MAP = {
    "t2v": [],  # 仅 text
    "i2v": ["first_frame"],  # text + first_frame
    "fl2v": ["first_frame", "last_frame"],  # text + first_frame + last_frame
    "ref": ["reference_image", "reference_video", "reference_audio"],  # text + references
}


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_content(request: dict[str, Any]) -> list[dict[str, Any]]:
    """构建 MiniMax API 的 content 数组"""
    content: list[dict[str, Any]] = []

    # 必须的 text 项
    text = str(request.get("prompt") or "").strip()
    if not text:
        raise ValueError("提示词不能为空")
    content.append({"type": "text", "text": text})

    mode = request.get("mode", "t2v")

    # 首帧 / 尾帧
    if mode in ("i2v", "fl2v"):
        first_frame = request.get("first_frame")
        if first_frame:
            content.append({
                "type": "image_url",
                "image_url": {"url": _resolve_media_url(first_frame)},
                "role": "first_frame",
            })
    if mode == "fl2v":
        last_frame = request.get("last_frame")
        if last_frame:
            content.append({
                "type": "image_url",
                "image_url": {"url": _resolve_media_url(last_frame)},
                "role": "last_frame",
            })

    # 多模态参考
    if mode == "ref":
        refs = request.get("references") or []
        # 图片参考
        image_refs = [r for r in refs if r.get("kind") == "image"]
        for ref in image_refs:
            content.append({
                "type": "image_url",
                "image_url": {"url": _resolve_media_url(ref["file"])},
                "role": "reference_image",
            })
        # 视频参考
        video_refs = [r for r in refs if r.get("kind") == "video"]
        for ref in video_refs:
            content.append({
                "type": "video_url",
                "video_url": {"url": _resolve_media_url(ref["file"])},
                "role": "reference_video",
            })
        # 音频参考
        audio_refs = [r for r in refs if r.get("kind") == "audio"]
        for ref in audio_refs:
            content.append({
                "type": "audio_url",
                "audio_url": {"url": _resolve_media_url(ref["file"])},
                "role": "reference_audio",
            })

    return content


def _resolve_media_url(file_path: str) -> str:
    """将本地文件路径解析为 MiniMax API 可用的 URL"""
    # 如果已经是 URL，直接返回
    if file_path.startswith(("http://", "https://", "data:")):
        return file_path

    # 如果是本地文件，尝试上传到公网可访问的地址
    # 对于 ComfyUI 场景，文件路径是 ComfyUI 的 output 路径
    # 返回一个可访问的 URL 或直接使用文件路径
    # MiniMax 支持公网 URL，不支持本地文件路径
    # 这里返回文件路径，由调用方确保可访问性
    if os.path.isfile(file_path):
        # 如果文件存在，用户需要自己确保文件可公网访问
        # 或者使用 MiniMax 的 upload 机制
        return file_path

    # 对于 ComfyUI output 路径格式，保留原样
    return file_path


def _get_ratio(request: dict[str, Any]) -> str:
    """获取宽高比"""
    mode = request.get("mode", "t2v")
    ratio = str(request.get("aspect_ratio") or "16:9")

    # 图生视频模式，ratio 固定为 adaptive
    if mode in ("i2v", "fl2v"):
        return "adaptive"

    # 多模态参考模式，默认 adaptive
    if mode == "ref":
        return ratio if ratio in RATIO_OPTIONS else "adaptive"

    # 文生视频，ratio 必填
    return ratio if ratio in RATIO_OPTIONS else "16:9"


def submit_h3_task(
    request: dict[str, Any],
    api_key: str,
    poll_interval: float = 2.0,
    max_poll_time: float = 600.0,
) -> dict[str, Any]:
    """
    提交 MiniMax H3 视频生成任务并等待结果

    Args:
        request: 标准化后的生成请求
        api_key: MiniMax API Key
        poll_interval: 轮询间隔（秒）
        max_poll_time: 最大等待时间（秒）

    Returns:
        dict: 任务结果，包含状态、输出 URL 等
    """
    if not api_key:
        raise ValueError("MiniMax API Key 未配置")

    # 构建 content
    content = _build_content(request)

    # 分辨率
    resolution = RESOLUTION_MAP.get(str(request.get("resolution", "768P")), "768P")

    # 时长
    duration = int(request.get("duration", 5))
    if duration not in DURATION_OPTIONS:
        duration = 5

    # 宽高比
    ratio = _get_ratio(request)

    # 构建 payload
    payload = {
        "model": "MiniMax-H3",
        "content": content,
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
    }

    logger.info(f"MiniMax API: 提交 H3 任务, resolution={resolution}, duration={duration}s, ratio={ratio}")

    # 提交任务
    try:
        resp = requests.post(
            MINIMAX_VIDEO_ENDPOINT,
            headers=_headers(api_key),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        task_id = result.get("task_id")
        if not task_id:
            raise RuntimeError(f"MiniMax API 未返回 task_id: {result}")
        logger.info(f"MiniMax API: 任务已提交, task_id={task_id}")
    except requests.exceptions.RequestException as e:
        error_detail = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.text[:500]
            except Exception:
                pass
        raise RuntimeError(f"MiniMax API 提交失败: {e} {error_detail}") from e

    # 轮询任务状态
    start_time = time.time()
    last_status = ""
    while time.time() - start_time < max_poll_time:
        try:
            query_resp = requests.get(
                MINIMAX_QUERY_ENDPOINT,
                headers=_headers(api_key),
                params={"task_id": task_id},
                timeout=10,
            )
            query_resp.raise_for_status()
            query_result = query_resp.json()
            task = query_result.get("task", {})

            status = task.get("status", "unknown")
            if status != last_status:
                logger.info(f"MiniMax API: 任务 {task_id} 状态={status}")
                last_status = status

            if status == "succeeded":
                video_url = task.get("content", {}).get("url", "")
                logger.info(f"MiniMax API: 任务完成, video_url={video_url}")
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "video_url": video_url,
                    "resolution": task.get("resolution", resolution),
                    "duration": task.get("duration", duration),
                    "ratio": task.get("ratio", ratio),
                    "usage": task.get("usage", {}),
                }

            if status == "failed":
                error_msg = task.get("error", {}).get("message", "MiniMax API 任务失败")
                logger.error(f"MiniMax API: 任务失败, error={error_msg}")
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error_msg,
                }

            if status == "cancelled":
                logger.warning(f"MiniMax API: 任务已取消, task_id={task_id}")
                return {
                    "task_id": task_id,
                    "status": "cancelled",
                    "error": "任务已取消",
                }

        except requests.exceptions.RequestException as e:
            logger.warning(f"MiniMax API: 查询任务状态失败: {e}")

        time.sleep(poll_interval)

    logger.error(f"MiniMax API: 任务超时, task_id={task_id}")
    return {
        "task_id": task_id,
        "status": "failed",
        "error": f"任务超时（{max_poll_time}s）",
    }


def query_h3_task(task_id: str, api_key: str) -> dict[str, Any]:
    """查询 MiniMax H3 任务状态"""
    if not api_key:
        raise ValueError("MiniMax API Key 未配置")

    try:
        resp = requests.get(
            MINIMAX_QUERY_ENDPOINT,
            headers=_headers(api_key),
            params={"task_id": task_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"MiniMax API 查询失败: {e}") from e