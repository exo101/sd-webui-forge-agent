"""MiniMax H3 cloud API client.

Wraps the asynchronous video-generation endpoints so the Forge H3 Studio
workbench can run without a local ComfyUI installation when the user
chooses backend_mode == "api".
"""
from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from .config import DATA_DIR, load_config
from .errors import H3StudioError

CLOUD_UPLOAD_DIR = DATA_DIR / "cloud_uploads"

# MiniMax H3 public limits
MIN_DURATION = 4
MAX_DURATION = 15
H3_FPS = 24

_ASPECT_MAP = {
    "16:9": "16:9",
    "9:16": "9:16",
    "1:1": "1:1",
    "4:3": "4:3",
    "3:4": "3:4",
    "21:9": "21:9",
    "adaptive": "adaptive",
}


def _asset_path(filename: str) -> Path | None:
    """Locate an uploaded asset file.

    In cloud mode assets are stored under data/cloud_uploads. We also fall
    back to ComfyUI's input directory when a portable ComfyUI path is
    configured, which keeps older workflows working.
    """
    if not filename:
        return None
    candidate = CLOUD_UPLOAD_DIR / Path(filename).name
    if candidate.is_file():
        return candidate
    config = load_config()
    comfy_path = str(config.get("comfy_path") or "").strip()
    if comfy_path:
        input_dir = Path(comfy_path) / "ComfyUI" / "input"
        if not input_dir.is_dir():
            input_dir = Path(comfy_path) / "input"
        fallback = input_dir / filename
        if fallback.is_file():
            return fallback
    return None


def _image_data_url(filename: str) -> str | None:
    path = _asset_path(filename)
    if path is None:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class CloudClient:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.api_key = str(self.config.get("minimax_api_key") or "").strip()
        # MiniMax domestic platform (platform.minimax.cn) uses api.minimaxi.com
        self.base = str(self.config.get("minimax_api_base") or "https://api.minimaxi.com").rstrip("/")
        self.timeout = float(self.config.get("request_timeout") or 30)

    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        """Translate the Forge H3 Studio request into MiniMax H3 V2 API format.

        Official spec: https://platform.minimax.cn/docs/api-reference/video-generation-v2-create
        - content is a multimodal array with text + image_url items
        - first/last frame images use role="first_frame"/"last_frame" inside content
        - resolution: 768P or 2K; duration: 4-15 integer seconds
        - ratio: required for t2v (non-adaptive); forced adaptive for i2v
        """
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise H3StudioError("提示词不能为空")

        width = int(request.get("width") or 1344)
        height = int(request.get("height") or 768)
        resolution = "2K" if height >= 1440 else "768P"

        frames = int(request.get("frames") or (5 * H3_FPS))
        duration = max(MIN_DURATION, min(MAX_DURATION, round(frames / H3_FPS)))

        aspect = str(request.get("aspect_ratio") or "").strip()

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        first = str(request.get("first_frame") or "").strip()
        last = str(request.get("last_frame") or "").strip()
        has_frames = bool(first or last)

        # First/last frame images go INSIDE content array with role (official V2 spec)
        if first:
            first_url = _image_data_url(first)
            if first_url:
                content.append({"type": "image_url", "role": "first_frame", "image_url": {"url": first_url}})
        if last:
            last_url = _image_data_url(last)
            if last_url:
                content.append({"type": "image_url", "role": "last_frame", "image_url": {"url": last_url}})

        # Reference images (multimodal reference mode)
        has_ref = False
        for ref in request.get("references") or []:
            if ref.get("kind") != "image":
                continue
            data_url = _image_data_url(str(ref.get("file") or ""))
            if data_url:
                content.append({"type": "image_url", "role": "reference_image", "image_url": {"url": data_url}})
                has_ref = True

        # Ratio rules:
        # - t2v (no images): ratio required, cannot be adaptive
        # - i2v (first/last frame): ratio forced adaptive
        # - ref mode: ratio optional, default adaptive
        if has_frames:
            ratio = "adaptive"
        elif has_ref:
            ratio = _ASPECT_MAP.get(aspect, "adaptive")
        else:
            # Pure text-to-video: ratio must be a concrete value
            ratio = _ASPECT_MAP.get(aspect) if aspect in _ASPECT_MAP else "16:9"
            if ratio == "adaptive":
                ratio = "16:9"

        payload: dict[str, Any] = {
            "model": "MiniMax-H3",
            "content": content,
            "resolution": resolution,
            "duration": duration,
            "ratio": ratio,
        }

        # Optional seed (not in V2 spec but harmless if server ignores)
        seed = int(request.get("seed") or 0)
        if seed > 0:
            payload["seed"] = seed

        return payload

    def submit(self, request: dict[str, Any]) -> str:
        """Submit a video generation task, return the task_id."""
        if not self.enabled():
            raise H3StudioError("尚未配置 MiniMax API Key，请在设置中填写")
        payload = self._build_payload(request)
        try:
            response = httpx.post(
                f"{self.base}/v2/video_generation",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise H3StudioError(f"MiniMax API 连接失败：{exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:500]
            raise H3StudioError(f"MiniMax API 提交失败（HTTP {response.status_code}）：{detail}")
        data = response.json()
        task_id = str(data.get("task_id") or data.get("id") or "").strip()
        if not task_id:
            raise H3StudioError(f"MiniMax API 没有返回 task_id：{data}")
        return task_id

    def query(self, task_id: str) -> dict[str, Any]:
        """Poll task status. Returns {status, video_url, error}.

        Official V2 query: GET /v2/query/video_generation/{task_id}
        Response: {"task": {"status": "succeeded", "content": {"url": "..."}}}
        """
        if not self.enabled():
            raise H3StudioError("尚未配置 MiniMax API Key")
        try:
            response = httpx.get(
                f"{self.base}/v2/query/video_generation/{task_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise H3StudioError(f"MiniMax API 查询失败：{exc}") from exc
        if response.status_code >= 400:
            raise H3StudioError(f"MiniMax API 查询失败（HTTP {response.status_code}）：{response.text[:300]}")
        data = response.json()
        task = data.get("task") or {}
        status = str(task.get("status") or "").lower()
        content = task.get("content") or {}
        video_url = str(content.get("url") or "").strip()
        error = str(task.get("error") or task.get("fail_reason") or "").strip()
        return {"status": status, "video_url": video_url, "error": error, "raw": data}

    def download(self, url: str) -> tuple[bytes, str]:
        """Download the generated video, return (content, filename)."""
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise H3StudioError(f"下载云端视频失败：{exc}") from exc
        filename = f"h3_cloud_{int(time.time())}.mp4"
        return response.content, filename
