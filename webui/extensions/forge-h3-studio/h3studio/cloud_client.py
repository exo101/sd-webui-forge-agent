"""MiniMax H3 cloud API client.

Wraps the asynchronous video-generation endpoints so the Forge H3 Studio
workbench can run without a local ComfyUI installation when the user
chooses backend_mode == "api".
"""
from __future__ import annotations

import base64
import mimetypes
import time
import uuid
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

_YOBOX_2K_SIZES = {
    "21:9": (3360, 1440),
    "16:9": (2560, 1440),
    "4:3": (1920, 1440),
    "1:1": (1440, 1440),
    "3:4": (1440, 1920),
    "9:16": (1440, 2560),
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
        if self.base.lower().endswith("/v1"):
            self.base = self.base[:-3].rstrip("/")
        self.timeout = float(self.config.get("request_timeout") or 30)

    def enabled(self) -> bool:
        return bool(self.api_key)

    def _is_yobox(self) -> bool:
        return "yoboxai.com" in self.base.lower()

    def _headers(self, *, yobox: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if yobox:
            headers["Idempotency-Key"] = f"h3-{uuid.uuid4().hex}"
        return headers

    @staticmethod
    def _json_response(response: httpx.Response, action: str) -> dict[str, Any]:
        """Decode an API response without leaking a raw JSONDecodeError."""
        text = response.text.strip()
        if not text:
            raise H3StudioError(
                f"MiniMax API {action}返回空响应（HTTP {response.status_code}）。"
                "请检查 API 地址、API Key 和上游服务状态"
            )
        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            preview = " ".join(text.split())[:500]
            raise H3StudioError(
                f"MiniMax API {action}返回的不是 JSON（HTTP {response.status_code}）：{preview}"
            ) from exc
        if not isinstance(data, dict):
            raise H3StudioError(
                f"MiniMax API {action}返回格式无效（HTTP {response.status_code}）：{text[:500]}"
            )
        return data

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

    def _build_yobox_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        """Build the YoboxAI /v1/videos request shape."""
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise H3StudioError("提示词不能为空")
        frames = int(request.get("frames") or (5 * H3_FPS))
        duration = max(MIN_DURATION, min(MAX_DURATION, round(frames / H3_FPS)))
        aspect = str(request.get("aspect_ratio") or "16:9").strip()
        aspect = aspect if aspect in _YOBOX_2K_SIZES else "16:9"
        width, height = _YOBOX_2K_SIZES[aspect]
        payload_input: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect,
            # YoboxAI's MiniMax-H3 endpoint accepts 2K for async billing.
            # Do not send the native MiniMax "768P" value here.
            "resolution": "2K",
            "width": width,
            "height": height,
            "duration": duration,
            "audio": True,
            "n": 1,
        }

        first = str(request.get("first_frame") or "").strip()
        last = str(request.get("last_frame") or "").strip()
        if first:
            first_url = _image_data_url(first)
            if first_url:
                payload_input["start_frames"] = [{"url": first_url}]
        if last:
            last_url = _image_data_url(last)
            if last_url:
                payload_input["end_frames"] = [{"url": last_url}]

        references = []
        audio_references = []
        for ref in request.get("references") or []:
            data_url = _image_data_url(str(ref.get("file") or ""))
            if not data_url:
                continue
            if ref.get("kind") == "image":
                references.append({"url": data_url, "strength": "MID"})
            elif ref.get("kind") == "audio":
                audio_references.append({"url": data_url})
        if references and "start_frames" not in payload_input:
            payload_input["image_references"] = references[:5]
            # Yobox requires at least one image reference when audio references
            # are supplied; silently omit audio-only references.
            if audio_references:
                payload_input["audio_references"] = audio_references[:3]

        return {"model": "MiniMax-H3", "input": payload_input}

    def submit(self, request: dict[str, Any]) -> str:
        """Submit a video generation task, return the task_id."""
        if not self.enabled():
            raise H3StudioError("尚未配置 MiniMax API Key，请在设置中填写")
        yobox = self._is_yobox()
        payload = self._build_yobox_payload(request) if yobox else self._build_payload(request)
        try:
            response = httpx.post(
                f"{self.base}/v1/videos" if yobox else f"{self.base}/v2/video_generation",
                json=payload,
                headers=self._headers(yobox=yobox),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise H3StudioError(f"MiniMax API 连接失败：{exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:500]
            if response.status_code == 503:
                raise H3StudioError(
                    "YoboxAI 当前暂无可用账号（HTTP 503），请稍后重试，"
                    "或在 H3 设置中切换到其他可用的视频供应商"
                )
            raise H3StudioError(f"MiniMax API 提交失败（HTTP {response.status_code}）：{detail}")
        data = self._json_response(response, "提交")
        result = data.get("data") if isinstance(data.get("data"), dict) else {}
        task_id = str(result.get("task_id") or data.get("task_id") or data.get("id") or "").strip()
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
                f"{self.base}/v1/videos/{task_id}" if self._is_yobox() else f"{self.base}/v2/query/video_generation/{task_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise H3StudioError(f"MiniMax API 查询失败：{exc}") from exc
        if response.status_code >= 400:
            raise H3StudioError(f"MiniMax API 查询失败（HTTP {response.status_code}）：{response.text[:300]}")
        data = self._json_response(response, "查询")
        task = data.get("task") or data.get("data") or {}
        if not isinstance(task, dict):
            task = {}
        status = str(task.get("status") or "").lower()
        content = task.get("content") or task.get("output") or {}
        if not isinstance(content, dict):
            content = {}
        video_url = str(content.get("url") or "").strip()
        if not video_url:
            video_url = str(task.get("video_url") or task.get("url") or "").strip()
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
