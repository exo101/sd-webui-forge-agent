from __future__ import annotations

import json
import mimetypes
import threading
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Callable
from urllib.parse import urlencode, urlparse, urlunparse

import httpx

from .config import load_config
from .errors import BackendUnavailable, H3StudioError

REQUIRED_H3_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "LoadImage",
    "ImageCrop",
    "ImageScale",
    "LoadVideo",
    "GetVideoComponents",
    "LoadAudio",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
    "RandomNoise",
    "BasicGuider",
    "KSamplerSelect",
    "BasicScheduler",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "ImageFromBatch",
    "PreviewImage",
    "CreateVideo",
    "SaveVideo",
)


class ComfyEventStream:
    """Small lifecycle wrapper around ComfyUI's synchronous WebSocket feed."""

    def __init__(self, connection: Any, callback: Callable[[dict[str, Any]], None]):
        self.connection = connection
        self.callback = callback
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="h3studio-comfy-events", daemon=True)

    def start(self) -> "ComfyEventStream":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                payload = self.connection.recv()
            except TimeoutError:
                continue
            except Exception as exc:
                if exc.__class__.__name__ == "WebSocketTimeoutException":
                    continue
                break
            if payload in (None, "", b""):
                break
            try:
                if isinstance(payload, str):
                    message = json.loads(payload)
                    if isinstance(message, dict):
                        self.callback(message)
                elif isinstance(payload, (bytes, bytearray)) and len(payload) > 8:
                    image_type = int.from_bytes(payload[4:8], "big")
                    mime = {1: "image/jpeg", 2: "image/png", 3: "image/webp"}.get(image_type, "image/jpeg")
                    self.callback({
                        "type": "preview_image",
                        "data": {"mime": mime, "bytes": bytes(payload[8:])},
                    })
            except Exception:
                # A malformed/non-H3 event must never stop job monitoring.
                continue

    def close(self) -> None:
        self._stopped.set()
        try:
            self.connection.close()
        except Exception:
            pass
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.5)


def normalize_base_url(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise H3StudioError("ComfyUI 地址必须是有效的 http:// 或 https:// 地址")
    if parsed.username or parsed.password:
        raise H3StudioError("ComfyUI 地址中不能包含用户名或密码")
    return value


def safe_media_path(filename: str, subfolder: str = "") -> tuple[str, str]:
    filename = (filename or "").strip()
    subfolder = (subfolder or "").replace("\\", "/").strip("/")
    if not filename or PurePosixPath(filename).name != filename:
        raise H3StudioError("非法媒体文件名")
    parts = PurePosixPath(subfolder).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise H3StudioError("非法媒体子目录")
    return filename, subfolder


class ComfyClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        config = load_config()
        self.base_url = normalize_base_url(base_url or config["comfy_url"])
        self.timeout = float(timeout or config.get("request_timeout", 30))

    def _url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _raise(self, response: httpx.Response, action: str) -> None:
        if response.is_success:
            return
        detail = ""
        try:
            payload = response.json()
            detail = json.dumps(payload, ensure_ascii=False)[:4000]
        except Exception:
            detail = response.text[:1000]
        raise H3StudioError(f"{action}失败（HTTP {response.status_code}）{': ' + detail if detail else ''}")

    def get_json(self, path: str, *, timeout: float | None = None) -> Any:
        try:
            response = httpx.get(self._url(path), timeout=timeout or self.timeout)
            self._raise(response, "读取 ComfyUI")
            return response.json()
        except H3StudioError:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"无法连接 ComfyUI：{exc}") from exc

    def post_json(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        try:
            response = httpx.post(self._url(path), json=payload, timeout=timeout or self.timeout)
            self._raise(response, "调用 ComfyUI")
            if not response.content:
                return {}
            return response.json()
        except H3StudioError:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"无法连接 ComfyUI：{exc}") from exc

    def health(self) -> dict[str, Any]:
        try:
            stats = self.get_json("/system_stats", timeout=3.0)
            return {"ok": True, "base_url": self.base_url, "system_stats": stats}
        except Exception as exc:
            return {"ok": False, "base_url": self.base_url, "error": str(exc)}

    def object_info(self) -> dict[str, Any]:
        payload = self.get_json("/object_info", timeout=max(30.0, self.timeout))
        if not isinstance(payload, dict):
            raise H3StudioError("ComfyUI /object_info 返回格式异常")
        return payload

    @staticmethod
    def _input_options(info: dict[str, Any], input_name: str) -> list[str]:
        inputs = info.get("input", {}) if isinstance(info, dict) else {}
        for group in ("required", "optional"):
            item = inputs.get(group, {}).get(input_name)
            if isinstance(item, (list, tuple)) and item:
                options = item[0]
                if isinstance(options, (list, tuple)):
                    return [str(value) for value in options]
        return []

    def catalog(self) -> dict[str, Any]:
        info = self.object_info()
        missing = [name for name in REQUIRED_H3_NODES if name not in info]
        return {
            "models": self._input_options(info.get("UNETLoader", {}), "unet_name"),
            "text_encoders": self._input_options(info.get("CLIPLoader", {}), "clip_name"),
            "vaes": self._input_options(info.get("VAELoader", {}), "vae_name"),
            "loras": self._input_options(info.get("LoraLoader", {}), "lora_name")
            or self._input_options(info.get("LoraLoaderModelOnly", {}), "lora_name"),
            "samplers": self._input_options(info.get("KSamplerSelect", {}), "sampler_name"),
            "schedulers": self._input_options(info.get("BasicScheduler", {}), "scheduler"),
            "nodes": sorted(info.keys()),
            "h3_ready": not missing,
            "missing_nodes": missing,
            "supports_lora_model_only": "LoraLoaderModelOnly" in info,
            "supports_lora_model_clip": "LoraLoader" in info,
        }

    def upload(self, stream: BinaryIO, filename: str, content_type: str | None = None) -> dict[str, Any]:
        filename, _ = safe_media_path(filename)
        mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        files = {"image": (filename, stream, mime)}
        data = {"type": "input", "subfolder": "forge_h3_studio", "overwrite": "false"}
        try:
            response = httpx.post(
                self._url("/upload/image"),
                data=data,
                files=files,
                timeout=max(300.0, self.timeout),
            )
            self._raise(response, "上传素材")
            result = response.json()
            result["content_type"] = mime
            return result
        except H3StudioError:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"上传素材时无法连接 ComfyUI：{exc}") from exc

    def submit(self, workflow: dict[str, Any], client_id: str, extra_data: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.post_json(
            "/prompt",
            {"prompt": workflow, "client_id": client_id, "extra_data": extra_data or {}},
            timeout=max(60.0, self.timeout),
        )

    def connect_events(
        self,
        client_id: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> ComfyEventStream | None:
        """Connect before prompt submission so no node/progress events are lost."""
        try:
            import websocket
        except Exception:
            return None
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = (parsed.path.rstrip("/") + "/ws") or "/ws"
        url = urlunparse((scheme, parsed.netloc, path, "", urlencode({"clientId": client_id}), ""))
        try:
            connection = websocket.create_connection(
                url,
                timeout=min(5.0, self.timeout),
                enable_multithread=True,
            )
            connection.settimeout(1.0)
        except Exception:
            return None
        return ComfyEventStream(connection, callback).start()

    def history(self, prompt_id: str) -> dict[str, Any]:
        payload = self.get_json(f"/history/{prompt_id}", timeout=10.0)
        return payload if isinstance(payload, dict) else {}

    def queue(self) -> dict[str, Any]:
        payload = self.get_json("/queue", timeout=10.0)
        return payload if isinstance(payload, dict) else {}

    def cancel(self, prompt_id: str) -> None:
        try:
            self.post_json("/interrupt", {"prompt_id": prompt_id}, timeout=10.0)
        finally:
            try:
                self.post_json("/queue", {"delete": [prompt_id]}, timeout=10.0)
            except Exception:
                pass

    def delete_queued(self, prompt_ids: list[str]) -> None:
        if prompt_ids:
            self.post_json("/queue", {"delete": prompt_ids}, timeout=10.0)

    def open_media(
        self,
        filename: str,
        subfolder: str = "",
        media_type: str = "output",
        byte_range: str = "",
    ) -> tuple[httpx.Response, Iterator[bytes]]:
        filename, subfolder = safe_media_path(filename, subfolder)
        if media_type not in {"input", "output", "temp"}:
            raise H3StudioError("非法媒体类型")
        client = httpx.Client(timeout=None)
        request = client.build_request(
            "GET",
            self._url("/view"),
            params={"filename": filename, "subfolder": subfolder, "type": media_type},
            headers={"Range": byte_range} if byte_range else None,
        )
        try:
            response = client.send(request, stream=True)
            self._raise(response, "读取媒体")
        except Exception:
            client.close()
            raise

        def iterator() -> Iterator[bytes]:
            try:
                yield from response.iter_bytes(chunk_size=1024 * 1024)
            finally:
                response.close()
                client.close()

        return response, iterator()
