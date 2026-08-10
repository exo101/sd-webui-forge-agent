from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from .backend_manager import backend_manager
from .comfy_client import ComfyClient, normalize_base_url
from .config import (
    DEFAULT_CONFIG,
    load_config,
    load_lora_presets,
    save_config,
    save_lora_presets,
)
from .errors import H3StudioError
from .jobs import job_store
from .workflow import build_h3_workflow

API_ROOT = "/h3studio/api"
ASSET_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg",
}


def _fail(exc: Exception, status: int = 400):
    if isinstance(exc, HTTPException):
        raise exc
    raise HTTPException(status_code=status, detail=str(exc)) from exc


def _asset_url(item: dict[str, Any]) -> str:
    return (
        f"{API_ROOT}/media?filename={quote(str(item.get('filename') or item.get('name') or ''))}"
        f"&subfolder={quote(str(item.get('subfolder') or ''))}&type={quote(str(item.get('type') or 'input'))}"
    )


def _public_config() -> dict[str, Any]:
    config = load_config()
    return {key: config.get(key) for key in DEFAULT_CONFIG}


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    for item in job.get("outputs", []):
        item["url"] = _asset_url(item)
    revision = int(job.get("progress", {}).get("previewRevision") or 0)
    if revision:
        job["preview_url"] = f"{API_ROOT}/jobs/{quote(str(job.get('id') or ''))}/preview?v={revision}"
    return job


def _validate_settings(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise H3StudioError("设置格式无效")
    cleaned: dict[str, Any] = {}
    if "backend_mode" in payload:
        if payload["backend_mode"] not in {"managed", "external"}:
            raise H3StudioError("后端模式无效")
        cleaned["backend_mode"] = payload["backend_mode"]
    for key in ("comfy_path", "python_executable", "extra_args", "output_prefix"):
        if key in payload:
            cleaned[key] = str(payload[key] or "").strip()
    if "comfy_url" in payload:
        cleaned["comfy_url"] = normalize_base_url(str(payload["comfy_url"] or ""))
    if "port" in payload:
        port = int(payload["port"])
        if not 1024 <= port <= 65535:
            raise H3StudioError("端口必须位于 1024–65535")
        cleaned["port"] = port
    if "startup_timeout" in payload:
        cleaned["startup_timeout"] = max(10, min(int(payload["startup_timeout"]), 900))
    if "request_timeout" in payload:
        cleaned["request_timeout"] = max(3, min(int(payload["request_timeout"]), 600))
    if "auto_start_on_tab" in payload:
        cleaned["auto_start_on_tab"] = bool(payload["auto_start_on_tab"])
    mode = cleaned.get("backend_mode", load_config().get("backend_mode"))
    port = cleaned.get("port", load_config().get("port", 8189))
    if mode == "managed":
        cleaned["comfy_url"] = f"http://127.0.0.1:{port}"
    return cleaned


def register_api(_: Any, app: FastAPI) -> None:
    if any(getattr(route, "path", None) == f"{API_ROOT}/bootstrap" for route in app.routes):
        return

    @app.get(f"{API_ROOT}/bootstrap")
    def bootstrap():
        return {
            "version": "0.2.2",
            "config": _public_config(),
            "backend": backend_manager.status(),
            "lora_presets": load_lora_presets(),
            "jobs": [_public_job(job) for job in job_store.list(30)],
        }

    @app.get(f"{API_ROOT}/settings")
    def get_settings():
        return _public_config()

    @app.post(f"{API_ROOT}/settings")
    def update_settings(payload: dict[str, Any] = Body(...)):
        try:
            return save_config(_validate_settings(payload))
        except Exception as exc:
            _fail(exc)

    @app.get(f"{API_ROOT}/backend/status")
    def backend_status():
        return backend_manager.status()

    @app.post(f"{API_ROOT}/backend/start")
    def backend_start():
        try:
            return backend_manager.start()
        except Exception as exc:
            _fail(exc)

    @app.post(f"{API_ROOT}/backend/stop")
    def backend_stop():
        try:
            return backend_manager.stop()
        except Exception as exc:
            _fail(exc)

    @app.get(f"{API_ROOT}/backend/logs")
    def backend_logs(limit: int = Query(200, ge=1, le=500)):
        return {"lines": backend_manager.logs(limit)}

    @app.get(f"{API_ROOT}/catalog")
    def catalog():
        try:
            return ComfyClient().catalog()
        except Exception as exc:
            _fail(exc, 503)

    @app.post(f"{API_ROOT}/assets/upload")
    def upload_asset(file: UploadFile = File(...)):
        try:
            filename = file.filename or ""
            suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if suffix not in ASSET_EXTENSIONS:
                raise H3StudioError("只允许上传常见的图片、视频或音频文件")
            result = ComfyClient().upload(file.file, file.filename or "asset.bin", file.content_type)
            result["file"] = "/".join(part for part in (result.get("subfolder"), result.get("name")) if part)
            result["url"] = _asset_url(result)
            return result
        except Exception as exc:
            _fail(exc)
        finally:
            try:
                file.file.close()
            except Exception:
                pass

    @app.get(f"{API_ROOT}/media")
    def media(
        request: Request,
        filename: str = Query(...),
        subfolder: str = Query(""),
        type: str = Query("output"),
    ):
        try:
            response, iterator = ComfyClient().open_media(
                filename,
                subfolder,
                type,
                request.headers.get("range", ""),
            )
            headers = {}
            for header in ("content-length", "content-range", "accept-ranges", "etag", "last-modified"):
                if response.headers.get(header):
                    headers["-".join(part.capitalize() for part in header.split("-"))] = response.headers[header]
            return StreamingResponse(
                iterator,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers=headers,
            )
        except Exception as exc:
            _fail(exc, 404)

    @app.post(f"{API_ROOT}/workflow/preview")
    def preview_workflow(payload: dict[str, Any] = Body(...)):
        try:
            workflow, summary = build_h3_workflow(payload)
            return {"workflow": workflow, "summary": summary}
        except Exception as exc:
            _fail(exc)

    @app.post(f"{API_ROOT}/jobs")
    def create_job(payload: dict[str, Any] = Body(...)):
        try:
            return _public_job(job_store.submit(payload))
        except Exception as exc:
            _fail(exc)

    @app.get(f"{API_ROOT}/jobs")
    def list_jobs(limit: int = Query(30, ge=1, le=100)):
        return {"jobs": [_public_job(job) for job in job_store.list(limit)]}

    @app.get(f"{API_ROOT}/jobs/{{job_id}}")
    def get_job(job_id: str, include_workflow: bool = Query(False)):
        try:
            job = job_store.get(job_id, include_workflow=include_workflow)
            return _public_job(job)
        except Exception as exc:
            _fail(exc, 404)

    @app.get(f"{API_ROOT}/jobs/{{job_id}}/preview")
    def get_job_preview(job_id: str):
        try:
            content, mime, revision = job_store.get_preview(job_id)
            return Response(
                content=content,
                media_type=mime,
                headers={"Cache-Control": "no-store", "X-H3-Preview-Revision": str(revision)},
            )
        except Exception as exc:
            _fail(exc, 404)

    @app.get(f"{API_ROOT}/jobs/{{job_id}}/metadata")
    def get_job_metadata(job_id: str):
        try:
            job = job_store.get(job_id, include_workflow=True)
            summary = job.get("summary") or {}
            seed = int(summary.get("seed") or 0)
            payload = {
                "schema": "forge-h3-studio/reproducible-v1",
                "job_id": job.get("id"),
                "prompt_id": job.get("prompt_id"),
                "seed": seed,
                "parameters": summary.get("reproducible") or {},
                "outputs": job.get("outputs") or [],
                "workflow": job.get("workflow") or {},
            }
            return Response(
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="Forge_H3_seed_{seed}.json"'},
            )
        except Exception as exc:
            _fail(exc, 404)

    @app.post(f"{API_ROOT}/jobs/{{job_id}}/cancel")
    def cancel_job(job_id: str):
        try:
            return _public_job(job_store.cancel(job_id))
        except Exception as exc:
            _fail(exc)

    @app.post(f"{API_ROOT}/jobs/clear")
    def clear_jobs(payload: dict[str, Any] = Body(...)):
        try:
            return job_store.clear(str(payload.get("scope") or "completed"))
        except Exception as exc:
            _fail(exc)

    @app.get(f"{API_ROOT}/lora-presets")
    def get_lora_presets():
        return {"presets": load_lora_presets()}

    @app.post(f"{API_ROOT}/lora-presets")
    def upsert_lora_preset(payload: dict[str, Any] = Body(...)):
        try:
            name = str(payload.get("name") or "").strip()
            loras = payload.get("loras")
            if not name or len(name) > 100 or "/" in name or "\\" in name or not isinstance(loras, list):
                raise H3StudioError("LoRA 预设名称或内容无效")
            presets = [item for item in load_lora_presets() if item.get("name") != name]
            presets.append({"name": name, "loras": loras[:16], "updated_at": time.time()})
            return {"presets": save_lora_presets(presets)}
        except Exception as exc:
            _fail(exc)

    @app.delete(f"{API_ROOT}/lora-presets/{{name}}")
    def delete_lora_preset(name: str):
        presets = [item for item in load_lora_presets() if item.get("name") != name]
        return {"presets": save_lora_presets(presets)}
