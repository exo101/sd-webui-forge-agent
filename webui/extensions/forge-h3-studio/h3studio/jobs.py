from __future__ import annotations

import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from .comfy_client import ComfyClient
from .errors import H3StudioError
from .workflow import build_h3_workflow


@dataclass
class StudioJob:
    id: str
    prompt_id: str
    client_id: str
    state: str
    created_at: float
    updated_at: float
    summary: dict[str, Any]
    outputs: list[dict[str, Any]] = field(default_factory=list)
    queue_position: int | None = None
    error: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    workflow: dict[str, Any] | None = None
    progress: dict[str, Any] = field(default_factory=lambda: {
        "nodeId": None,
        "nodeTitle": "等待 ComfyUI 开始",
        "nodePercent": 0.0,
        "overallPercent": 0.0,
        "step": 0,
        "maxSteps": 0,
        "completedNodes": [],
        "queueRemaining": None,
        "previewRevision": 0,
    })
    preview_bytes: bytes | None = field(default=None, repr=False)
    preview_mime: str = field(default="image/jpeg", repr=False)

    def public(self, include_workflow: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data.pop("preview_bytes", None)
        data.pop("preview_mime", None)
        if not include_workflow:
            data.pop("workflow", None)
        return data


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, StudioJob] = {}
        self._streams: dict[str, Any] = {}

    def _close_stream(self, job_id: str) -> None:
        with self._lock:
            stream = self._streams.pop(job_id, None)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _trim(self) -> None:
        if len(self._jobs) <= 200:
            return
        ordered = sorted(self._jobs.values(), key=lambda item: item.created_at)
        for item in ordered[: len(self._jobs) - 200]:
            if item.state not in {"queued", "running"}:
                self._jobs.pop(item.id, None)

    @staticmethod
    def _collect_outputs(value: Any, result: list[dict[str, Any]]) -> None:
        if isinstance(value, dict):
            filename = value.get("filename") or value.get("name")
            media_type = value.get("type")
            if filename and media_type in {"output", "temp", "input"}:
                item = {
                    "filename": str(filename),
                    "subfolder": str(value.get("subfolder") or ""),
                    "type": str(media_type),
                }
                if value.get("format"):
                    item["format"] = value["format"]
                if item not in result:
                    result.append(item)
            for child in value.values():
                JobStore._collect_outputs(child, result)
        elif isinstance(value, (list, tuple)):
            for child in value:
                JobStore._collect_outputs(child, result)

    @staticmethod
    def _history_error(entry: dict[str, Any]) -> str:
        status = entry.get("status") if isinstance(entry, dict) else None
        messages = status.get("messages", []) if isinstance(status, dict) else []
        for message in reversed(messages):
            if not isinstance(message, (list, tuple)) or len(message) < 2:
                continue
            if message[0] == "execution_error" and isinstance(message[1], dict):
                detail = message[1]
                return str(detail.get("exception_message") or detail.get("exception_type") or "ComfyUI 执行失败")
        if isinstance(status, dict) and status.get("status_str") == "error":
            return "ComfyUI 执行失败，请查看后端日志"
        return ""

    @staticmethod
    def _update_overall(job: StudioJob) -> None:
        progress = job.progress
        completed = {str(value) for value in progress.get("completedNodes", [])}
        progress["completedNodes"] = sorted(completed)
        node_count = max(1, int(job.summary.get("node_count") or 1))
        node_percent = max(0.0, min(100.0, float(progress.get("nodePercent") or 0)))
        progress["overallPercent"] = max(
            0.0,
            min(99.9, (len(completed) + node_percent / 100.0) / node_count * 100.0),
        )

    def _set_node(self, job: StudioJob, node_id: Any) -> None:
        progress = job.progress
        previous = progress.get("nodeId")
        completed = {str(value) for value in progress.get("completedNodes", [])}
        if previous is not None and str(previous) != str(node_id):
            completed.add(str(previous))
        progress["completedNodes"] = sorted(completed)
        progress["nodeId"] = None if node_id is None else str(node_id)
        progress["nodeTitle"] = (
            "正在整理输出"
            if node_id is None
            else job.summary.get("node_titles", {}).get(str(node_id), f"节点 {node_id}")
        )
        progress["nodePercent"] = 100.0 if node_id is None else 0.0
        self._update_overall(job)

    def _handle_event(self, job_id: str, message: dict[str, Any]) -> None:
        if not isinstance(message, dict) or not message.get("type"):
            return
        event_type = str(message["type"])
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            event_prompt = data.get("prompt_id")
            if job.prompt_id and event_prompt and str(event_prompt) != job.prompt_id:
                return
            now = time.time()
            if event_type == "preview_image":
                preview = data.get("bytes")
                if isinstance(preview, (bytes, bytearray)) and preview:
                    job.preview_bytes = bytes(preview)
                    job.preview_mime = str(data.get("mime") or "image/jpeg")
                    job.progress["previewRevision"] = int(now * 1000)
                    job.updated_at = now
                return
            if event_type == "execution_start":
                job.state = "running"
                job.started_at = job.started_at or now
                job.error = ""
            elif event_type == "executing":
                self._set_node(job, data.get("node"))
                if data.get("node") is not None:
                    job.state = "running"
                    job.started_at = job.started_at or now
            elif event_type == "progress":
                if data.get("node") is not None and str(data.get("node")) != job.progress.get("nodeId"):
                    self._set_node(job, data.get("node"))
                maximum = max(0.0, float(data.get("max") or 0))
                value = max(0.0, float(data.get("value") or 0))
                job.progress["step"] = value
                job.progress["maxSteps"] = maximum
                job.progress["nodePercent"] = min(100.0, value / maximum * 100.0) if maximum else 0.0
                self._update_overall(job)
            elif event_type == "progress_state" and isinstance(data.get("nodes"), dict):
                entries = [(key, value) for key, value in data["nodes"].items() if isinstance(value, dict)]
                current = next((item for item in entries if item[1].get("state") == "running"), None)
                if current is None:
                    current = next((item for item in entries if float(item[1].get("value") or 0) < float(item[1].get("max") or 0)), None)
                if current:
                    node_id, value = current
                    if str(node_id) != job.progress.get("nodeId"):
                        self._set_node(job, node_id)
                    maximum = max(0.0, float(value.get("max") or 0))
                    step = max(0.0, float(value.get("value") or 0))
                    job.progress["step"] = step
                    job.progress["maxSteps"] = maximum
                    job.progress["nodePercent"] = min(100.0, step / maximum * 100.0) if maximum else 0.0
                    self._update_overall(job)
            elif event_type == "executed":
                completed = {str(value) for value in job.progress.get("completedNodes", [])}
                if data.get("node") is not None:
                    completed.add(str(data["node"]))
                job.progress["completedNodes"] = sorted(completed)
                self._update_overall(job)
            elif event_type == "execution_cached":
                completed = {str(value) for value in job.progress.get("completedNodes", [])}
                completed.update(str(value) for value in data.get("nodes", []))
                job.progress["completedNodes"] = sorted(completed)
                self._update_overall(job)
            elif event_type == "status":
                status = data.get("status") if isinstance(data.get("status"), dict) else {}
                exec_info = status.get("exec_info") if isinstance(status.get("exec_info"), dict) else {}
                job.progress["queueRemaining"] = exec_info.get("queue_remaining")
            elif event_type == "execution_error":
                job.error = str(data.get("exception_message") or data.get("exception_type") or "ComfyUI 执行失败")
                job.progress["fullError"] = data
                job.state = "failed"
                job.completed_at = now
            elif event_type == "execution_success":
                self._set_node(job, None)
            job.updated_at = now

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        client = ComfyClient()
        health = client.health()
        if not health.get("ok"):
            raise H3StudioError("H3 后端尚未就绪")
        workflow, summary = build_h3_workflow(request)
        requested_client_id = str(request.get("client_id") or "")
        client_id = requested_client_id if re.fullmatch(r"[A-Za-z0-9_-]{16,128}", requested_client_id) else uuid.uuid4().hex
        now = time.time()
        job = StudioJob(
            id=str(uuid.uuid4()),
            prompt_id="",
            client_id=client_id,
            state="queued",
            created_at=now,
            updated_at=now,
            summary=summary,
            workflow=workflow,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._trim()
        stream = client.connect_events(client_id, lambda message: self._handle_event(job.id, message))
        if stream is not None:
            with self._lock:
                self._streams[job.id] = stream
        try:
            response = client.submit(
                workflow,
                client_id,
                extra_data={
                    "extra_pnginfo": {
                        "seed": summary["seed"],
                        "forge_h3_studio": {
                            "version": 3,
                            "seed": summary["seed"],
                            "summary": summary,
                        }
                    }
                },
            )
            prompt_id = response.get("prompt_id")
            if not prompt_id:
                raise H3StudioError(f"ComfyUI 没有返回 prompt_id：{response}")
        except Exception:
            self._close_stream(job.id)
            with self._lock:
                self._jobs.pop(job.id, None)
            raise
        with self._lock:
            job.prompt_id = str(prompt_id)
            job.updated_at = time.time()
        threading.Thread(target=self._monitor, args=(job.id,), daemon=True).start()
        return job.public()

    def _monitor(self, job_id: str) -> None:
        client = ComfyClient()
        started = time.time()
        while time.time() - started < 24 * 60 * 60:
            terminal = False
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                terminal = job.state in {"completed", "failed", "cancelled"}
                prompt_id = job.prompt_id
            if terminal:
                self._close_stream(job_id)
                return
            try:
                history = client.history(prompt_id)
                entry = history.get(prompt_id)
                if isinstance(entry, dict):
                    outputs: list[dict[str, Any]] = []
                    self._collect_outputs(entry.get("outputs", {}), outputs)
                    error = self._history_error(entry)
                    status = entry.get("status", {})
                    completed = bool(status.get("completed", not error)) if isinstance(status, dict) else not error
                    with self._lock:
                        job = self._jobs.get(job_id)
                        if job is None:
                            return
                        job.outputs = outputs
                        job.error = error
                        job.state = "failed" if error or not completed else "completed"
                        if job.started_at is None:
                            job.started_at = job.created_at
                        job.updated_at = time.time()
                        job.completed_at = job.updated_at
                        job.progress["nodePercent"] = 100.0
                        job.progress["overallPercent"] = 100.0 if job.state == "completed" else job.progress.get("overallPercent", 0.0)
                        job.progress["nodeTitle"] = "生成完成" if job.state == "completed" else "执行失败"
                    self._close_stream(job_id)
                    return

                queue = client.queue()
                running = queue.get("queue_running", [])
                pending = queue.get("queue_pending", [])
                running_ids = [str(item[1]) for item in running if isinstance(item, (list, tuple)) and len(item) > 1]
                pending_ids = [str(item[1]) for item in pending if isinstance(item, (list, tuple)) and len(item) > 1]
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None:
                        return
                    if prompt_id in running_ids:
                        job.state = "running"
                        job.queue_position = 0
                        job.error = ""
                        if job.started_at is None:
                            job.started_at = time.time()
                        if not job.progress.get("nodeId"):
                            job.progress["nodeTitle"] = "ComfyUI 已开始执行"
                    elif prompt_id in pending_ids:
                        job.state = "queued"
                        job.queue_position = pending_ids.index(prompt_id) + 1
                        job.error = ""
                    job.updated_at = time.time()
            except Exception as exc:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None and not job.error:
                        job.error = f"状态检查暂时失败：{exc}"
                        job.updated_at = time.time()
            time.sleep(1.0)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.state not in {"completed", "failed", "cancelled"}:
                job.state = "failed"
                job.error = "任务监控超过 24 小时"
                job.updated_at = time.time()
        self._close_stream(job_id)

    def get(self, job_id: str, *, include_workflow: bool = False) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise H3StudioError("任务不存在")
            return deepcopy(job.public(include_workflow=include_workflow))

    def get_preview(self, job_id: str) -> tuple[bytes, str, int]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise H3StudioError("任务不存在")
            if not job.preview_bytes:
                raise H3StudioError("该任务还没有可用预览")
            return bytes(job.preview_bytes), job.preview_mime, int(job.progress.get("previewRevision") or 0)

    def list(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [deepcopy(item.public()) for item in jobs[: max(1, min(limit, 100))]]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise H3StudioError("任务不存在")
            prompt_id = job.prompt_id
            current_state = job.state
            if current_state not in {"queued", "running"}:
                raise H3StudioError("该任务已经结束，无法中断")
        client = ComfyClient()
        if current_state == "queued":
            client.delete_queued([prompt_id])
        elif current_state == "running":
            client.cancel(prompt_id)
        with self._lock:
            job = self._jobs[job_id]
            job.state = "cancelled"
            job.error = "任务已由用户中断"
            job.updated_at = time.time()
            job.completed_at = job.updated_at
            result = deepcopy(job.public())
        self._close_stream(job_id)
        return result

    def clear(self, scope: str) -> dict[str, Any]:
        if scope not in {"completed", "queue"}:
            raise H3StudioError("不支持的清理范围")
        with self._lock:
            if scope == "completed":
                ids = [
                    item.id for item in self._jobs.values()
                    if item.state in {"completed", "failed", "cancelled"}
                ]
                for job_id in ids:
                    self._jobs.pop(job_id, None)
            else:
                ids = []
            queued = [item for item in self._jobs.values() if item.state == "queued"]
        if scope == "completed":
            for job_id in ids:
                self._close_stream(job_id)
            return {"cleared": len(ids), "scope": scope}
        client = ComfyClient()
        client.delete_queued([item.prompt_id for item in queued])
        try:
            queue = client.queue()
            running_ids = {
                str(item[1]) for item in queue.get("queue_running", [])
                if isinstance(item, (list, tuple)) and len(item) > 1
            }
        except Exception:
            running_ids = set()
        now = time.time()
        cleared = 0
        closed_ids: list[str] = []
        with self._lock:
            for item in queued:
                job = self._jobs.get(item.id)
                if job is not None and job.state == "queued" and job.prompt_id not in running_ids:
                    job.state = "cancelled"
                    job.error = "排队任务已清除"
                    job.updated_at = now
                    job.completed_at = now
                    closed_ids.append(item.id)
                    cleared += 1
        for job_id in closed_ids:
            self._close_stream(job_id)
        return {"cleared": cleared, "scope": scope}


job_store = JobStore()
