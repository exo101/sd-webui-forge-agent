from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from h3studio.jobs import JobStore, StudioJob


def make_job(identifier: str, state: str) -> StudioJob:
    now = time.time()
    return StudioJob(
        id=identifier,
        prompt_id=f"prompt-{identifier}",
        client_id=f"client-{identifier}",
        state=state,
        created_at=now,
        updated_at=now,
        summary={"mode": "t2v"},
    )


class JobStoreTests(unittest.TestCase):
    @patch("h3studio.jobs.ComfyClient")
    def test_submit_connects_events_before_queueing_prompt(self, client_class):
        request = {
            "mode": "t2v",
            "model": "h3.safetensors",
            "text_encoder": "clip.safetensors",
            "video_vae": "video_vae.safetensors",
            "audio_vae": "audio_vae.safetensors",
            "prompt": "A slow camera move",
            "width": 608,
            "height": 352,
            "frames": 124,
            "steps": 30,
            "seed": 987654,
            "loras": [],
            "output": {"filename_prefix": "video/test"},
        }
        order = []
        client = client_class.return_value
        client.health.return_value = {"ok": True}
        client.connect_events.side_effect = lambda *_: order.append("connect")
        client.submit.side_effect = lambda *args, **kwargs: order.append("submit") or {"prompt_id": "prompt-live"}
        store = JobStore()
        with patch.object(store, "_monitor", return_value=None):
            job = store.submit(request)

        self.assertEqual(order, ["connect", "submit"])
        self.assertEqual(job["prompt_id"], "prompt-live")
        extra_data = client.submit.call_args.kwargs["extra_data"]["extra_pnginfo"]
        self.assertEqual(extra_data["seed"], 987654)
        self.assertTrue(job["summary"]["filename_prefix"].endswith("_seed_987654"))

    def test_realtime_events_update_node_progress_and_preview(self):
        store = JobStore()
        job = make_job("live", "queued")
        job.summary = {"mode": "i2v", "node_count": 10, "node_titles": {"7": "自定义采样器（高级）"}}
        store._jobs[job.id] = job

        store._handle_event(job.id, {"type": "execution_start", "data": {"prompt_id": job.prompt_id}})
        store._handle_event(job.id, {"type": "executing", "data": {"prompt_id": job.prompt_id, "node": "7"}})
        store._handle_event(job.id, {"type": "progress", "data": {"prompt_id": job.prompt_id, "node": "7", "value": 12, "max": 30}})
        store._handle_event(job.id, {"type": "preview_image", "data": {"mime": "image/jpeg", "bytes": b"preview"}})

        public = store.get(job.id)
        self.assertEqual(public["state"], "running")
        self.assertEqual(public["progress"]["nodeTitle"], "自定义采样器（高级）")
        self.assertEqual(public["progress"]["nodePercent"], 40)
        self.assertGreater(public["progress"]["overallPercent"], 0)
        self.assertGreater(public["progress"]["previewRevision"], 0)
        self.assertNotIn("preview_bytes", public)
        content, mime, _ = store.get_preview(job.id)
        self.assertEqual(content, b"preview")
        self.assertEqual(mime, "image/jpeg")

    def test_clear_completed_keeps_active_jobs(self):
        store = JobStore()
        store._jobs = {
            "done": make_job("done", "completed"),
            "failed": make_job("failed", "failed"),
            "running": make_job("running", "running"),
        }
        result = store.clear("completed")
        self.assertEqual(result["cleared"], 2)
        self.assertEqual(list(store._jobs), ["running"])

    @patch("h3studio.jobs.ComfyClient")
    def test_clear_queue_deletes_only_pending_prompts(self, client_class):
        store = JobStore()
        store._jobs = {
            "queued": make_job("queued", "queued"),
            "running": make_job("running", "running"),
        }
        client = client_class.return_value
        client.queue.return_value = {"queue_running": [[0, "prompt-running"]]}
        result = store.clear("queue")
        client.delete_queued.assert_called_once_with(["prompt-queued"])
        self.assertEqual(result["cleared"], 1)
        self.assertEqual(store._jobs["queued"].state, "cancelled")
        self.assertEqual(store._jobs["running"].state, "running")


if __name__ == "__main__":
    unittest.main()
