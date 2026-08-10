"""Pixapi API client for Nano Banana image generation."""

import json
import time
import requests
from typing import Optional

PIXAPI_BASE = "https://api.pixapi.ai"


class PixapiError(Exception):
    """Base exception for Pixapi API errors."""


class PixapiClient:
    """Client for Pixapi API (OpenAI-compatible)."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def set_api_key(self, key: str):
        self.api_key = key

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def list_models(self) -> list[dict]:
        """Fetch available models from /v1/models.

        Falls back to a hardcoded list of known Gemini models if the
        endpoint returns 404 (not available for all API tiers).
        """
        resp = requests.get(
            f"{PIXAPI_BASE}/v1/models",
            headers=self._headers(),
            timeout=15,
        )
        if resp.status_code == 401:
            raise PixapiError("Invalid API key - authentication failed")
        if resp.status_code == 404:
            # Fall back to known Gemini models
            return self._default_models()
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    @staticmethod
    def _default_models() -> list[dict]:
        """Return hardcoded list of known Pixapi Gemini image models."""
        return [
            {"id": "gemini-3.1-flash-image-preview", "object": "model", "category": "image", "capabilities": ["text-to-image", "image-to-image"]},
            {"id": "gemini-3.1-flash-lite-image", "object": "model", "category": "image", "capabilities": ["text-to-image", "image-to-image"]},
            {"id": "gemini-3-pro-image-preview", "object": "model", "category": "image", "capabilities": ["text-to-image", "image-to-image"]},
        ]

    # ------------------------------------------------------------------
    # Text-to-Image  (sync)
    # ------------------------------------------------------------------
    def generate_image(
        self,
        model: str,
        prompt: str,
        size: str = "1:1",
        n: int = 1,
        quality: Optional[str] = None,
    ) -> list[dict]:
        """Generate image(s) from text prompt.

        Returns list of dicts with keys: url (and optionally revised_prompt).
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if quality:
            payload["quality"] = quality

        resp = requests.post(
            f"{PIXAPI_BASE}/v1/images/generations",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        if resp.status_code == 401:
            raise PixapiError("Invalid API key - authentication failed")
        if not resp.ok:
            err = self._parse_error(resp)
            raise PixapiError(f"Image generation failed: {err}")

        result = resp.json()
        return result.get("data", [])

    # ------------------------------------------------------------------
    # Image Editing  (sync)
    # ------------------------------------------------------------------
    def edit_image(
        self,
        model: str,
        prompt: str,
        image_url: str,
        size: str = "1:1",
        n: int = 1,
        quality: Optional[str] = None,
    ) -> list[dict]:
        """Edit an image via image URL.

        Returns list of dicts with keys: url.
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "image": image_url,
            "n": n,
            "size": size,
        }
        if quality:
            payload["quality"] = quality

        resp = requests.post(
            f"{PIXAPI_BASE}/v1/images/edits",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        if resp.status_code == 401:
            raise PixapiError("Invalid API key - authentication failed")
        if not resp.ok:
            err = self._parse_error(resp)
            raise PixapiError(f"Image editing failed: {err}")

        result = resp.json()
        return result.get("data", [])

    # ------------------------------------------------------------------
    # Async support
    # ------------------------------------------------------------------
    def submit_async_generation(
        self, model: str, prompt: str, size: str = "1:1", n: int = 1, quality: Optional[str] = None
    ) -> str:
        """Submit async text-to-image task and return task id."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if quality:
            payload["quality"] = quality

        resp = requests.post(
            f"{PIXAPI_BASE}/v1/async/images/generations",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            err = self._parse_error(resp)
            raise PixapiError(f"Async submission failed: {err}")
        data = resp.json()
        return data["id"]

    def submit_async_edit(
        self, model: str, prompt: str, image_url: str, size: str = "1:1", n: int = 1, quality: Optional[str] = None
    ) -> str:
        """Submit async image edit task and return task id."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "image": image_url,
            "n": n,
            "size": size,
        }
        if quality:
            payload["quality"] = quality

        resp = requests.post(
            f"{PIXAPI_BASE}/v1/async/images/edits",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            err = self._parse_error(resp)
            raise PixapiError(f"Async edit submission failed: {err}")
        data = resp.json()
        return data["id"]

    def get_task(self, task_id: str) -> dict:
        """Poll task status via GET /v1/tasks/{id}."""
        resp = requests.get(
            f"{PIXAPI_BASE}/v1/tasks/{task_id}",
            headers=self._headers(),
            timeout=15,
        )
        if not resp.ok:
            err = self._parse_error(resp)
            raise PixapiError(f"Task query failed: {err}")
        return resp.json()

    def wait_for_task(self, task_id: str, poll_interval: float = 2.0, timeout: float = 120.0) -> dict:
        """Blocking poll until task completes or fails."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            task = self.get_task(task_id)
            status = task.get("status")
            if status in ("completed", "failed"):
                return task
            time.sleep(poll_interval)
        raise PixapiError(f"Task {task_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_error(resp: requests.Response) -> str:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        return f"[{resp.status_code}] {msg}"

    @staticmethod
    def download_image(url: str, timeout: float = 30) -> bytes:
        """Download image bytes from URL."""
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content

    @staticmethod
    def upload_to_hosting(file_path: str) -> str:
        """Upload a local image file to a temporary hosting service and return the public URL.

        Uses 0x0.st (free, anonymous, no API key required). Files are auto-deleted after ~30 days.
        """
        with open(file_path, "rb") as f:
            r = requests.post(
                "https://0x0.st",
                files={"file": f},
                timeout=60,
            )
        if not r.ok:
            raise PixapiError(f"Failed to upload file to hosting: [{r.status_code}] {r.text}")
        return r.text.strip()