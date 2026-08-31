import json
import logging
import os
import ssl
import time
from io import BytesIO
from enum import Enum

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.logging import setup_logger

logger = logging.getLogger("api_providers")
setup_logger(logger)

# Create a custom SSLContext that ignores unexpected EOF errors
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE
try:
    _ssl_context.options |= ssl.OP_IGNORE_UNEXPECTED_EOF
except AttributeError:
    pass


class _SSLAdapter(HTTPAdapter):
    """HTTPAdapter that uses the custom SSL context."""
    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = _ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def send(self, *args, **kwargs):
        kwargs["verify"] = False
        return super().send(*args, **kwargs)


def _make_session() -> requests.Session:
    """Create a requests Session that bypasses proxy and uses custom SSL context."""
    sess = requests.Session()
    sess.trust_env = False
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = _SSLAdapter()
    adapter.max_retries = retry_strategy
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

# In-memory session API key (not persisted to disk)
_session_api_key = ""


def set_session_api_key(key: str) -> None:
    """Store the API key in memory for the current session."""
    global _session_api_key
    _session_api_key = key


def get_session_api_key() -> str:
    """Get the current session's API key from memory."""
    return _session_api_key


# In-memory reference images (from image_stitch plugin, for API editing models)
_reference_images: list[Image.Image] = []


def set_reference_images(images: list[Image.Image]) -> None:
    """Store reference images for API image editing models."""
    global _reference_images
    _reference_images = images


def get_reference_images() -> list[Image.Image]:
    """Get the current reference images."""
    return _reference_images


def clear_reference_images() -> None:
    """Clear reference images."""
    global _reference_images
    _reference_images = []


def _pil_to_data_url(img: Image.Image) -> str:
    """Convert a PIL Image to a compressed base64 data URL."""
    import base64
    from io import BytesIO

    # Resize to max 1024px on longest side to reduce payload size
    max_size = 1024
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    # Use JPEG with quality 80 for smaller payload
    buf = BytesIO()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class ApiProvider(Enum):
    modelscope = 1
    dashscope = 2
    pixapi = 3
    openai = 4

    @staticmethod
    def choices() -> list[str]:
        return [p.name for p in ApiProvider]

    @staticmethod
    def display_name(name: str) -> str:
        names = {
            "modelscope": "ModelScope API",
            "dashscope": "DashScope (Qwen-Image)",
            "pixapi": "Pixapi.ai",
            "openai": "OpenAI (GPT-Image)",
        }
        return names.get(name, name)


API_PROVIDER_CONFIGS = {
    "modelscope": {
        "api_root": "https://api-inference.modelscope.cn",
        "base_url": "https://api-inference.modelscope.cn/v1/images/generations",
        "doc_url": "https://www.modelscope.cn/docs/API-Inference/Overview",
        "fallback_models": [
            "Qwen/Qwen-Image-Edit-2511",
            "FireRedTeam/FireRed-Image-Edit-1.1",
            "krea/Krea-2-Turbo",
        ],
    },
    "dashscope": {
        "api_root": "https://dashscope.aliyuncs.com",
        "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        "doc_url": "https://help.aliyun.com/zh/model-studio/qwen-image-generation-and-editing-api-reference",
        "fallback_models": [
            "qwen-image-3.0",
            "qwen-image-3.0-pro",
        ],
    },
    "pixapi": {
        "api_root": "https://api.pixapi.ai",
        "base_url": "https://api.pixapi.ai/v1/images/generations",
        "doc_url": "https://pixapi.ai/docs",
        "fallback_models": [
            "gpt-image-2",
            "nano-banana",
            "gemini-3-pro-image-preview",
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image-preview",
        ],
    },
    "openai": {
        "api_root": "https://api.openai.com",
        "base_url": "https://api.openai.com/v1/images/generations",
        "doc_url": "https://platform.openai.com/docs/api-reference/images",
        "fallback_models": [
            "gpt-image-2",
            "dall-e-3",
            "dall-e-2",
        ],
    },
}


def fetch_models_from_api(provider: str, api_key: str) -> tuple[list[str], bool]:
    """Fetch available models from the API's /v1/models endpoint.
    Returns (models, from_api) where from_api=True if list came from API, False if fallback."""
    config = API_PROVIDER_CONFIGS.get(provider)
    if not config:
        return [], False

    api_root = config.get("api_root", "")
    if not api_root or not api_key:
        return [], False

    # ModelScope and DashScope use curated model lists, not /v1/models
    if provider in ("modelscope", "dashscope"):
        fallback = config.get("fallback_models", [])
        return fallback, False

    # Pixapi and OpenAI use /v1/models endpoint
    url = f"{api_root}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    logger.info(f"Fetching models from {url}")
    try:
        sess = _make_session()
        resp = sess.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        logger.debug(f"Raw /v1/models response: {json.dumps(result, ensure_ascii=False)[:1000]}")
    except Exception as e:
        logger.warning(f"Failed to fetch models from API: {e}")
        fallback = config.get("fallback_models", [])
        return fallback, False

    # Parse OpenAI-compatible /v1/models response: { "data": [ { "id": "xxx", ... }, ... ] }
    image_models = []
    if "data" in result and isinstance(result["data"], list):
        for item in result["data"]:
            if isinstance(item, dict) and "id" in item:
                model_id = item["id"]
                # If the API provides a category field, use it to filter image models
                category = item.get("category", "")
                if category:
                    if category.lower() == "image":
                        image_models.append(model_id)
                # Otherwise, use heuristic keyword matching
                else:
                    image_keywords = {"image", "img", "dall-e", "flux", "stable-diffusion", "kandinsky", "ideogram", "pix", "gemini"}
                    if any(kw in model_id.lower() for kw in image_keywords):
                        image_models.append(model_id)

    if not image_models:
        # If the API returned models but none are image-related (e.g. ModelScope returns only text models),
        # use the curated fallback list instead
        logger.warning("No image-generation models identified from API, using fallback")
        fallback = config.get("fallback_models", [])
        return fallback, False

    logger.info(f"Fetched {len(image_models)} image models from API")
    return sorted(image_models), True


def call_api_modelscope(prompt: str, negative_prompt: str, api_key: str, model: str, width: int, height: int, n_iter: int, batch_size: int, reference_images: list[Image.Image] = None) -> list[Image.Image]:
    config = API_PROVIDER_CONFIGS["modelscope"]
    base_url = config["api_root"]

    common_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "n": n_iter * batch_size,
        "size": f"{width}x{height}",
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    # Add reference images for editing models (e.g. Qwen-Image-Edit-2511)
    is_edit_model = "edit" in model.lower()
    logger.info(f"ModelScope: model={model}, is_edit={is_edit_model}, has_ref={bool(reference_images)}, ref_count={len(reference_images) if reference_images else 0}")
    if is_edit_model:
        if not reference_images:
            raise RuntimeError(f"编辑模型 {model} 需要上传参考图片，请先在多图参考插件中上传参考图片")
        image_urls = [_pil_to_data_url(img) for img in reference_images]
        payload["image_url"] = image_urls
        logger.info(f"Added {len(reference_images)} reference image(s) to ModelScope edit model payload")

    # Use a session that ignores proxy env vars and uses custom SSL context
    session = _make_session()

    # Step 1: Submit task with async mode
    submit_url = f"{base_url}/v1/images/generations"
    async_headers = {**common_headers, "X-ModelScope-Async-Mode": "true"}

    logger.info(f"Submitting ModelScope async task: {submit_url}")
    logger.info(f"Payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

    try:
        resp = session.post(
            submit_url,
            headers=async_headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response is not None else str(e)
        status_code = e.response.status_code if e.response is not None else "unknown"
        logger.error(f"ModelScope HTTP error {status_code}: {error_body}")
        raise RuntimeError(f"ModelScope API error {status_code}: {error_body}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"ModelScope connection error: {e}")
        raise RuntimeError(f"ModelScope connection error: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"ModelScope timeout: {e}")
        raise RuntimeError(f"ModelScope timeout: {e}")

    logger.debug(f"Submit response: {json.dumps(result, ensure_ascii=False)[:500]}")

    task_id = result.get("task_id")
    if not task_id:
        raise RuntimeError(f"ModelScope API did not return task_id. Response: {result}")

    logger.info(f"Task submitted, task_id: {task_id}")

    # Step 2: Poll task status
    # NOTE: Always use "image_generation" as the task type for polling,
    # even for edit models. The official ModelScope SDK/examples confirm
    # that the polling endpoint only recognizes "image_generation".
    task_headers = {**common_headers, "X-ModelScope-Task-Type": "image_generation"}
    logger.info(f"Polling task {task_id} with task type: image_generation")
    poll_url = f"{base_url}/v1/tasks/{task_id}"

    max_retries = 120
    retry_count = 0

    while retry_count < max_retries:
        time.sleep(5)
        retry_count += 1

        try:
            resp = session.get(poll_url, headers=task_headers, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response is not None else str(e)
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.error(f"Poll HTTP error {status_code}: {error_body}")
            raise RuntimeError(f"ModelScope poll error {status_code}: {error_body}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Retry on transient connection/timeout errors
            if retry_count < 3:
                logger.warning(f"Poll connection error (attempt {retry_count}), retrying: {e}")
                continue
            logger.error(f"Poll connection error: {e}")
            raise RuntimeError(f"ModelScope poll connection error: {e}")

        logger.debug(f"Poll response ({retry_count}): {json.dumps(result, ensure_ascii=False)[:300]}")

        task_status = result.get("task_status", "")

        if task_status == "SUCCEED":
            logger.info(f"Task {task_id} completed successfully")
            images = []
            output_images = result.get("output_images", [])
            if not output_images:
                output_images = result.get("images", [])
            for img_url in output_images:
                images.append(_download_image(img_url))
            return images

        elif task_status == "FAILED":
            error_msg = result.get("message", "Unknown error")
            logger.error(f"Task {task_id} failed: {error_msg}")
            raise RuntimeError(f"ModelScope image generation failed: {error_msg}")

        logger.info(f"Task {task_id} status: {task_status}, retrying in 5s...")

    raise RuntimeError(f"ModelScope task {task_id} timed out after {max_retries * 5} seconds")


def call_api_dashscope(prompt: str, api_key: str, model: str, width: int, height: int, n_iter: int, batch_size: int) -> list[Image.Image]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ],
                }
            ],
        },
        "parameters": {
            "size": f"{width}x{height}",
            "n": n_iter * batch_size,
        },
    }

    config = API_PROVIDER_CONFIGS["dashscope"]
    return _call_dashscope_api(config["base_url"], headers, payload, n_iter * batch_size)


def _call_dashscope_api(url: str, headers: dict, payload: dict, expected_count: int) -> list[Image.Image]:
    logger.info(f"Calling DashScope API: {url}")
    logger.info(f"Payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

    sess = _make_session()

    try:
        resp = sess.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response is not None else str(e)
        status_code = e.response.status_code if e.response is not None else "unknown"
        logger.error(f"DashScope HTTP error {status_code}: {error_body}")
        raise RuntimeError(f"DashScope API error {status_code}: {error_body}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"DashScope connection error: {e}")
        raise RuntimeError(f"DashScope API connection error: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"DashScope timeout: {e}")
        raise RuntimeError(f"DashScope API timeout: {e}")

    logger.debug(f"DashScope response: {json.dumps(result, ensure_ascii=False)[:500]}")

    images = []
    # DashScope response format: { "output": { "choices": [ { "message": { "content": [ { "image": "base64" } ] } } ] } }
    output = result.get("output", {})
    choices = output.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content", [])
        for item in content:
            if "image" in item:
                img_data = item["image"]
                # Check if it's a base64 string or a URL
                if img_data.startswith("http"):
                    images.append(_download_image(img_data))
                else:
                    import base64
                    images.append(Image.open(BytesIO(base64.b64decode(img_data))))

    return images


def _download_image(url: str) -> Image.Image:
    sess = _make_session()
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def _call_openai_images_api(url: str, headers: dict, payload: dict, expected_count: int) -> list[Image.Image]:
    """Call an OpenAI-compatible /v1/images/generations endpoint."""
    logger.info(f"Calling OpenAI-compatible API: {url}")
    logger.info(f"Payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

    sess = _make_session()

    try:
        resp = sess.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response is not None else str(e)
        status_code = e.response.status_code if e.response is not None else "unknown"
        logger.error(f"OpenAI API HTTP error {status_code}: {error_body}")
        raise RuntimeError(f"API error {status_code}: {error_body}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"OpenAI API connection error: {e}")
        raise RuntimeError(f"API connection error: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"OpenAI API timeout: {e}")
        raise RuntimeError(f"API timeout: {e}")

    logger.debug(f"OpenAI API response: {json.dumps(result, ensure_ascii=False)[:500]}")

    images = []
    # Standard OpenAI response: { "data": [ { "b64_json": "..." } ] } or { "data": [ { "url": "..." } ] }
    data_list = result.get("data", [])
    for item in data_list:
        if "b64_json" in item:
            import base64
            img_data = base64.b64decode(item["b64_json"])
            images.append(Image.open(BytesIO(img_data)))
        elif "url" in item:
            images.append(_download_image(item["url"]))

    return images


def _pixels_to_aspect_ratio(width: int, height: int) -> str:
    """Convert pixel dimensions to the closest supported aspect ratio string."""
    supported_ratios = [
        "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2",
        "4:5", "5:4", "21:9", "1:4", "4:1", "8:1", "1:8",
    ]
    target = width / height
    closest = min(supported_ratios, key=lambda r: abs(target - (int(r.split(":")[0]) / int(r.split(":")[1]))))
    return closest


def _upload_image_to_hosting(img: Image.Image) -> str:
    """Upload a PIL Image to a temporary hosting service and return the public URL.

    Tries multiple hosting services in order, falls back to data URL if all fail.
    """
    import tempfile
    tmp_path = None
    try:
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(tmp, format="JPEG", quality=85)
        tmp_path = tmp.name
        tmp.close()

        sess = _make_session()

        # Try multiple hosting services in order
        upload_services = [
            {
                "name": "catbox.moe",
                "url": "https://catbox.moe/user/api.php",
                "method": "post",
                "data": {"reqtype": "fileupload"},
                "files": "fileToUpload",
                "parse": lambda r: r.text.strip(),
            },
            {
                "name": "litterbox.catbox.moe",
                "url": "https://litterbox.catbox.moe/resources/internals/api.php",
                "method": "post",
                "data": {"reqtype": "fileupload", "time": "1h"},
                "files": "fileToUpload",
                "parse": lambda r: r.text.strip(),
            },
            {
                "name": "0x0.st",
                "url": "https://0x0.st",
                "method": "post",
                "data": {},
                "files": "file",
                "parse": lambda r: r.text.strip(),
            },
        ]

        for service in upload_services:
            try:
                with open(tmp_path, "rb") as f:
                    files = {service["files"]: f}
                    if service["method"] == "post":
                        resp = sess.post(
                            service["url"],
                            data=service["data"],
                            files=files,
                            timeout=30,
                        )
                    else:
                        continue
                if resp.ok:
                    url = service["parse"](resp)
                    if url and url.startswith("http"):
                        logger.info(f"Uploaded reference image via {service['name']}: {url[:60]}...")
                        return url
                logger.debug(f"{service['name']} upload failed ({resp.status_code})")
            except Exception as e:
                logger.debug(f"{service['name']} upload error: {e}")
                continue

        logger.warning("All hosting services failed, falling back to data URL")
    except Exception as e:
        logger.warning(f"Hosting upload error: {e}, falling back to data URL")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    # Fallback: use data URL
    return _pil_to_data_url(img)


def call_api_pixapi(prompt: str, negative_prompt: str, api_key: str, model: str, width: int, height: int, n_iter: int, batch_size: int, reference_images: list[Image.Image] = None) -> list[Image.Image]:
    config = API_PROVIDER_CONFIGS["pixapi"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "n": n_iter * batch_size,
        "size": _pixels_to_aspect_ratio(width, height),
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    if reference_images:
        # Use the edits endpoint when reference images are provided.
        # Pixapi supports both /v1/images/generations (text-to-image) and
        # /v1/images/edits (image-to-image). The edits endpoint accepts
        # an "image" parameter (single image URL) instead of "image_url" (list).
        # Also remove negative_prompt as Gemini models don't support it on edits endpoint.
        # Upload image to hosting service first (smaller payload than data URL)
        image_url = _upload_image_to_hosting(reference_images[0])
        payload["image"] = image_url
        payload.pop("image_url", None)
        payload.pop("negative_prompt", None)
        logger.info(f"Added {len(reference_images)} reference image(s) to Pixapi payload, using edits endpoint")
        return _call_openai_images_api(config["api_root"] + "/v1/images/edits", headers, payload, n_iter * batch_size)

    return _call_openai_images_api(config["base_url"], headers, payload, n_iter * batch_size)


def call_api_openai(prompt: str, negative_prompt: str, api_key: str, model: str, width: int, height: int, n_iter: int, batch_size: int, reference_images: list[Image.Image] = None) -> list[Image.Image]:
    config = API_PROVIDER_CONFIGS["openai"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "n": n_iter * batch_size,
        "size": f"{width}x{height}",
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    if reference_images:
        image_urls = [_pil_to_data_url(img) for img in reference_images]
        payload["image_url"] = image_urls
        logger.info(f"Added {len(reference_images)} reference image(s) to OpenAI payload")

    return _call_openai_images_api(config["base_url"], headers, payload, n_iter * batch_size)


def generate_with_api(provider: str, prompt: str, negative_prompt: str, api_key: str, model: str, width: int, height: int, n_iter: int, batch_size: int, reference_images: list[Image.Image] = None) -> list[Image.Image]:
    if provider == "modelscope":
        return call_api_modelscope(prompt, negative_prompt, api_key, model, width, height, n_iter, batch_size, reference_images)
    elif provider == "dashscope":
        return call_api_dashscope(prompt, api_key, model, width, height, n_iter, batch_size)
    elif provider == "pixapi":
        return call_api_pixapi(prompt, negative_prompt, api_key, model, width, height, n_iter, batch_size, reference_images)
    elif provider == "openai":
        return call_api_openai(prompt, negative_prompt, api_key, model, width, height, n_iter, batch_size, reference_images)
    else:
        raise ValueError(f"Unknown API provider: {provider}")