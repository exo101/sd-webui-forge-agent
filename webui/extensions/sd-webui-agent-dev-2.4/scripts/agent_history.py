"""聊天历史中的文本、图片路径及图片编码辅助函数。"""

import base64
import io
import os

from PIL import Image

from scripts.agent_config import _save_pil_to_tempfile


def _extract_text_from_history_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    if isinstance(content, dict) and "path" in content:
        return ""
    return str(content)


def _normalize_image_path(image_value):
    if image_value is None:
        return None
    if isinstance(image_value, str):
        return image_value if os.path.isfile(image_value) else None
    if isinstance(image_value, dict):
        path = image_value.get("path")
        return path if path and os.path.isfile(path) else None
    if isinstance(image_value, Image.Image):
        return _save_pil_to_tempfile(image_value)
    return None


def _lenient_image_path(p):
    """宽容校验上传图片，兼容单路径和 Gradio 多图路径列表。"""
    if isinstance(p, (list, tuple)):
        valid = []
        for item in p:
            checked = _lenient_image_path(item)
            if checked:
                valid.append(checked)
        return valid or None
    if not p or not isinstance(p, str) or not os.path.isfile(p):
        return None
    try:
        with Image.open(p) as im:
            im.verify()
        return p
    except Exception:
        return None


def _normalize_image_paths(image_value):
    if image_value is None:
        return []
    if isinstance(image_value, (list, tuple)):
        return [path for item in image_value if (path := _normalize_image_path(item))]
    path = _normalize_image_path(image_value)
    return [path] if path else []


def _latest_image_paths_from_history(history, limit=5):
    paths = []
    for item in reversed(history or []):
        if not isinstance(item, dict) or not isinstance(item.get("content"), dict):
            continue
        path = _normalize_image_path(item["content"])
        if path and os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            paths.append(path)
            if len(paths) >= limit:
                break
    return paths


def _image_to_compressed_base64(img_path, max_size=1024, quality=85):
    try:
        file_size = os.path.getsize(img_path)
        img = Image.open(img_path)
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        compressed = buf.getvalue()
        if len(compressed) >= file_size:
            with open(img_path, "rb") as f:
                compressed = f.read()
            mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
        else:
            mime = "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(compressed).decode('utf-8')}", file_size, len(compressed)
    except Exception as e:
        print(f"[Agent] ⚠️ 图片压缩失败，回退原图编码: {e}")
        try:
            file_size = os.path.getsize(img_path)
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
            return f"data:{mime};base64,{b64}", file_size, file_size
        except Exception:
            return None, 0, 0
