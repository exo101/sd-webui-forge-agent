# =============================================================================
# Agent Tools — 公共辅助（UI 画面比例 / 上传图尺寸）
# =============================================================================

import os

from PIL import Image


_UI_ASPECT_SIZES = {
    "1:1": (1024, 1024),
    "9:16": (1024, 1792),
    "16:9": (1792, 1024),
}


def _ui_aspect_size(cfg):
    """Return the global UI aspect-ratio size used by every image model."""
    ratio = str((cfg or {}).get("image_aspect_ratio") or "1:1").strip()
    return _UI_ASPECT_SIZES.get(ratio, _UI_ASPECT_SIZES["1:1"])


def _uploaded_image_size(image):
    """Return the uploaded image dimensions for the keep-original-ratio mode."""
    try:
        source = image[0] if isinstance(image, (list, tuple)) and image else image
        if isinstance(source, Image.Image):
            return source.size
        if isinstance(source, str) and os.path.isfile(source):
            with Image.open(source) as img:
                return img.size
    except Exception:
        pass
    return None


def _apply_ui_aspect_size(size, cfg, *, allow_auto=True):
    """Apply the UI ratio only when the caller did not provide an explicit size."""
    value = str(size or "").strip().lower()
    defaults = ("", "1024x1024", "1:1", "square") if not allow_auto else ("auto", "none", "null")
    if value in defaults:
        width, height = _ui_aspect_size(cfg)
        return f"{width}x{height}"
    return size
