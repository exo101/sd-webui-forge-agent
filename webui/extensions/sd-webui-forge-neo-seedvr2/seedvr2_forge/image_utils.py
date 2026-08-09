from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


RESAMPLE_MAP = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
}


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_to_long_edge(image: Image.Image, target_long_edge: int, method: str = "lanczos") -> Image.Image:
    image = ensure_rgb(image)
    w, h = image.size
    long_edge = max(w, h)
    if long_edge == target_long_edge:
        return image
    scale = target_long_edge / float(long_edge)
    new_w = max(8, int(round(w * scale / 8.0) * 8))
    new_h = max(8, int(round(h * scale / 8.0) * 8))
    return image.resize((new_w, new_h), RESAMPLE_MAP.get(method, Image.LANCZOS))


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(ensure_rgb(image)).astype(np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def alpha_weight_mask(width: int, height: int, feather: int) -> np.ndarray:
    feather = max(1, feather)
    wx = np.ones(width, dtype=np.float32)
    wy = np.ones(height, dtype=np.float32)

    ramp_x = np.linspace(0.0, 1.0, feather, dtype=np.float32)
    ramp_y = np.linspace(0.0, 1.0, feather, dtype=np.float32)

    wx[:feather] = np.minimum(wx[:feather], ramp_x)
    wx[-feather:] = np.minimum(wx[-feather:], ramp_x[::-1])
    wy[:feather] = np.minimum(wy[:feather], ramp_y)
    wy[-feather:] = np.minimum(wy[-feather:], ramp_y[::-1])

    mask = wy[:, None] * wx[None, :]
    return np.clip(mask, 1e-6, 1.0)


def crop_to_size(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    w, h = size
    return image.crop((0, 0, w, h))
