from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from .image_utils import ensure_rgb


@dataclass
class Tile:
    image: Image.Image
    left: int
    top: int
    right: int
    bottom: int

    @property
    def w(self) -> int:
        return self.right - self.left

    @property
    def h(self) -> int:
        return self.bottom - self.top


@dataclass
class TileLayout:
    original_size: Tuple[int, int]
    grid_size: Tuple[int, int]
    tile_size: int
    blend_padding: int
    tiles: List[Tile]


def _calculate_step(size: int, tile_size: int) -> Tuple[int, int]:
    """Equivalent layout math to TTP_Image_Tile_Batch.

    It chooses ceil(size/tile_size) tiles, then distributes the unavoidable
    overlap across the gaps. This is why a 6080px edge with 1024px tiles gives
    exactly 6 tiles rather than 7.
    """
    if size <= tile_size:
        return 1, 0
    num_tiles = (size + tile_size - 1) // tile_size
    overlap = (num_tiles * tile_size - size) // (num_tiles - 1)
    step = tile_size - overlap
    return num_tiles, step


def _axis_boxes(size: int, tile_size: int) -> List[Tuple[int, int]]:
    count, step = _calculate_step(size, tile_size)
    boxes: List[Tuple[int, int]] = []
    for index in range(count):
        start = index * step
        end = min(start + tile_size, size)
        if end - start < tile_size:
            start = max(0, size - tile_size)
        boxes.append((start, end))
    return boxes


def split_image(image: Image.Image, tile_size: int = 1024, tile_padding: int = 64) -> TileLayout:
    image = ensure_rgb(image)
    width, height = image.size
    x_boxes = _axis_boxes(width, tile_size)
    y_boxes = _axis_boxes(height, tile_size)

    tiles: List[Tile] = []
    for top, bottom in y_boxes:
        for left, right in x_boxes:
            crop = image.crop((left, top, right, bottom))
            if crop.size != (tile_size, tile_size) and width >= tile_size and height >= tile_size:
                raise RuntimeError(
                    f"Unexpected tile size {crop.size}; expected {(tile_size, tile_size)} "
                    f"for box {(left, top, right, bottom)} from image {image.size}"
                )
            tiles.append(Tile(crop, left, top, right, bottom))

    return TileLayout(
        original_size=(width, height),
        grid_size=(len(x_boxes), len(y_boxes)),
        tile_size=tile_size,
        blend_padding=max(0, int(tile_padding)),
        tiles=tiles,
    )


def _gradient_blend(a: Image.Image, b: Image.Image, overlap: int, direction: str, padding: int) -> Image.Image:
    """Blend neighboring tiles like TTP_Image_Assy, but using NumPy for stability."""
    a = ensure_rgb(a)
    b = ensure_rgb(b)
    overlap = max(0, int(overlap))
    blend_size = min(max(0, int(padding)), overlap)

    if overlap <= 0:
        if direction == "horizontal":
            canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)))
            canvas.paste(a, (0, 0))
            canvas.paste(b, (a.width, 0))
        else:
            canvas = Image.new("RGB", (max(a.width, b.width), a.height + b.height))
            canvas.paste(a, (0, 0))
            canvas.paste(b, (0, a.height))
        return canvas

    if blend_size == 0:
        if direction == "horizontal":
            canvas = Image.new("RGB", (a.width + b.width - overlap, max(a.height, b.height)))
            canvas.paste(a.crop((0, 0, a.width - overlap, a.height)), (0, 0))
            canvas.paste(b, (a.width - overlap, 0))
        else:
            canvas = Image.new("RGB", (max(a.width, b.width), a.height + b.height - overlap))
            canvas.paste(a.crop((0, 0, a.width, a.height - overlap)), (0, 0))
            canvas.paste(b, (0, a.height - overlap))
        return canvas

    # TTP only feathers `padding` pixels centered inside the physical overlap.
    leftover = overlap - blend_size
    before = leftover // 2
    after = leftover - before

    if direction == "horizontal":
        if a.height != b.height:
            raise ValueError(f"Horizontal tile heights differ: {a.size} vs {b.size}")
        a_np = np.asarray(a, dtype=np.float32)
        b_np = np.asarray(b, dtype=np.float32)
        a_blend = a_np[:, a.width - overlap + before : a.width - after, :]
        b_blend = b_np[:, before : before + blend_size, :]
        if a_blend.shape != b_blend.shape:
            raise ValueError(f"Horizontal blend shapes differ: {a_blend.shape} vs {b_blend.shape}")
        alpha = np.linspace(1.0, 0.0, blend_size, endpoint=True, dtype=np.float32)[None, :, None]
        blended = a_blend * alpha + b_blend * (1.0 - alpha)

        left_part = a_np[:, : a.width - overlap + before, :]
        right_part = b_np[:, before + blend_size :, :]
        result = np.concatenate([left_part, blended, right_part], axis=1)
    else:
        if a.width != b.width:
            raise ValueError(f"Vertical tile widths differ: {a.size} vs {b.size}")
        a_np = np.asarray(a, dtype=np.float32)
        b_np = np.asarray(b, dtype=np.float32)
        a_blend = a_np[a.height - overlap + before : a.height - after, :, :]
        b_blend = b_np[before : before + blend_size, :, :]
        if a_blend.shape != b_blend.shape:
            raise ValueError(f"Vertical blend shapes differ: {a_blend.shape} vs {b_blend.shape}")
        alpha = np.linspace(1.0, 0.0, blend_size, endpoint=True, dtype=np.float32)[:, None, None]
        blended = a_blend * alpha + b_blend * (1.0 - alpha)

        top_part = a_np[: a.height - overlap + before, :, :]
        bottom_part = b_np[before + blend_size :, :, :]
        result = np.concatenate([top_part, blended, bottom_part], axis=0)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def merge_tiles(layout: TileLayout, processed_tiles: List[Image.Image]) -> Image.Image:
    if len(layout.tiles) != len(processed_tiles):
        raise ValueError(f"Tile count mismatch: {len(layout.tiles)} vs {len(processed_tiles)}")

    num_cols, num_rows = layout.grid_size
    padding = layout.blend_padding

    # First assemble each row, matching TTP_Image_Assy's ordering.
    row_images: List[Image.Image] = []
    for row in range(num_rows):
        first_idx = row * num_cols
        row_image = ensure_rgb(processed_tiles[first_idx])
        for col in range(1, num_cols):
            idx = row * num_cols + col
            tile_image = ensure_rgb(processed_tiles[idx])
            prev_right = layout.tiles[idx - 1].right
            left = layout.tiles[idx].left
            overlap = prev_right - left
            row_image = _gradient_blend(row_image, tile_image, overlap, "horizontal", padding)
        row_images.append(row_image)

    final_image = row_images[0]
    for row in range(1, num_rows):
        prev_idx = (row - 1) * num_cols
        cur_idx = row * num_cols
        overlap = layout.tiles[prev_idx].bottom - layout.tiles[cur_idx].top
        final_image = _gradient_blend(final_image, row_images[row], overlap, "vertical", padding)

    # Defensive crop in case a third-party PIL version rounds differently.
    return final_image.crop((0, 0, layout.original_size[0], layout.original_size[1]))
