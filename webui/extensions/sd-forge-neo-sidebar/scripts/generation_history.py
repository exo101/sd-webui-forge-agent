"""
生成历史 API - 提供最近生成的图像列表
"""

import os
import glob
from pathlib import Path
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse
from modules import script_callbacks

router = APIRouter(prefix="/neo-history", tags=["neo-history"])

# 输出目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # webui/
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# 支持的图片格式
IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')

# 最大返回数量
MAX_IMAGES = 200


def get_recent_images(tab_type=None, limit=50):
    """获取最近的生成图像列表"""
    if tab_type == "txt2img":
        base_dirs = [os.path.join(OUTPUT_DIR, "txt2img-images")]
    elif tab_type == "img2img":
        base_dirs = [os.path.join(OUTPUT_DIR, "img2img-images")]
    else:
        base_dirs = [
            os.path.join(OUTPUT_DIR, "txt2img-images"),
            os.path.join(OUTPUT_DIR, "img2img-images"),
        ]

    images = []
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        # 遍历所有子目录（日期文件夹）
        for date_dir in sorted(os.listdir(base_dir), reverse=True):
            date_path = os.path.join(base_dir, date_dir)
            if not os.path.isdir(date_path):
                continue

            for ext in IMAGE_EXTENSIONS:
                pattern = os.path.join(date_path, ext)
                for img_path in glob.glob(pattern):
                    try:
                        mtime = os.path.getmtime(img_path)
                        # 获取相对路径（相对于 outputs 目录）
                        rel_path = os.path.relpath(img_path, OUTPUT_DIR)
                        # 确定类型
                        if "txt2img" in img_path:
                            img_type = "txt2img"
                        else:
                            img_type = "img2img"
                        images.append({
                            "path": rel_path,
                            "full_path": img_path,
                            "mtime": mtime,
                            "type": img_type,
                            "filename": os.path.basename(img_path),
                        })
                    except OSError:
                        continue

    # 按修改时间排序（最新的在前）
    images.sort(key=lambda x: x["mtime"], reverse=True)
    return images[:limit]


@router.get("/images")
async def list_images(limit: int = 50, tab: str = "all"):
    """列出最近的生成图像"""
    tab_type = tab if tab in ("txt2img", "img2img") else None
    images = get_recent_images(tab_type=tab_type, limit=limit)
    return {"images": images}


@router.get("/image/{path:path}")
async def serve_image(path: str):
    """提供图像文件"""
    full_path = os.path.join(OUTPUT_DIR, path)

    # 安全检查：确保路径在 outputs 目录内
    real_path = os.path.realpath(full_path)
    real_output = os.path.realpath(OUTPUT_DIR)
    if not real_path.startswith(real_output):
        return {"error": "Access denied"}

    if not os.path.exists(full_path):
        return {"error": "File not found"}

    return FileResponse(full_path)


def on_app_started(demo, app: FastAPI):
    """注册 API 路由"""
    app.include_router(router)


script_callbacks.on_app_started(on_app_started)