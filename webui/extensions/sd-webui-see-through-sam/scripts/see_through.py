from __future__ import annotations

import os
import sys
import logging
import subprocess
import gradio as gr
from modules import script_callbacks
from modules import scripts
from modules import shared
from modules.processing import StableDiffusionProcessing
import torch
import numpy as np
from PIL import Image

from layer_separation_module import create_layer_separation_ui, on_ui_settings, on_app_started, install_dependencies

logger = logging.getLogger("See-Through")
logger.setLevel(logging.WARNING)

see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
see_through_path = os.path.abspath(see_through_path)
sys.path.insert(0, see_through_path)

inference_path = os.path.join(see_through_path, "inference")
inference_path = os.path.abspath(inference_path)
if inference_path not in sys.path:
    sys.path.insert(0, inference_path)
    logger.info(f"[Path] Added to sys.path: {inference_path}")

if not os.path.exists(inference_path):
    logger.error(f"[Path Error] Inference path does not exist: {inference_path}")
else:
    logger.info(f"[Path OK] Inference path exists: {inference_path}")


def install_dependencies():
    """自动安装依赖"""
    try:
        import importlib

        dependencies = {
            'psd_tools': 'psd-tools',
            'pycocotools': 'pycocotools'
        }

        missing_deps = []
        for module_name, package_name in dependencies.items():
            try:
                importlib.import_module(module_name)
                logger.info(f"{package_name} 已安装")
            except ImportError:
                logger.warning(f"{package_name} 未安装，正在安装...")
                missing_deps.append(package_name)

        if missing_deps:
            logger.info("正在安装缺失的依赖...")
            for package in missing_deps:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    logger.info(f"{package} 安装成功")
                except subprocess.CalledProcessError as e:
                    logger.error(f"{package} 安装失败: {e}")
    except Exception as e:
        logger.error(f"依赖安装检查失败: {e}")


def on_app_started(demo, app):
    """WebUI启动时自动安装依赖"""
    logger.info("See-Through: 检查依赖...")
    install_dependencies()


def on_ui_settings():
    shared.opts.add_option(
        key="see_through_enabled",
        info=shared.OptionInfo(
            default=False,
            label="启用See-Through插件",
            section=("see_through", "See-Through")
        )
    )
    webui_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    default_output_dir = os.path.join(webui_dir, "output", "See-Through", "layerdiff_output")
    shared.opts.add_option(
        key="see_through_output_dir",
        info=shared.OptionInfo(
            default=default_output_dir,
            label="See-Through输出目录",
            section=("see_through", "See-Through")
        )
    )

# Imported from layer_separation_module: on_ui_settings, on_app_started, script_callbacks
# (Registered via layer_separation_module to avoid duplication)