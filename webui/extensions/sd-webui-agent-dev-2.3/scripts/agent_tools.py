# =============================================================================
# Agent Tools — 工具门面模块（兼容性入口）
# 实现已按领域拆分到以下子模块：
#   agent_tools_common.py    公共辅助（画面比例 / 上传图尺寸）
#   agent_tools_models.py    模型管理工具（checkpoint / VAE / TE / 设置）
#   agent_tools_image.py     本地生图与图像处理工具
#   agent_tools_api.py       远程 API 图像生成/编辑工具
#   agent_tools_video.py     视频生成工具（H3 / Dreamina）
#   agent_tools_workspace.py 工作区 / 文档 / 联网 / 插件调研工具
#   agent_tools_schemas.py   Function Calling schema (TOOLS)
# 本模块保持原有导入接口不变：TOOLS / TOOL_FUNCTIONS / 全部工具函数。
# =============================================================================

import os
import sys
import json
import time
import base64
import traceback
import tempfile
import csv
import zipfile
import subprocess
import re
import html
import urllib.request
import urllib.error
import urllib.parse
from xml.etree import ElementTree
from pathlib import Path
from io import BytesIO

from PIL import Image

from modules import shared, scripts, sd_models, sd_samplers, postprocessing
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    process_images,
)
from modules.shared import device, opts
from modules.images import flatten
from modules.sd_samplers_common import images_tensor_to_samples
from backend.args import dynamic_args
import numpy as np
import torch

# 从配置模块导入共享函数
from scripts.agent_config import (
    load_config, _save_pil_to_tempfile, _get_current_checkpoint, _get_sampler, _scan_model_dir,
    _REGISTRY_AVAILABLE, get_registered_tools, get_tool_function, list_registered_tools
)
from scripts.agent_tools_common import (
    _UI_ASPECT_SIZES, _ui_aspect_size, _uploaded_image_size, _apply_ui_aspect_size,
)
from scripts.agent_tools_models import (
    switch_model_tool, update_settings_tool,
    list_models_tool, list_vae_tool, list_text_encoders_tool, list_controlnet_tool,
    MODEL_GUIDE, get_model_guide_tool, _forge_update_additional_modules,
    set_vae_tool, set_text_encoder_tool, set_model_components_tool,
    list_samplers_tool, list_upscalers_tool, get_current_settings_tool, list_loras_tool,
)
from scripts.agent_tools_image import (
    txt2img_tool, img2img_tool, _edit_with_image_stitch, upscale_tool,
    video_keyframe_extract_tool, trellis2_image_to_3d_tool, video_to_frames_tool,
    stitch_images_tool, list_preprocessors_tool, apply_adetailer_tool,
    remove_background_tool, layer_separation_tool, edit_image_tool,
    change_background_tool, generate_with_lora_tool,
)
from scripts.agent_tools_api import (
    api_image_edit_tool, _modelscope_poll_task, _ASPECT_RATIO_TO_PIXELS,
    _resolve_image_size, _pixels_to_gemini_config, _infer_image_size_from_prompt,
    api_image_generate_tool, _select_image_api_key, _get_image_api_base_url,
    _is_gemini_image_model, _call_gemini_generate, _format_api_http_error,
)
from scripts.agent_tools_video import (
    _get_webui_base_url, h3_video_generate_tool, dreamina_video_generate_tool,
)
from scripts.agent_tools_workspace import (
    list_extensions_tool, research_extension_tool,
    _TEXT_FILE_EXTENSIONS, _BLOCKED_FILE_EXTENSIONS,
    _get_workspace_root, _get_webui_root, _resolve_webui_path, _limit_chars,
    _read_document_text, read_workspace_file_tool, analyze_document_tool,
    _workspace_backup_path, repair_workspace_file_tool, diagnose_workspace_tool,
    audit_extensions_tool, _fetch_web_text, _strip_html_text,
    web_search_tool, web_read_url_tool, explore_webui_tool,
)
from scripts.agent_tools_schemas import TOOLS

# 每个导入实例持有独立的 TOOLS 列表（与原始模块级定义语义一致：
# WebUI 顶层导入与 scripts.agent_tools 是独立实例，仅后者会被
# agent_chat 的注册表合并逻辑追加注册工具）。
TOOLS = list(TOOLS)


TOOL_FUNCTIONS = {
    "txt2img": txt2img_tool,
    "img2img": img2img_tool,
    "switch_model": switch_model_tool,
    "update_settings": update_settings_tool,
    "upscale": upscale_tool,
    "generate_with_lora": generate_with_lora_tool,
    "list_models": list_models_tool,
    "list_vae": list_vae_tool,
    "list_text_encoders": list_text_encoders_tool,
    "list_controlnet": list_controlnet_tool,
    "get_model_guide": get_model_guide_tool,
    "set_vae": set_vae_tool,
    "set_text_encoder": set_text_encoder_tool,
    "set_model_components": set_model_components_tool,
    "list_samplers": list_samplers_tool,
    "list_upscalers": list_upscalers_tool,
    "get_current_settings": get_current_settings_tool,
    "list_loras": list_loras_tool,
    "video_keyframe_extract": video_keyframe_extract_tool,
    "trellis2_image_to_3d": trellis2_image_to_3d_tool,
    "video_to_frames": video_to_frames_tool,
    "h3_video_generate": h3_video_generate_tool,
    "dreamina_video_generate": dreamina_video_generate_tool,
    "stitch_images": stitch_images_tool,
    "list_preprocessors": list_preprocessors_tool,
    "apply_adetailer": apply_adetailer_tool,
    "remove_background": remove_background_tool,
    "layer_separation": layer_separation_tool,
    "api_image_edit": api_image_edit_tool,
    "api_image_generate": api_image_generate_tool,
    "edit_image": edit_image_tool,
    "change_background": change_background_tool,
    "web_search": web_search_tool,
    "web_read_url": web_read_url_tool,
    "list_extensions": list_extensions_tool,
    "research_extension": research_extension_tool,
    "explore_webui": explore_webui_tool,
    "read_workspace_file": read_workspace_file_tool,
    "analyze_document": analyze_document_tool,
    "audit_extensions": audit_extensions_tool,
    "repair_workspace_file": repair_workspace_file_tool,
    "diagnose_workspace": diagnose_workspace_tool,
}
