"""模型目录与模型选择辅助函数。"""

LLM_MODELS_BY_PROVIDER = {
    "ModelScope": [
        "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next", "ZhipuAI/GLM-5.3",
        "ZhipuAI/GLM-5.3-Flash", "deepseek-ai/DeepSeek-V4-Pro-0813",
        "deepseek-ai/DeepSeek-V4-Pro", "moonshotai/Kimi-K3",
    ],
}

IMAGE_GENERATION_MODELS_BY_PROVIDER = {
    "ModelScope": [
        "krea/Krea-2-Turbo", "Tongyi-MAI/Z-Image",
        "Qwen/Qwen-Image-2.1",
    ],
}

IMAGE_GENERATION_MODELS = [
    "krea/Krea-2-Turbo",
    "Tongyi-MAI/Z-Image",
    "Qwen/Qwen-Image-2.1",
]

VIDEO_GENERATION_MODELS = []


def _image_models_for_provider(provider):
    models = IMAGE_GENERATION_MODELS_BY_PROVIDER.get(str(provider or "").strip())
    return models or IMAGE_GENERATION_MODELS


def _models_for_provider(cfg, kind, provider, defaults):
    values = list(defaults)
    custom = (cfg.get("custom_models", {}) if isinstance(cfg, dict) else {}).get(kind, {})
    values.extend(custom.get(provider, []) if isinstance(custom, dict) else [])
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))


def _is_saved_image_model(cfg, model):
    if model in IMAGE_GENERATION_MODELS:
        return True
    custom = cfg.get("custom_models", {}).get("image", {}) if isinstance(cfg, dict) else {}
    return any(model in (values or []) for values in custom.values() if isinstance(values, list))


def _default_image_model_for_provider(provider, current_model=None):
    models = _image_models_for_provider(provider)
    current_model = str(current_model or "").strip()
    return current_model if current_model in models else models[0]
