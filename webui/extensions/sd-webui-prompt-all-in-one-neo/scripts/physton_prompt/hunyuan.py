import os
import time
import warnings
from scripts.physton_prompt.get_lang import get_lang

# 忽略 transformers 的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*token_type_ids.*')

# 支持的 Hy-MT2 模型列表
SUPPORTED_MODELS = {
    "tencent/Hy-MT2-1.8B": {
        "name": "Hy-MT2-1.8B",
        "repo_id": "tencent/Hy-MT2-1.8B",
        "local_path": "Hy-MT2-1.8B",
        "description": "基础版，适合低显存环境"
    },
    "tencent/Hy-MT2-7B-FP8": {
        "name": "Hy-MT2-7B-FP8",
        "repo_id": "tencent/Hy-MT2-7B-FP8",
        "local_path": "Hy-MT2-7B-FP8",
        "description": "7B 版本，FP8 量化，平衡性能和质量"
    },
    "tencent/Hy-MT2-7B": {
        "name": "Hy-MT2-7B",
        "repo_id": "tencent/Hy-MT2-7B",
        "local_path": "Hy-MT2-7B",
        "description": "7B 版本，全精度，更高质量"
    },
    "tencent/Hy-MT2-30B-A3B": {
        "name": "Hy-MT2-30B-A3B",
        "repo_id": "tencent/Hy-MT2-30B-A3B",
        "local_path": "Hy-MT2-30B-A3B",
        "description": "30B 版本，A3B 量化，适合高端显卡"
    },
    "tencent/Hy-MT2-30B-A3B-FP8": {
        "name": "Hy-MT2-30B-A3B-FP8",
        "repo_id": "tencent/Hy-MT2-30B-A3B-FP8",
        "local_path": "Hy-MT2-30B-A3B-FP8",
        "description": "30B 版本，A3B+FP8 量化，最优性能"
    }
}

# 兼容旧版 Hy-MT2 分支保存的 ModelScope 风格模型 ID。
# Hugging Face 官方仓库位于 tencent/*；旧配置仍可继续使用，无需用户手动重置。
LEGACY_MODEL_ALIASES = {
    "Tencent-Hunyuan/Hy-MT2-1.8B": "tencent/Hy-MT2-1.8B",
    "Tencent-Hunyuan/Hy-MT2-7B-FP8": "tencent/Hy-MT2-7B-FP8",
    "Tencent-Hunyuan/Hy-MT2-7B": "tencent/Hy-MT2-7B",
    "Tencent-Hunyuan/Hy-MT2-30B-A3B": "tencent/Hy-MT2-30B-A3B",
    "Tencent-Hunyuan/Hy-MT2-30B-A3B-FP8": "tencent/Hy-MT2-30B-A3B-FP8",
}

def normalize_model_key(model_key):
    if not model_key:
        return "tencent/Hy-MT2-1.8B"
    return LEGACY_MODEL_ALIASES.get(model_key, model_key)

model = None
tokenizer = None
current_model_key = "tencent/Hy-MT2-1.8B"
loaded_model_key = None
# 模型目录设置在 webui 的 models 目录下
# 从 scripts/physton_prompt/ 到 webui/models 需要 4 个 ../
cache_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../../../../models')
loading = False

def get_supported_models():
    """获取支持的模型列表"""
    return list(SUPPORTED_MODELS.keys())

def get_model_info(model_key):
    """获取模型信息"""
    model_key = normalize_model_key(model_key)
    return SUPPORTED_MODELS.get(model_key, SUPPORTED_MODELS["tencent/Hy-MT2-1.8B"])

def initialize(model_key=None, reload=False):
    global model, tokenizer, current_model_key, loaded_model_key, cache_dir, loading
    
    # 使用指定的模型或当前模型
    if model_key is None:
        model_key = current_model_key
    model_key = normalize_model_key(model_key)
    current_model_key = model_key
    
    # 获取模型信息
    model_info = get_model_info(model_key)
    model_name = model_info["name"]
    repo_id = model_info["repo_id"]
    local_model_path = os.path.join(cache_dir, model_info["local_path"])
    
    # 延迟导入 torch，只在需要时导入
    import torch
    
    if loading:
        while loading:
            time.sleep(0.1)
        if model is None or tokenizer is None:
            raise Exception(get_lang('model_not_initialized'))
        # Another request may have been loading a different Hy-MT2 variant.
        # Only return early when the model that just finished is the one we need.
        if loaded_model_key == model_key and not reload:
            return
    if not reload and model is not None and tokenizer is not None and loaded_model_key == model_key:
        return
    loading = True
    model = None
    tokenizer = None

    # 检查本地模型是否存在。若不存在，先显式下载到插件的模型目录。
    # 这样可以对 HF_ENDPOINT 镜像失效做自动回退，而不修改 Forge Neo 全局设置。
    config_file = os.path.join(local_model_path, "config.json")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if os.path.exists(local_model_path) and os.path.exists(config_file):
            model_name = local_model_path
            print(f'[sd-webui-prompt-all-in-one] Loading local model from {local_model_path}...')
        else:
            from huggingface_hub import snapshot_download

            os.makedirs(local_model_path, exist_ok=True)
            configured_endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
            plugin_endpoint = os.environ.get("PAIO_HYMT2_ENDPOINT", "").strip().rstrip("/")
            endpoints = []
            for endpoint in (plugin_endpoint, configured_endpoint, "https://huggingface.co"):
                if endpoint and endpoint not in endpoints:
                    endpoints.append(endpoint)

            errors = []
            downloaded = False
            for endpoint in endpoints:
                try:
                    print(f'[sd-webui-prompt-all-in-one] Downloading {repo_id} via {endpoint} ...')
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_model_path,
                        endpoint=endpoint,
                    )
                    downloaded = True
                    print(f'[sd-webui-prompt-all-in-one] Download completed via {endpoint}.')
                    break
                except Exception as download_error:
                    errors.append(f'{endpoint}: {download_error}')
                    print(f'[sd-webui-prompt-all-in-one] Download via {endpoint} failed: {download_error}')

            if not downloaded:
                # 避免留下一个只有部分文件的目录被误判为完整模型。
                raise RuntimeError(
                    'Hy-MT2 download failed on all configured endpoints.\n' + '\n'.join(errors)
                )

            model_name = local_model_path

        print(f'[sd-webui-prompt-all-in-one] Loading model {model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        loaded_model_key = model_key
        print(f'[sd-webui-prompt-all-in-one] Model {model_name} loaded.')
        loading = False
    except Exception as e:
        loading = False
        raise e

def translate(text, src_lang, target_lang):
    global model, tokenizer
    
    # 在翻译时导入 torch
    import torch

    if not text:
        if isinstance(text, list):
            return []
        else:
            return ''

    if model is None:
        raise Exception(get_lang('model_not_initialized'))

    if tokenizer is None:
        raise Exception(get_lang('model_not_initialized'))

    if src_lang == target_lang:
        return text

    # Use the requested TARGET language. The Hy-MT2 fork used src_lang here,
    # which made many non-Chinese/English direction pairs translate to the wrong language.
    lang_map = {
        'zh_CN': '中文', 'zh_HK': '繁体中文', 'zh_TW': '繁体中文',
        'en_US': '英语', 'en_GB': '英语', 'ja_JP': '日语', 'ko_KR': '韩语',
        'de_DE': '德语', 'fr_FR': '法语', 'ru_RU': '俄语', 'es_ES': '西班牙语',
        'pt_PT': '葡萄牙语', 'it_IT': '意大利语', 'ar_SA': '阿拉伯语',
        'hi_IN': '印地语', 'id_ID': '印尼语', 'vi_VN': '越南语',
        'th_TH': '泰语', 'ms_MY': '马来语', 'tr_TR': '土耳其语',
    }
    tgt = lang_map.get(target_lang, target_lang)

    if isinstance(text, list):
        results = []
        for t in text:
            prompt = f"将以下文本翻译成{tgt},注意只需要输出翻译后的结果,不要额外解释:\n\n{t}"
            print(f'[sd-webui-prompt-all-in-one] Hunyuan translating: {prompt}')
            try:
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=500,
                        temperature=0.7,
                        top_p=0.6,
                        top_k=20,
                        repetition_penalty=1.05,
                    )
                result = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                print(f'[sd-webui-prompt-all-in-one] Hunyuan result: {result}')
                results.append(result.strip())
            except Exception as e:
                print(f'[sd-webui-prompt-all-in-one] Hunyuan translate error: {str(e)}')
                import traceback
                traceback.print_exc()
                results.append(t)  # 如果翻译失败，返回原文
        return results
    else:
        prompt = f"将以下文本翻译成{tgt},注意只需要输出翻译后的结果,不要额外解释:\n\n{text}"
        print(f'[sd-webui-prompt-all-in-one] Hunyuan translating: {prompt}')
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.6,
                    top_k=20,
                    repetition_penalty=1.05,
                )
            result = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            print(f'[sd-webui-prompt-all-in-one] Hunyuan result: {result}')
            return [result.strip()]
        except Exception as e:
            print(f'[sd-webui-prompt-all-in-one] Hunyuan translate error: {str(e)}')
            import traceback
            traceback.print_exc()
            return [text]  # 如果翻译失败，返回原文