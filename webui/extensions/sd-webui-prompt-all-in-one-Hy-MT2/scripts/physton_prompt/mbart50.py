import os
import time
from scripts.physton_prompt.get_lang import get_lang

model = None
tokenizer = None
model_name = "facebook/mbart-large-50-many-to-many-mmt"
# 模型目录设置在 webui 的 models 目录下
# 从 scripts/physton_prompt/ 到 webui/models 需要：
# scripts/physton_prompt/ → scripts/ (../)
# scripts/ → extensions/sd-webui-prompt-all-in-one-forgeneo/ (../../)
# extensions/sd-webui-prompt-all-in-one-forgeneo/ → extensions/ (../../../)
# extensions/ → webui/ (../../../../)
# webui/ → webui/models (../../../../models)
_possible_paths = [
    # webui/models (正确路径：向上 4 层)
    os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../../../../models'),
    # 插件目录下的 models (备用：向上 2 层)
    os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../../models'),
]
# 选择存在的路径，或者使用第一个路径
cache_dir = _possible_paths[0]
for p in _possible_paths:
    model_test_path = os.path.join(p, "mbart-large-50-many-to-many-mmt")
    if os.path.exists(model_test_path):
        cache_dir = p
        break
loading = False

def initialize(reload=False):
    global model, tokenizer, model_name, cache_dir, loading
    if loading:
        while not loading:
            time.sleep(0.1)
            pass
        if model is None or tokenizer is None:
            raise Exception('error')
        # raise Exception(get_lang('model_is_loading'))
        return
    if not reload and model is not None:
        return
    loading = True
    model = None
    tokenizer = None

    model_path = os.path.join(cache_dir, "mbart-large-50-many-to-many-mmt")
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_path) and os.path.exists(model_file):
        model_name = model_path

    try:
        from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
        print(f'[sd-webui-prompt-all-in-one] Loading model {model_name} from {cache_dir}...')
        model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
        print(f'[sd-webui-prompt-all-in-one] Model {model_name} loaded.')
        loading = False
    except Exception as e:
        loading = False
        raise e

def translate(text, src_lang, target_lang):
    global model, tokenizer

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

    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(text, return_tensors="pt", padding=True)
    generated_tokens = model.generate(
        **encoded_input, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_new_tokens=500
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
