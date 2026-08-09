from scripts.physton_prompt.translator.base_translator import BaseTranslator
from scripts.physton_prompt.get_lang import get_lang
from scripts.physton_prompt.hunyuan import initialize as hunyuan_initialize, translate as hunyuan_translate, get_supported_models


class HunyuanTranslator(BaseTranslator):
    def __init__(self):
        super().__init__('hunyuan')

    def translate(self, text):
        if not text:
            if isinstance(text, list):
                return []
            else:
                return ''

        # 从 api_config 中获取模型配置
        model_key = None
        if self.api_config and 'model' in self.api_config:
            model_key = self.api_config['model']

        # 确保模型已初始化（支持选择不同模型）
        hunyuan_initialize(model_key=model_key)

        result = hunyuan_translate(text=text, src_lang=self.from_lang, target_lang=self.to_lang)
        if not result:
            raise Exception(get_lang('response_is_empty', {'0': 'Hunyuan MT2'}))

        if isinstance(text, list):
            return result
        else:
            return result[0]

    def translate_batch(self, texts):
        return self.translate(texts)