from modules import extra_networks, prompt_parser, sd_models
from functools import reduce


def get_token_counter(text, steps):
    # FIX: Use try-except to safely handle PyTorch/model access errors (TypeError NoneType)
    # that occur during model loading/switching when the token counter API is triggered.
    try:
        # copy from modules.ui.py
        try:
            text, _ = extra_networks.parse_prompt(text)

            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

        except Exception:
            # a parsing error can happen here during typing, and we don't want to bother the user with
            # messages related to it in console
            prompt_schedules = [[[steps, text]]]

        # forge-neo removed modules/sd_hijack.py; use the model-level API introduced in forge-neo
        try:
            get_prompt_lengths_on_ui = sd_models.model_data.sd_model.get_prompt_lengths_on_ui
            assert get_prompt_lengths_on_ui is not None
        except Exception:
            return {"token_count": 0, "max_length": 0}

        flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
        prompts = [prompt_text for step, prompt_text in flat_prompts]

        token_count, max_length = max([get_prompt_lengths_on_ui(prompt) for prompt in prompts],
                                      key=lambda args: args[0])

        return {"token_count": token_count, "max_length": max_length}

    except Exception as e:
        # return 0 token count if any error (model instability, parsing error, etc.) occurs during calculation
        return {"token_count": 0, "max_length": 0}
