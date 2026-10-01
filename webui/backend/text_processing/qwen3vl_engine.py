# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/krea2.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen35.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen3vl.py

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from modules.shared import opts

KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


def _quantized_matmul_context(model):
    # Delayed import avoids operations <-> operations_mixed_precision cycle
    # during WebUI startup.
    from backend.operations_mixed_precision import use_quantized_matmul
    return use_quantized_matmul(model)


def _quant_stats():
    from backend.operations_mixed_precision import get_quant_stats
    return get_quant_stats()


def _reset_quant_stats():
    from backend.operations_mixed_precision import reset_quant_stats
    reset_quant_stats()


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class Qwen3VLTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.max_length = 99999999
        self.min_length = 1
        self.id_pad = 151643
        self.id_template = 151644
        self.id_image = 151655
        self.last_image_slots: list[int] = []
        self.system_prompt = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        self.qwen21_template = self.system_prompt + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.image_template = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

    def tokenize(self, texts, images=[]):
        if dynamic_args.qwen21:
            if images:
                # Qwen-Image-2.1 keeps the system turn and puts an <imageN> label before each vision block
                refs = " ".join("<image{}>{}".format(i + 1, self.vision_block) for i in range(len(images)))
                image_template = self.qwen21_template.replace("{}", refs + "{}", 1)
            else:
                image_template = self.qwen21_template
        elif images:
            image_template = self.image_template.replace(self.vision_block, self.vision_block * len(images), 1)
        else:
            image_template = self.llama_template
        llama_texts = [(image_template.format(text)) if text else " " for text in texts]
        return self.tokenizer(llama_texts)["input_ids"]

    def tokenize_line(self, line: str, images=[]):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        tokenized = self.tokenize([text for text, _ in parsed], images)

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            embed_count = 0
            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.id_image:
                    token = {"type": "image", "data": images[embed_count], "original_type": "image"}
                    embed_count += 1

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts, images=None):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs = []
        cache = {}
        slots = []
        
        # Handle None images parameter
        if images is None:
            images = []

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks = self.tokenize_line(line, images)
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    z, info = self.process_tokens([tokens], [multipliers])
                    z, chunk_slots = self.strip_template(z, tokens, info)
                    if chunk_slots:
                        slots = chunk_slots
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        self.last_image_slots = slots
        return zs

    def strip_template(self, out, tokens, info=[]):
        template_end = 0
        count_im_start = 0

        for i, v in enumerate(tokens):
            try:
                elem = int(v)
                if elem == self.id_template and count_im_start < 2:
                    template_end = i
                    count_im_start += 1
            except TypeError:
                continue

        # Qwen-Image-2.1 keeps the opening user turn, mirroring the reference implementation
        if out.shape[2] > (template_end + 3) and not dynamic_args.qwen21:
            if int(tokens[template_end + 1]) == 872:
                if int(tokens[template_end + 2]) == 198:
                    template_end += 3

        # vision tokens are replaced by reference latents in the DiT: record where each image sits
        # in the stripped context, then drop the spans (qwen21 only; other models keep the legacy slice)
        slots = []
        if dynamic_args.qwen21:
            # `template_end` is an index in the original token list, while
            # process_embeds() has already expanded every vision placeholder
            # into thousands of embeddings.  ComfyUI adjusts the second
            # <|im_start|> by the expansion of all preceding image blocks.
            # Without this correction single-image edits can appear to work,
            # but multi-image conditioning is cut at the wrong position.
            expanded_template_end = template_end
            image_positions = [
                index for index, token in enumerate(tokens)
                if not isinstance(token, int) and isinstance(token, dict) and token.get("type") == "image"
            ]
            for image_index, image_position in enumerate(image_positions):
                if image_position < template_end and image_index < len(info):
                    expanded_template_end += int(info[image_index].get("size", 1)) - 1

            keep = torch.ones(out.shape[2], dtype=torch.bool)
            keep[:expanded_template_end] = False
            for e in info:
                if e.get("type") != "image":
                    continue
                start = e["index"]
                keep[start:start + e["size"]] = False
                slots.append(int(keep[:start].sum()))
            if slots:
                out = out[:, :, keep.to(out.device)]
            else:
                out = out[:, :, template_end:]
        else:
            out = out[:, :, template_end:]

        b, n, seq, h = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)
        # webui stacks per-prompt conds into the batch dim (torch.stack), so return 2D (seq, dim)
        out = out.squeeze(0)

        return out, slots

    def process_embeds(self, batch_tokens):
        # Qwen Image 2.1's 8B joint text/vision encoder must execute on the
        # CUDA device.  The generic text_encoder_device() may intentionally
        # return CPU under Forge low-VRAM policy, which makes the whole
        # Qwen3-VL forward take minutes even though the model is loaded.
        device = memory_management.get_torch_device() if dynamic_args.qwen21 else memory_management.text_encoder_device()

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0

            for t in tokens:
                try:
                    token = int(t)
                    attention_mask.append(0 if eos else 1)
                    tokens_temp += [token]
                    if not eos and token == self.id_pad:
                        eos = True
                except TypeError:
                    other_embeds.append((index, t))
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            index = 0
            embeds_info = []

            for o in other_embeds:
                # ComfyUI keeps image/vision preprocessing under the same
                # quantized-matmul scope as the language model.  Running the
                # Qwen3-VL vision tower outside this scope makes every visual
                # Linear fall back to materialized BF16/FP32 weights.
                with _quantized_matmul_context(self.text_encoder):
                    emb, extra = self.text_encoder.preprocess_embed(o[1], device=device)
                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]

                assert emb.shape[-1] == tokens_embed.shape[-1]
                tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                index += emb_shape - 1
                emb_type = o[1].get("type", None)
                embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def process_tokens(self, batch_tokens, batch_multipliers):
        _reset_quant_stats()
        embeds, mask, count, info = self.process_embeds(batch_tokens)

        # Expand multipliers to match expanded embedding size (with image tokens inserted)
        # For each batch item, insert 1.0 multipliers at image token positions
        expanded_multipliers = []
        for batch_idx, tokens in enumerate(batch_tokens):
            mults = batch_multipliers[batch_idx]
            expanded = []
            offset = 0
            for i, t in enumerate(tokens):
                try:
                    int(t)
                    # Regular token - use original multiplier
                    expanded.append(mults[i])
                except TypeError:
                    # Image token - find its expanded size from info
                    img_info = next((x for x in info if x.get("index") is not None and any(
                        isinstance(tok, dict) and tok.get("type") == "image"
                        for tok in tokens[:i+1]
                        if not isinstance(tok, int)
                    )), None)
                    # Use 1.0 for image tokens (no emphasis)
                    # Find the corresponding info entry
                    img_idx = sum(1 for tok in tokens[:i] if not isinstance(tok, int) and isinstance(tok, dict))
                    if img_idx < len(info):
                        emb_size = info[img_idx]["size"]
                        expanded.extend([1.0] * emb_size)
                    else:
                        expanded.append(1.0)
            expanded_multipliers.append(expanded)

        # Flatten all expanded multipliers into a single tensor
        flat_multipliers = []
        for m in expanded_multipliers:
            flat_multipliers.extend(m)
        self.emphasis.multipliers = torch.tensor(flat_multipliers, device=embeds.device, dtype=embeds.dtype)
        self.emphasis.tokens = batch_tokens
        self.emphasis.z = embeds
        self.emphasis.after_transformers()
        embeds = self.emphasis.z

        if dynamic_args.qwen21:
            # Qwen-Image-2.1 reads the text encoder from its final hidden state (single 4096-dim layer),
            # which is the first return value. Unsqueeze to 4D to match strip_template's (b, n, seq, h) contract.
            with _quantized_matmul_context(self.text_encoder):
                z, _ = self.text_encoder(None, embeds=embeds, attention_mask=mask, num_tokens=count, embeds_info=info, intermediate_output=None)
            z = z.unsqueeze(1)
        else:
            with _quantized_matmul_context(self.text_encoder):
                _, z = self.text_encoder(None, embeds=embeds, attention_mask=mask, num_tokens=count, embeds_info=info, intermediate_output=KREA2_TAP_LAYERS, final_layer_norm_intermediate=False)
        print(
            f"[Qwen21] TE diagnostic: device={embeds.device}, tokens={embeds.shape[1]}, "
            f"convrot_linear={_quant_stats()['convrot']}, linear_fallback={_quant_stats()['fallback']}",
            flush=True,
        )
        return z, info
