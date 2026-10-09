from functools import partial

import torch

from .modeling import ImageEncoderViT, PromptEncoder
from .modeling.tiny_vit_sam import TinyViT

model_type_registry = dict(
    vit_l = dict(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            global_attn_indexes=[5, 11, 17, 23]
    ),

    vit_h = dict(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
    ),

    vit_b = dict(
            embed_dim=768,
            depth=12,
            num_heads=12,
            global_attn_indexes=[2, 5, 8, 11],),
)

def build_image_encoder(model_type: str):

    if model_type == 'vit_t':
        image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )
    else:
        assert model_type in model_type_registry
        image_encoder = ImageEncoderViT(
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            window_size=14,
            out_chans=256,
            **model_type_registry[model_type]
        )

    return image_encoder

def build_prompt_encoder(image_size = 1024, vit_patch_size = 16):
    image_embedding_size = image_size // vit_patch_size
    prompt_encoder=PromptEncoder(
        embed_dim=256,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    return prompt_encoder

    