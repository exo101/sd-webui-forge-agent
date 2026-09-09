from typing import List, Tuple

from torch import nn
import torch
from einops import rearrange

from .sam.modeling import Sam
from .sam.modeling.mask_decoder import MLP
from .sam import sam_model_registry
from .extend_sam import BaseExtendSam, BaseMaskDecoderAdapter, MaskDecoder


class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):

    def __init__(self, sam_mask_decoder: MaskDecoder, fix=False, class_num=20, init_from_sam=True):
        super(SemMaskDecoderAdapter, self).__init__(sam_mask_decoder, fix)

        self.class_num = class_num
        self.is_hq = self.sam_mask_decoder.is_hq
        self.num_mask_tokens = self.sam_mask_decoder.num_mask_tokens
        
        transformer_dim = self.sam_mask_decoder.transformer_dim

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.class_num)
            ]
        )
        if init_from_sam:
            target_sd = self.sam_mask_decoder.output_hypernetworks_mlps[1].state_dict()
            for ii in range(class_num):
                self.output_hypernetworks_mlps[ii].load_state_dict(target_sd)
        del self.sam_mask_decoder.output_hypernetworks_mlps

        if self.is_hq:
            self.hf_mlps = nn.ModuleList(
                [
                    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for _ in range(self.class_num)
                ]
            )
            if init_from_sam:
                target_sd = self.sam_mask_decoder.hf_mlp.state_dict()
                for ii in range(class_num):
                    self.hf_mlps[ii].load_state_dict(target_sd)
            del self.sam_mask_decoder.hf_mlp
            # input cond tokens: cat[1 x iou tokens, 4 x original mask tokens, 1 x hf token]
            # num_mask_tokens: 4 + 1
            self.hf_token_idx = self.num_mask_tokens

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        hq_token_only: bool = False,
        interm_embeddings: torch.Tensor = None,
        mask_scale=1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        hq_features = None

        # token processing
        if self.is_hq:
            vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
            hq_features = self.sam_mask_decoder.embedding_encoder(image_embeddings) + self.sam_mask_decoder.compress_vit_feat(vit_features)
            output_tokens = [self.sam_mask_decoder.iou_token.weight, self.sam_mask_decoder.mask_tokens.weight, self.sam_mask_decoder.hf_token.weight]
        else:
            output_tokens = [self.sam_mask_decoder.iou_token.weight, self.sam_mask_decoder.mask_tokens.weight]
        output_tokens = torch.cat(output_tokens, dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # tokens: (batch size, model preserved tokens (iou*1, mask*4, hf token * 1) + user prompts, token dim)
        # Expand per-image data in batch direction to be per-mask. multiple user prompts for the same image are divide along batch channel 
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        src = src.to(dtype=pos_src.dtype)
        tokens = tokens.to(dtype=pos_src.dtype)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.sam_mask_decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Decode tokens, mask tokens -> iou preds, src tokens (input image tokens) to masks
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.sam_mask_decoder.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        hyper_hq_list: List[torch.Tensor] = []
        for i in range(self.class_num):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, mask_scale, :]))
            if self.is_hq:
                hyper_hq_list.append(self.hf_mlps[i](hs[:, self.hf_token_idx, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.sam_mask_decoder.iou_prediction_head(iou_token_out)

        if self.is_hq:
            hyper_hq = torch.stack(hyper_hq_list, dim=1)
            upscaled_embedding_hq = self.sam_mask_decoder.embedding_maskfeature(upscaled_embedding_sam) + hq_features
            masks_hq = (hyper_hq @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
            if hq_token_only:
                masks = masks_hq
            else:
                masks = masks_sam + masks_hq

        else:
            masks = masks_sam

        # Generate mask quality predictions
        # Prepare output
        return masks, iou_pred


class SemMaskDecoderAdapterTokenVariant(BaseMaskDecoderAdapter):

    def __init__(self, sam_mask_decoder: MaskDecoder, fix=False, class_num=20, init_from_sam=True):
        super(SemMaskDecoderAdapterTokenVariant, self).__init__(sam_mask_decoder, fix)

        self.class_num = class_num
        self.is_hq = self.sam_mask_decoder.is_hq
        # self.num_mask_tokens = self.sam_mask_decoder.num_mask_tokens
        
        
        
        transformer_dim = self.sam_mask_decoder.transformer_dim
        self.sem_mask_tokens = nn.Embedding(class_num, transformer_dim)

        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        if init_from_sam:
            target_sd = self.sam_mask_decoder.output_hypernetworks_mlps[1].state_dict()
            self.output_hypernetworks_mlp.load_state_dict(target_sd)
            target_sd = self.sam_mask_decoder.mask_tokens.state_dict()
            target_sd = {'weight': target_sd['weight'][[1]].repeat(class_num, 1)}
            self.sem_mask_tokens.load_state_dict(target_sd)
            pass

        del self.sam_mask_decoder.mask_tokens
        del self.sam_mask_decoder.output_hypernetworks_mlps

        if self.is_hq:
            self.hq_mask_tokens = nn.Embedding(class_num, transformer_dim)
            self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            if init_from_sam:
                target_sd = self.sam_mask_decoder.hf_mlp.state_dict()
                self.hf_mlp.load_state_dict(target_sd)
                target_sd = self.sam_mask_decoder.hf_token.state_dict()
                target_sd = {'weight': target_sd['weight'].repeat(class_num, 1)}
                self.hq_mask_tokens.load_state_dict(target_sd)
            del self.sam_mask_decoder.hf_mlp
            del self.sam_mask_decoder.hf_token
            # input cond tokens: cat[1 x iou tokens, 4 x original mask tokens, 1 x hf token]
            # num_mask_tokens: 4 + 1
            # self.hf_token_idx = self.num_mask_tokens

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        hq_token_only: bool = False,
        interm_embeddings: torch.Tensor = None,
        mask_scale=1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        hq_features = None

        # token processing
        if self.is_hq:
            vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
            hq_features = self.sam_mask_decoder.embedding_encoder(image_embeddings) + self.sam_mask_decoder.compress_vit_feat(vit_features)
            output_tokens = [self.sam_mask_decoder.iou_token.weight, self.sem_mask_tokens.weight, self.hq_mask_tokens.weight]
        else:
            output_tokens = [self.sam_mask_decoder.iou_token.weight, self.sem_mask_tokens.weight]
        output_tokens = torch.cat(output_tokens, dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # tokens: (batch size, model preserved tokens (iou*1, mask*4, hf token * 1) + user prompts, token dim)
        # Expand per-image data in batch direction to be per-mask. multiple user prompts for the same image are divide along batch channel 
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        src = src.to(dtype=pos_src.dtype)
        tokens = tokens.to(dtype=pos_src.dtype)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.sam_mask_decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.class_num), :]

        # Decode tokens, mask tokens -> iou preds, src tokens (input image tokens) to masks
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.sam_mask_decoder.output_upscaling(src)

        hyper_in = self.output_hypernetworks_mlp(rearrange(mask_tokens_out, 'b c d -> (b c) d'))
        hyper_in = rearrange(hyper_in, '(b c) d -> b c d', b=b)
        # for i in range(self.class_num):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, mask_scale, :]))
        if self.is_hq:
            # hyper_hq_list.append(self.hf_mlps[i](hs[:, self.hf_token_idx, :]))
            hyper_hq = self.hf_mlp(rearrange(hs[:, 1 + self.class_num: (1 + 2 * self.class_num), :], 'b c d -> (b c) d'))
            hyper_hq = rearrange(hyper_hq, '(b c) d -> b c d', b=b)

        # hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.sam_mask_decoder.iou_prediction_head(iou_token_out)

        if self.is_hq:
            # hyper_hq = torch.stack(hyper_hq_list, dim=1)
            upscaled_embedding_hq = self.sam_mask_decoder.embedding_maskfeature(upscaled_embedding_sam) + hq_features
            masks_hq = (hyper_hq @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
            if hq_token_only:
                masks = masks_hq
            else:
                masks = masks_sam + masks_hq

        else:
            masks = masks_sam

        # Generate mask quality predictions
        # Prepare output
        return masks, iou_pred


class SemanticSam(BaseExtendSam):

    def __init__(self,
                 class_num,
                 sam: Sam = None, 
                 fix_img_en=False, 
                 fix_prompt_en=False, 
                 fix_mask_de=False,
                 model_type: str = 'h_hq',
                 mask_decoder='mlp_variant',
                 **kwargs):
        
        init_from_sam = sam is not None
        if sam is None:
            build_sam = sam_model_registry[model_type]['build']
            sam = build_sam()
        
        super().__init__(sam=sam, fix_img_en=fix_img_en, fix_mask_de=fix_mask_de, fix_prompt_en=fix_prompt_en)
        sam_mask_decoder = self.mask_adapter.sam_mask_decoder
        del self.mask_adapter
        if mask_decoder == 'mlp_variant':
            self.mask_adapter = SemMaskDecoderAdapter(sam_mask_decoder=sam_mask_decoder, fix=fix_mask_de, class_num=class_num, init_from_sam=init_from_sam)
        elif mask_decoder == 'token_variant':
            self.mask_adapter = SemMaskDecoderAdapterTokenVariant(sam_mask_decoder=sam_mask_decoder, fix=fix_mask_de, class_num=class_num, init_from_sam=init_from_sam)
        else:
            raise Exception(f'Invalid mask decoder: {mask_decoder}')