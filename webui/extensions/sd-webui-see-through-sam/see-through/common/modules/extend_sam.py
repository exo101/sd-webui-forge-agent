# Extended SAMs are structured following https://github.com/ziqi-jin/finetune-anything
# but are re-writing for flexibility

from typing import List, Tuple

from torch import nn
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_functional

from utils.torch_utils import fix_params, img2tensor, tensor2img
from .sam.build_sam import sam_model_registry, Sam
from .sam.modeling.mask_decoder import MaskDecoder
from .sam.modeling.prompt_encoder import PromptEncoder
from .sam.modeling.image_encoder import ImageEncoderViT
from .sam.utils.transforms import resize_longside_torch

def pair_params(self, target_model: nn.Module):
    src_dict = self.sam_mask_decoder.state_dict()
    for name, value in target_model.named_parameters():
        if name in src_dict.keys():
            value.data.copy_(src_dict[name].data)


class BaseImgEncodeAdapter(nn.Module):

    def __init__(self, sam_img_encoder: ImageEncoderViT, fix=False):
        super(BaseImgEncodeAdapter, self).__init__()
        self.sam_img_encoder = sam_img_encoder
        if fix:
            fix_params(self.sam_img_encoder)

    def forward(self, *args, **kwargs):
        return self.sam_img_encoder(*args, **kwargs)


class BaseMaskDecoderAdapter(nn.Module):
    '''
      multimask_output (bool): If true, the model will return three masks.
    For ambiguous input prompts (such as a single click), this will often
    produce better masks than a single prediction. If only a single
    mask is needed, the model's predicted quality score can be used
    to select the best mask. For non-ambiguous prompts, such as multiple
    input prompts, multimask_output=False can give better results.
    '''

    _hidden_param_keywords = ['transformer']
    # _hidden_param_exclude_keywords = ['transformer', 'hf_mlp', 'output_hypernetworks_mlps']

    # is fix and load params
    def __init__(self, sam_mask_decoder: MaskDecoder, fix=False):
        super(BaseMaskDecoderAdapter, self).__init__()
        # mask_decoder = ori_sam.mask_decoder
        self.sam_mask_decoder: MaskDecoder = sam_mask_decoder
        if fix:
            fix_params(self.sam_mask_decoder)  # move to runner to implement

    def forward(self, *args, **kwargs):
        return self.sam_mask_decoder(*args, **kwargs)

    def get_muon_training_params(self):
        hidden_weights, nonhidden_params = [], []
        for pname, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_hidden_weights = False
            for hidden_param_name in self._hidden_param_keywords:
                if hidden_param_name in pname:
                    is_hidden_weights = True
                    break
            if is_hidden_weights and p.ndim >= 2:
                hidden_weights.append(p)
            else:
                nonhidden_params.append(p)
        return hidden_weights, nonhidden_params


class BasePromptEncodeAdapter(nn.Module):

    def __init__(self, sam_prompt_encoder: PromptEncoder, fix=False):
        super(BasePromptEncodeAdapter, self).__init__()
        self.sam_prompt_encoder = sam_prompt_encoder
        if fix:
            fix_params(self.sam_prompt_encoder)

    def forward(self, *args, **kwargs):
        return self.sam_prompt_encoder(*args, **kwargs)



class BaseExtendSam(nn.Module):

    def __init__(self, 
                 sam: Sam, 
                 fix_img_en=False, 
                 fix_prompt_en=False, 
                 fix_mask_de=False):
        super(BaseExtendSam, self).__init__()
        # self.ori_sam: Sam = sam
        self.img_adapter = BaseImgEncodeAdapter(sam.image_encoder, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(sam.prompt_encoder, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(sam.mask_decoder, fix=fix_mask_de)
        del sam.mask_decoder
        del sam.image_encoder
        del sam.prompt_encoder

    @property
    def img_size(self):
        return self.img_adapter.sam_img_encoder.img_size

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def inference(self, batch_imgs, normalize=True, output_dtype='tensor'):
        
        if isinstance(batch_imgs, (Image.Image, np.ndarray)):
            batch_imgs = [batch_imgs]

        preprocess_lst = []
        _batch_imgs = []
        device = self.device
        dtype = self.dtype
        for x in batch_imgs:
            if isinstance(x, (Image.Image, np.ndarray)):
                x = img2tensor(x)
            ori_sz = x.shape[-2:]
            x = resize_longside_torch(x, target_length=self.img_size)
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x1 = padw // 2
            y1 = padh // 2
            preprocess_lst.append(((y1, x1, ori_sz[0], ori_sz[1]), (h, w)))
            if padh > 0 or padw > 0:
                x = F.pad(x, (x1, padw - x1, y1, padh - y1))
            _batch_imgs.append(x)

        _batch_imgs = torch.cat(_batch_imgs).to(device=self.device, dtype=self.dtype)
        batch_imgs = _batch_imgs
        if normalize:
            batch_imgs = tv_functional.normalize(batch_imgs, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        rst, _ = self(batch_imgs)
        rst_imgs = []
        for ii, pred in enumerate(rst):
            pred = F.interpolate(
                pred[None],
                (self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            ori_sz, input_size = preprocess_lst[ii]
            pred = pred[..., ori_sz[0]: ori_sz[0] + input_size[0], ori_sz[1]: ori_sz[1] + input_size[1]]
            pred = F.interpolate(pred, (ori_sz[2], ori_sz[3]), mode="bilinear", align_corners=False)[0]
            if output_dtype == 'numpy':
                pred = pred.to(device='cpu', dtype=torch.float32).numpy()
            rst_imgs.append(pred)
        return rst_imgs

    def forward(
            self, 
            img,
            hq_token_only=False,
            multimask_output=True
        ):
        image_embeddings, interm_embeddings = self.img_adapter(img, get_interm_embeds=self.mask_adapter.is_hq)
        points = None
        boxes = None
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        
        image_pe = self.prompt_adapter.sam_prompt_encoder.get_dense_pe()
        
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
        )

        return low_res_masks, iou_predictions
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    

    @property
    def device(self):
        return next(self.parameters()).device


    def get_muon_training_params(self):
        def _get_adapter_muon_params(adapter):
            if hasattr(adapter, 'get_muon_training_params'):
                return adapter.get_muon_training_params()
            else:
                hidden_weights = [p for p in adapter.parameters() if p.ndim >= 2 and p.requires_grad]
                hidden_gains_biases = [p for p in adapter.parameters() if p.ndim < 2 and p.requires_grad]
                return hidden_weights, hidden_gains_biases

        hidden_weights = []
        nonhidden_params = []
        for module_name in ['img_adapter', 'prompt_adapter', 'mask_adapter']:
            h, n = _get_adapter_muon_params(getattr(self, module_name))
            hidden_weights += h
            nonhidden_params += n

        return hidden_weights, nonhidden_params

        # hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        # hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        # nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]