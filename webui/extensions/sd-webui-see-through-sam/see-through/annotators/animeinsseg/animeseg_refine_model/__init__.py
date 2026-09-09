# modified from https://github.com/SkyTNT/anime-segmentation/blob/main/train.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from torch.cuda import amp

try:
    import pytorch_lightning as pl
    AnimeSegmentationCLS = pl.LightningModule
except:
    AnimeSegmentationCLS = torch.nn.Module

from .isnet import ISNetDIS, ISNetGTEncoder
from utils.torch_utils import init_model_from_pretrained


def load_refinenet(refine_method = 'refinenet_isnet', device: str = 'cuda'):
    if refine_method == 'animeseg':
        def _patch_statedict(model, state_dict):
            klist = list(state_dict.keys())
            new_sd = {}
            for k in klist:
                if k.startswith('net.'):
                    new_sd[k.replace('net.', '')] = state_dict[k]
            return new_sd
        model: ISNetDIS = init_model_from_pretrained('skytnt/anime-seg', ISNetDIS, patch_state_dict_func=_patch_statedict, weights_name='isnetis.ckpt', )
        # model = AnimeSegmentation.try_load('isnet_is', model_path, device)
    elif refine_method == 'refinenet_isnet':
        model: ISNetDIS = init_model_from_pretrained('dreMaz/AnimeInstanceSegmentation', ISNetDIS, 
                                                     model_args={'in_ch': 4}, weights_name='refine_last.ckpt')
    else:
        raise NotImplementedError
    return model.eval().to(device)


def get_mask(model, input_img, use_amp=True, s=640):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
        return pred