from typing import Union
from collections import OrderedDict

import torch
import torchvision
import numpy as np
from einops import rearrange
import kornia
import torch.nn as nn
import torchvision.transforms as TT


import json
def jread(fn, mode='r'):
    with open(fn, mode) as handle:
        return json.load(handle)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # read rulebook
        # see ./hack/snips/danbooru_intently_combatively.py for preprocessing

        # self.fn_rules = f'E:/gitclones/wdv3-timm/bizarre-pose-estimator/_data/danbooru/_filters/intently_combatively_rules.json'
        # self.rules = jread(self.fn_rules)

        # setup resnet
        self.resnet = torchvision.models.resnet50()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1062)
        self.resnet_preprocess = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # loss
        return


    def forward(self, rgb, return_more=True):
        normed = self.resnet_preprocess(rgb)
        out_raw = self.resnet(normed)
        out = {'raw': out_raw}
        if return_more:
            out['prob'] = torch.sigmoid(out_raw)
            out['pred'] = out_raw>0
        return out



