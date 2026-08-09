from typing import Union
from collections import OrderedDict

import torch
import torchvision
import numpy as np
import kornia
import torch.nn as nn
import torchvision.transforms as TT
import detectron2

from .tagger import Model as Classifier


class ResBlock(nn.Module):
    def __init__(self,
            depth, channels, kernel,
            channels_in=None,  # in case different from channels
            activation=nn.ReLU,
            normalization=nn.BatchNorm2d,
                ):
        # activation()
        # normalization(channels)
        super(ResBlock, self).__init__()
        self.depth = depth
        self.channels = channels
        self.channels_in = channels_in
        self.kernel = kernel
        self.activation = activation
        self.normalization = normalization

        # create sequential network
        od = OrderedDict()
        for i in range(depth):
            chin = channels_in \
                if channels_in is not None and i==0 \
                else channels
            od[f'conv{i}'] = nn.Conv2d(
                chin, channels,
                kernel_size=kernel, padding=kernel//2,
                bias=True, padding_mode='replicate',
            )
            if activation is not None:
                od[f'act{i}'] = activation()
            if normalization is not None:
                od[f'norm{i}'] = normalization(channels)
        self.net = nn.Sequential(od)

        # last activation/normalization
        od_tail = OrderedDict()
        if activation is not None:
            od_tail[f'act{depth}'] = activation()
        if normalization is not None:
            od_tail[f'norm{depth}'] = normalization(channels)
        self.net_tail = nn.Sequential(od_tail)
        return

    def forward(self, x):
        if self.channels_in is None:
            return self.net_tail(x + self.net(x))
        else:
            head = self.net[0](x)
            t = head
            for body in self.net[1:]:
                t = body(t)
            return self.net_tail(head + t)


class ResnetFeatureConverter(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.resizer = TT.Resize(self.size, antialias=False)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.layer1 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.layer2 = nn.Conv2d(512, 64, kernel_size=1, padding=0)
        self.layer3 = nn.Conv2d(1024, 64, kernel_size=1, padding=0)
        self.head = nn.Conv2d(64*4, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm2d(64*4)

    def forward(self, feats_resnet):
        return self.head(self.relu(self.batchnorm(torch.cat([
            self.resizer(self.conv1(feats_resnet['conv1'])),
            self.resizer(self.layer1(feats_resnet['layer1'])),
            self.resizer(self.layer2(feats_resnet['layer2'])),
            self.resizer(self.layer3(feats_resnet['layer3'])),
            # self.resizer(self.layer4(feats_resnet['layer4'])),
        ], dim=1))))


class ResnetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        base = Classifier()

        self.resize = TT.Resize(256, antialias=False)
        self.resnet_preprocess = base.resnet_preprocess
        self.conv1 = base.resnet.conv1
        self.bn1 = base.resnet.bn1
        self.relu = base.resnet.relu      #   64ch, 128p (assuming 256p input)
        self.maxpool = base.resnet.maxpool
        self.layer1 = base.resnet.layer1  #  256ch,  64p
        self.layer2 = base.resnet.layer2  #  512ch,  32p
        self.layer3 = base.resnet.layer3  # 1024ch,  16p


    def forward(self, x):
        ans = {}
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ans['conv1'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        ans['layer1'] = x
        x = self.layer2(x)
        ans['layer2'] = x
        x = self.layer3(x)
        ans['layer3'] = x
        # x = self.layer4(x)
        # ans['layer4'] = x
        return ans

from detectron2 import model_zoo as _
from detectron2 import engine as _
from detectron2 import config as _
from detectron2.modeling import META_ARCH_REGISTRY

def detectron2_build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    return model

class PretrainedKeypointDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # setup rcnn model
        self.cfg = detectron2.config.get_cfg()
        self.cfg.merge_from_file(detectron2.model_zoo.get_config_file(
            'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
        ))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        # self.cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(
        #     'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
        # )
        self.cfg['MODEL']['DEVICE'] = 'cpu'
        self.model = detectron2_build_model(self.cfg)

        # preprocessing
        self.size = 800
        self.resize = TT.Resize(self.size, antialias=True)


    def forward(self, img, return_more=False, return_match_feats=False):
        # assumes img.shape = (bs, rgb, h, w)
        h,w = img.shape[2:]
        x = [{'image': i, 'height': h, 'width': w} for i in 255*self.resize(img).flip(1)]
        images = self.model.preprocess_image(x)
        features = self.model.backbone(images.tensor)

        # forces them to use my bboxes
        h,w = images[0].shape[1:]
        detected_instances = [
            detectron2.structures.instances.Instances(
                image_size=(h,w),
                pred_boxes=detectron2.structures.boxes.Boxes(torch.tensor([
                    0, 0, h, w,
                ], device=images.device)[None]),
                pred_classes=torch.tensor([0,], device=images.device),
            )
            for _ in range(img.shape[0])
        ]

        if return_match_feats:
            roi = self.model.roi_heads
            _instances = detected_instances
            if roi.keypoint_pooler is not None:
                _features = [features[f] for f in roi.keypoint_in_features]
                #boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
                _boxes = [x.pred_boxes for x in _instances]
                _features = roi.keypoint_pooler(_features, _boxes)
            else:
                _features = {f: features[f] for f in roi.keypoint_in_features}

            # roi_heads.keypoint_head.forward, in BaseKeypointRCNNHead
            # pred_keypoint_logits = roi.keypoint_head.layers(_features)
            # return {'keypoint_heatmaps': pred_keypoint_logits, 'locals': locals()}

            # get last feature layer
            fl = _features
            for i in range(len(roi.keypoint_head)-1):
                fl = roi.keypoint_head[i](fl)
            feats_last = fl  # 256p input -> 512ch, 14p
            return {'features_last': feats_last}

        else:
            results = self.model.roi_heads.forward_with_given_boxes(features, detected_instances)
            results = self.model._postprocess(results, x, images.image_sizes)
            hms = torch.cat([r['instances'].pred_keypoint_heatmaps for r in results])
            ans = {'keypoint_heatmaps': hms}
            if return_more:
                ans['results'] = results
                ans['images'] = images
                ans['features'] = features
                ans['detected_instances'] = detected_instances
                ans['bboxes'] = [
                    ((b,a), (d-b,c-a))
                    for r in results
                    for a,b,c,d in r['instances'].pred_boxes.tensor.detach().cpu().numpy()
                ]
                ans['keypoints'] = torch.stack([
                    k[:,:2].flip(1)
                    for r in results
                    for k in r['instances'].pred_keypoints
                ])
            return ans


class ModelFeatConcat(nn.Module):
    def __init__(self, size=128):

        super().__init__()
        self.size = size

        # set up frozen pretrained networks
        self.resnet = ResnetFeatureExtractor()
        self.rcnn = PretrainedKeypointDetector()

        # set up keypoint detector head
        self.size = size
        self.resizer = TT.Resize(self.size, antialias=False)
        self.keypoint_head = nn.ModuleDict({
            'converter_resnet': ResnetFeatureConverter(self.size),
            'converter_rcnn': nn.Conv2d(17, 32, kernel_size=1, padding=0),
            'head': nn.Sequential(
                nn.Conv2d(128+32+3, 128, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                ResBlock(3, 64, 3, channels_in=128),
                ResBlock(3, 32, 3, channels_in=64),
                nn.Conv2d(32, 17+8, kernel_size=3, padding=1, padding_mode='replicate'),
            ),
        })

    def _forward_model(self, x, return_more=False):
        feats_resnet = self.resnet(x)
        feats_rcnn = self.rcnn(x, return_more=return_more)

        feats = torch.cat([
            self.resizer(x),
            self.resizer(self.keypoint_head['converter_resnet'](feats_resnet)),
            self.resizer(self.keypoint_head['converter_rcnn'](feats_rcnn['keypoint_heatmaps'])),
        ], dim=1)
        hms = self.keypoint_head['head'](feats)
        return hms

    def forward(self, x, smoothing=None, return_more=False, heat_map_aug=False):
        # feats_resnet = self.resnet(x)
        # feats_rcnn = self.rcnn(x, return_more=return_more)

        # feats = torch.cat([
        #     self.resizer(x),
        #     self.resizer(self.keypoint_head['converter_resnet'](feats_resnet)),
        #     self.resizer(self.keypoint_head['converter_rcnn'](feats_rcnn['keypoint_heatmaps'])),
        # ], dim=1)
        # hms = self.keypoint_head['head'](feats)
        hms = self._forward_model(x)
        if heat_map_aug:
            hflip = torchvision.transforms.functional.hflip
            hms_flip = self._forward_model(hflip(x))
            lst = np.arange(1, 17).reshape((-1, 2))
            for l in lst:
                hms[:, l] = (hms[:, l] + hflip(hms_flip[:, [l[1], l[0]]])) / 2

            # hms = (hms + hflip(hms_flip)) / 2
        ans = {'keypoint_heatmaps': hms}

        if return_more:
            # ans['features_resnet'] = feats_resnet
            # ans['features_rcnn'] = feats_rcnn
            if smoothing is None:
                ans['keypoint_heatmaps_prob'] = torch.sigmoid(hms)
                kps = hms.view(hms.shape[0], hms.shape[1], -1).argmax(-1)
                kps = torch.stack([
                    kps // hms.shape[2] * (x.shape[2]/hms.shape[2]),
                    kps % hms.shape[3] * (x.shape[3]/hms.shape[3]),
                ], dim=2)
            else:
                hmp = torch.sigmoid(hms)
                ksig = max(hmp.shape[-2:]) * smoothing
                kern = max(3, int(ksig)*2+1)
                hmps = kornia.filters.gaussian_blur2d(
                    hmp,
                    kernel_size=(kern,kern),
                    sigma=(ksig,ksig),
                    border_type='reflect',
                )
                hmps_flatten = hmps.view(hmps.shape[0], hmps.shape[1], -1)
                kps = hmps_flatten.argmax(-1)
                scores = [hmps_flatten[0, i, kps[0, i]].item() for i in range(kps.shape[1])]
                kps = torch.stack([
                    kps // hmps.shape[2] * (x.shape[2]/hmps.shape[2]),
                    kps % hmps.shape[3] * (x.shape[3]/hmps.shape[3]),
                ], dim=2)
                ans['keypoint_heatmaps_prob'] = hmps
                ans['keypoint_scores'] = scores
            ans['keypoints'] = kps
        return ans


class ModelFeatMatch(nn.Module):
    def __init__(self, size=128):
        super().__init__()

        # set up frozen pretrained networks
        self.resnet = ResnetFeatureExtractor()
        self.size = size
        self.resizer = TT.Resize(self.size, antialias=False)
        self.keypoint_head = nn.ModuleDict({
            'converter_resnet': ResnetFeatureConverter(self.size),
            'matcher_rcnn': nn.Sequential(
                TT.Resize(14, antialias=False),
                nn.Conv2d(128, 512, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.ReLU(),
                nn.BatchNorm2d(512),
            ),
            'converter_rcnn': nn.Sequential(
                nn.Conv2d(512, 32, kernel_size=1, padding=0),
                self.resizer,
            ),
            'head': nn.Sequential(
                nn.Conv2d(128+32+3, 128, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                ResBlock(3, 64, 3, channels_in=128),
                ResBlock(3, 32, 3, channels_in=64),
                nn.Conv2d(32, 17+8, kernel_size=3, padding=1, padding_mode='replicate'),
            ),
        })

    def forward(self, x, smoothing=None, return_more=False, heat_map_aug=False):
        # extract backbone features
        feats_resnet = self.resnet(x)
        feats_resnet_convert = self.resizer(self.keypoint_head['converter_resnet'](feats_resnet))

        # matching
        feats_match = self.keypoint_head['matcher_rcnn'](feats_resnet_convert)
        feats = torch.cat([
            self.resizer(x),
            feats_resnet_convert,
            self.keypoint_head['converter_rcnn'](feats_match),
        ], dim=1)

        # apply head
        hms = self.keypoint_head['head'](feats)
        ans = {
            'keypoint_heatmaps': hms,
            # 'features_match': (feats_rcnn, feats_match),
        }
        if return_more:
            ans['features_resnet'] = feats_resnet
            # ans['features_rcnn'] = feats_rcnn
            if smoothing is None:
                ans['keypoint_heatmaps_prob'] = torch.sigmoid(hms)
                kps = hms.view(hms.shape[0], hms.shape[1], -1).argmax(-1)
                kps = torch.stack([
                    kps // hms.shape[2] * (x.shape[2]/hms.shape[2]),
                    kps % hms.shape[3] * (x.shape[3]/hms.shape[3]),
                ], dim=2)
            else:
                hmp = torch.sigmoid(hms)
                ksig = max(hmp.shape[-2:]) * smoothing
                kern = max(3, int(ksig)*2+1)
                hmps = kornia.filters.gaussian_blur2d(
                    hmp,
                    kernel_size=(kern,kern),
                    sigma=(ksig,ksig),
                    border_type='reflect',
                )
                kps = hmps.view(hmps.shape[0], hmps.shape[1], -1).argmax(-1)
                kps = torch.stack([
                    kps // hmps.shape[2] * (x.shape[2]/hmps.shape[2]),
                    kps % hmps.shape[3] * (x.shape[3]/hmps.shape[3]),
                ], dim=2)
                ans['keypoint_heatmaps_prob'] = hmps
            ans['keypoints'] = kps
        return ans
