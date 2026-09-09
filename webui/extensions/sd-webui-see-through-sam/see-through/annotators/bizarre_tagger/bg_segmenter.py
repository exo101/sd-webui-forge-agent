import torchvision
import torch.nn as nn
import torchvision.transforms as TT
import torch


class CharacterBGSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        # setup deeplab
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(weights_backbone=None)
        self.deeplab.aux_classifier = None
        # for param in self.deeplab.backbone.parameters():
        #     param.requires_grad = False
        self.deeplab.classifier = nn.Sequential(
            # tv.models.segmentation.deeplabv3.DeepLabHead(2048, 2)[0],
            torchvision.models.segmentation.deeplabv3.ASPP(2048, [12, 24, 36]),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )
        self.final_head = nn.Sequential(
            nn.Conv2d(16+3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 2, kernel_size=1, stride=1),
        )
        self.deeplab_preprocess = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, rgb, return_more=True):
        # preprocess
        normed = self.deeplab_preprocess(rgb)
        stackin = normed

        # forward pass
        out_dl = self.deeplab(normed)['out']
        out_fin = self.final_head(torch.cat([
            out_dl, stackin,
        ], dim=1))
        out = {'raw': out_fin}
        if return_more:
            out['softmax'] = torch.softmax(out_fin, dim=1)
            out['max'] = torch.max(out_fin, dim=1).indices
        return out
