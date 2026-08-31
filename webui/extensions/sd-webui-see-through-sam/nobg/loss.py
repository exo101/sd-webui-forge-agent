import torch
import torch.nn as nn
import torch.nn.functional as F


class BiRefNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IOULoss()

    def forward(self, inputs, targets):
        # inputs: list of multi-scale outputs from the model
        # targets: ground truth masks
        total_loss = 0.0
        for pred in inputs:
            # Resize prediction to match target spatial dimensions
            pred = F.interpolate(pred, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            bce_loss = self.bce(pred, targets)
            iou_loss = self.iou(pred, targets)
            total_loss += bce_loss + iou_loss
        return total_loss / len(inputs)


class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = (inter + 1e-7) / (union + 1e-7)
        return 1 - iou.mean()


def birefnet_loss():
    return BiRefNetLoss()