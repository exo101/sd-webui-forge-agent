import torch
import torch.nn.functional as F


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU between predicted and target masks."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    inter = (pred_bin * target).sum(dim=(2, 3))
    union = (pred_bin + target).sum(dim=(2, 3)) - inter
    iou = (inter + 1e-7) / (union + 1e-7)
    return iou.mean()


def compute_mae(pred, target):
    """Compute Mean Absolute Error."""
    pred_prob = torch.sigmoid(pred)
    return F.l1_loss(pred_prob, target)


def compute_max_f1(pred, target, thresholds=None):
    """Compute maximum F1 score across thresholds."""
    if thresholds is None:
        thresholds = torch.linspace(0, 1, 101)
    pred_prob = torch.sigmoid(pred)
    best_f1 = 0.0
    for t in thresholds:
        pred_bin = (pred_prob > t).float()
        tp = (pred_bin * target).sum().float()
        fp = (pred_bin * (1 - target)).sum().float()
        fn = ((1 - pred_bin) * target).sum().float()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        best_f1 = max(best_f1, f1)
    return best_f1