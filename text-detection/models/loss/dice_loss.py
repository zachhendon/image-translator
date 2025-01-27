import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        gt = torch.clamp(gt, 0, 1)

        pred = pred * mask

        a = torch.sum(pred * gt, dim=(1, 2))
        b = torch.sum(pred * pred, dim=(1, 2))
        c = torch.sum(gt * gt, dim=(1, 2))
        loss = 1 - (2 * a) / (b + c + 1e-6)
        return loss.mean()
