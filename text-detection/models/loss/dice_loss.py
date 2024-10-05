import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, masks):
        pred = pred * masks
        gt = gt * masks

        batch_size = pred.size(0)

        pred = pred.reshape(batch_size, -1)
        gt = gt.reshape(batch_size, -1)

        a = torch.sum(pred * gt, dim=1)
        b = torch.sum(pred * pred, dim=1)
        c = torch.sum(gt * gt, dim=1)
        d = (2 * a) / (b + c + 1e-6)

        loss = 1 - d
        loss = loss.mean()
        return loss
