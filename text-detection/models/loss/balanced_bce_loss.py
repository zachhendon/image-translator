import torch.nn as nn
import torch.nn.functional as F


class BalancedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, masks):
        pred = pred * masks
        gt = gt * masks

        loss = F.binary_cross_entropy(pred, gt)
        return loss
