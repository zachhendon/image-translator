import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        pred = pred * mask
        gt = torch.clamp(gt, 0, 1)

        loss = F.binary_cross_entropy(pred, gt, reduction="none")
        balanced_loss = torch.sum(loss) / (torch.sum(mask) + 1e-6)
        return balanced_loss
