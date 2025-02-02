import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        loss = F.binary_cross_entropy(pred * mask, gt.float(), reduction="none")
        balanced_loss = torch.sum(loss, dim=(1, 2)) / (
            torch.sum(mask, dim=(1, 2)) + 1e-6
        )
        return balanced_loss.mean()
