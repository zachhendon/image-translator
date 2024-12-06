import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, masks):
        ratio = 3.0
        pos = gt * masks
        neg = (1 - gt) * masks
        num_pos = int((gt * masks).sum())
        num_neg = int(min((1 - gt).sum(), num_pos * ratio))

        loss = F.binary_cross_entropy(pred, gt, reduction="none")
        positive_loss = loss * pos
        negative_loss = loss * neg
        negative_loss, _ = torch.topk(negative_loss.view(-1), num_neg)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            num_pos + num_neg + 1e-6
        )
        return balance_loss
