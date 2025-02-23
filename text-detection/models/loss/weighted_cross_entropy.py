import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        N = gt[0].numel()
        N_pos = gt.sum(dim=(1, 2), keepdim=True)
        N_neg = N - N_pos
        pos_weight = torch.where(N_pos == 0, 0, N / N_pos)
        neg_weight = torch.where(N_neg == 0, 0, N / N_neg)

        pos_loss = -pos_weight * gt * torch.log(pred + 1e-6) * mask
        neg_loss = -neg_weight * (1 - gt) * torch.log(1 - pred + 1e-6) * mask
        loss = (pos_loss + neg_loss).sum() / mask.sum()
        return loss
