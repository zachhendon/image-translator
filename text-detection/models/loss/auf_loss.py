import torch
import torch.nn as nn


class AUFLoss(nn.Module):
    def __init__(self, lmb=0.5, delta=0.6, gamma=0.5):
        super().__init__()
        self.eps = 1e-10
        self.lmb = lmb
        self.delta = delta
        self.gamma = gamma

    def forward(self, pred, gt, mask):
        pred = torch.clamp(pred * mask, self.eps, 1 - self.eps) 
        gt = torch.clamp(gt * mask, self.eps, 1 - self.eps) 

        N = torch.sum(mask, dim=(1, 2))
        maf_loss = (-self.delta / N * torch.sum(gt * torch.log(pred), dim=(1, 2))) - (
            (1 - self.delta)
            / N
            * torch.sum((pred) ** self.gamma * torch.log(pred), dim=(1, 2))
        )

        mTI_neg = torch.sum((1 - pred) * (1 - gt), dim=(1, 2)) / (
            torch.sum((1 - pred) * (1 - gt), dim=(1, 2))
            + self.delta * torch.sum((1 - pred) * gt, dim=(1, 2))
            + (1 - self.delta) * torch.sum(pred * (1 - gt), dim=(1, 2))
        )
        mTI_pos = torch.sum(pred * gt, dim=(1, 2)) / (
            torch.sum(pred * gt, dim=(1, 2))
            + self.delta * torch.sum(pred * (1 - gt), dim=(1, 2))
            + (1 - self.delta) * torch.sum((1 - pred) * gt, dim=(1, 2))
        )
        maft_loss = (1 - mTI_neg) + (1 - mTI_pos) ** (1 - self.gamma)

        auf_loss = self.lmb * maf_loss + (1 - self.lmb) * maft_loss
        return auf_loss.mean()
