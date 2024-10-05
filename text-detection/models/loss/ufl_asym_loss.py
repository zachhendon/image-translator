import torch
import torch.nn as nn


class UFLAsymLoss(nn.Module):
    def __init__(self):
        super(UFLAsymLoss, self).__init__()

    def forward(self, input, target):
        delta = 0.6
        gamma = 0.5
        lmd = 0.5

        pt = target * input + (1 - target) * (1 - input)
        L_maF = -delta * target * torch.log(pt + 1e-8) - (1 - delta) * torch.pow(
            input + 1e-8, gamma
        ) * torch.log(pt + 1e-8)
        L_maF = -delta * input * torch.log(pt + 1e-8)
        L_maF = L_maF.mean()

        mTI = (input * target).sum() / (
            input * target
            + delta * (input * (1 - target))
            + (1 - delta) * ((1 - input) * target)
        ).sum()
        L_maFT = (1 - mTI) + torch.pow(1 - mTI + 1e-8, 1 - gamma)

        return lmd * L_maF + (1 - lmd) * L_maFT
