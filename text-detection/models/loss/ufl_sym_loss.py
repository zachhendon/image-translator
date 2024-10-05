import torch.nn as nn
import torch.nn.functional as F


class UFLSymLoss(nn.Module):
    def __init__(self):
        super(UFLSymLoss, self).__init__()

    def forward(self, input, target):
        delta = 0.6
        gamma = 0.5
        lmd = 0.5

        pt = target * input + (1 - target) * (1 - input)
        L_mF = (
            delta
            * (1 - pt + 1e-8).pow(gamma)
            * F.binary_cross_entropy(input, target, reduction="none")
        )
        # L_mF = L_mF * edge_weight
        L_mF = L_mF.mean()

        # mTI = (edge_weight * pred * true).sum() / (
        #     edge_weight
        #     * (
        #         pred * true
        #         + delta * (pred * (1 - true))
        #         + (1 - delta) * ((1 - pred) * true)
        #     )
        # ).sum()
        mTI = (input * target).sum() / (
            input * target
            + delta * (input * (1 - target))
            + (1 - delta) * ((1 - input) * target)
        ).sum()
        L_mFT = (1 - mTI).pow(gamma)

        return lmd * L_mF + (1 - lmd) * L_mFT
