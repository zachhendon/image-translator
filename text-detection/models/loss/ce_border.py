import torch
import torch.nn as nn
import torch.nn.functional as F


class CEBorderLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gt_kernels, gt_text, training_masks):
        N = gt_kernels[0].numel()
        N_kernel = gt_kernels.sum(dim=(1, 2), keepdim=True)
        N_border = (gt_text - gt_kernels).sum(dim=(1, 2), keepdim=True)
        N_nontext = (1 - gt_text).sum(dim=(1, 2), keepdim=True)
        kernel_weight = torch.where(N_kernel == 0, 0, N / N_kernel)
        border_weight = torch.where(N_border == 0, 0, N / N_border)
        nontext_weight = torch.where(N_nontext == 0, 0, N / N_nontext)

        kernel_loss = (
            -kernel_weight * gt_kernels * torch.log(preds[:, 0] + 1e-6) * training_masks
        )
        border_loss = (
            -border_weight
            * (gt_text - gt_kernels)
            * torch.log(preds[:, 1] + 1e-6)
            * training_masks
        )
        nontext_loss = (
            -nontext_weight
            * (1 - gt_text)
            * torch.log(preds[:, 2] + 1e-6)
            * training_masks
        )

        loss = (kernel_loss + border_loss + nontext_loss).sum(
            dim=(1, 2)
        ) / training_masks.sum(dim=(1, 2))
        return loss.mean()
