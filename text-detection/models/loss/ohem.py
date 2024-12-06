import torch


def get_ohem_masks(pred, gt, masks, ratio=3.0):
    ohem_masks = []
    for i in range(len(pred)):
        num_pos = int(torch.sum(gt[i] > 0.5)) - int(
            torch.sum((gt[i] > 0.5) & (masks[i] <= 0.5))
        )
        num_neg = int(min(torch.sum(gt[i] <= 0.5), num_pos * ratio))
        if num_pos == 0 or num_neg == 0:
            ohem_masks.append(masks[i])
            continue

        neg_pred = pred[i][gt[i] <= 0.5]
        threshold = torch.sort(neg_pred.view(-1), descending=True)[0][
            num_neg - 1
        ].item()
        mask = torch.bitwise_and(
            torch.bitwise_or((pred[i] >= threshold), gt[i] > 0.5), (masks[i] > 0.5)
        )
        ohem_masks.append(mask)
    return torch.stack(ohem_masks)
