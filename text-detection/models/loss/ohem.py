import torch


def get_ohem_masks(preds, gt_texts, gt_texts_ignore, ratio=3.0):
    ohem_masks = []
    gt_texts = torch.clamp(gt_texts, 0, 1)
    gt_texts_ignore = 1 - torch.clamp(gt_texts_ignore, 0, 1)
    for i in range(len(preds)):
        num_pos = int(torch.sum(gt_texts[i]))
        num_neg = int(min(ratio * num_pos, torch.sum(1 - gt_texts[i])))
        if num_pos == 0 or num_neg == 0:
            ohem_masks.append(gt_texts_ignore[i])
            continue

        neg_pred = preds[i] * (1 - gt_texts[i])
        threshold = torch.sort(neg_pred.view(-1), descending=True)[0][
            num_neg - 1
        ].item()

        mask = torch.bitwise_and(
            torch.bitwise_or(preds[i] >= threshold, gt_texts[i] > 0.5),
            gt_texts_ignore[i] > 0.5,
        )
        ohem_masks.append(mask.float())
    return torch.stack(ohem_masks)
