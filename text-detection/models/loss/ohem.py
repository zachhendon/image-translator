import torch


# def get_ohem_masks(preds, gt_texts, gt_texts_ignore, ratio=3.0):
#     ohem_masks = []
#     gt_texts = torch.clamp(gt_texts, 0, 1)
#     gt_texts_ignore = 1 - torch.clamp(gt_texts_ignore, 0, 1)
#     for i in range(len(preds)):
#         num_pos = int(torch.sum(gt_texts[i]))
#         num_neg = int(min(ratio * num_pos, torch.sum(1 - gt_texts[i])))
#         if num_pos == 0 or num_neg == 0:
#             ohem_masks.append(gt_texts_ignore[i])
#             continue

#         neg_pred = preds[i] * (1 - gt_texts[i])
#         threshold = torch.sort(neg_pred.view(-1), descending=True)[0][
#             num_neg - 1
#         ].item()

#         mask = torch.bitwise_and(
#             torch.bitwise_or(preds[i] >= threshold, gt_texts[i] > 0.5),
#             gt_texts_ignore[i] > 0.5,
#         )
#         ohem_masks.append(mask.float())
#     return torch.stack(ohem_masks)


def get_ohem_masks(preds, gt, training_mask):
    ohem_masks = []
    
    for i in range(len(preds)):
        num_pos = int(
            torch.sum(gt[i] >= 0.5)
            - torch.sum((gt[i] >= 0.5) & (training_mask[i] < 0.5))
        )
        num_neg = int(min(torch.sum(gt[i] < 0.5), num_pos * 3))
        if num_pos == 0 or num_neg == 0:
            ohem_masks.append(training_mask[i])
            continue

        neg_pred = preds[i][gt[i] < 0.5]
        neg_pred_sorted, _ = torch.sort(neg_pred.view(-1), descending=True)
        threshold = neg_pred_sorted[num_neg - 1]

        ohem_mask = ((preds[i] >= threshold) | (gt[i] >= 0.5)) & (
            training_mask[i] >= 0.5
        )
        ohem_masks.append(ohem_mask.float())
    return torch.stack(ohem_masks)
