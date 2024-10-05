import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import torch
import torch.nn.functional as F
from tqdm import tqdm


def dilate_polygons(polygons):
    dilated_polygons = []
    for poly in polygons:
        if len(poly) < 4:
            continue
        Gs = Polygon(poly)
        D = Gs.area * 1.5 / Gs.length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

        dilated_polygon = pco.Execute(D)
        if dilated_polygon == []:
            continue
        dilated_polygon = np.array(dilated_polygon[0]).reshape(-1, 2)
        dilated_polygons.append(dilated_polygon)
    return dilated_polygons


def seg_to_polygons(map):
    bin_map = (map >= 0.2).astype(np.uint8).squeeze()
    contours, _ = cv.findContours(bin_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 25 or area > 200000:
            continue
        approx = cv.approxPolyDP(cnt, 0.005 * cv.arcLength(cnt, True), True)
        polygons.append(approx.reshape(-1, 2))
    return polygons


def polygons_to_seg(polygons, size):
    map = np.zeros(size, np.float32)
    for poly in polygons:
        cv.fillPoly(map, np.expand_dims(poly, 0), 1)
    return map


def get_seg_and_poly(output_maps):
    output_maps = output_maps.detach().cpu().numpy()
    bin_maps = (output_maps >= 0.2).astype(np.uint8)

    gt_polygons = []
    gt_maps = np.empty_like(bin_maps, dtype=np.float32)
    for i, map in enumerate(bin_maps):
        polygons = seg_to_polygons(map)
        dilated_polygons = dilate_polygons(polygons)
        gt_polygons.append(dilated_polygons)
        gt_maps[i] = polygons_to_seg(dilated_polygons, (640, 640))
    return gt_polygons, gt_maps


def get_components(img):
    # labels = []
    _, labels = cv.connectedComponents(img)
    # labels.append(label)
    labels = np.array(labels)
    labels = torch.from_numpy(labels).to(dtype=torch.float32)

    for i in range(1, int(labels.max()) + 1):
        if (labels == i).sum() <= 10:
            labels[labels == i] = 0

    labels = labels.cuda()
    labels = F.max_pool2d(labels.unsqueeze(0), 9, stride=1, padding=4).squeeze(0)
    return labels


def get_iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids):
    iou_matrix = np.empty((len(pred_ids), len(gt_ids)))
    iou_matrix = torch.from_numpy(iou_matrix).cuda()

    for i, pred_id in enumerate(pred_ids):
        for j, gt_id in enumerate(gt_ids):
            intersection = ((pred_labels == pred_id) & (gt_labels == gt_id)).sum()
            union = ((pred_labels == pred_id) | (gt_labels == gt_id)).sum()
            iou_matrix[i, j] = intersection / union
    return iou_matrix


def match_bboxes(iou_matrix, pred_ids, gt_ids):
    matches = []
    matched_gt = set()

    iou_pairs = [
        (pred_ids[i], gt_ids[j], iou_matrix[i, j])
        for i, j in zip(*torch.where(iou_matrix > 0.5))
    ]
    iou_pairs.sort(key=lambda x: x[2], reverse=True)

    for pred_id, gt_id, iou in iou_pairs:
        if gt_id in matched_gt:
            continue
        matches.append((pred_id, gt_id))
        matched_gt.add(gt_id)
    return matches


def get_gt_labels(bboxes, size):
    gt_labels = np.zeros(size)

    for i, bbox in enumerate(bboxes):
        cv.fillPoly(gt_labels, np.expand_dims(bbox, 0).astype(np.int32), i + 1)

    gt_labels = torch.from_numpy(gt_labels).cuda()
    return gt_labels


def blur_image(image):
    kernel = (
        torch.tensor(
            [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]], device="cuda"
        )
        / 16
    )

    return (
        F.conv2d(
            image.unsqueeze(0),
            kernel,
            stride=1,
            padding=1,
        )
        .cpu()
        .numpy()
        .squeeze(0)
    )


def get_metrics(pred, bboxes):
    threshold = 0.7

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    for i in range(len(pred)):
        binary = (pred[i] > threshold).to(dtype=torch.float32)
        binary = (blur_image(binary) > 0.1).astype(np.uint8)
        pred_labels = get_components(binary)
        gt_labels = get_gt_labels(bboxes[i], pred_labels.shape)

        pred_ids = np.unique(pred_labels.cpu())[1:].tolist()
        gt_ids = np.unique(gt_labels.cpu())[1:].tolist()

        iou_matrix = get_iou_matrix(pred_labels, gt_labels, pred_ids, gt_ids)
        matches = match_bboxes(iou_matrix, pred_ids, gt_ids)

        true_positives = len(matches)
        false_positives = len(pred_ids) - true_positives
        false_negatives = len(gt_ids) - true_positives

        if len(gt_ids) == 0:
            precision = 1 if len(pred_ids) == 0 else 0
            recall = 1
            f1 = precision
        else:
            precision = (
                0
                if true_positives == 0
                else true_positives / (true_positives + false_positives)
            )
            recall = (
                0
                if true_positives == 0
                else true_positives / (true_positives + false_negatives)
            )
            f1 = (
                0
                if precision + recall == 0
                else (2 * precision * recall) / (precision + recall)
            )

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    return (total_precision, total_recall, total_f1)


def evaluate(model, loader):
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    dataset_size = 0

    model.eval()

    with torch.no_grad():
        for images, (_, _, kernel_masks, text_masks, bboxes) in loader:
            images = images.to(dtype=torch.float32, device="cuda")
            kernel_masks = kernel_masks.to(dtype=torch.float32, device="cuda")
            text_masks = text_masks.to(dtype=torch.float32, device="cuda")

            batch_size = len(images)

            preds = model(images)
            # preds = preds * kernel_masks
            preds = preds * text_masks
            precision, recall, f1 = get_metrics(preds, bboxes)

            running_precision += precision
            running_recall += recall
            running_f1 += f1
            dataset_size += batch_size

    avg_precision = running_precision / dataset_size
    avg_recall = running_recall / dataset_size
    avg_f1 = running_f1 / dataset_size
    return avg_precision, avg_recall, avg_f1
