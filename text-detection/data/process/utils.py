import numpy as np
import albumentations as A
import cv2 as cv
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon
import time

resize = A.Compose([A.Resize(640, 640)], keypoint_params=A.KeypointParams(format="xy"))


def get_maps(bboxes, size):
    r = 0.1

    gt_text = np.zeros((640, 640), dtype=np.uint8)
    gt_kernel = np.zeros((640, 640), dtype=np.uint8)
    area_weight = np.ones((640, 640), dtype=np.float32)
    edge_weight = np.ones((640, 640), dtype=np.float32)

    transformed = resize(image=np.zeros(size), keypoints=bboxes.reshape(-1, 2))
    bboxes = transformed["keypoints"].reshape(-1, 4, 2).astype(np.int32)

    for bbox in bboxes:
        # fill gt_text
        gt_text_poly = np.zeros((640, 640), dtype=np.uint8)
        cv.fillPoly(gt_text_poly, [bbox], 1)
        gt_text = np.maximum(gt_text, gt_text_poly)

        # vatti clipping
        temp1 = np.zeros_like(gt_kernel, dtype=np.uint8)
        poly = Polygon(bbox)
        if poly.area == 0:
            continue
        offset = poly.area * (1 - r * r) / poly.length
        shrink_poly = poly.buffer(-offset)
        cv.fillPoly(temp1, [np.array(shrink_poly.exterior.coords, dtype=np.int32)], 1)

        # morphological erosion
        temp2 = (
            -F.max_pool2d(
                -torch.from_numpy(gt_text_poly)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(dtype=torch.float16, device="cuda"),
                9,
                stride=1,
                padding=4,
            )
            .to(dtype=torch.uint8)
            .squeeze()
            .cpu()
            .numpy()
        )

        # combine vatti clipping and morphological erosion to get gt_kernel
        gt_kernel_bbox = np.maximum(temp1, temp2)
        gt_kernel = np.maximum(gt_kernel, gt_kernel_bbox)

        # area weights
        # fill_value = 1 + 4 * np.exp(-(poly.area / poly.length) / 10)
        fill_value = 1 + 4 * np.exp(-np.sqrt(poly.area) / 50)
        area_weight = np.maximum(area_weight, fill_value * gt_kernel_bbox)

        # edge weights
        edge_bbox = F.avg_pool2d(
            torch.from_numpy(gt_kernel_bbox)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(dtype=torch.float16, device="cuda"),
            7,
            stride=1,
            padding=3,
        )
        edge_bbox = (
            torch.where((edge_bbox == 0) | (edge_bbox == 1), 1, 2)
            .squeeze()
            .cpu()
            .numpy()
        )
        edge_weight = np.maximum(edge_weight, edge_bbox)

    return gt_kernel, gt_text, area_weight * edge_weight


def get_maps2(bboxes, size):
    r = 0.1
    num_bboxes = len(bboxes)

    transformed = resize(image=np.zeros(size), keypoints=bboxes.reshape(-1, 2))
    bboxes = transformed["keypoints"].reshape(-1, 4, 2).astype(np.int32)

    gt_text = torch.zeros((num_bboxes, 640, 640), dtype=torch.uint8, device="cuda")
    gt_kernel = torch.zeros((num_bboxes, 640, 640), dtype=torch.uint8, device="cuda")
    fill_values = torch.empty((num_bboxes, 1, 1), dtype=torch.float32, device="cuda")

    for i, bbox in enumerate(bboxes):
        gt_text[i] = torch.from_numpy(cv.fillPoly(gt_text[i].cpu().numpy(), [bbox], 1))

        poly = Polygon(bbox)
        if poly.area == 0:
            continue
        offset = poly.area * (1 - r * r) / poly.length
        shrink_poly = poly.buffer(-offset)
        gt_kernel[i] = torch.from_numpy(
            cv.fillPoly(
                gt_kernel[i].cpu().numpy(),
                [np.array(shrink_poly.exterior.coords, dtype=np.int32)],
                1,
            )
        )

        fill_values[i] = 1 + 4 * np.exp(-np.sqrt(poly.area) / 50)

    gt_kernel = torch.maximum(
        gt_kernel,
        -F.max_pool2d(
            -gt_text
            .unsqueeze(0)
            .to(dtype=torch.float16),
            9,
            stride=1,
            padding=4,
        )
        .to(dtype=torch.uint8)
        .squeeze(),
    )
    
    edge_bbox = F.avg_pool2d(
        gt_kernel.unsqueeze(0).to(dtype=torch.float16),
        7,
        stride=1,
        padding=3,    
    )
    edge_bbox = torch.where((edge_bbox == 0) | (edge_bbox == 1), 1, 2).squeeze()
    edge_weights = gt_kernel * fill_values * edge_bbox
    edge_weights = torch.clamp(edge_weights, 1)

    gt_kernel = torch.max(gt_kernel, dim=0).values
    gt_kernel = gt_kernel.cpu().numpy()
    gt_text = torch.max(gt_text, dim=0).values
    gt_text = gt_text.cpu().numpy()
    edge_weights = torch.max(edge_weights, dim=0).values
    edge_weights = edge_weights.cpu().numpy()

    return gt_kernel, gt_text, edge_weights
