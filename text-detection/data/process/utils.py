import numpy as np
import albumentations as A
import cv2 as cv
import torch
import torch.nn.functional as F


transform = A.Compose([A.Resize(960, 960)])


def get_maps(gt, size):
    gt_map = np.zeros(size, dtype=np.float32)
    gt_map = cv.fillPoly(gt_map, gt, 1)

    eroded_map = np.zeros_like(gt_map)
    for poly in gt:
        poly_map = np.zeros(size, dtype=np.float32)
        poly_map = cv.fillPoly(poly_map, np.expand_dims(poly, 0), 1)

        eroded_poly_map = -F.max_pool2d(
            -torch.from_numpy(poly_map).cuda().view(1, 1, *size),
            9, stride=1, padding=4
        ).squeeze().cpu().numpy()
        eroded_map += eroded_poly_map
    eroded_map = np.clip(eroded_map, 0, 1)

    return gt_map, eroded_map


def resize_image(image):
    return transform(image=image)["image"]
