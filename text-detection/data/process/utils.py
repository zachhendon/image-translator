import numpy as np
import albumentations as A
import cv2 as cv
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon


transform = A.Compose([A.Resize(960, 960)])


def get_maps(gt, size):
    r = 0.1

    gt_text = np.zeros(size)
    gt_kernel = np.zeros_like(gt_text)

    for bbox in gt:
        gt_text_poly = np.zeros(size)
        gt_text_poly = cv.fillPoly(gt_text_poly, np.expand_dims(bbox, 0), 1)
        gt_text = np.maximum(gt_text, gt_text_poly)

        # vatti clipping
        temp1 = np.zeros_like(gt_kernel)
        poly = Polygon(bbox)
        offset = poly.area * (1 - r * r) / poly.length
        shrink_poly = poly.buffer(-offset)
        cv.fillPoly(temp1, [np.array(shrink_poly.exterior.coords, dtype=np.int32)], 1)

        # morphological erosion
        temp2 = (
            -F.max_pool2d(
                -torch.from_numpy(gt_text_poly).cuda().view(1, 1, *size),
                9,
                stride=1,
                padding=4,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        gt_kernel = np.maximum(gt_kernel, np.maximum(temp1, temp2))

    return gt_text, gt_kernel


def resize_image(image):
    return transform(image=image)["image"]
