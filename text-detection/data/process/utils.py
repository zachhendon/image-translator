import numpy as np
import albumentations as A
import cv2 as cv
import torch
import torch.nn.functional as F


transform = A.Compose([A.Resize(960, 960)])


def get_maps(gt, size):
    gt_text = np.zeros(size, dtype=np.float32)
    gt_kernel = np.zeros_like(gt_text)

    for poly in gt:
        gt_text_poly = np.zeros(size, dtype=np.float32)
        gt_text_poly = cv.fillPoly(gt_text_poly, np.expand_dims(poly, 0), 1)
        gt_text += gt_text_poly 

        gt_kernel_poly = (
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
        gt_kernel += gt_kernel_poly 
    gt_kernel = np.clip(gt_kernel, 0, 1)
    gt_text = np.clip(gt_text, 0, 1)

    return gt_text, gt_kernel 


def resize_image(image):
    return transform(image=image)["image"]
