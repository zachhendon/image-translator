import numpy as np
import albumentations as A
import cv2 as cv
import torch
import torch.nn.functional as F


transform = A.Compose([A.Resize(960, 960)])


def get_maps(gt, size):
    gt_map = np.zeros(size, dtype=np.float32)
    gt_map = cv.fillPoly(gt_map, gt, 1)

    s = 9
    kernel = torch.ones((1, 1, s, s), dtype=torch.float32, device='cuda')
    eroded_map = F.conv2d(torch.from_numpy(gt_map).cuda().view(
        1, 1, *size), kernel, padding='same')
    eroded_map = torch.where(eroded_map < 81, 0, 1)
    eroded_map = eroded_map.squeeze().cpu().numpy()

    return gt_map, eroded_map


def resize_image(image):
    return transform(image=image)["image"]
