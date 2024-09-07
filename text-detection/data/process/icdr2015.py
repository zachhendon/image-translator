import pandas as pd
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from utils import resize_image, get_maps

TRAIN_IMG_DIR = "../raw/icdr2015/train_images"
TRAIN_GT_DIR = "../raw/icdr2015/train_gts"
TEST_IMG_DIR = "../raw/icdr2015/test_images"
TEST_GT_DIR = "../raw/icdr2015/test_gts"

train_image_paths = sorted(os.listdir(TRAIN_IMG_DIR))
train_gt_paths = sorted(os.listdir(TRAIN_GT_DIR))
test_image_paths = sorted(os.listdir(TEST_IMG_DIR))
test_gt_paths = sorted(os.listdir(TEST_GT_DIR))

N = 0


def process_images(image_paths, gt_paths):
    global N

    for i in tqdm(range(len(image_paths))):
        image = cv.imread(f"{TRAIN_IMG_DIR}/{image_paths[i]}")
        h, w, _ = image.shape
        image = resize_image(image)
        cv.imwrite(
            f"../processed/icdr2015/images/image_{str(N).zfill(4)}.jpg", image)

        with open(f"{TRAIN_GT_DIR}/{gt_paths[i]}", encoding="utf-8-sig") as file:
            gt = [line.rstrip().split(",")[:8] for line in file]
        gt = np.array(gt, dtype=np.float32)
        ratios = np.empty(8)
        ratios[::2] = 960 / w
        ratios[1::2] = 960 / h
        gt *= ratios

        size = (960, 960)
        thresh_map, bin_map, gt_map = get_maps(gt, size)

        cv.imwrite(
            f"../processed/icdr2015/gts/thresh_map_{str(N).zfill(4)}.jpg",
            thresh_map * 255,
        )
        cv.imwrite(
            f"../processed/icdr2015/gts/bin_map_{str(N).zfill(4)}.jpg",
            bin_map * 255,
        )
        cv.imwrite(
            f"../processed/icdr2015/gts/gt_map_{str(N).zfill(4)}.jpg",
            gt_map * 255,
        )
        N += 1


process_images(train_image_paths, train_gt_paths)
process_images(test_image_paths, test_gt_paths)
