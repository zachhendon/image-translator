import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from utils import resize_image, get_maps
import lmdb

TRAIN_IMG_DIR = "../raw/icdr2015/train_images"
TRAIN_GT_DIR = "../raw/icdr2015/train_gts"
TEST_IMG_DIR = "../raw/icdr2015/test_images"
TEST_GT_DIR = "../raw/icdr2015/test_gts"

train_image_paths = sorted(os.listdir(TRAIN_IMG_DIR))
train_gt_paths = sorted(os.listdir(TRAIN_GT_DIR))
test_image_paths = sorted(os.listdir(TEST_IMG_DIR))
test_gt_paths = sorted(os.listdir(TEST_GT_DIR))

N = 0
# size = (960, 960)


def process_images(image_paths, gt_paths):
    global N
    env = lmdb.open(f'../processed/lmdb_dataset', map_size=150 * 2**30)

    with env.begin(write=True) as txn:
        for i in tqdm(range(len(image_paths))):
            image = cv.imread(f"{TRAIN_IMG_DIR}/{image_paths[i]}")
            h, w, _ = image.shape

            encoded_image = bytes(cv.imencode('.jpg', image)[1])
            txn.put(f'icdr2015_image_{
                    str(N).zfill(4)}'.encode(), encoded_image)

            with open(f"{TRAIN_GT_DIR}/{gt_paths[i]}", encoding="utf-8-sig") as file:
                gt = [line.rstrip().split(",")[:8] for line in file]
            gt = np.array(gt, dtype=np.float32)
            gt = gt.reshape(-1, 4, 2)
            gt = np.round(gt).astype(np.int64)

            txn.put(f'icdr2015_nbounds_{str(N).zfill(4)}'.encode(), str(len(gt)).encode())
            txn.put(f'icdr2015_bounds_{str(N).zfill(4)}'.encode(), gt)

            gt_map, eroded_map = get_maps(gt, (h, w))
            encoded_gt_map = bytes(cv.imencode('.jpg', gt_map * 255)[1])
            txn.put(f'icdr2015_gt_{str(N).zfill(4)}'.encode(), encoded_gt_map)
            encoded_eroded_map = bytes(cv.imencode('.jpg', eroded_map * 255)[1])
            txn.put(f'icdr2015_eroded_{str(N).zfill(
                4)}'.encode(), encoded_eroded_map)

            N += 1


process_images(train_image_paths, train_gt_paths)
process_images(test_image_paths, test_gt_paths)
