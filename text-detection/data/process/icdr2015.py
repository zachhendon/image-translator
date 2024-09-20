import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from utils import resize_image, get_maps
import lmdb
import albumentations as A

TRAIN_IMG_DIR = "../raw/icdr2015/train_images"
TRAIN_GT_DIR = "../raw/icdr2015/train_gts"
TEST_IMG_DIR = "../raw/icdr2015/test_images"
TEST_GT_DIR = "../raw/icdr2015/test_gts"

train_image_paths = sorted(os.listdir(TRAIN_IMG_DIR))
train_gt_paths = sorted(os.listdir(TRAIN_GT_DIR))
test_image_paths = sorted(os.listdir(TEST_IMG_DIR))
test_gt_paths = sorted(os.listdir(TEST_GT_DIR))

N = 0
resize = A.Compose([
    A.Resize(640, 640)
], keypoint_params=A.KeypointParams(format="xy"))


def process_images(image_paths, gt_paths):
    global N
    env = lmdb.open(f'../processed/lmdb_dataset', map_size=150 * 2**30)

    with env.begin(write=True) as txn:
        for i in tqdm(range(len(image_paths))):
            image = cv.imread(f"{TRAIN_IMG_DIR}/{image_paths[i]}")
            with open(f"{TRAIN_GT_DIR}/{gt_paths[i]}", encoding="utf-8-sig") as file:
                gt = [line.rstrip().split(",")[:8] for line in file]
            gt = np.array(gt, dtype=np.float32)
            gt = gt.reshape(-1, 4, 2)

            transformed = resize(image=image, keypoints=gt.reshape(-1, 2))
            image = transformed["image"]
            gt = transformed["keypoints"].reshape(-1, 4, 2).astype(np.int32)          

            gt_text, gt_kernel = get_maps(gt, image.shape[:2])
            encoded_image = bytes(cv.imencode('.jpg', image)[1])
            encoded_gt_text = bytes(cv.imencode('.jpg', gt_text * 255)[1])
            encoded_gt_kernel = bytes(cv.imencode('.jpg', gt_kernel * 255)[1])

            txn.put(f'icdr2015_image_{
                    str(N).zfill(4)}'.encode(), encoded_image)
            # txn.put(f'icdr2015_nbounds_{str(N).zfill(4)}'.encode(), str(len(gt)).encode())
            txn.put(f'icdr2015_bounds_{str(N).zfill(4)}'.encode(), gt)
            txn.put(f'icdr2015_gt_{str(N).zfill(4)}'.encode(), encoded_gt_text)
            txn.put(f'icdr2015_eroded_{str(N).zfill(4)}'.encode(), encoded_gt_kernel)

            N += 1


process_images(train_image_paths, train_gt_paths)
process_images(test_image_paths, test_gt_paths)
