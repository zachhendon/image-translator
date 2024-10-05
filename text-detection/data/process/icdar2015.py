import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
from utils import get_maps2
import lmdb
import albumentations as A

TRAIN_IMG_DIR = "../raw/icdar2015/train_images"
TRAIN_GT_DIR = "../raw/icdar2015/train_gts"
TEST_IMG_DIR = "../raw/icdar2015/test_images"
TEST_GT_DIR = "../raw/icdar2015/test_gts"

train_image_paths = sorted(os.listdir(TRAIN_IMG_DIR))
train_gt_paths = sorted(os.listdir(TRAIN_GT_DIR))
test_image_paths = sorted(os.listdir(TEST_IMG_DIR))
test_gt_paths = sorted(os.listdir(TEST_GT_DIR))

N = 0
resize = A.Compose([A.Resize(640, 640)], keypoint_params=A.KeypointParams(format="xy"))

env = lmdb.open(f"../processed/lmdb_dataset", map_size=150 * 2**30)


def process_images(image_paths, gt_paths):
    global N

    for i in tqdm(range(len(image_paths))):
        image = cv.imread(f"{TRAIN_IMG_DIR}/{image_paths[i]}")
        bboxes = []
        with open(f"{TRAIN_GT_DIR}/{gt_paths[i]}", encoding="utf-8-sig", mode="r") as f:
            for line in f:
                gt = line.rstrip().split(",")
                if gt[8] in ["###", "*"]:
                    continue 
                bboxes.append([gt[:8]])
        bboxes = np.array(bboxes, dtype=np.int32).reshape(-1, 4, 2)

        # transformed = resize(image=image, keypoints=bboxes.reshape(-1, 2))
        # image = transformed["image"]
        # bboxes = transformed["keypoints"].reshape(-1, 4, 2).astype(np.int32)

        encoded_image = bytes(cv.imencode(".jpg", image)[1])
        gt_kernel, gt_text, edge_weight = get_maps2(bboxes, image.shape[:-1])
        edge_weight = (edge_weight - edge_weight.min()) / (
            edge_weight.max() - edge_weight.min()
        )

        labels = np.stack([edge_weight, gt_text, gt_kernel], axis=-1) * 255
        encoded_maps = bytes(cv.imencode(".jpg", labels)[1])

        with env.begin(write=True) as txn:
            txn.put(f"icdar2015_image_{str(N).zfill(4)}".encode(), encoded_image)
            txn.put(f"icdar2015_bboxes_{str(N).zfill(4)}".encode(), bboxes)
            txn.put(f"icdar2015_maps_{str(N).zfill(4)}".encode(), encoded_maps)

        N += 1


process_images(train_image_paths, train_gt_paths)
process_images(test_image_paths, test_gt_paths)
