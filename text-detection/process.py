import cv2 as cv
from glob import glob
import numpy as np
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


def create_masks(bboxes, size):
    gt_kernels = []
    gt_text = np.zeros(size, dtype=np.uint8)
    if len(bboxes) == 0:
        return gt_text, gt_text
    for bbox in bboxes:
        gt_kernel = np.zeros(size, dtype=np.uint8)
        gt_kernel = np.pad(gt_kernel, 4)
        cv.fillPoly(gt_kernel, [bbox + 4], 1)
        cv.fillPoly(gt_text, [bbox], 1)
        gt_kernels.append(gt_kernel)
    gt_kernel = np.array(gt_kernels)
    gt_kernel = (
        -F.max_pool2d(
            -torch.from_numpy(gt_kernel)
            .to(dtype=torch.float16, device="cuda")
            .unsqueeze(0),
            kernel_size=9,
            stride=1,
            padding=0,
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )
    gt_kernel = np.max(gt_kernel, axis=0)
    return gt_kernel, gt_text


def get_min_bboxes(bboxes):
    rate = 0.1**2
    shrunk_bboxes = []
    for bbox in bboxes:
        poly = Polygon(bbox)
        offset = poly.area * (1 - rate) / poly.length
        shrunk_poly = poly.buffer(-offset)
        if shrunk_poly.is_empty:
            shrunk_bboxes.append(bbox)
            continue
        shrunk_bboxes.append(list(shrunk_poly.exterior.coords)[:4])
    return np.array(shrunk_bboxes).reshape(-1, 4, 2).astype(np.int32)


def process_ic15_data(image_paths, gt_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    images_dir = f"{save_dir}/images"
    bboxes_dir = f"{save_dir}/bboxes"
    # min_bboxes_dir = f"{save_dir}/min_bboxes"
    ignore_bboxes_dir = f"{save_dir}/ignore_bboxes"
    # min_ignore_bboxes_dir = f"{save_dir}/min_ignore_bboxes"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)
    # os.makedirs(min_bboxes_dir, exist_ok=True)
    os.makedirs(ignore_bboxes_dir, exist_ok=True)
    # os.makedirs(min_ignore_bboxes_dir, exist_ok=True)
    for i, (image_path, gt_path) in tqdm(enumerate(zip(image_paths, gt_paths))):
        id = str(i).zfill(6)
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)

        bboxes = []
        ignore_bboxes = []
        with open(gt_path, encoding="utf-8-sig", mode="r") as f:
            for line in f:
                gt = line.rstrip().split(",")
                if gt[8] == "###":
                    ignore_bboxes.append([gt[:8]])
                else:
                    bboxes.append([gt[:8]])
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4, 2)
        ignore_bboxes = np.array(ignore_bboxes, dtype=np.float32).reshape(-1, 4, 2)

        cv.imwrite(f"{images_dir}/{id}.jpg", image)
        np.save(f"{bboxes_dir}/{id}.npy", bboxes)
        np.save(f"{ignore_bboxes_dir}/{id}.npy", ignore_bboxes)


def process_ic15():
    # process training data
    train_images_dir = "data/raw/ic15/train_images"
    train_gts_dir = "data/raw/ic15/train_gts"
    train_image_paths = sorted(glob(f"{train_images_dir}/*"))
    train_gt_paths = sorted(glob(f"{train_gts_dir}/*"))
    train_image_paths, val_image_paths = train_test_split(
        train_image_paths, test_size=0.2, random_state=42
    )
    train_gt_paths, val_gt_paths = train_test_split(
        train_gt_paths, test_size=0.2, random_state=42
    )

    train_save_dir = "data/processed/ic15/train"
    val_save_dir = "data/processed/ic15/val"

    process_ic15_data(train_image_paths, train_gt_paths, train_save_dir)
    process_ic15_data(val_image_paths, val_gt_paths, val_save_dir)

    # process test data
    test_images_dir = "data/raw/ic15/test_images"
    test_gts_dir = "data/raw/ic15/test_gts"
    test_image_paths = sorted(glob(f"{test_images_dir}/*"))
    test_gt_paths = sorted(glob(f"{test_gts_dir}/*"))
    test_save_dir = "data/processed/ic15/test"
    process_ic15_data(test_image_paths, test_gt_paths, test_save_dir)


def process_synthtext_data(image_paths, bboxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    images_dir = f"{save_dir}/images"
    bboxes_dir = f"{save_dir}/bboxes"
    ignore_bboxes_dir = f"{save_dir}/ignore_bboxes"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)
    os.makedirs(ignore_bboxes_dir, exist_ok=True)

    for i, (image_path, bbox) in tqdm(enumerate(zip(image_paths, bboxes))):
        id = str(i).zfill(6)
        image = cv.imread(image_path, cv.IMREAD_UNCHANGED)

        cv.imwrite(f"{images_dir}/{id}.jpg", image)
        np.save(f"{bboxes_dir}/{id}.npy", bbox)
        np.save(f"{ignore_bboxes_dir}/{id}.npy", np.array([]).reshape(0, 4, 2))


def process_synthtext():
    data = loadmat("data/raw/synthtext/gt.mat")
    indices = list(range(data["imnames"].shape[1]))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.05, random_state=42
    )

    train_image_paths = [
        f"data/raw/synthtext/{data["imnames"][0, i].item()}" for i in train_indices
    ]
    val_image_paths = [
        f"data/raw/synthtext/{data["imnames"][0, i].item()}" for i in val_indices
    ]
    train_bboxes = [data["wordBB"][0, i].T.reshape(-1, 4, 2) for i in train_indices]
    val_bboxes = [data["wordBB"][0, i].T.reshape(-1, 4, 2) for i in val_indices]

    train_save_dir = f"data/processed/synthtext/train"
    val_save_dir = f"data/processed/synthtext/val"

    process_synthtext_data(train_image_paths, train_bboxes, train_save_dir)
    process_synthtext_data(val_image_paths, val_bboxes, val_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True)

    args = parser.parse_args()
    if args.dataset == "ic15":
        process_ic15()
    elif args.dataset == "synthtext":
        process_synthtext()
