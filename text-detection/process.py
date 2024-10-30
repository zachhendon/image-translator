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


def shrink_bboxes(bboxes):
    rate = 0.4**2
    shrunk_bboxes = []

    for bbox in bboxes:
        poly = Polygon(bbox)
        offset = poly.area * (1 - rate) / poly.length
        shrunk_poly = poly.buffer(-offset)
        shrunk_bboxes.append(list(shrunk_poly.exterior.coords)[:4])
    return np.array(shrunk_bboxes).reshape(-1, 4, 2).astype(np.float32)


def process_data(image_paths, gt_paths, subdir):
    os.makedirs(subdir, exist_ok=True)

    images_dir = f"{subdir}/images"
    bboxes_dir = f"{subdir}/bboxes"
    min_bboxes_dir = f"{subdir}/min_bboxes"
    ignore_bboxes_dir = f"{subdir}/ignore_bboxes"
    min_ignore_bboxes_dir = f"{subdir}/min_ignore_bboxes"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)
    os.makedirs(min_bboxes_dir, exist_ok=True)
    os.makedirs(ignore_bboxes_dir, exist_ok=True)
    os.makedirs(min_ignore_bboxes_dir, exist_ok=True)

    for i, (image_path, gt_path) in tqdm(enumerate(zip(image_paths, gt_paths))):
        id = str(i).zfill(6)
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        cv.imwrite(f"{images_dir}/{id}.jpg", image)

        bboxes = []
        ignore_bboxes = []
        with open(gt_path, encoding="utf-8-sig", mode="r") as f:
            for line in f:
                gt = line.rstrip().split(",")
                if gt[8] in ["###", "*"]:
                    ignore_bboxes.append([gt[:8]])
                else:
                    bboxes.append([gt[:8]])
        bboxes = np.array(bboxes, dtype=np.int32).reshape(-1, 4, 2)
        min_bboxes = shrink_bboxes(bboxes)
        np.save(f"{bboxes_dir}/{id}.npy", bboxes)
        np.save(f"{min_bboxes_dir}/{id}.npy", min_bboxes)

        ignore_bboxes = np.array(ignore_bboxes, dtype=np.int32).reshape(-1, 4, 2)
        min_ignore_bboxes = shrink_bboxes(ignore_bboxes)
        np.save(f"{ignore_bboxes_dir}/{id}.npy", ignore_bboxes)
        np.save(f"{min_ignore_bboxes_dir}/{id}.npy", min_ignore_bboxes)


def main(root_dir, save_dir):
    raw_images_dir = f"{root_dir}_images"
    raw_gts_dir = f"{root_dir}_gts"

    # images_dir = f"{save_dir}/images"
    # bboxes_dir = f"{save_dir}/bboxes"
    # min_bboxes_dir = f"{save_dir}/min_bboxes"
    # ignore_bboxes_dir = f"{save_dir}/ignore_bboxes"
    # min_ignore_bboxes_dir = f"{save_dir}/min_ignore_bboxes"
    # gt_kernels_dir = f"{save_dir}/gt_kernels"
    # gt_texts_dir = f"{save_dir}/gt_texts"
    # ignore_kernels_dir = f"{save_dir}/ignore_kernels"
    # ignore_texts_dir = f"{save_dir}/ignore_texts"

    # os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(images_dir, exist_ok=True)
    # os.makedirs(bboxes_dir, exist_ok=True)
    # os.makedirs(min_bboxes_dir, exist_ok=True)
    # os.makedirs(ignore_bboxes_dir, exist_ok=True)
    # os.makedirs(min_ignore_bboxes_dir, exist_ok=True)
    # os.makedirs(gt_kernels_dir, exist_ok=True)
    # os.makedirs(gt_texts_dir, exist_ok=True)
    # os.makedirs(ignore_kernels_dir, exist_ok=True)
    # os.makedirs(ignore_texts_dir, exist_ok=True)

    image_paths = sorted(glob(f"{raw_images_dir}/*"))
    gt_paths = sorted(glob(f"{raw_gts_dir}/*"))
    train_image_paths, val_image_paths = train_test_split(
        image_paths, test_size=0.2, random_state=42
    )
    train_gt_paths, val_gt_paths = train_test_split(
        gt_paths, test_size=0.2, random_state=42
    )
    train_subdir = f"{save_dir}/train"
    val_subdir = f"{save_dir}/val"
    process_data(train_image_paths, train_gt_paths, train_subdir)
    process_data(val_image_paths, val_gt_paths, val_subdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    main(args.root_dir, args.save_dir)
