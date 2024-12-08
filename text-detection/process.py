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
    rate = 0.4**2
    shrunk_bboxes = []
    for bbox in bboxes:
        poly = Polygon(bbox)
        offset = poly.area * (1 - rate) / poly.length
        shrunk_poly = poly.buffer(-offset)
        shrunk_bboxes.append(list(shrunk_poly.exterior.coords)[:4])
    return np.array(shrunk_bboxes).reshape(-1, 4, 2).astype(np.float32)


def process_icdar2015_data(image_paths, gt_paths, subdir):
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
        min_bboxes = get_min_bboxes(bboxes)
        torch.save(torch.from_numpy(bboxes).float(), f"{bboxes_dir}/{id}.pt")
        torch.save(torch.from_numpy(min_bboxes).float(), f"{min_bboxes_dir}/{id}.pt")
        ignore_bboxes = np.array(ignore_bboxes, dtype=np.int32).reshape(-1, 4, 2)
        min_ignore_bboxes = get_min_bboxes(ignore_bboxes)
        torch.save(
            torch.from_numpy(ignore_bboxes).float(), f"{ignore_bboxes_dir}/{id}.pt"
        )
        torch.save(
            torch.from_numpy(min_ignore_bboxes).float(),
            f"{min_ignore_bboxes_dir}/{id}.pt",
        )


def process_icdar2015(save_dir):
    raw_images_dir = "data/raw/icdar2015/train_images"
    raw_gts_dir = f"data/raw/icdar2015/train_gts"
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

    process_icdar2015_data(train_image_paths, train_gt_paths, train_subdir)
    process_icdar2015_data(val_image_paths, val_gt_paths, val_subdir)


def process_synthtext_data(image_paths, bboxes, subdir):
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

    for i, (image_path, bbox) in enumerate(tqdm(zip(image_paths, bboxes))):
        id = str(i).zfill(6)
        image = cv.imread(image_path)
        cv.imwrite(f"{images_dir}/{id}.jpg", image)

        min_bbox = get_min_bboxes(bbox)
        torch.save(torch.from_numpy(bbox).float(), f"{bboxes_dir}/{id}.pt")
        torch.save(torch.from_numpy(min_bbox).float(), f"{min_bboxes_dir}/{id}.pt")

        torch.save(torch.tensor([]).reshape(0, 4, 2), f"{ignore_bboxes_dir}/{id}.pt")
        torch.save(torch.tensor([]).reshape(0, 4, 2), f"{min_ignore_bboxes_dir}/{id}.pt")


def process_synthtext(save_dir):
    data = loadmat("data/raw/synthtext/gt.mat")
    indices = list(range(data["imnames"].shape[1]))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, random_state=42
    )

    train_img_paths = [
        f"data/raw/synthtext/{data["imnames"][0, i].item()}" for i in train_indices
    ]
    val_img_paths = [
        f"data/raw/synthtext/{data["imnames"][0, i].item()}" for i in val_indices
    ]
    train_bboxes = [data["wordBB"][0, i].T.reshape(-1, 4, 2) for i in train_indices]
    val_bboxes = [data["wordBB"][0, i].T.reshape(-1, 4, 2) for i in val_indices]

    train_subdir = f"{save_dir}/train"
    val_subdir = f"{save_dir}/val"

    # process_synthtext_data(train_img_paths, train_bboxes, train_subdir)
    process_synthtext_data(val_img_paths, val_bboxes, val_subdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    args = parser.parse_args()
    save_dir = f"data/processed/{args.save}"
    if args.dataset == "icdar2015":
        process_icdar2015(save_dir)
    elif args.dataset == "synthtext":
        process_synthtext(save_dir)
