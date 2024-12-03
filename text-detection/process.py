import cv2 as cv
from glob import glob
import numpy as np
import os
import json
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
    return np.array(shrunk_bboxes).astype(np.float32)


def get_bbox(polygons):
    xmin = np.min(polygons[:, 0])
    xmax = np.max(polygons[:, 0])
    ymin = np.min(polygons[:, 1])
    ymax = np.max(polygons[:, 1])
    return np.array([xmin, ymin, xmax - xmin, ymax - ymin])


def add_annotation(annotations, polygons, category_id, image_id, ann_cnt):
    for poly in polygons:
        annotations["annotations"].append(
            {
                "id": ann_cnt,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": get_bbox(poly).tolist(),
                "segmentation": poly.reshape(-1, 8).tolist(),
            }
        )
        ann_cnt += 1
    return ann_cnt


def process_data(image_paths, gt_paths, subdir):
    os.makedirs(subdir, exist_ok=True)
    os.makedirs(f"{subdir}/images", exist_ok=True)

    annotations = {
        "categories": [
            {"id": 1, "name": "bboxes"},
            {"id": 2, "name": "min_bboxes"},
            {"id": 3, "name": "ignore_bboxes"},
            {"id": 4, "name": "min_ignore_bboxes"},
        ],
        "images": [],
        "annotations": [],
    }

    ann_cnt = 1
    for i, (image_path, gt_path) in tqdm(enumerate(zip(image_paths, gt_paths))):
        id = str(i).zfill(6)
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        cv.imwrite(f"{subdir}/images/{id}.jpg", image)

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
        ignore_bboxes = np.array(ignore_bboxes, dtype=np.int32).reshape(-1, 4, 2)
        min_bboxes = shrink_bboxes(bboxes)
        min_ignore_bboxes = shrink_bboxes(ignore_bboxes)

        # add image and bboxes to annotations dictionary (COCO format)
        annotations["images"].append(
            {
                "id": i + 1,
                "file_name": f"{id}.jpg",
                "width": image.shape[1],
                "height": image.shape[0],
            }
        )

        ann_cnt = add_annotation(annotations, bboxes, 1, i + 1, ann_cnt)
        ann_cnt = add_annotation(annotations, min_bboxes, 2, i + 1, ann_cnt)
        ann_cnt = add_annotation(annotations, ignore_bboxes, 3, i + 1, ann_cnt)
        ann_cnt = add_annotation(annotations, min_ignore_bboxes, 4, i + 1, ann_cnt)

    with open(f"{subdir}/annotations.json", "w") as f:
        json.dump(annotations, f)


def process_icdar2015_coco():
    # process training/validation data
    root_dir = "data/raw/icdar2015/train"
    save_dir = "data/processed/icdar2015_coco"
    raw_images_dir = f"{root_dir}_images"
    raw_gts_dir = f"{root_dir}_gts"

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
    parser.add_argument("--dataset", "-d", type=str, required=True)

    args = parser.parse_args()
    if args.dataset == "icdar2015":
        process_icdar2015_coco()
