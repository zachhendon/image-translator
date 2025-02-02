import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2 as cv
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon
import time
from torchvision.io import read_file, decode_jpeg
import kornia.augmentation as K
from kornia.utils import draw_convex_polygon
import math


def random_scale(image, scales=[0.5, 2.0], aspects=[0.9, 1.1]):
    scale = np.random.uniform(scales[0], scales[1])
    h, w = image.shape[2:]
    if aspects:
        aspect = np.random.uniform(aspects[0], aspects[1])
        h_scale = scale * aspect
        w_scale = scale / aspect
    else:
        h_scale = scale
        w_scale = scale

    image = F.interpolate(image, size=(int(h * h_scale), int(w * w_scale)))
    return image, (h_scale, w_scale)


def random_crop(image, gt_instances, crop_size=(640, 640)):
    h, w = image.shape[1:]
    h_crop, w_crop = crop_size
    device = "cuda"

    unique_instances = torch.unique(gt_instances)
    num_instances = len(unique_instances)

    if num_instances == 0:
        p = torch.ones((num_instances, h, w), device=device)
    else:
        y_indices = torch.arange(h, device=device).view(-1, 1)
        x_indices = torch.arange(w, device=device).view(1, -1)
        num_y = torch.clamp(y_indices + 1, max=h_crop)
        num_x = torch.clamp(x_indices + 1, max=w_crop)
        factors = ((h_crop * w_crop) - (num_y * num_x)) / (
            torch.clamp(y_indices + 1, max=h_crop)
            * torch.clamp(x_indices + 1, max=w_crop)
        ) + 1

        instance_contributions = []
        for val in unique_instances[1:]:
            mask = gt_instances == val
            M = mask.float() * factors

            integral = torch.zeros((h + 1, w + 1), device=device)
            integral[1:, 1:] = torch.cumsum(torch.cumsum(M, 0), 1)

            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )

            y2 = torch.clamp(yy + h_crop - 1, max=h - 1)
            x2 = torch.clamp(xx + w_crop - 1, max=w - 1)

            sum_contribution = (
                integral[y2 + 1, x2 + 1]
                - integral[yy, x2 + 1]
                - integral[y2 + 1, xx]
                + integral[yy, xx]
            )

            instance_pixels = mask.sum()
            if instance_pixels > 0:
                sum_contribution /= instance_pixels
                instance_contributions.append(sum_contribution)

        p = torch.clamp(
            (
                torch.stack(instance_contributions).sum(dim=0)
                if instance_contributions
                else torch.zeros((h, w), device=device)
            ),
            min=0,
        )

    if p.sum() == 0:
        p = torch.ones((h, w), device=device)
    p /= p.sum()
    p = p**2

    p[h - h_crop + 1 :, :] = 0
    p[:, w - w_crop + 1 :] = 0
    p /= p.sum()

    coord = torch.multinomial(p.view(-1), 1).item()
    y_coord = coord // w
    x_coord = coord % w
    return y_coord, x_coord


def scale_bboxes(bboxes, scale):
    if type(scale) == int:
        h_scale = scale
        w_scale = scale
    else:
        h_scale, w_scale = scale

    bboxes[:, :, 0] *= w_scale
    bboxes[:, :, 1] *= h_scale
    return bboxes


def shrink_bboxes(bboxes):
    rate = 0.1**2
    shrunk_bboxes = []
    for bbox in bboxes:
        poly = Polygon(bbox.cpu().numpy())
        offset = poly.area * (1 - rate) / poly.length
        shrunk_poly = poly.buffer(-offset)
        if shrunk_poly.is_empty:
            shrunk_bboxes.append(bbox)
            continue
        shrunk_bboxes.append(list(shrunk_poly.exterior.coords)[:4])
    shrunk_bboxes = np.array(shrunk_bboxes).reshape(-1, 4, 2).astype(np.int32)
    return torch.from_numpy(shrunk_bboxes).to(device="cuda")


class FAST_IC15(Dataset):
    def __init__(self, type, short_size=640):
        self.type = type
        self.short_size = short_size
        self.data_keys = ["image", "keypoints", "keypoints"]
        self.transforms = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=30, shear=20, translate=(0.1, 0.1), p=1.0),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
            same_on_batch=False,
            data_keys=self.data_keys,
        )

    def __len__(self):
        return len(glob(f"data/processed/ic15/{self.type}/images/*"))

    def __getitem__(self, idx):
        # load image and bboxes
        id = str(idx).zfill(6)
        bytes = read_file(f"data/processed/ic15/{self.type}/images/{id}.jpg")
        image = decode_jpeg(bytes, device="cuda") / 255
        image = image.unsqueeze(0)
        bboxes = torch.from_numpy(
            np.load(f"data/processed/ic15/{self.type}/bboxes/{id}.npy")
        ).to(device="cuda", dtype=torch.float32)
        ignore_bboxes = torch.from_numpy(
            np.load(f"data/processed/ic15/{self.type}/ignore_bboxes/{id}.npy")
        ).to(device="cuda", dtype=torch.float32)

        # scale/pad image and bboxes for training
        if self.type == "train":
            image, (h_scale, w_scale) = random_scale(image, scales=[0.5, 2.0])
            bboxes = scale_bboxes(bboxes, (h_scale, w_scale))
            ignore_bboxes = scale_bboxes(ignore_bboxes, (h_scale, w_scale))

        if self.type == "train":
            margin_h = max(0, self.short_size - image.shape[2])
            margin_w = max(0, self.short_size - image.shape[3])
            pad_top = np.random.randint(0, margin_h + 1)
            pad_bottom = margin_h - pad_top
            pad_left = np.random.randint(0, margin_w + 1)
            pad_right = margin_w - pad_left
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            image = F.pad(image, padding, value=0)
            bboxes[:, :, 0] += pad_left
            bboxes[:, :, 1] += pad_top
            ignore_bboxes[:, :, 0] += pad_left
            ignore_bboxes[:, :, 1] += pad_top

        # image augmentations
        if self.type == "train":
            image, bboxes, ignore_bboxes = self.transforms(image, bboxes, ignore_bboxes)
        else:
            h, w = image.shape[2:]
            if h <= w:
                long_size = math.ceil((w * self.short_size / h) / 4) * 4
                new_size = (self.short_size, long_size)
            else:
                long_size = math.ceil((h * self.short_size / w) / 4) * 4
                new_size = (long_size, self.short_size)
            resize = K.AugmentationSequential(
                K.Resize(new_size),
                same_on_batch=True,
                data_keys=self.data_keys,
            )
            image, bboxes, ignore_bboxes = resize(image, bboxes, ignore_bboxes)

        # create masks
        min_bboxes = shrink_bboxes(bboxes)
        gt_instances = torch.zeros(
            (1, 1, *image.shape[2:]), dtype=torch.int8, device="cuda"
        )
        training_mask = torch.ones(
            (1, 1, *image.shape[2:]), dtype=torch.int8, device="cuda"
        )
        for i, bbox in enumerate(bboxes):
            gt_instances = draw_convex_polygon(
                gt_instances,
                [bbox],
                torch.tensor([i + 1], dtype=torch.uint8, device="cuda"),
            )
        for bbox in ignore_bboxes:
            training_mask = draw_convex_polygon(
                training_mask,
                [bbox],
                torch.tensor([0], dtype=torch.uint8, device="cuda"),
            )
        gt_instances = gt_instances.squeeze(0).squeeze(0)
        training_mask = training_mask.squeeze(0).squeeze(0)

        gt_kernels = [
            torch.zeros((1, 1, *image.shape[2:]), dtype=torch.int8, device="cuda")
        ]
        for bbox, min_bbox in zip(bboxes, min_bboxes):
            gt_kernel = torch.zeros(
                (1, 1, *image.shape[2:]), dtype=torch.int8, device="cuda"
            )
            gt_kernel = draw_convex_polygon(
                gt_kernel, [bbox], torch.tensor([1], dtype=torch.uint8, device="cuda")
            )
            gt_kernels.append(gt_kernel)
        gt_kernels = torch.cat(gt_kernels, dim=1).to(dtype=torch.float16)
        overlap = (gt_kernels.sum(dim=1) > 1).to(dtype=torch.float16)
        overlap = F.max_pool2d(overlap, kernel_size=3, stride=1, padding=1)
        gt_kernel = -F.max_pool2d(-gt_kernels, kernel_size=9, stride=1, padding=4)
        gt_kernel = torch.clamp(torch.sum(gt_kernel, dim=1), 0, 1)
        gt_kernel[overlap > 0] = 0
        gt_kernel = gt_kernel.unsqueeze(0)
        for min_bbox in min_bboxes:
            gt_kernel = draw_convex_polygon(
                gt_kernel,
                [min_bbox],
                torch.tensor([1], dtype=torch.uint8, device="cuda"),
            )
        gt_kernel = gt_kernel.squeeze(0).squeeze(0)

        gt_text = torch.clamp(gt_instances, 0, 1)
        image = image.squeeze(0)

        # random crop
        if self.type == "train":
            y_coord, x_coord = random_crop(
                image, gt_instances, crop_size=(self.short_size, self.short_size)
            )
            image = image[
                :,
                y_coord : y_coord + self.short_size,
                x_coord : x_coord + self.short_size,
            ]
            gt_kernel = gt_kernel[
                y_coord : y_coord + self.short_size, x_coord : x_coord + self.short_size
            ]
            gt_text = gt_text[
                y_coord : y_coord + self.short_size, x_coord : x_coord + self.short_size
            ]
            training_mask = training_mask[
                y_coord : y_coord + self.short_size, x_coord : x_coord + self.short_size
            ]
            gt_instances = gt_instances[
                y_coord : y_coord + self.short_size, x_coord : x_coord + self.short_size
            ]

        # convert to float32
        image = image.to(dtype=torch.float32)
        gt_kernel = gt_kernel.to(dtype=torch.float32)
        gt_text = gt_text.to(dtype=torch.float32)
        training_mask = training_mask.to(dtype=torch.float32)
        gt_instances = gt_instances.to(dtype=torch.float32)

        data = {
            "images": image,
            "gt_kernels": gt_kernel,
            "gt_texts": gt_text,
            "training_masks": training_mask,
            "gt_instances": gt_instances,
        }
        return data


class DataLoaderIterator:
    def __init__(self, dataset, shuffle, batch_size=16):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
        self.iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)

    def __len__(self):
        return len(self.loader)
