from torch.utils.data import Dataset
import cv2 as cv
import os
import albumentations as A
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import random


class ICDR2015Dataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = f"{data_dir}/processed/icdr2015/images"
        self.gt_dir = f"{data_dir}/processed/icdr2015/gts"

        self.num_images = len(os.listdir(self.image_dir))
        self.train = True
        self.normalize = A.Normalize(mean=[0.3315, 0.3530, 0.3724],
                                     std=[0.2262, 0.2232, 0.2290])
        self.train = True

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        maps = {}

        image = cv.imread(
            f"{self.image_dir}/image_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED)
        image = self.normalize(image=image)['image']

        gt_map = cv.imread(
            f"{self.gt_dir}/gt_map_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        maps['gt_map'] = gt_map

        thresh_map = cv.imread(
            f"{self.gt_dir}/thresh_map_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        maps['thresh_map'] = thresh_map
        bin_map = cv.imread(
            f"{self.gt_dir}/bin_map_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        maps['bin_map'] = bin_map
        return image, maps


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, maps = self.dataset[idx]
        transform_data = self.transform(image=image)
        image = transform_data['image'].cuda()

        for key, val in maps.items():
            maps[key] = A.ReplayCompose.replay(transform_data['replay'], image=val)[
                'image'].squeeze()
        return image, maps


class ICDR2015Dataset2(Dataset):
    def __init__(self, data_dir):
        self.image_dir = f"{data_dir}/processed/icdr2015/images"
        self.gt_dir = f"{data_dir}/processed/icdr2015/gts"

        self.num_images = len(os.listdir(self.image_dir))
        self.train = True
        self.normalize = v2.Normalize(mean=[0.3315, 0.3530, 0.3724],
                                      std=[0.2262, 0.2232, 0.2290])
        self.train = True

    def __len__(self):
        return self.num_images

    def __getitems__(self, idxs):
        images_bytes = []
        gt_maps = []
        thresh_maps = []
        bin_maps = []

        for idx in idxs:
            image_path = f"{self.image_dir}/image_{str(idx).zfill(4)}.jpg"
            with open(image_path, 'rb') as f:
                image_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
            images_bytes.append(image_bytes)

            with open(f"{self.gt_dir}/gt_map_{str(idx).zfill(4)}.jpg", 'rb') as f:
                gt_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
            gt_map = torchvision.io.decode_jpeg(gt_bytes, device='cuda') / 255
            gt_maps.append(gt_map)

            with open(f"{self.gt_dir}/thresh_map_{str(idx).zfill(4)}.jpg", 'rb') as f:
                thresh_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
            thresh_map = torchvision.io.decode_jpeg(
                thresh_bytes, device='cuda') / 255
            thresh_maps.append(thresh_map)

            with open(f"{self.gt_dir}/bin_map_{str(idx).zfill(4)}.jpg", 'rb') as f:
                bin_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
            bin_map = torchvision.io.decode_jpeg(
                bin_bytes, device='cuda') / 255
            bin_maps.append(bin_map)

        images = torchvision.io.decode_jpeg(images_bytes, device='cuda')
        collated_images = torch.utils.data.default_collate(images) / 255
        collated_images = self.normalize(collated_images)

        maps = {}
        collated_gt = torch.utils.data.default_collate(gt_maps)
        maps['gt_map'] = collated_gt
        collated_thresh = torch.utils.data.default_collate(thresh_maps)
        maps['thresh_map'] = collated_thresh
        collated_bin = torch.utils.data.default_collate(bin_maps)
        maps['bin_map'] = collated_bin

        return collated_images, maps

    def __getitem__(self, idx):
        maps = {}

        with open(f"{self.image_dir}/image_{str(idx).zfill(4)}.jpg", 'rb') as f:
            image_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
        image = torchvision.io.decode_jpeg(image_bytes, device='cuda') / 255
        image = self.normalize(image)

        with open(f"{self.gt_dir}/gt_map_{str(idx).zfill(4)}.jpg", 'rb') as f:
            gt_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
        gt_map = torchvision.io.decode_jpeg(gt_bytes, device='cuda') / 255
        maps['gt_map'] = gt_map

        with open(f"{self.gt_dir}/thresh_map_{str(idx).zfill(4)}.jpg", 'rb') as f:
            thresh_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
        thresh_map = torchvision.io.decode_jpeg(
            thresh_bytes, device='cuda') / 255
        maps['thresh_map'] = thresh_map

        with open(f"{self.gt_dir}/bin_map_{str(idx).zfill(4)}.jpg", 'rb') as f:
            bin_bytes = torch.frombuffer(f.read(), dtype=torch.uint8)
        bin_map = torchvision.io.decode_jpeg(bin_bytes, device='cuda') / 255
        maps['bin_map'] = bin_map

        return image, maps


def apply_transformation(images, maps, transform_fn, **kwargs):
    images = transform_fn(images, **kwargs)
    for key, val in maps.items():
        maps[key] = transform_fn(val, **kwargs)
    return images, maps


class TransformDataset2(Dataset):
    def __init__(self, dataset, train=True):
        self.dataset = dataset
        self.train = train

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, idx):
    #     image, maps = self.dataset[idx]
    #     image = self.transform(image)

    #     for key, val in maps.items():
    #         maps[key] = self.transform(val)
    #     return image, maps

    def __getitems__(self, idxs):
        images, maps = self.dataset.__getitems__(idxs)

        if self.train:
            rot = random.random() * 20 - 10
            images, maps = apply_transformation(images, maps, TF.rotate, angle=rot)

            top = random.randrange(0, 960 - 640 + 1)
            left = random.randrange(0, 960 - 640 + 1)
            images, maps = apply_transformation(
                images, maps, TF.crop, top=top, left=left, height=640, width=640)
        else:
            images, maps = apply_transformation(images, maps, TF.resize, size=(640, 640))

        return images, maps
