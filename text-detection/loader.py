import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_file, decode_jpeg
import kornia.augmentation as K
from kornia.utils import draw_convex_polygon
from glob import glob
import math


class FastDataset(Dataset):
    def __init__(self, datadir, train, ignore):
        self.datadir = datadir
        self.train = train
        self.ignore = ignore
        self.num_images = len(glob(f"{datadir}/images/*.jpg"))

        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )

        self.data_keys = ["image", "keypoints", "keypoints"]
        if ignore:
            self.data_keys += ["keypoints", "keypoints"]
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomAffine(
                degrees=(-20, 20),
                translate=(0.2, 0.2),
                shear=(-20, 20),
                scale=(0.8, 1.2),
                p=1.0,
            ),
            K.RandomPerspective(distortion_scale=0.3, p=0.3),
            K.RandomCrop((640, 640)),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
            K.RandomSharpness(0.5),
            K.RandomGaussianNoise(mean=0, std=0.05, p=1),
            data_keys=self.data_keys,
            same_on_batch=False,
        )
        self.upscale = K.AugmentationSequential(
            K.SmallestMaxSize(640),
            data_keys=self.data_keys,
        )
        self.resize = K.AugmentationSequential(
            K.Resize((640, 640)),
            data_keys=self.data_keys,
        )

    def __len__(self):
        return self.num_images

    def load_bboxes(self, type, idx):
        return torch.load(
            f"{self.datadir}/{type}/{str(idx).zfill(6)}.pt",
            weights_only=True,
            map_location="cuda",
        )

    def preprocess_bboxes(self, bboxes, num_bboxes, batch_size):
        max_len = max(num_bboxes)
        for i, kp in enumerate(bboxes):
            bboxes[i] = torch.cat([kp, torch.zeros(max_len - len(kp), 4, 2).cuda()])
        bboxes = torch.stack(bboxes).reshape(batch_size, -1, 2)
        return bboxes

    def get_masks(self, bboxes, min_bboxes, size):
        batch_bboxes = torch.vstack(bboxes)
        batch_min_bboxes = torch.vstack(min_bboxes)
        if batch_bboxes.shape[0] == 0:
            return (
                torch.zeros((len(bboxes), *size)),
                torch.zeros((len(bboxes), *size)),
            )

        fill_value = torch.tensor([1.0], device="cuda")

        text_masks = torch.zeros((batch_bboxes.shape[0], 1, *size), device="cuda")
        text_masks = draw_convex_polygon(text_masks, batch_bboxes, fill_value)

        kernel_masks = -F.max_pool2d(-text_masks, kernel_size=9, stride=1, padding=4)

        min_kernel_masks = torch.zeros((batch_bboxes.shape[0], 1, *size), device="cuda")
        min_kernel_masks = draw_convex_polygon(
            min_kernel_masks, batch_min_bboxes, fill_value
        )

        kernel_masks = kernel_masks.squeeze(1)
        text_masks = text_masks.squeeze(1)
        min_kernel_masks = min_kernel_masks.squeeze(1)

        batch_kernel_masks = []
        batch_text_masks = []
        l = 0
        for i in range(len(bboxes)):
            r = l + len(bboxes[i])
            if l == r:
                batch_kernel_masks.append(torch.zeros(size, device="cuda"))
                batch_text_masks.append(torch.zeros(size, device="cuda"))
            else:
                batch_kernel_masks.append(
                    torch.clamp(
                        (kernel_masks[l:r] + min_kernel_masks[l:r]).sum(axis=0), 0, 1
                    )
                )
                batch_text_masks.append(torch.clamp(text_masks[l:r].sum(axis=0), 0, 1))
            l = r
        return batch_kernel_masks, batch_text_masks

    def apply_augmentations(self, images, bboxes, min_bboxes, aug):
        if type(images) == list:
            images_aug = []
            bboxes_aug = []
            min_bboxes_aug = []

            for image, bbox, min_bbox in zip(images, bboxes, min_bboxes):
                if len(bbox.shape) == 2:
                    bbox = bbox.unsqueeze(0)
                    min_bbox = min_bbox.unsqueeze(0)
                image, bbox, min_bbox = aug(
                    image, bbox, min_bbox
                )
                images_aug.append(image)
                bboxes_aug.append(bbox)
                min_bboxes_aug.append(min_bbox)
        else:
            images_aug, bboxes_aug, min_bboxes_aug = aug(images, bboxes, min_bboxes)
        return images_aug, bboxes_aug, min_bboxes_aug

    def apply_augmentations_ignore(
        self, images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes, aug
    ):
        if type(images) == list:
            images_aug = []
            bboxes_aug = []
            min_bboxes_aug = []
            ignore_bboxes_aug = []
            min_ignore_bboxes_aug = []

            for image, bbox, min_bbox, ignore_bbox, min_ignore_bbox in zip(
                images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes
            ):
                image, bbox, min_bbox, ignore_bbox, min_ignore_bbox = aug(
                    image,
                    bbox,
                    min_bbox,
                    ignore_bbox,
                    min_ignore_bbox,
                )
                images_aug.append(image)
                bboxes_aug.append(bbox)
                min_bboxes_aug.append(min_bbox)
                ignore_bboxes_aug.append(ignore_bbox)
                min_ignore_bboxes_aug.append(min_ignore_bbox)
        else:
            (
                images_aug,
                bboxes_aug,
                min_bboxes_aug,
                ignore_bboxes_aug,
                min_ignore_bboxes_aug,
            ) = aug(images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes)
        return (
            images_aug,
            bboxes_aug,
            min_bboxes_aug,
            ignore_bboxes_aug,
            min_ignore_bboxes_aug,
        )

    def __getitems__(self, idxs):
        batch_size = len(idxs)

        # load images and bboxes
        bytes = []
        for idx in idxs:
            bytes.append(read_file(f"{self.datadir}/images/{str(idx).zfill(6)}.jpg"))
        images = decode_jpeg(bytes, device="cuda")

        bboxes = [self.load_bboxes("bboxes", idx) for idx in idxs]
        num_bboxes = [len(bb) for bb in bboxes]
        bboxes = self.preprocess_bboxes(bboxes, num_bboxes, batch_size)
        min_bboxes = [self.load_bboxes("min_bboxes", idx) for idx in idxs]
        num_min_bboxes = [len(bb) for bb in min_bboxes]
        min_bboxes = self.preprocess_bboxes(min_bboxes, num_min_bboxes, batch_size)

        if self.ignore:
            ignore_bboxes = [self.load_bboxes("ignore_bboxes", idx) for idx in idxs]
            num_ignore_bboxes = [len(bb) for bb in ignore_bboxes]
            ignore_bboxes = self.preprocess_bboxes(
                ignore_bboxes, num_ignore_bboxes, batch_size
            )
            min_ignore_bboxes = [
                self.load_bboxes("min_ignore_bboxes", idx) for idx in idxs
            ]
            num_min_ignore_bboxes = [len(bb) for bb in min_ignore_bboxes]
            min_ignore_bboxes = self.preprocess_bboxes(
                min_ignore_bboxes, num_min_ignore_bboxes, batch_size
            )

        # handle images of different sizes
        equal_size = True
        size = images[0].shape[1:]
        for img in images[1:]:
            if img.shape[1:] != size:
                equal_size = False
                break
        if equal_size:
            images = torch.stack(images) / 255
            images = self.normalize(images)
        else:
            images = [self.normalize(img / 255) for img in images]
            bboxes = list(bboxes)
            min_bboxes = list(min_bboxes)
            if self.ignore:
                ignore_bboxes = list(ignore_bboxes)
                min_ignore_bboxes = list(min_ignore_bboxes)
                images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes = (
                    self.apply_augmentations_ignore(
                        images,
                        bboxes,
                        min_bboxes,
                        ignore_bboxes,
                        min_ignore_bboxes,
                        self.upscale,
                    )
                )
            else:
                images, bboxes, min_bboxes = self.apply_augmentations(
                    images,
                    bboxes,
                    min_bboxes,
                    self.upscale,
                )

        # apply augmentations
        if self.train:
            if self.ignore:
                images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes = (
                    self.apply_augmentations_ignore(
                        images,
                        bboxes,
                        min_bboxes,
                        ignore_bboxes,
                        min_ignore_bboxes,
                        self.aug,
                    )
                )
                if type(images) == list:
                    images = torch.cat(images)
                    bboxes = torch.cat(bboxes)
                    min_bboxes = torch.cat(min_bboxes)
                    ignore_bboxes = torch.cat(ignore_bboxes)
                    min_ignore_bboxes = torch.cat(min_ignore_bboxes)
            else:
                images, bboxes, min_bboxes = self.apply_augmentations(
                    images,
                    bboxes,
                    min_bboxes,
                    self.aug,
                )
                if type(images) == list:
                    images = torch.cat(images)
                    bboxes = torch.cat(bboxes)
                    min_bboxes = torch.cat(min_bboxes)
        else:
            if type(images) == list:
                h = sum([img.shape[2] for img in images]) // len(images)
                w = sum([img.shape[3] for img in images]) // len(images)
            else:
                h, w = images[0].shape[2:]
            
            long_size = math.ceil((w * 640 / h) / 4) * 4
            resize = K.AugmentationSequential(
                K.Resize((640, long_size)),
                data_keys=self.data_keys,
                same_on_batch=True,
            )
            if self.ignore:
                images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes = (
                    self.apply_augmentations_ignore(
                        images,
                        bboxes,
                        min_bboxes,
                        ignore_bboxes,
                        min_ignore_bboxes,
                        resize,
                    )
                )
                if type(images) == list:
                    images = torch.cat(images)
                    bboxes = torch.cat(bboxes)
                    min_bboxes = torch.cat(min_bboxes)
                    ignore_bboxes = torch.cat(ignore_bboxes)
                    min_ignore_bboxes = torch.cat(min_ignore_bboxes)
            else:
                images, bboxes, min_bboxes = self.apply_augmentations(
                    images,
                    bboxes,
                    min_bboxes,
                    resize,
                )
                if type(images) == list:
                    images = torch.cat(images)
                    bboxes = torch.cat(bboxes)
                    min_bboxes = torch.cat(min_bboxes)
        # process augmented bboxes
        bboxes = [
            bb[: num_bboxes[i]]
            for i, bb in enumerate(bboxes.reshape(batch_size, -1, 4, 2))
        ]
        min_bboxes = [
            bb[: num_min_bboxes[i]]
            for i, bb in enumerate(min_bboxes.reshape(batch_size, -1, 4, 2))
        ]
        if self.ignore:
            ignore_bboxes = [
                bb[: num_ignore_bboxes[i]]
                for i, bb in enumerate(ignore_bboxes.reshape(batch_size, -1, 4, 2))
            ]
            min_ignore_bboxes = [
                bb[: num_min_ignore_bboxes[i]]
                for i, bb in enumerate(min_ignore_bboxes.reshape(batch_size, -1, 4, 2))
            ]

        # get masks
        size = images.shape[2:]
        kernel_masks, text_masks = self.get_masks(bboxes, min_bboxes, size)
        if self.ignore:
            ignore_kernel_masks, ignore_text_masks = self.get_masks(
                ignore_bboxes, min_ignore_bboxes, size
            )
            ignore_kernel_masks = [1 - mask for mask in ignore_kernel_masks]
            ignore_text_masks = [1 - mask for mask in ignore_text_masks]
        else:
            ignore_kernel_masks = [torch.ones(size, device="cuda") for _ in bboxes]
            ignore_text_masks = [torch.ones(size, device="cuda") for _ in bboxes]

        return (
            list(images),
            kernel_masks,
            ignore_kernel_masks,
            text_masks,
            ignore_text_masks,
        )


def collate_fn(batch):
    images = torch.stack(batch[0])
    text_masks = torch.stack(batch[1])
    kernel_masks = torch.stack(batch[2])
    ignore_text_masks = torch.stack(batch[3])
    ignore_kernel_masks = torch.stack(batch[4])
    return images, text_masks, kernel_masks, ignore_text_masks, ignore_kernel_masks


class DataLoaderIterator:
    def __init__(self, dataset, batch_size=16, train=True):
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, collate_fn=collate_fn
        )
        self.iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)


def get_icdar2015_loaders(batch_size=16):
    train_dataset = FastDataset(
        "data/processed/icdar2015/train", train=True, ignore=True
    )
    val_dataset = FastDataset("data/processed/icdar2015/val", train=False, ignore=True)
    return (
        DataLoaderIterator(
            train_dataset,
            batch_size,
            train=True,
        ),
        DataLoaderIterator(
            val_dataset,
            batch_size,
            train=False,
        ),
    )


def get_synthtext_loaders(batch_size=16):
    train_dataset = FastDataset(
        "data/processed/synthtext/train", train=True, ignore=False
    )
    val_dataset = FastDataset("data/processed/synthtext/val", train=False, ignore=False)
    return (
        DataLoaderIterator(
            train_dataset,
            batch_size,
            train=True,
        ),
        DataLoaderIterator(
            val_dataset,
            batch_size,
            train=False,
        ),
    )
