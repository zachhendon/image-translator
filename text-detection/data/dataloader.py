import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.dataset import *


train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.2, 0.2),
            rotate=(-20, 20),
            shear=(-20, 20),
            p=1.0,
        ),
        A.PadIfNeeded(640, 640, border_mode=cv.BORDER_CONSTANT, value=0, p=1.0),
        A.RandomCrop(640, 640, p=1.0),
        A.OneOf(
            [
                A.AdvancedBlur(p=1.0),
                A.Sharpen(p=1.0),
            ],
            p=1.0,
        ),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
        A.GaussNoise(p=1.0),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)

val_transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2(),
    ],
)


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch])
    gt_kernel = torch.stack([x[1][0] for x in batch])
    gt_text = torch.stack([x[1][1] for x in batch])
    kernel_masks = torch.stack([x[1][2] for x in batch])
    text_masks = torch.stack([x[1][3] for x in batch])
    bboxes = [x[1][4] for x in batch]

    return images, (gt_kernel, gt_text, kernel_masks, text_masks, bboxes)


def get_loaders(datadir, batch_size=16, train=False):
    train_dset = ICDAR2015Dataset2(datadir, train_transform, train=train)
    val_dset = ICDAR2015Dataset2(datadir, val_transform, train=False)

    train_loader = DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_eval_loader(datadir, batch_size=16):
    dataset = ICDAR2015Dataset(datadir)
    dataset = TransformDataset(dataset, val_transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return loader
