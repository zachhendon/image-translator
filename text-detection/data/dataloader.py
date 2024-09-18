import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.dataset import *


# train_transform = A.Compose(
#     [
#         A.HorizontalFlip(),
#         A.VerticalFlip(),
#         A.ShiftScaleRotate(),
#         # A.RandomResizedCrop(
#         #     size=(640, 640),
#         #     scale=(0.3, 1.0),
#         #     ratio=(2 / 3, 3 / 2),
#         # ),
#         # A.RandomResizedCrop(
#         #     size=(640, 640),
#         #     scale=(0.7, 1.0),
#         #     ratio=(3 / 4, 4 / 3),
#         # ),
#         A.Resize(640, 640),
#         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5),
#         # A.AdvancedBlur(),
#         # A.Sharpen(),
#         A.GaussNoise(p=1.0),
#         A.Normalize(),
#         # ToTensorV2(),
#     ],
#     keypoint_params=A.KeypointParams(format="xy"),
# )


train_transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Affine(
            scale=(2 / 3, 3 / 2),
            translate_percent=(-0.2, 0.2),
            rotate=(-45, 45),
            shear=(-10, 10),
            balanced_scale=True,
        ),
        A.RandomResizedCrop(size=(640, 640), scale=(0.9, 1.0), ratio=(2 / 3, 3 / 2)),
        A.Normalize(),
        # A.GaussNoise(p=1.0),
        # A.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0),
        # A.AdvancedBlur(),
        # A.Sharpen(),
    ],
    # keypoint_params=A.KeypointParams(format="xy"),
)

val_transform = A.Compose(
    [
        A.Resize(640, 640),
        A.Normalize(),
        # ToTensorV2(),
    ],
    # keypoint_params=A.KeypointParams(format="xy"),
)


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch])

    labels = {}
    labels["gt_text"] = torch.stack([x[1]["gt_text"] for x in batch])
    labels["gt_kernel"] = torch.stack([x[1]["gt_kernel"] for x in batch])
    labels["bboxes"] = [x[1]["bboxes"] for x in batch]
    return images, labels


def get_loaders(datadir, batch_size=16, train=False):
    dataset = ICDR2015Dataset(datadir)
    # dataset = SynthtextDataset(datadir)
    generator = torch.Generator().manual_seed(42)
    train_dset, val_dset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)
    # train_dset, val_dset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator)

    if train:
        train_dset = TransformDataset(train_dset, train_transform)
    else:
        train_dset = TransformDataset(train_dset, val_transform)
    train_loader = DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dset = TransformDataset(val_dset, val_transform)
    val_loader = DataLoader(
        val_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_loaders2(datadir, batch_size=16, train=False):
    pass


def get_eval_loader(datadir, batch_size=16):
    dataset = ICDR2015Dataset(datadir)
    dataset = TransformDataset(dataset, val_transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return loader
