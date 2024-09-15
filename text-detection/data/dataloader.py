import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.dataset import *


train_transform = A.ReplayCompose([
    A.Rotate((-10, 10)),
    A.RandomCrop(640, 640),
    ToTensorV2()
])
val_transform = A.ReplayCompose([
    A.Resize(640, 640),
    ToTensorV2()
])


def get_loaders(datadir, batch_size=16, train=False):
    dataset = ICDR2015Dataset(datadir)
    generator = torch.Generator().manual_seed(42)
    train_dset, val_dset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator)
    
    if train:
        train_dset = TransformDataset(train_dset, train_transform)
    else:
        train_dset = TransformDataset(train_dset, val_transform)
    train_loader = DataLoader(train_dset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_dset = TransformDataset(val_dset, val_transform)
    val_loader = DataLoader(val_dset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def get_eval_loader(datadir, batch_size=16):
    dataset = ICDR2015Dataset(datadir)
    dataset = TransformDataset(dataset, val_transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    return loader
