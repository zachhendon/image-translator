import torch
from torch.utils.data import DataLoader
from data.dataset import *


def collate_fn(batch):
    return batch


def get_loaders(datadir, batch_size=16, train=False):
    dataset = ICDR2015Dataset2(datadir)
    generator = torch.Generator().manual_seed(42)
    train_dset, val_dset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator)

    if train:
        train_dset = TransformDataset2(train_dset, train=True)
    else:
        train_dset = TransformDataset2(train_dset, train=True)
    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dset = TransformDataset2(val_dset, train=False)
    val_loader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def get_eval_loader(datadir, batch_size=16):
    dataset = ICDR2015Dataset2(datadir)
    dataset = TransformDataset2(dataset, train=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return loader
