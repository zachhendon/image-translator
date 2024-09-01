import torch
from torch.utils.data import DataLoader
from data.dataset import ICDR2015Dataset


def get_dataloader(batch_size=32):
    dataset = ICDR2015Dataset("data")
    generator = torch.Generator().manual_seed(42)
    train_dset, val_dset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)

    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader
