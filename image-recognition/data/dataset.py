import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class ICDR2015Dataset(Dataset):
    def __init__(self, datadir, train=True):
        if train:
            imagedir = f"{datadir}/train_images"
            gtdir = f"{datadir}/train_gts"
        else:
            imagedir = f"{datadir}/test_images"
            gtdir = f"{datadir}/test_gts"

        self.image_paths = [f"{imagedir}/img_{i+1}.jpg" for i in range(len(os.listdir(imagedir)))]
        self.gt_paths = [f"{gtdir}/gt_img_{i+1}.txt" for i in range(len(os.listdir(gtdir)))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        gt = pd.read_csv(self.gt_paths[idx], header=None)
        gt = gt.to_numpy()
        return image, gt
        