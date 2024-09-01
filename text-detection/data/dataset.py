import torch
from torch.utils.data import Dataset
import cv2 as cv
import os


class ICDR2015Dataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = f"{data_dir}/processed/icdr2015/images"
        self.gt_dir = f"{data_dir}/processed/icdr2015/gts"

        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = cv.imread(f"{self.image_dir}/image_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        # image = torch.from_numpy(image).to(device='cuda', dtype=torch.float32)
        # image = image.permute(2, 0, 1)

        bin_map = cv.imread(f"{self.gt_dir}/bin_map_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        # bin_map = torch.from_numpy(bin_map).to(device='cuda', dtype=torch.float32)
        prob_map = cv.imread(f"{self.gt_dir}/prob_map_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        # prob_map = torch.from_numpy(prob_map).to(device='cuda', dtype=torch.float32)
        thresh_map = cv.imread(f"{self.gt_dir}/thresh_map_{str(idx).zfill(4)}.jpg", cv.IMREAD_UNCHANGED) / 255
        # thresh_map = torch.from_numpy(thresh_map).to(device='cuda', dtype=torch.float32)
        maps = {"bin_map": bin_map, "prob_map": prob_map, "thresh_map": thresh_map}
        return image, maps
