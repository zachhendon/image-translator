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
        image = cv.imread(f"{self.image_dir}/image_{str(idx).zfill(4)}.jpg")
        bin_map = cv.imread(f"{self.gt_dir}/bin_map_{str(idx).zfill(4)}.jpg")
        prob_map = cv.imread(f"{self.gt_dir}/prob_map_{str(idx).zfill(4)}.jpg")
        thresh_map = cv.imread(f"{self.gt_dir}/thresh_map_{str(idx).zfill(4)}.jpg")
        return image, (bin_map, prob_map, thresh_map)
