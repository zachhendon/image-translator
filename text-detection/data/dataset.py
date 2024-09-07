from torch.utils.data import Dataset
import cv2 as cv
import os
import albumentations as A


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
        image = transform_data['image']

        for key, val in maps.items():
            maps[key] = A.ReplayCompose.replay(transform_data['replay'], image=val)['image'].squeeze()
        return image, maps

        