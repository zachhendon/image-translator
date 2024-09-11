from torch.utils.data import Dataset
import albumentations as A
import torch
import torchvision
import lmdb
import numpy as np
from torch.utils.data import Dataset


class ICDR2015Dataset(Dataset):
    def __init__(self, datadir):
        env = lmdb.open(f'{datadir}/processed/lmdb_dataset', map_size=2**30)
        self.txn = env.begin()

    def __len__(self):
        return 1500

    def get_images(self, path, idxs):
        image_tensors = []
        for idx in idxs:
            image_bytes = self.txn.get(f'{path}_{idx}'.encode())
            image_tensor = torch.frombuffer(
                bytearray(image_bytes), dtype=torch.uint8)
            image_tensors.append(image_tensor)
        images = torchvision.io.decode_jpeg(image_tensors)
        images = [img.permute(1, 2, 0).numpy() for img in images]
        return images

    def __getitems__(self, idxs):
        idxs = [str(idx).zfill(4) for idx in idxs]
        images = self.get_images('icdr2015_image', idxs)

        maps = {}
        maps['gt_map'] = self.get_images('icdr2015_gt_map', idxs)
        maps['thresh_map'] = self.get_images('icdr2015_thresh_map', idxs)
        maps['bin_map'] = self.get_images('icdr2015_bin_map', idxs)

        return images, maps


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.normalize = A.Normalize(mean=[0.3315, 0.3530, 0.3724],
                                     std=[0.2262, 0.2232, 0.2290])

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, idxs):
        images, maps = self.dataset.__getitems__(idxs)
        images = self.normalize(images=images)['images']

        transform_data = self.transform(images=images)
        images = torch.from_numpy(
            np.array(transform_data['images'])).permute(0, 3, 1, 2)
        for key, val in maps.items():
            maps[key] = torch.from_numpy(np.array(
                A.ReplayCompose.replay(
                    transform_data['replay'], images=val
                )['images']
            )).permute(0, 3, 1, 2) / 255

        return images, maps


class SimpleICDR2015Dataset(Dataset):
    def __init__(self, datadir):
        self.env = lmdb.open(
            f'{datadir}/processed/lmdb_dataset', map_size=2**30)

    def __len__(self):
        return 1500

    def get_image(self, path):
        with self.env.begin() as txn:
            image_bytes = txn.get(path.encode())
            image_tensor = torch.frombuffer(
                bytearray(image_bytes), dtype=torch.uint8)
            image = torchvision.io.decode_jpeg(
                image_tensor).permute(1, 2, 0).numpy()
            return image

    def __getitem__(self, id):
        id = str(id).zfill(4)
        image = self.get_image(f'icdr2015_image_{id}')

        maps = {}
        maps['gt_map'] = self.get_image(f'icdr2015_gt_{id}')
        maps['eroded_map'] = self.get_image(f'icdr2015_eroded_{id}')
        return image, maps
