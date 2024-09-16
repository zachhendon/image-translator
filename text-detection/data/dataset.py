from torch.utils.data import Dataset
import albumentations as A
import torch
import torchvision
import lmdb
from torch.utils.data import Dataset
import numpy as np


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        image, labels = self.dataset.__getitem__(id)

        transformed_data = self.transform(image=image, masks=[labels['gt_maps'], labels['eroded_maps']])
        image = transformed_data['image']
        labels['gt_maps'] = transformed_data['masks'][0].permute(2, 0, 1).float()
        labels['eroded_maps'] = transformed_data['masks'][1].permute(2, 0, 1).float()

        # transform_data = self.transform(image=image)
        # image = transform_data["image"]
        # image = A.Normalize()(image=transform_data["image"].numpy())["image"]
        # labels["gt_maps"] = np.round(
        #     A.ReplayCompose.replay(transform_data["replay"], image=labels["gt_maps"])[
        #         "image"
        #     ]
        #     / 255
        # )
        # labels["eroded_maps"] = np.round(
        #     A.ReplayCompose.replay(
        #         transform_data["replay"], image=labels["eroded_maps"]
        #     )["image"]
        #     / 255
        # )
        return image, labels


class ICDR2015Dataset(Dataset):
    def __init__(self, datadir):
        self.env = lmdb.open(f"{datadir}/processed/lmdb_dataset", map_size=2**30)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return 1500

    def get_image(self, path):
        image_bytes = self.txn.get(path.encode())
        image_tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        image = torchvision.io.decode_jpeg(image_tensor).permute(1, 2, 0).numpy()
        return image

    def __getitem__(self, id):
        id = str(id).zfill(4)
        image = self.get_image(f"icdr2015_image_{id}")

        labels = {}
        labels["gt_maps"] = self.get_image(f"icdr2015_gt_{id}") / 255
        labels["eroded_maps"] = self.get_image(f"icdr2015_eroded_{id}") / 255
        return image, labels
