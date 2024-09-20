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

        transformed_data = self.transform(
            image=image,
            masks=[labels["gt_text"], labels["gt_kernel"]],
            # keypoints=labels["bboxes"].reshape(-1, 2)
        )
        image = (
            torch.from_numpy(transformed_data["image"].copy()).permute(2, 0, 1).float()
        )
        labels["gt_text"] = (
            torch.from_numpy(transformed_data["masks"][0].copy())
            .permute(2, 0, 1)
            .float()
        )
        labels["gt_kernel"] = (
            torch.from_numpy(transformed_data["masks"][1].copy())
            .permute(2, 0, 1)
            .float()
        )
        # labels["bboxes"] = transformed_data["keypoints"].reshape(-1, 4, 2)
        return image, labels


class ICDR2015Dataset(Dataset):
    def __init__(self, datadir):
        self.env = lmdb.open(f"{datadir}/processed/lmdb_dataset", map_size=150 * 2**30)
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
        labels["gt_text"] = self.get_image(f"icdr2015_gt_{id}") / 255
        labels["gt_kernel"] = self.get_image(f"icdr2015_eroded_{id}") / 255

        # nbounds = int(self.txn.get(f"icdr2015_nbounds_{id}".encode()))
        bboxes = (
            np.frombuffer(
                self.txn.get(f"icdr2015_bounds_{id}".encode()), dtype=np.int32
            )
            .reshape(-1, 4, 2)
            .copy()
        )
        labels["bboxes"] = bboxes
        return image, labels


resize = A.Resize(640, 640)


class SynthtextDataset(Dataset):
    def __init__(self, datadir):
        self.env = lmdb.open(f"{datadir}/processed/lmdb_dataset", map_size=150 * 2**30)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return 858750
        # return 1000

    def get_image(self, path):
        image_bytes = self.txn.get(path.encode())
        image_tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        image = torchvision.io.decode_jpeg(image_tensor).permute(1, 2, 0).numpy()
        image = resize(image=image)["image"].copy()
        return image

    def __getitem__(self, id):
        id = str(id).zfill(6)
        image = self.get_image(f"synthtext_image_{id}")

        labels = {}
        labels["gt_text"] = self.get_image(f"synthtext_text_{id}") / 255
        labels["gt_kernel"] = self.get_image(f"synthtext_kernel_{id}") / 255

        nbounds = int(self.txn.get(f"synthtext_nbounds_{id}".encode()))
        bounds = np.frombuffer(
            self.txn.get(f"synthtext_bounds_{id}".encode()), dtype=np.int32
        ).reshape(nbounds, -1, 2)
        labels["bboxes"] = bounds
        return image, labels
