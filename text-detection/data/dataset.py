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
        self.resize = A.Compose(
            [A.Resize(640, 640)], keypoint_params=A.KeypointParams(format="xy")
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        image, labels = self.dataset.__getitem__(id)
        resized_data = self.resize(
            image=image, keypoints=labels["bboxes"].reshape(-1, 2)
        )
        image = resized_data["image"]
        labels["bboxes"] = resized_data["keypoints"].reshape(-1, 4, 2)

        transformed_data = self.transform(
            image=image,
            masks=[*labels["maps"]],
            # keypoints=labels["bboxes"].reshape(-1, 2)
        )
        # image = (
        #     transformed_data["image"]).permute(
        # )
        image = transformed_data["image"]

        # maps = torch.empty((3, 640, 640), dtype=torch.float32)
        # # maps[0] = torch.from_numpy(transformed_data["masks"][0]).float()
        # # maps[1] = torch.from_numpy(transformed_data["masks"][1]).float()
        # # maps[2] = torch.from_numpy(transformed_data["masks"][2]).float()
        # # print(*transformed_data["masks"])
        # maps[0] = transformed_data["masks"][0]
        # maps[1] = transformed_data["masks"][1]
        # maps[2] = transformed_data["masks"][2]

        maps = torch.stack(transformed_data["masks"], dim=0).float()
        labels["maps"] = maps
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
        image = torchvision.io.decode_jpeg(image_tensor).numpy()
        return image

    def __getitem__(self, id):
        id = str(id).zfill(4)
        image = self.get_image(f"icdar2015_image_{id}")
        image = image.transpose(1, 2, 0)

        labels = {}
        # labels["gt_text"] = self.get_image(f"icdr2015_gt_{id}") / 255
        # labels["gt_kernel"] = self.get_image(f"icdr2015_eroded_{id}") / 255
        labels["maps"] = self.get_image(f"icdar2015_maps_{id}") / 255
        labels["maps"][2] = 1 + 8 * labels["maps"][2]

        # nbounds = int(self.txn.get(f"icdr2015_nbounds_{id}".encode()))
        bboxes = np.frombuffer(
            self.txn.get(f"icdar2015_bboxes_{id}".encode()), dtype=np.int32
        ).reshape(-1, 4, 2)
        labels["bboxes"] = bboxes.copy()
        return image, labels


resize = A.Resize(640, 640)


class SynthtextDataset(Dataset):
    def __init__(self, datadir):
        self.env = lmdb.open(f"{datadir}/processed/lmdb_dataset", map_size=150 * 2**30)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return 858750
        # return 1000
        # return 5000

    def get_image(self, path):
        image_bytes = self.txn.get(path.encode())
        image_tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        image = torchvision.io.decode_jpeg(image_tensor).numpy()
        # image = resize(image=image)["image"].copy()
        return image

    def __getitem__(self, id):
        id = str(id).zfill(6)
        image = self.get_image(f"synthtext_image_{id}")
        image = image.transpose(1, 2, 0)

        labels = {}
        labels["maps"] = self.get_image(f"synthtext_maps_{id}") / 255
        labels["maps"][2] = 1 + 8 * labels["maps"][2]

        bboxes = np.frombuffer(
            self.txn.get(f"synthtext_bboxes_{id}".encode()), dtype=np.int32
        ).reshape(-1, 4, 2)
        labels["bboxes"] = bboxes.copy()
        # labels['bboxes'][:, :, 0] = np.clip(bounds[:, :, 0], 0, image.shape[1] - 1)
        # labels['bboxes'][:, :, 1] = np.clip(bounds[:, :, 1], 0, image.shape[0] - 1)
        return image, labels
