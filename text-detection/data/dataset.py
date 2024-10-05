from torch.utils.data import Dataset
import albumentations as A
import torch
import torchvision
import lmdb
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
import pyclipper


class TransformDataset(Dataset):
    def __init__(self, dataset, transform, train=False):
        self.dataset = dataset
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        image, labels = self.dataset.__getitem__(id)

        if self.train:
            size = (640, 640)
        else:
            size = (640, 4 * int(image.shape[1] * 640 / image.shape[0] / 4))
        # size = (640, 640)
        resize = A.Compose(
            [A.Resize(*size)], keypoint_params=A.KeypointParams(format="xy")
        )

        resized_data = resize(
            image=image,
            masks=[*labels["maps"]],
            keypoints=labels["bboxes"].reshape(-1, 2),
        )
        image = resized_data["image"]
        labels["maps"] = np.stack(resized_data["masks"])
        labels["bboxes"] = resized_data["keypoints"].reshape(-1, 4, 2)

        transformed_data = self.transform(
            image=image,
            masks=[*labels["maps"]],
        )
        image = transformed_data["image"]

        maps = torch.stack(transformed_data["masks"]).float()
        labels["maps"] = maps
        return image, labels


class ICDAR2015Dataset(Dataset):
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
        labels["maps"] = self.get_image(f"icdar2015_maps_{id}") / 255
        labels["maps"][2] = 1 + 8 * labels["maps"][2]

        bboxes = np.frombuffer(
            self.txn.get(f"icdar2015_bboxes_{id}".encode()), dtype=np.int32
        ).reshape(-1, 4, 2)
        labels["bboxes"] = bboxes.copy()
        return image, labels


class ICDAR2015Dataset2(Dataset):
    def __init__(self, datadir, transform, train=True):
        super().__init__()
        self.datadir = datadir
        self.transform = transform
        self.train = train

        self.train_resize = A.Compose(
            [
                A.RandomScale(0.5, p=1.0),
                # A.PadIfNeeded(640, 640, border_mode=cv.BORDER_CONSTANT, value=0, p=1.0),
                # A.RandomCrop(640, 640, p=1.0),
                # A.RandomResizedCrop(
                #     size=(640, 640), scale=(1/ 2, 1), ratio=(0.9, 1.1), p=1.0
                # ),
                # A.RandomCrop(640, 640, p=1.0),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        self.eval_resize = A.Compose(
            [
                A.Resize(640, 1140, p=1.0),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __len__(self):
        if self.train:
            return 800
        else:
            return 200

    def get_gt_kernel(self, bboxes, size):
        gt_kernels = []
        gt_text = np.zeros(size, dtype=np.uint8)
        if len(bboxes) == 0:
            return gt_text, gt_text

        for bbox in bboxes:
            gt_kernel = np.zeros(size, dtype=np.uint8)
            gt_kernel = np.pad(gt_kernel, 4)
            cv.fillPoly(gt_kernel, [bbox + 4], 1)
            cv.fillPoly(gt_text, [bbox], 1)
            gt_kernels.append(gt_kernel)
        gt_kernel = np.array(gt_kernels)

        gt_kernel = (
            -F.max_pool2d(
                -torch.from_numpy(gt_kernel)
                .to(dtype=torch.float16, device="cuda")
                .unsqueeze(0),
                kernel_size=9,
                stride=1,
                padding=0,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

        gt_kernel = np.max(gt_kernel, axis=0)
        return gt_kernel, gt_text

    def get_gt_kernel_shrinked(self, bboxes, size):
        rate = 0.1 * 0.1
        gt_kernel_shrinked = np.zeros(size, dtype=np.uint8)
        shrinked_bboxes = []
        for bbox in bboxes:
            poly = Polygon(bbox)

            try:
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                offset = poly.area * (1 - rate) / poly.length
                shrinked_bbox = pco.Execute(-offset)[0]
                shrinked_bbox = np.array(shrinked_bbox)
                shrinked_bboxes.append(shrinked_bbox)
            except:
                shrinked_bboxes.append(bbox)

        for shrinked_bbox in shrinked_bboxes:
            cv.fillPoly(gt_kernel_shrinked, [shrinked_bbox], 1)
        return gt_kernel_shrinked

    def get_masks(self, bboxes, size):
        masks = np.ones(size, dtype=np.uint8)
        for bbox in bboxes:
            cv.fillPoly(masks, [bbox], 0)
        return masks

    def apply_transform(self, transform, image, bboxes=None, ignore_bboxes=None):
        if bboxes is None:
            transformed = transform(image=image)
            return transformed["image"]

        num_bboxes = len(bboxes)
        transformed = transform(
            image=image,
            keypoints=np.vstack([bboxes.reshape(-1, 2), ignore_bboxes.reshape(-1, 2)]),
        )
        image = transformed["image"]
        bboxes = transformed["keypoints"].reshape(-1, 4, 2).astype(np.int32)
        ignore_bboxes = bboxes[num_bboxes:]
        bboxes = bboxes[:num_bboxes]
        return image, bboxes, ignore_bboxes

    def __getitem__(self, idx):
        if self.train:
            image_path = f"{self.datadir}/processed/icdar2015/train_images/img_{str(idx).zfill(4)}.jpg"
            gt_path = f"{self.datadir}/processed/icdar2015/train_gts/gt_{str(idx).zfill(4)}.txt"
            gt_ignore_path = f"{self.datadir}/processed/icdar2015/train_gts/gt_ignore_{str(idx).zfill(4)}.txt"
        else:
            image_path = f"{self.datadir}/processed/icdar2015/val_images/img_{str(idx).zfill(4)}.jpg"
            gt_path = (
                f"{self.datadir}/processed/icdar2015/val_gts/gt_{str(idx).zfill(4)}.txt"
            )
            gt_ignore_path = f"{self.datadir}/processed/icdar2015/val_gts/gt_ignore_{str(idx).zfill(4)}.txt"
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        with open(gt_path, encoding="utf-8-sig", mode="r") as f:
            bboxes = [line.rstrip().split(",")[:8] for line in f]
        bboxes = np.array(bboxes, dtype=np.int32).reshape(-1, 4, 2)
        with open(gt_ignore_path, encoding="utf-8-sig", mode="r") as f:
            ignore_bboxes = [line.rstrip().split(",")[:8] for line in f]
        ignore_bboxes = np.array(ignore_bboxes, dtype=np.int32).reshape(-1, 4, 2)

        if self.train:
            # resized = self.train_resize(
            #     image=image,
            #     keypoints=np.vstack(
            #         [bboxes.reshape(-1, 2), ignore_bboxes.reshape(-1, 2)]
            #     ),
            # )
            # image = resized["image"]
            # bboxes = resized["keypoints"].reshape(-1, 4, 2)
            # ignore_bboxes = bboxes[num_bboxes:]
            # bboxes = bboxes[:num_bboxes]
            image, bboxes, ignore_bboxes = self.apply_transform(
                self.train_resize, image, bboxes, ignore_bboxes
            )

            # transformed = self.transform(image=image, keypoints=bboxes.reshape(-1, 2))
            # image = transformed["image"]
            # bboxes = transformed["keypoints"].reshape(-1, 4, 2).astype(np.int32)
            image, bboxes, ignore_bboxes = self.apply_transform(
                self.transform, image, bboxes, ignore_bboxes
            )
        else:
            # resized = self.eval_resize(image=image, keypoints=bboxes.reshape(-1, 2))
            # image = resized["image"]
            # bboxes = resized["keypoints"].reshape(-1, 4, 2).astype(np.int32)
            image, bboxes, ignore_bboxes = self.apply_transform(
                self.eval_resize, image, bboxes, ignore_bboxes
            )

            # transformed = self.transform(image=image)
            # image = transformed["image"]
            image = self.transform(image=image)["image"]

        gt_kernel, gt_text = self.get_gt_kernel(bboxes, image.shape[1:])
        gt_kernel_shrinked = self.get_gt_kernel_shrinked(bboxes, image.shape[1:])
        gt_kernel = np.maximum(gt_kernel, gt_kernel_shrinked)
        gt_kernel = torch.from_numpy(gt_kernel)
        gt_text = torch.from_numpy(gt_text)

        kernel_masks, text_masks = self.get_gt_kernel(ignore_bboxes, image.shape[1:])
        kernel_masks_shrinked = self.get_gt_kernel_shrinked(ignore_bboxes, image.shape[1:])
        kernel_masks = np.maximum(kernel_masks, kernel_masks_shrinked)
        kernel_masks = 1 - torch.from_numpy(kernel_masks)
        text_masks = 1 - torch.from_numpy(text_masks)
        return image, (gt_kernel, gt_text, kernel_masks, text_masks, bboxes)


class SynthtextDataset(Dataset):
    def __init__(self, datadir):
        self.env = lmdb.open(f"{datadir}/processed/lmdb_dataset", map_size=150 * 2**30)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return 858750

    def get_image(self, path):
        image_bytes = self.txn.get(path.encode())
        image_tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        image = torchvision.io.decode_jpeg(image_tensor).numpy()
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
        return image, labels
