import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_file, decode_jpeg
import kornia.augmentation as K
from kornia.utils import draw_convex_polygon
from glob import glob


class ICDAR2015Dataset(Dataset):
    def __init__(self, datadir, train):
        self.datadir = datadir
        self.train = train
        self.num_images = len(glob(f"{datadir}/images/*.jpg"))

        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
        if train:
            self.aug = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomAffine(
                    degrees=(-20, 20),
                    translate=(0.2, 0.2),
                    shear=(-20, 20),
                    scale=(0.8, 1.2),
                    p=1.0,
                ),
                K.RandomPerspective(distortion_scale=0.3, p=0.3),
                K.RandomCrop((640, 640)),
                K.ColorJitter(0.1, 0.1, 0.1, 0.1),
                K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
                K.RandomSharpness(0.5),
                K.RandomGaussianNoise(mean=0, std=0.05, p=1),
                data_keys=["image", "keypoints", "keypoints", "keypoints", "keypoints"],
                same_on_batch=False,
            )
        else:
            self.aug = K.AugmentationSequential(
                K.SmallestMaxSize(640),
                data_keys=["image", "keypoints", "keypoints", "keypoints", "keypoints"],
                same_on_batch=True,
            )

    def __len__(self):
        return self.num_images

    def load_bboxes(self, type, idx):
        return torch.load(
            f"{self.datadir}/{type}/{str(idx).zfill(6)}.pt",
            weights_only=True,
            map_location="cuda",
        )

    def preprocess_bboxes(self, bboxes, num_bboxes, batch_size):
        max_len = max(num_bboxes)
        for i, kp in enumerate(bboxes):
            bboxes[i] = torch.cat([kp, torch.zeros(max_len - len(kp), 4, 2).cuda()])
        bboxes = torch.stack(bboxes).reshape(batch_size, -1, 2)
        return bboxes

    def get_masks(self, bboxes, min_bboxes, size):
        batch_bboxes = torch.vstack(bboxes)
        batch_min_bboxes = torch.vstack(min_bboxes)
        if batch_bboxes.shape[0] == 0:
            return (
                torch.zeros((len(bboxes), *size)),
                torch.zeros((len(bboxes), *size)),
            )

        fill_value = torch.tensor([1.0], device="cuda")

        text_masks = torch.zeros((batch_bboxes.shape[0], 1, *size), device="cuda")
        text_masks = draw_convex_polygon(text_masks, batch_bboxes, fill_value)

        kernel_masks = -F.max_pool2d(-text_masks, kernel_size=9, stride=1, padding=4)

        min_kernel_masks = torch.zeros((batch_bboxes.shape[0], 1, *size), device="cuda")
        min_kernel_masks = draw_convex_polygon(
            min_kernel_masks, batch_min_bboxes, fill_value
        )

        kernel_masks = kernel_masks.squeeze(1)
        text_masks = text_masks.squeeze(1)
        min_kernel_masks = min_kernel_masks.squeeze(1)

        batch_kernel_masks = []
        batch_text_masks = []
        l = 0
        for i in range(len(bboxes)):
            r = l + len(bboxes[i])
            if l == r:
                batch_kernel_masks.append(torch.zeros(size, device="cuda"))
                batch_text_masks.append(torch.zeros(size, device="cuda"))
            else:
                batch_kernel_masks.append(
                    torch.clamp(
                        (kernel_masks[l:r] + min_kernel_masks[l:r]).sum(axis=0), 0, 1
                    )
                )
                batch_text_masks.append(torch.clamp(text_masks[l:r].sum(axis=0), 0, 1))
            l = r
        return batch_kernel_masks, batch_text_masks

    def __getitems__(self, idxs):
        batch_size = len(idxs)

        # load images and bboxes
        bytes = []
        for idx in idxs:
            bytes.append(read_file(f"{self.datadir}/images/{str(idx).zfill(6)}.jpg"))
        images = decode_jpeg(bytes, device="cuda")
        images = torch.stack(images) / 255

        bboxes = [self.load_bboxes("bboxes", idx) for idx in idxs]
        num_bboxes = [len(bb) for bb in bboxes]
        bboxes = self.preprocess_bboxes(bboxes, num_bboxes, batch_size)

        ignore_bboxes = [self.load_bboxes("ignore_bboxes", idx) for idx in idxs]
        num_ignore_bboxes = [len(bb) for bb in ignore_bboxes]
        ignore_bboxes = self.preprocess_bboxes(
            ignore_bboxes, num_ignore_bboxes, batch_size
        )

        min_bboxes = [self.load_bboxes("min_bboxes", idx) for idx in idxs]
        num_min_bboxes = [len(bb) for bb in min_bboxes]
        min_bboxes = self.preprocess_bboxes(min_bboxes, num_min_bboxes, batch_size)

        min_ignore_bboxes = [self.load_bboxes("min_ignore_bboxes", idx) for idx in idxs]
        num_min_ignore_bboxes = [len(bb) for bb in min_ignore_bboxes]
        min_ignore_bboxes = self.preprocess_bboxes(
            min_ignore_bboxes, num_min_ignore_bboxes, batch_size
        )

        # apply augmentations
        images = self.normalize(images)
        images, bboxes, ignore_bboxes, min_bboxes, min_ignore_bboxes = self.aug(
            images, bboxes, ignore_bboxes, min_bboxes, min_ignore_bboxes
        )

        # process augmented bboxes
        bboxes = [
            bb[: num_bboxes[i]]
            for i, bb in enumerate(bboxes.reshape(batch_size, -1, 4, 2))
        ]
        ignore_bboxes = [
            bb[: num_ignore_bboxes[i]]
            for i, bb in enumerate(ignore_bboxes.reshape(batch_size, -1, 4, 2))
        ]
        min_bboxes = [
            bb[: num_min_bboxes[i]]
            for i, bb in enumerate(min_bboxes.reshape(batch_size, -1, 4, 2))
        ]
        min_ignore_bboxes = [
            bb[: num_min_ignore_bboxes[i]]
            for i, bb in enumerate(min_ignore_bboxes.reshape(batch_size, -1, 4, 2))
        ]

        # get masks
        size = images.shape[2:]
        kernel_masks, text_masks = self.get_masks(bboxes, min_bboxes, size)
        ignore_kernel_masks, ignore_text_masks = self.get_masks(
            ignore_bboxes, min_ignore_bboxes, size
        )

        return (
            list(images),
            kernel_masks,
            ignore_kernel_masks,
            text_masks,
            ignore_text_masks,
        )


def collate_fn(batch):
    images = torch.stack(batch[0])
    text_masks = torch.stack(batch[1])
    kernel_masks = torch.stack(batch[2])
    ignore_text_masks = torch.stack(batch[3])
    ignore_kernel_masks = torch.stack(batch[4])
    return images, text_masks, kernel_masks, ignore_text_masks, ignore_kernel_masks


class DataLoaderIterator:
    def __init__(self, dataset, datadir, batch_size=16, train=True):
        dataset = dataset(datadir, train)
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=train, collate_fn=collate_fn
        )
        self.iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)


def get_icdar2015_loaders(root_dir, batch_size=16):
    return (
        DataLoaderIterator(
            ICDAR2015Dataset, f"{root_dir}/icdar2015/train", batch_size, train=True
        ),
        DataLoaderIterator(
            ICDAR2015Dataset, f"{root_dir}/icdar2015/val", batch_size, train=False
        ),
    )
