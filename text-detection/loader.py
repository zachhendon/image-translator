import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_file, decode_jpeg
import kornia.augmentation as K
from kornia.geometry.transform import rescale
from kornia.utils import draw_convex_polygon
from glob import glob
import math
import random


class FastDataset(Dataset):
    def __init__(self, datadir, train, ignore, batch):
        self.datadir = datadir
        self.train = train
        self.ignore = ignore
        self.batch = batch

        self.num_images = len(glob(f"{datadir}/images/*.jpg"))

        self.data_keys = [
            "image",
            "mask",
            "keypoints",
            "keypoints",
            "keypoints",
            "keypoints",
        ]
        self.aug = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0, std=0.05, p=1),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
            K.RandomSharpness(sharpness=0.1, p=1),
            K.ColorJiggle(0.2, 0.2, 0.2, 0.2, p=0.5),
            K.RandomPerspective(distortion_scale=0.1, p=1),
            K.RandomAffine(
                degrees=(-20, 20),
                translate=(0.1, 0.1),
                shear=(-5, 5),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=self.data_keys,
            same_on_batch=False,
        )
        self.upscale = K.AugmentationSequential(
            K.SmallestMaxSize(640),
            data_keys=self.data_keys,
        )
        self.resize = K.AugmentationSequential(
            K.Resize((640, 640)),
            data_keys=self.data_keys,
        )

        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )

    def __len__(self):
        return self.num_images

    def load_bboxes(self, type, idx):
        return torch.load(
            f"{self.datadir}/{type}/{str(idx).zfill(6)}.pt",
            weights_only=True,
            map_location="cuda",
        )

    def preprocess_bboxes(self, bboxes, batch_size):
        max_len = max([len(bb) for bb in bboxes])
        for i, kp in enumerate(bboxes):
            bboxes[i] = torch.cat([kp, torch.zeros(max_len - len(kp), 4, 2).cuda()])
        bboxes = torch.stack(bboxes).reshape(batch_size, -1, 2)
        return bboxes

    def get_masks(self, bboxes, min_bboxes, size):
        if bboxes.shape[0] == 0:
            return (
                torch.zeros((1, *size), device="cuda"),
                torch.zeros((1, *size), device="cuda"),
            )

        fill_value = torch.tensor([1.0], device="cuda")

        text_masks = torch.zeros((bboxes.shape[0], 1, *size), device="cuda")
        text_masks = draw_convex_polygon(text_masks, bboxes, fill_value)

        kernel_masks = -F.max_pool2d(-text_masks, kernel_size=9, stride=1, padding=4)
        min_kernel_masks = torch.zeros((bboxes.shape[0], 1, *size), device="cuda")
        min_kernel_masks = draw_convex_polygon(min_kernel_masks, min_bboxes, fill_value)

        for i in range(len(kernel_masks)):
            kernel_masks[i] = torch.clamp(
                kernel_masks[i] + min_kernel_masks[i], 0, 1
            ) * (i + 1)
            text_masks[i] = text_masks[i] * (i + 1)
        kernel_masks, _ = torch.max(kernel_masks, dim=0)
        text_masks, _ = torch.max(text_masks, dim=0)

        return kernel_masks, text_masks

    def apply_augmentations(self, images, bboxes, min_bboxes, aug):
        if self.batch:
            images_aug, bboxes_aug, min_bboxes_aug = aug(images, bboxes, min_bboxes)
        else:
            images_aug = []
            bboxes_aug = []
            min_bboxes_aug = []

            for image, bbox, min_bbox in zip(images, bboxes, min_bboxes):
                if len(bbox.shape) == 2:
                    bbox = bbox.unsqueeze(0)
                    min_bbox = min_bbox.unsqueeze(0)
                image, bbox, min_bbox = aug(image, bbox, min_bbox)
                images_aug.append(image)
                bboxes_aug.append(bbox)
                min_bboxes_aug.append(min_bbox)
        return images_aug, bboxes_aug, min_bboxes_aug

    def random_scale_and_crop(self, images, target_size):
        crop_h, crop_w = target_size
        resize = K.SmallestMaxSize(min(crop_h, crop_w))

        # scale images
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1)
        else:
            scale = random.uniform(1, 2)

        margin_y = max(0, (crop_h - int(scale * crop_h)))
        margin_x = max(0, (crop_w - int(scale * crop_w)))
        pad_left = random.randint(0, margin_x)
        pad_right = margin_x - pad_left
        pad_top = random.randint(0, margin_y)
        pad_bottom = margin_y - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        images_scale = []
        for image in images:
            image_scale = resize(image)
            image_scale = rescale(image_scale, (scale, scale))
            image_scale = F.pad(image_scale, padding)
            images_scale.append(image_scale)

        # crop images
        h, w = images_scale[0].shape[2:]
        y_coords, x_coords = torch.where(images_scale[1][0, 0] > 0)
        if y_coords.numel() == 0:
            y_min, y_max = 0, h
            x_min, x_max = 0, w
        else:
            y_min, y_max = y_coords.min().item(), y_coords.max().item()
            x_min, x_max = x_coords.min().item(), x_coords.max().item()

        if y_max - y_min <= crop_h:
            margin_y = crop_h - (y_max - y_min)
            start_y = min(random.randint(max(0, y_min - margin_y), y_min), h - crop_h)
        else:
            start_y = random.randint(y_min, y_max - crop_h)
        if x_max - x_min <= crop_w:
            margin_x = crop_w - (x_max - x_min)
            start_x = min(random.randint(max(0, x_min - margin_x), x_min), w - crop_w)
        else:
            start_x = random.randint(x_min, x_max - crop_w)

        images_out = []
        for image in images_scale:
            image_crop = image[
                :, :, start_y : start_y + crop_h, start_x : start_x + crop_w
            ]
            images_out.append(image_crop)
        return images_out

    def random_scale_and_crop2(
        self,
        image,
        mask,
        bboxes,
        min_bboxes,
        ignore_bboxes,
        ignore_min_bboxes,
        target_size,
    ):
        # scale images
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1)
        else:
            scale = random.uniform(1, 2)
        # scale = random.uniform(0.5, 2)

        image_scaled = F.interpolate(image, scale_factor=scale, mode="bilinear")
        mask_scaled = F.interpolate(mask, scale_factor=scale, mode="bilinear")
        bboxes_scaled = bboxes * scale
        min_bboxes_scaled = min_bboxes * scale
        ignore_bboxes_scaled = ignore_bboxes * scale
        ignore_min_bboxes_scaled = ignore_min_bboxes * scale

        # pad or crop images
        if scale < 1:
            # pad images if scale is less than 1
            margin_left = image.shape[3] - image_scaled.shape[3]
            margin_top = image.shape[2] - image_scaled.shape[2]

            pad_left = random.randint(0, margin_left)
            pad_right = margin_left - pad_left
            pad_top = random.randint(0, margin_top)
            pad_bottom = margin_top - pad_top

            image_scaled = F.pad(
                image_scaled, (pad_left, pad_right, pad_top, pad_bottom), value=0
            )
            mask_scaled = F.pad(
                mask_scaled, (pad_left, pad_right, pad_top, pad_bottom), value=0
            )
            bbox_padding = torch.tensor([pad_left, pad_top], device="cuda")
            bboxes_scaled += bbox_padding
            min_bboxes_scaled += bbox_padding
            ignore_bboxes_scaled += bbox_padding
            ignore_min_bboxes_scaled += bbox_padding

        # crop images
        crop_size = min(image.shape[2:])
        crop_mask = torch.ones(
            (
                image_scaled.shape[2] - crop_size + 1,
                image_scaled.shape[3] - crop_size + 1,
                len(bboxes),
            ),
            device="cuda",
        )

        xmin, _ = torch.min(bboxes_scaled[:, :, 0], dim=1)
        xmax, _ = torch.max(bboxes_scaled[:, :, 0], dim=1)
        ymin, _ = torch.min(bboxes_scaled[:, :, 1], dim=1)
        ymax, _ = torch.max(bboxes_scaled[:, :, 1], dim=1)

        start_x = torch.clamp(xmax - crop_size, 0, image_scaled.shape[3]).int()
        end_x = torch.clamp(xmin, 0, image_scaled.shape[3]).int()
        start_y = torch.clamp(ymax - crop_size, 0, image_scaled.shape[2]).int()
        end_y = torch.clamp(ymin, 0, image_scaled.shape[2]).int()

        scale_factor = 1 + ((xmax - xmin) + (ymax - ymin)) / (
            (end_x - start_x) * (end_y - start_y)
        )
        for i in range(len(bboxes_scaled)):
            crop_mask[start_y[i] : end_y[i], start_x[i] : end_x[i], i] *= scale_factor[
                i
            ]
        crop_mask = torch.prod(crop_mask, dim=2)
        crop_mask += 0.01 * (torch.sum(crop_mask, dim=0) + 1e-6)

        # sample a crop location based on the mask
        flattened_mask = crop_mask.flatten()
        probabilities = flattened_mask / flattened_mask.sum()
        sampled_index = torch.multinomial(probabilities, 1).item()
        sampled_y, sampled_x = divmod(sampled_index, crop_mask.shape[1])

        image_crop = image_scaled[
            :, :, sampled_y : sampled_y + crop_size, sampled_x : sampled_x + crop_size
        ]
        mask_crop = mask_scaled[
            :, :, sampled_y : sampled_y + crop_size, sampled_x : sampled_x + crop_size
        ]
        crop = torch.tensor([sampled_x, sampled_y], device="cuda")
        bboxes_crop = bboxes_scaled - crop
        min_bboxes_crop = min_bboxes_scaled - crop
        ignore_bboxes_crop = ignore_bboxes_scaled - crop
        ignore_min_bboxes_crop = ignore_min_bboxes_scaled - crop

        # resize images to target size
        image_out = F.interpolate(image_crop, size=target_size, mode="bilinear")
        mask_out = F.interpolate(mask_crop, size=target_size, mode="bilinear")
        bbox_resize = target_size / image_crop.shape[2]
        bboxes_out = bboxes_crop * bbox_resize
        min_bboxes_out = min_bboxes_crop * bbox_resize
        ignore_bboxes_out = ignore_bboxes_crop * bbox_resize
        ignore_min_bboxes_out = ignore_min_bboxes_crop * bbox_resize

        return (
            image_out,
            mask_out,
            bboxes_out,
            min_bboxes_out,
            ignore_bboxes_out,
            ignore_min_bboxes_out,
        )

    def __getitem__(self, id):
        # load image and bboxes
        bytes = read_file(f"{self.datadir}/images/{str(id).zfill(6)}.jpg")
        image = decode_jpeg(bytes, device="cuda") / 255
        # mask = torch.ones_like(image)
        mask = torch.ones((1, 1, *image.shape[1:]), device="cuda")

        bboxes = self.load_bboxes("bboxes", id)
        min_bboxes = self.load_bboxes("min_bboxes", id)
        if self.ignore:
            ignore_bboxes = self.load_bboxes("ignore_bboxes", id)
            ignore_min_bboxes = self.load_bboxes("min_ignore_bboxes", id)
        else:
            ignore_bboxes = torch.zeros((0, 4, 2), device="cuda")
            ignore_min_bboxes = torch.zeros((0, 4, 2), device="cuda")

        if self.train:
            # apply geometric and color augmentations
            image, mask, bboxes, min_bboxes, ignore_bboxes, ignore_min_bboxes = (
                self.aug(
                    image, mask, bboxes, min_bboxes, ignore_bboxes, ignore_min_bboxes
                )
            )

            # scale and crop
            image, mask, bboxes, min_bboxes, ignore_bboxes, ignore_min_bboxes = (
                self.random_scale_and_crop2(
                    image,
                    mask,
                    bboxes,
                    min_bboxes,
                    ignore_bboxes,
                    ignore_min_bboxes,
                    640,
                )
            )
        else:
            h, w = image.shape[1:]
            short_size = 736
            long_size = math.ceil((w * short_size / h) / 4) * 4
            resize = K.AugmentationSequential(
                K.Resize((short_size, long_size)),
                data_keys=self.data_keys,
                same_on_batch=True,
            )
            image, mask, bboxes, min_bboxes, ignore_bboxes, ignore_min_bboxes = resize(
                image, mask, bboxes, min_bboxes, ignore_bboxes, ignore_min_bboxes
            )

        # normalize images
        image = self.normalize(image)

        # get bbox masks
        kernel_masks, text_masks = self.get_masks(bboxes, min_bboxes, image.shape[2:])
        ignore_kernel_masks, ignore_text_masks = self.get_masks(
            ignore_bboxes, ignore_min_bboxes, image.shape[2:]
        )
        ignore_kernel_masks[mask[0] == 0] = 1
        ignore_text_masks[mask[0] == 0] = 1

        return (
            image.squeeze(0),
            kernel_masks.squeeze(0),
            ignore_kernel_masks.squeeze(0),
            text_masks.squeeze(0),
            ignore_text_masks.squeeze(0),
        )

    # def __getitems__(self, idxs):
    #     # load images and bboxes
    #     bytes = []
    #     for idx in idxs:
    #         bytes.append(read_file(f"{self.datadir}/images/{str(idx).zfill(6)}.jpg"))
    #     images = decode_jpeg(bytes, device="cuda")

    #     bboxes = [self.load_bboxes("bboxes", idx) for idx in idxs]
    #     min_bboxes = [self.load_bboxes("min_bboxes", idx) for idx in idxs]
    #     masks = [
    #         self.get_masks(bbox, min_bbox, images[i].shape[1:])
    #         for i, (bbox, min_bbox) in enumerate(zip(bboxes, min_bboxes))
    #     ]
    #     kernel_masks, text_masks = zip(*masks)

    #     if self.ignore:
    #         ignore_bboxes = [self.load_bboxes("ignore_bboxes", idx) for idx in idxs]
    #         ignore_min_bboxes = [
    #             self.load_bboxes("min_ignore_bboxes", idx) for idx in idxs
    #         ]
    #         ignore_masks = [
    #             self.get_masks(bbox, min_bbox, images[i].shape[1:])
    #             for i, (bbox, min_bbox) in enumerate(
    #                 zip(ignore_bboxes, ignore_min_bboxes)
    #             )
    #         ]
    #         ignore_kernel_masks, ignore_text_masks = zip(*ignore_masks)
    #     else:
    #         ignore_kernel_masks = [torch.zeros_like(mask) for mask in kernel_masks]
    #         ignore_text_masks = [torch.zeros_like(mask) for mask in text_masks]

    #     # Check if images are different sizes
    #     resize_images = len(set([img.shape for img in images])) > 1
    #     if resize_images:
    #         avg_h = sum([img.shape[1] for img in images]) // len(images)
    #         avg_w = sum([img.shape[2] for img in images]) // len(images)
    #         resize = K.Resize((avg_h, avg_w))

    #         images = torch.cat([resize(img / 255) for img in images])
    #         kernel_masks = torch.cat([resize(mask) for mask in kernel_masks])
    #         text_masks = torch.cat([resize(mask) for mask in text_masks])
    #         ignore_kernel_masks = torch.cat(
    #             [resize(mask) for mask in ignore_kernel_masks]
    #         )
    #         ignore_text_masks = torch.cat([resize(mask) for mask in ignore_text_masks])
    #     else:
    #         images = torch.stack(images) / 255
    #         kernel_masks = torch.stack(kernel_masks)
    #         text_masks = torch.stack(text_masks)
    #         ignore_kernel_masks = torch.stack(ignore_kernel_masks)
    #         ignore_text_masks = torch.stack(ignore_text_masks)

    #     # apply augmentations
    #     if self.train:
    #         images, kernel_masks, text_masks, ignore_kernel_masks, ignore_text_masks = (
    #             self.aug(
    #                 images,
    #                 kernel_masks,
    #                 text_masks,
    #                 ignore_kernel_masks,
    #                 ignore_text_masks,
    #             )
    #         )
    #     else:
    #         h, w = images.shape[2:]
    #         short_size = 736
    #         long_size = math.ceil((w * short_size / h) / 4) * 4
    #         resize = K.AugmentationSequential(
    #             K.Resize((short_size, long_size)),
    #             data_keys=self.data_keys,
    #             same_on_batch=True,
    #         )
    #         images, kernel_masks, text_masks, ignore_kernel_masks, ignore_text_masks = (
    #             resize(
    #                 images,
    #                 kernel_masks,
    #                 text_masks,
    #                 ignore_kernel_masks,
    #                 ignore_text_masks,
    #             )
    #         )

    #     # crop images
    #     if self.train:
    #         images, kernel_masks, text_masks, ignore_kernel_masks, ignore_text_masks = (
    #             self.random_scale_and_crop(
    #                 [
    #                     images,
    #                     kernel_masks,
    #                     text_masks,
    #                     ignore_kernel_masks,
    #                     ignore_text_masks,
    #                 ],
    #                 (640, 640),
    #             )
    #         )

    #     # normalize images
    #     images = (
    #         images - torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    #     ) / torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    #     kernel_masks = torch.round(kernel_masks).squeeze(1)
    #     text_masks = torch.round(text_masks).squeeze(1)
    #     ignore_kernel_masks = torch.round(ignore_kernel_masks).squeeze(1)
    #     ignore_text_masks = torch.round(ignore_text_masks).squeeze(1)

    #     batch = [
    #         images,
    #         kernel_masks,
    #         ignore_kernel_masks,
    #         text_masks,
    #         ignore_text_masks,
    #     ]
    #     return batch


def collate_fn(batch):
    images = torch.cat(batch[0])
    kernel_masks = torch.stack(batch[1])
    ignore_kernel_masks = torch.stack(batch[2])
    text_masks = torch.stack(batch[3])
    ignore_text_masks = torch.stack(batch[4])

    return_batch = [
        images,
        kernel_masks,
        ignore_kernel_masks,
        text_masks,
        ignore_text_masks,
    ]
    return return_batch


class DataLoaderIterator:
    def __init__(self, dataset, batch_size=16, train=True):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
        self.iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)


def get_icdar2015_loaders(batch_size=16):
    train_dataset = FastDataset(
        "data/processed/icdar2015/train", train=True, ignore=True, batch=False
    )
    val_dataset = FastDataset(
        "data/processed/icdar2015/val", train=False, ignore=True, batch=False
    )
    return (
        DataLoaderIterator(
            train_dataset,
            batch_size,
            train=True,
        ),
        DataLoaderIterator(
            val_dataset,
            batch_size,
            train=False,
        ),
    )


def get_synthtext_loaders(batch_size=16):
    train_dataset = FastDataset(
        "data/processed/synthtext/train", train=True, ignore=False, batch=False
    )
    val_dataset = FastDataset(
        "data/processed/synthtext/val", train=False, ignore=False, batch=False
    )
    return (
        DataLoaderIterator(
            train_dataset,
            batch_size,
            train=True,
        ),
        DataLoaderIterator(
            val_dataset,
            1,
            train=False,
        ),
    )
