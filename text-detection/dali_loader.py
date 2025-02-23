from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.auto_aug import trivial_augment, augmentations, rand_augment
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import torch
import torch.nn.functional as F
import kornia
from kornia.utils import draw_convex_polygon
from shapely.geometry import Polygon
import math
import time
import numpy as np


def pad_images(image, bboxes, ignore_bboxes):
    h, w = image.shape[:2]
    extra_padding = int(min(h, w) * 0.25)
    if h > w:
        pad_left = (h - w) // 2
        pad_right = (h - w) - pad_left
        pad_top = extra_padding
        pad_bottom = extra_padding
    elif w >= h:
        pad_left = extra_padding
        pad_right = extra_padding
        pad_top = (w - h) // 2
        pad_bottom = (w - h) - pad_top
    image = F.pad(
        image.permute(2, 0, 1), (pad_left, pad_right, pad_top, pad_bottom)
    ).permute(1, 2, 0)
    bbox_offset = torch.tensor([pad_left, pad_top], device="cuda")
    bboxes += bbox_offset
    ignore_bboxes += bbox_offset

    crop_size = torch.tensor([min(h, w)], device="cuda")
    return image, bboxes, ignore_bboxes, crop_size


def random_flip(image, bboxes, ignore_bboxes):
    if np.random.rand() < 0.5:
        return image, bboxes, ignore_bboxes

    w = image.shape[1]
    image = kornia.geometry.transform.hflip(image.permute(2, 0, 1)).permute(1, 2, 0)
    bboxes[:, :, 0] = w - bboxes[:, :, 0]
    ignore_bboxes[:, :, 0] = w - ignore_bboxes[:, :, 0]
    return image, bboxes, ignore_bboxes


def fast_binary_avgpool2d(mask, kernel_size):
    cumsum_h = torch.cumsum(mask, dim=2)
    cumsum_2d = torch.cumsum(cumsum_h, dim=3)

    top_left = cumsum_2d[..., : -kernel_size + 1, : -kernel_size + 1]
    bottom_right = cumsum_2d[..., kernel_size - 1 :, kernel_size - 1 :]
    top_right = cumsum_2d[..., : -kernel_size + 1, kernel_size - 1 :]
    bottom_left = cumsum_2d[..., kernel_size - 1 :, : -kernel_size + 1]

    window_sums = bottom_right + top_left - top_right - bottom_left
    return window_sums / kernel_size**2


def random_crop(image, bboxes, crop_size, gamma=2):
    device = image.device
    h, w = image.shape[:2]

    if bboxes.numel() == 0:
        p = torch.ones((1, h - crop_size + 1, w - crop_size + 1), device=device, dtype=torch.float32)
    else:
        mask = torch.zeros((len(bboxes), 1, h, w), device=device, dtype=torch.uint8)
        mask = draw_convex_polygon(mask, bboxes, torch.tensor([1], device=device))
        p = fast_binary_avgpool2d(mask, kernel_size=crop_size).squeeze(1)
        p = torch.clamp(p, min=0)
        p /= torch.sum(p, dim=(1, 2), keepdim=True) + 1e-6
        p = torch.sum(p, dim=0) ** gamma
        if p.sum() == 0:
            p += 1
    coord = torch.multinomial(p.view(-1), 1)
    y_coord, x_coord = coord // p.shape[1], coord % p.shape[1]
    return x_coord, y_coord


def random_scale_and_crop(image, bboxes, ignore_bboxes, crop_size):
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.5, 1)
    else:
        scale = np.random.uniform(1, 2)
    crop_scale = int(crop_size / scale)

    # pad image if necessary for crop
    h, w = image.shape[:2]
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if h < crop_scale:
        pad_top = np.random.randint(0, crop_scale - h)
        pad_bottom = crop_scale - h - pad_top
    if w < crop_scale:
        pad_left = np.random.randint(0, crop_scale - w)
        pad_right = crop_scale - w - pad_left
    image = F.pad(
        image.permute(2, 0, 1),
        (pad_left, pad_right, pad_top, pad_bottom),
    ).permute(1, 2, 0)
    bbox_offset = torch.tensor([[pad_left, pad_top]], device="cuda")
    if bboxes.numel() > 0:
        bboxes += bbox_offset
    if ignore_bboxes.numel() > 0:
        ignore_bboxes += bbox_offset

    # crop image
    start_x, start_y = random_crop(image, bboxes, crop_scale)
    image = image[start_y : start_y + crop_scale, start_x : start_x + crop_scale]
    image = (
        F.interpolate(
            image.permute(2, 0, 1).unsqueeze(0), size=(crop_size, crop_size)
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    if bboxes.numel() > 0:
        bboxes = (bboxes - torch.tensor([start_x, start_y], device="cuda")) * scale
    if ignore_bboxes.numel() > 0:
        ignore_bboxes = (
            ignore_bboxes - torch.tensor([start_x, start_y], device="cuda")
        ) * scale
    return image, bboxes, ignore_bboxes


def resize(images, bboxes, ignore_bboxes, train, short_size):
    h, w = images.shape[:2]
    if train:
        resize_h, resize_w = (short_size, short_size)
    else:
        if h <= w:
            resize_h = short_size
            resize_w = math.ceil((w * short_size / h) / 4) * 4
        else:
            resize_h = math.ceil((h * short_size / w) / 4) * 4
            resize_w = short_size

    bbox_scale = torch.tensor([resize_w / w, resize_h / h], device="cuda")
    images = (
        F.interpolate(images.permute(2, 0, 1).unsqueeze(0), size=(resize_h, resize_w))
        .squeeze(0)
        .permute(1, 2, 0)
    )
    if bboxes.numel() > 0:
        bboxes *= bbox_scale
    if ignore_bboxes.numel() > 0:
        ignore_bboxes *= bbox_scale
    return images, bboxes, ignore_bboxes


def shrink_bboxes(bboxes):
    rate = 0.5**2
    shrunk_bboxes_list = []
    for bbox in bboxes:
        poly = Polygon(bbox.cpu().numpy())
        offset = poly.area * (1 - rate) / poly.length
        shrunk_poly = poly.buffer(-offset)
        coords = torch.tensor(shrunk_poly.exterior.coords[:4], dtype=torch.float32, device="cuda")
        if not shrunk_poly.is_empty and len(coords) == 4:
            shrunk_bboxes_list.append(coords)
        else:
            shrunk_bboxes_list.append(bbox)
    shrunk_bboxes = torch.stack(shrunk_bboxes_list)
    return shrunk_bboxes


def generate_masks(image, bboxes, ignore_bboxes):
    h, w = image.shape[:2]

    gt_kernels = torch.zeros((h, w), device="cuda")
    gt_text = torch.zeros((h, w), device="cuda")
    training_masks = torch.ones((h, w), device="cuda")
    gt_instances = torch.zeros((h, w), device="cuda")

    if bboxes.numel() > 0:
        min_bboxes = shrink_bboxes(bboxes)

        # create gt instances
        gt_instances = torch.zeros(
            (len(bboxes), 1, h, w), device="cuda", dtype=torch.uint8
        )
        color = torch.arange(1, len(bboxes) + 1, device="cuda", dtype=torch.uint8).view(
            -1, 1
        )
        gt_instances = draw_convex_polygon(gt_instances, bboxes, color)
        gt_kernels = torch.clamp(gt_instances, max=1)
        gt_instances = torch.max(gt_instances, dim=0)[0].squeeze(0)
        gt_text = torch.clamp(gt_instances, max=1)

        # create gt kernels
        # overlap = (gt_kernels.sum(axis=0) > 1).to(dtype=torch.float16).unsqueeze(0)
        # overlap = F.max_pool2d(overlap, kernel_size=3, stride=1, padding=1)
        gt_kernels = -F.max_pool2d(
            -gt_kernels.to(dtype=torch.float16), kernel_size=9, stride=1, padding=4
        ).to(dtype=torch.uint8)
        gt_kernels = torch.max(gt_kernels, dim=0)[0].unsqueeze(0)
        # gt_kernels[overlap > 0] = 0

        gt_kernels_min = torch.zeros((len(bboxes), 1, h, w), device="cuda")
        gt_kernels_min = draw_convex_polygon(
            gt_kernels_min, min_bboxes, torch.tensor([1], device="cuda")
        )
        gt_kernels = gt_kernels.squeeze(0).squeeze(0)

    else:
        gt_instances = torch.zeros((h, w), device="cuda")
        gt_kernels = torch.zeros((h, w), device="cuda")
        gt_text = torch.zeros((h, w), device="cuda")

    if ignore_bboxes.numel() > 0:
        min_ignore_bboxes = shrink_bboxes(ignore_bboxes)

        # create training masks
        training_masks = torch.ones((len(ignore_bboxes), 1, h, w), device="cuda")
        training_masks = draw_convex_polygon(
            training_masks, ignore_bboxes, torch.tensor([0], device="cuda")
        )
        training_masks = draw_convex_polygon(
            training_masks, min_ignore_bboxes, torch.tensor([0], device="cuda")
        )
        training_masks = torch.min(training_masks, dim=0)[0].squeeze(0)
    else:
        training_masks = torch.ones((h, w), device="cuda", dtype=torch.uint8)

    return (
        gt_kernels.float(),
        gt_text.float(),
        training_masks.float(),
        gt_instances.float(),
    )


@pipeline_def(enable_conditionals=True)
def data_pipeline(data_dir, train=True, short_size=640):
    jpegs, _ = fn.readers.file(
        name="ImageReader", file_root=data_dir, shuffle_after_epoch=True, seed=42
    )
    images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB).gpu()
    images1 = fn.reshape(images, layout="CHW")
    bboxes = fn.readers.numpy(
        file_root=f"{data_dir}/bboxes", shuffle_after_epoch=True, seed=42
    ).gpu()
    ignore_bboxes = fn.readers.numpy(
        file_root=f"{data_dir}/ignore_bboxes",
        shuffle_after_epoch=True,
        seed=42,
    ).gpu()

    if train:
        images, bboxes, ignore_bboxes, crop_size = torch_python_function(
            images1,
            bboxes,
            ignore_bboxes,
            function=pad_images,
            num_outputs=4,
        )

        shapes = fn.peek_image_shape(jpegs)
        center = shapes[1::-1] / 2

        images, bboxes, ignore_bboxes = torch_python_function(
            images, bboxes, ignore_bboxes, function=random_flip, num_outputs=3
        )

        rotate = fn.transforms.rotation(
            angle=fn.random.uniform(range=[-15, 15]), center=center
        )
        shear = fn.transforms.shear(
            angles=fn.random.uniform(range=[-15, 15], shape=(2,)), center=center
        )
        mt = fn.transforms.combine(rotate, shear)
        images = fn.warp_affine(images, matrix=mt, fill_value=0, inverse_map=False)
        bboxes = fn.coord_transform(bboxes, MT=mt)
        ignore_bboxes = fn.coord_transform(ignore_bboxes, MT=mt)

        images, bboxes, ignore_bboxes = torch_python_function(
            images,
            bboxes,
            ignore_bboxes,
            crop_size,
            function=random_scale_and_crop,
            num_outputs=3,
            batch_processing=False,
        )

        images, bboxes, ignore_bboxes = torch_python_function(
            images,
            bboxes,
            ignore_bboxes,
            train,
            short_size,
            function=resize,
            num_outputs=3,
        )

        images = fn.reshape(images, layout="HWC")
        images = rand_augment.apply_rand_augment(
            [
                augmentations.brightness,
                augmentations.contrast,
                augmentations.color,
                augmentations.sharpness,
                augmentations.solarize,
                augmentations.invert,
                augmentations.equalize,
                augmentations.auto_contrast,
                augmentations.identity,
            ],
            data=images,
            n=1,
            m=np.random.randint(31),
        )
    else:
        images, bboxes, ignore_bboxes = torch_python_function(
            images,
            bboxes,
            ignore_bboxes,
            train,
            short_size,
            function=resize,
            num_outputs=3,
        )
    images = fn.normalize(images)

    gt_kernels, gt_texts, training_masks, gt_instances = torch_python_function(
        images, bboxes, ignore_bboxes, function=generate_masks, num_outputs=4
    )

    images = torch_python_function(
        images, function=lambda x: x.permute(2, 0, 1), num_outputs=1
    )

    return images, gt_kernels, gt_texts, training_masks, gt_instances


class DALILoader:
    def __init__(self, iterator):
        self.iterator = iterator

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return next(self.iterator)

    def __len__(self):
        return len(self.iterator)


def get_loader(dataset, type, short_size, batch_size=16, num_threads=4, device_id=0):
    if type == "train":
        train = True
        batch_policy = LastBatchPolicy.FILL
    else:
        train = False
        batch_policy = LastBatchPolicy.PARTIAL
    data_dir = f"data/processed/{dataset}/{type}"

    pipe = data_pipeline(
        data_dir,
        train,
        short_size,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )
    data_iter = DALIGenericIterator(
        [pipe],
        ["images", "gt_kernels", "gt_texts", "training_masks", "gt_instances"],
        reader_name="ImageReader",
        auto_reset=True,
        last_batch_policy=batch_policy,
    )
    return DALILoader(data_iter)
