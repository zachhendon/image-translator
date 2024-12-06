import torch
import torch.nn.functional as F
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.fn import python_function
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
import kornia.augmentation as K
from kornia.utils import draw_convex_polygon
import numpy as np
import cv2 as cv

aug = K.AugmentationSequential(
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
    K.ColorJitter(0.4, 0.4, 0.4, 0.1),
    K.RandomGaussianBlur((3, 3), (0.1, 2.0)),
    K.RandomSharpness(0.5),
    # K.RandomGaussianNoise(mean=0, std=0.05, p=1),
    data_keys=["image", "mask", "mask", "mask", "mask"],
    same_on_batch=False,
)

resize = K.AugmentationSequential(
    K.SmallestMaxSize(640),
    data_keys=["image", "mask", "mask", "mask", "mask"],
    same_on_batch=True,
)

normalize = K.Normalize(
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225]),
)

call_counter = 0
CACHE_CLEAR_THRESHOLD = 10


def kornia_augment_batch(
    train,
    images,
    kernel_masks,
    ignore_kernel_masks,
    text_masks,
    ignore_text_masks,
):
    batch_size = len(images)
    batch_images = torch.cat(images).view(batch_size, *images[0].shape).half()
    batch_kernel_masks = torch.cat(kernel_masks).view(
        batch_size, *kernel_masks[0].shape
    )
    batch_ignore_kernel_masks = torch.cat(ignore_kernel_masks).view(
        batch_size, *ignore_kernel_masks[0].shape
    )
    batch_text_masks = torch.cat(text_masks).view(batch_size, *text_masks[0].shape)
    batch_ignore_text_masks = torch.cat(ignore_text_masks).view(
        batch_size, *ignore_text_masks[0].shape
    )

    (
        images_aug,
        kernel_masks_aug,
        ignore_kernel_masks_aug,
        text_masks_aug,
        ignore_text_masks_aug,
    ) = (
        aug(
            batch_images,
            batch_kernel_masks.unsqueeze(1),
            batch_ignore_kernel_masks.unsqueeze(1),
            batch_text_masks.unsqueeze(1),
            batch_ignore_text_masks.unsqueeze(1),
        )
        if train[0]
        else resize(
            batch_images,
            batch_kernel_masks.unsqueeze(1),
            batch_ignore_kernel_masks.unsqueeze(1),
            batch_text_masks.unsqueeze(1),
            batch_ignore_text_masks.unsqueeze(1),
        )
    )

    images_normalized = normalize(images_aug)
    kernel_masks_aug = kernel_masks_aug.squeeze(1)
    ignore_kernel_masks_aug = ignore_kernel_masks_aug.squeeze(1)
    text_masks_aug = text_masks_aug.squeeze(1)
    ignore_text_masks_aug = ignore_text_masks_aug.squeeze(1)

    return (
        list(images_normalized),
        list(kernel_masks_aug),
        list(ignore_kernel_masks_aug),
        list(text_masks_aug),
        list(ignore_text_masks_aug),
    )


def get_masks(images, polys, min_polys):
    h, w = images[0].shape[1:]
    batch_polys = np.vstack(polys)
    batch_min_polys = np.vstack(min_polys)
    if batch_polys.shape[0] == 0:
        return (
            list(np.zeros((len(polys), h, w))),
            list(np.zeros((len(polys), h, w))),
        )

    text_components = np.zeros(
        (batch_polys.shape[0], 1, h, w),
        dtype=np.float16,
    )
    cv.fillPoly(text_components, [poly.astype(np.int32) for poly in batch_polys], 1)

    kernel_components = (
        -F.max_pool2d(
            -torch.from_numpy(text_components).to(dtype=torch.float16, device="cuda"),
            kernel_size=9,
            stride=1,
            padding=4,
        )
        .cpu()
        .numpy()
    )
    min_components = np.zeros((batch_polys.shape[0], 1, h, w), dtype=np.float16)
    cv.fillPoly(min_components, [poly.astype(np.int32) for poly in batch_min_polys], 1)

    text_components = text_components.squeeze(1)
    kernel_components = kernel_components.squeeze(1)
    min_components = min_components.squeeze(1)

    text_masks = []
    kernel_masks = []
    l = 0
    for i in range(len(polys)):
        r = l + polys[i].shape[0]
        if l == r:
            text_masks.append(np.zeros((h, w), dtype=np.float16))
            kernel_masks.append(np.zeros((h, w), dtype=np.float16))
        else:
            text_masks.append(np.clip(text_components[l:r].sum(axis=0), 0, 1))
            kernel_masks.append(
                np.clip(
                    (kernel_components[l:r] + min_components[l:r]).sum(axis=0), 0, 1
                )
            )
            l = r

    text_masks = list(text_masks)
    kernel_masks = list(kernel_masks)

    return text_masks, kernel_masks


def pad_keypoints(polys, ignore_polys):
    num_polys = [torch.tensor(poly.shape[0], device="cuda") for poly in polys]
    num_ignore_polys = [
        torch.tensor(poly.shape[0], device="cuda") for poly in ignore_polys
    ]
    max_polys = max(num_polys).item()
    max_ignore_polys = max(num_ignore_polys).item()

    for i, poly in enumerate(polys):
        polys[i] = torch.vstack(
            [
                poly,
                torch.empty(
                    (max_polys - poly.shape[0], *poly.shape[1:]), device="cuda"
                ),
            ]
        )
    for i, poly in enumerate(ignore_polys):
        ignore_polys[i] = torch.vstack(
            [
                poly,
                torch.empty(
                    (max_ignore_polys - poly.shape[0], *poly.shape[1:]), device="cuda"
                ),
            ]
        )

    return (
        polys,
        num_polys,
        ignore_polys,
        num_ignore_polys,
    )


def stack_polys(polys):
    if not polys:
        return np.zeros((0, 4, 2), dtype=np.float16)
    else:
        return np.stack(polys).astype(np.float16)


def get_polygons(categories, poly_groups, keypoints):
    polys = []
    min_polys = []
    ignore_polys = []
    min_ignore_polys = []

    for i in range(len(categories)):
        c = categories[i].item()
        l, r = poly_groups[i][1:]
        poly = keypoints[l:r]

        if c == 1:
            polys.append(poly)
        elif c == 2:
            min_polys.append(poly)
        elif c == 3:
            ignore_polys.append(poly)
        elif c == 4:
            min_ignore_polys.append(poly)

    polys = stack_polys(polys)
    min_polys = stack_polys(min_polys)
    ignore_polys = stack_polys(ignore_polys)
    min_ignore_polys = stack_polys(min_ignore_polys)
    return polys, min_polys, ignore_polys, min_ignore_polys


@pipeline_def
def my_pipeline(datadir, train):
    jpegs, _, categories, poly_groups, keypoints = fn.readers.coco(
        file_root=f"{datadir}/images",
        annotations_file=f"{datadir}/annotations.json",
        polygon_masks=True,
        shuffle_after_epoch=True,
    )
    images = fn.decoders.image(jpegs, device="cpu", output_type=types.BGR)
    images = fn.transpose(images, perm=[2, 0, 1]) / 255

    # get polygons from keypoints
    polys, min_polys, ignore_polys, min_ignore_polys = python_function(
        categories,
        poly_groups,
        keypoints,
        function=get_polygons,
        num_outputs=4,
        batch_processing=False,
    )

    # convert polygons to masks
    text_masks, kernel_masks = python_function(
        images,
        polys,
        min_polys,
        function=get_masks,
        num_outputs=2,
        batch_processing=True,
    )
    ignore_text_masks, ignore_kernel_masks = python_function(
        images,
        ignore_polys,
        min_ignore_polys,
        function=get_masks,
        num_outputs=2,
        batch_processing=True,
    )

    images = images.gpu()
    kernel_masks = kernel_masks.gpu()
    ignore_kernel_masks = ignore_kernel_masks.gpu()
    text_masks = text_masks.gpu()
    ignore_text_masks = ignore_text_masks.gpu()

    images, kernel_masks, ignore_kernel_masks, text_masks, ignore_text_masks = (
        torch_python_function(
            train,
            images,
            kernel_masks,
            ignore_kernel_masks,
            text_masks,
            ignore_text_masks,
            function=kornia_augment_batch,
            num_outputs=5,
            batch_processing=True,
        )
    )
    return (
        images,
        kernel_masks,
        ignore_kernel_masks,
        text_masks,
        ignore_text_masks,
    )


def get_loader(batch_size, datadir):
    train_dir = f"{datadir}/train"
    val_dir = f"{datadir}/val"
    train_loader = DALIGenericIterator(
        [
            my_pipeline(
                train_dir,
                train=True,
                batch_size=batch_size,
                num_threads=2,
                device_id=0,
                prefetch_queue_depth=2,
            )
        ],
        [
            "images",
            "kernel_masks",
            "ignore_kernel_masks",
            "text_masks",
            "ignore_text_masks",
        ],
    )
    val_loader = DALIGenericIterator(
        [
            my_pipeline(
                val_dir,
                train=False,
                batch_size=batch_size,
                num_threads=2,
                device_id=0,
                prefetch_queue_depth=2,
            )
        ],
        [
            "images",
            "kernel_masks",
            "ignore_kernel_masks",
            "text_masks",
            "ignore_text_masks",
            # "polys",
            # "num_polys",
            # "ignore_polys",
            # "num_ignore_polys",
        ],
    )
    return train_loader, val_loader
