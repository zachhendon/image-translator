import torch
import torch.nn.functional as F
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
import kornia.augmentation as K
from kornia.utils import draw_convex_polygon
from glob import glob

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
    data_keys=["input", "keypoints", "keypoints", "keypoints", "keypoints"],
    same_on_batch=False,
)

resize = K.AugmentationSequential(
    K.SmallestMaxSize(640),
    data_keys=["input", "keypoints", "keypoints", "keypoints", "keypoints"],
    same_on_batch=True,
)

normalize = K.Normalize(
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225]),
)


def kornia_augment_batch(
    train,
    images,
    bboxes_pad,
    min_bboxes_pad,
    num_bboxes,
    ignore_bboxes_pad,
    min_ignore_bboxes_pad,
    num_ignore_bboxes,
):
    batch_size = len(images)
    batch_images = torch.stack(images, dim=0)
    batch_bboxes = torch.stack(bboxes_pad, dim=0).view(batch_size, -1, 2)
    batch_min_bboxes = torch.stack(min_bboxes_pad, dim=0).view(batch_size, -1, 2)
    batch_ignore_bboxes = torch.stack(ignore_bboxes_pad, dim=0).view(batch_size, -1, 2)
    batch_min_ignore_bboxes = torch.stack(min_ignore_bboxes_pad, dim=0).view(
        batch_size, -1, 2
    )
    if train[0]:
        transformed = aug(
            batch_images,
            batch_bboxes,
            batch_min_bboxes,
            batch_ignore_bboxes,
            batch_min_ignore_bboxes,
        )
    else:
        transformed = resize(
            batch_images,
            batch_bboxes,
            batch_min_bboxes,
            batch_ignore_bboxes,
            batch_min_ignore_bboxes,
        )
    images_aug, bboxes_aug, min_bboxes_aug, ignore_bboxes_aug, min_ignore_bboxes_aug = (
        transformed
    )
    images_normalized = normalize(images_aug)

    bboxes_aug = bboxes_aug.view(batch_size, -1, 4, 2)
    min_bboxes_aug = min_bboxes_aug.view(batch_size, -1, 4, 2)
    ignore_bboxes_aug = ignore_bboxes_aug.view(batch_size, -1, 4, 2)
    min_ignore_bboxes_aug = min_ignore_bboxes_aug.view(batch_size, -1, 4, 2)

    bboxes = []
    min_bboxes = []
    ignore_bboxes = []
    min_ignore_bboxes = []
    for i in range(len(num_bboxes)):
        bboxes.append(bboxes_aug[i][: num_bboxes[i].item()])
        min_bboxes.append(min_bboxes_aug[i][: num_bboxes[i].item()])
        # print(bboxes[i].shape)
    for i in range(len(num_ignore_bboxes)):
        ignore_bboxes.append(ignore_bboxes_aug[i][: num_ignore_bboxes[i].item()])
        min_ignore_bboxes.append(
            min_ignore_bboxes_aug[i][: num_ignore_bboxes[i].item()]
        )
    # print(len(bboxes))

    return (
        list(images_normalized.unbind(dim=0)),
        bboxes,
        min_bboxes,
        ignore_bboxes,
        min_ignore_bboxes,
    )


def get_masks(images, bboxes, min_bboxes):
    h, w = images[0].shape[1:]
    batch_bboxes = torch.vstack(bboxes)
    batch_min_bboxes = torch.vstack(min_bboxes)
    if batch_bboxes.shape[0] == 0:
        return (
            list(torch.zeros((len(bboxes), 1, h, w), device="cuda").unbind(dim=0)),
            list(torch.zeros((len(bboxes), 1, h, w), device="cuda").unbind(dim=0)),
        )

    text_components = torch.zeros(
        batch_bboxes.shape[0],
        1,
        h,
        w,
        dtype=torch.float32,
        device="cuda",
    )
    text_components = draw_convex_polygon(
        text_components, batch_bboxes, torch.tensor([1], device="cuda")
    )

    kernel_components = -F.max_pool2d(
        -text_components,
        kernel_size=9,
        stride=1,
        padding=4,
    )
    min_components = torch.zeros(
        batch_bboxes.shape[0],
        1,
        h,
        w,
        dtype=torch.float32,
        device="cuda",
    )
    min_components = draw_convex_polygon(
        min_components, batch_min_bboxes, torch.tensor([1], device="cuda")
    )

    text_masks = []
    kernel_masks = []
    l = 0
    for i in range(len(bboxes)):
        r = l + bboxes[i].shape[0]
        if l == r:
            text_masks.append(torch.zeros((1, h, w), device="cuda"))
            kernel_masks.append(torch.zeros((1, h, w), device="cuda"))
        else:
            text_masks.append(torch.clamp(text_components[l:r].sum(axis=0), 0, 1))
            kernel_masks.append(
                torch.clamp(
                    (kernel_components[l:r] + min_components[l:r]).sum(axis=0), 0, 1
                )
            )
        l = r
    return text_masks, kernel_masks


def pad_keypoints(bboxes, ignore_bboxes):
    num_bboxes = [torch.tensor(bbox.shape[0], device="cuda") for bbox in bboxes]
    num_ignore_bboxes = [
        torch.tensor(bbox.shape[0], device="cuda") for bbox in ignore_bboxes
    ]
    max_bboxes = max(num_bboxes).item()
    max_ignore_bboxes = max(num_ignore_bboxes).item()

    for i, bbox in enumerate(bboxes):
        bboxes[i] = torch.vstack(
            [
                bbox,
                torch.empty(
                    (max_bboxes - bbox.shape[0], *bbox.shape[1:]), device="cuda"
                ),
            ]
        )
    for i, bbox in enumerate(ignore_bboxes):
        ignore_bboxes[i] = torch.vstack(
            [
                bbox,
                torch.empty(
                    (max_ignore_bboxes - bbox.shape[0], *bbox.shape[1:]), device="cuda"
                ),
            ]
        )

    return (
        bboxes,
        num_bboxes,
        ignore_bboxes,
        num_ignore_bboxes,
    )


@pipeline_def
def my_pipeline(datadir, train):
    jpegs, _ = fn.readers.file(files=sorted(glob(f"{datadir}/images/*")))
    bboxes = fn.readers.numpy(files=sorted(glob(f"{datadir}/bboxes/*")), device="gpu")
    min_bboxes = fn.readers.numpy(
        files=sorted(glob(f"{datadir}/min_bboxes/*")),
        device="gpu",
    )
    ignore_bboxes = fn.readers.numpy(
        files=sorted(glob(f"{datadir}/ignore_bboxes/*")),
        device="gpu",
    )
    min_ignore_bboxes = fn.readers.numpy(
        files=sorted(glob(f"{datadir}/min_ignore_bboxes/*")),
        device="gpu",
    )

    images = fn.decoders.image(jpegs, device="cpu", output_type=types.BGR).gpu()

    images = fn.transpose(images, perm=[2, 0, 1]) / 255

    bboxes_pad, num_bboxes, ignore_bboxes_pad, num_ignore_bboxes = (
        torch_python_function(
            bboxes,
            ignore_bboxes,
            function=pad_keypoints,
            num_outputs=4,
            batch_processing=True,
        )
    )
    min_bboxes_pad, _, min_ignore_bboxes_pad, _ = torch_python_function(
        min_bboxes,
        min_ignore_bboxes,
        function=pad_keypoints,
        num_outputs=4,
        batch_processing=True,
    )

    images, bboxes, min_bboxes, ignore_bboxes, min_ignore_bboxes = (
        torch_python_function(
            train,
            images,
            bboxes_pad,
            min_bboxes_pad,
            num_bboxes,
            ignore_bboxes_pad,
            min_ignore_bboxes_pad,
            num_ignore_bboxes,
            function=kornia_augment_batch,
            num_outputs=5,
            batch_processing=True,
        )
    )

    text_masks, kernel_masks = torch_python_function(
        images,
        bboxes,
        min_bboxes,
        function=get_masks,
        num_outputs=2,
        batch_processing=True,
    )
    ignore_text_masks, ignore_kernel_masks = torch_python_function(
        images,
        ignore_bboxes,
        min_ignore_bboxes,
        function=get_masks,
        num_outputs=2,
        batch_processing=True,
    )

    if train:
        return (
            images,
            text_masks,
            kernel_masks,
            ignore_text_masks,
            ignore_kernel_masks,
        )
    else:
        bboxes, num_bboxes, ignore_bboxes, num_ignore_bboxes = torch_python_function(
            bboxes,
            ignore_bboxes,
            function=pad_keypoints,
            num_outputs=4,
            batch_processing=True,
        )

        return (
            images,
            bboxes_pad,
            num_bboxes,
            ignore_bboxes_pad,
            num_ignore_bboxes,
            text_masks,
            kernel_masks,
            ignore_text_masks,
            ignore_kernel_masks,
        )


def get_loader(batch_size):
    train_dir = "data/processed/icdar2015/train"
    val_dir = "data/processed/icdar2015/val"
    train_loader = DALIGenericIterator(
        [
            my_pipeline(
                train_dir,
                train=True,
                batch_size=batch_size,
                num_threads=1,
                device_id=0,
                prefetch_queue_depth=2,
            )
        ],
        [
            "data",
            "text_masks",
            "kernel_masks",
            "ignore_text_masks",
            "ignore_kernel_masks",
        ],
    )
    val_loader = DALIGenericIterator(
        [
            my_pipeline(
                val_dir,
                train=False,
                batch_size=batch_size,
                num_threads=1,
                device_id=0,
                prefetch_queue_depth=2,
            )
        ],
        [
            "data",
            "bboxes",
            "num_bboxes",
            "ignore_bboxes",
            "num_ignore_bboxes",
            "text_masks",
            "kernel_masks",
            "ignore_text_masks",
            "ignore_kernel_masks",
        ],
    )
    return train_loader, val_loader
