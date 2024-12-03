import torch
import torch.nn.functional as F
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
import kornia.augmentation as K
from kornia.utils import draw_convex_polygon

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
    polys_pad,
    min_polys_pad,
    num_polys,
    ignore_polys_pad,
    min_ignore_polys_pad,
    num_ignore_polys,
):
    batch_size = len(images)
    batch_images = torch.stack(images, dim=0)
    batch_polys = torch.stack(polys_pad, dim=0).view(batch_size, -1, 2)
    batch_min_polys = torch.stack(min_polys_pad, dim=0).view(batch_size, -1, 2)
    batch_ignore_polys = torch.stack(ignore_polys_pad, dim=0).view(batch_size, -1, 2)
    batch_min_ignore_polys = torch.stack(min_ignore_polys_pad, dim=0).view(
        batch_size, -1, 2
    )
    if train[0]:
        transformed = aug(
            batch_images,
            batch_polys,
            batch_min_polys,
            batch_ignore_polys,
            batch_min_ignore_polys,
        )
    else:
        transformed = resize(
            batch_images,
            batch_polys,
            batch_min_polys,
            batch_ignore_polys,
            batch_min_ignore_polys,
        )
    images_aug, polys_aug, min_polys_aug, ignore_polys_aug, min_ignore_polys_aug = (
        transformed
    )
    images_normalized = normalize(images_aug)

    polys_aug = polys_aug.view(batch_size, -1, 4, 2)
    min_polys_aug = min_polys_aug.view(batch_size, -1, 4, 2)
    ignore_polys_aug = ignore_polys_aug.view(batch_size, -1, 4, 2)
    min_ignore_polys_aug = min_ignore_polys_aug.view(batch_size, -1, 4, 2)

    polys = []
    min_polys = []
    ignore_polys = []
    min_ignore_polys = []
    for i in range(len(num_polys)):
        polys.append(polys_aug[i][: num_polys[i].item()])
        min_polys.append(min_polys_aug[i][: num_polys[i].item()])
    for i in range(len(num_ignore_polys)):
        ignore_polys.append(ignore_polys_aug[i][: num_ignore_polys[i].item()])
        min_ignore_polys.append(min_ignore_polys_aug[i][: num_ignore_polys[i].item()])

    return (
        list(images_normalized.unbind(dim=0)),
        polys,
        min_polys,
        ignore_polys,
        min_ignore_polys,
    )


def get_masks(images, polys, min_polys):
    h, w = images[0].shape[1:]
    batch_polys = torch.vstack(polys)
    batch_min_polys = torch.vstack(min_polys)
    if batch_polys.shape[0] == 0:
        return (
            list(torch.zeros((len(polys), 1, h, w), device="cuda").unbind(dim=0)),
            list(torch.zeros((len(polys), 1, h, w), device="cuda").unbind(dim=0)),
        )

    text_components = torch.zeros(
        batch_polys.shape[0],
        1,
        h,
        w,
        dtype=torch.float32,
        device="cuda",
    )
    text_components = draw_convex_polygon(
        text_components, batch_polys, torch.tensor([1], device="cuda")
    )

    kernel_components = -F.max_pool2d(
        -text_components,
        kernel_size=9,
        stride=1,
        padding=4,
    )
    min_components = torch.zeros(
        batch_polys.shape[0],
        1,
        h,
        w,
        dtype=torch.float32,
        device="cuda",
    )
    min_components = draw_convex_polygon(
        min_components, batch_min_polys, torch.tensor([1], device="cuda")
    )

    text_masks = []
    kernel_masks = []
    l = 0
    for i in range(len(polys)):
        r = l + polys[i].shape[0]
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
        return torch.zeros((0, 4, 2), device="cuda")
    else:
        return torch.stack(polys)


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
        shuffle_after_epoch=train
    )
    images = fn.decoders.image(jpegs, device="cpu", output_type=types.BGR).gpu()
    images = fn.transpose(images, perm=[2, 0, 1]) / 255

    categories = categories.gpu()
    poly_groups = poly_groups.gpu()
    keypoints = keypoints.gpu()
    polys, min_polys, ignore_polys, min_ignore_polys = torch_python_function(
        categories,
        poly_groups,
        keypoints,
        function=get_polygons,
        num_outputs=4,
        batch_processing=False,
    )

    polys_pad, num_polys, ignore_polys_pad, num_ignore_polys = torch_python_function(
        polys,
        ignore_polys,
        function=pad_keypoints,
        num_outputs=4,
        batch_processing=True,
    )
    min_polys_pad, _, min_ignore_polys_pad, _ = torch_python_function(
        min_polys,
        min_ignore_polys,
        function=pad_keypoints,
        num_outputs=4,
        batch_processing=True,
    )

    images, polys, min_polys, ignore_polys, min_ignore_polys = torch_python_function(
        train,
        images,
        polys_pad,
        min_polys_pad,
        num_polys,
        ignore_polys_pad,
        min_ignore_polys_pad,
        num_ignore_polys,
        function=kornia_augment_batch,
        num_outputs=5,
        batch_processing=True,
    )

    text_masks, kernel_masks = torch_python_function(
        images,
        polys,
        min_polys,
        function=get_masks,
        num_outputs=2,
        batch_processing=True,
    )
    ignore_text_masks, ignore_kernel_masks = torch_python_function(
        images,
        ignore_polys,
        min_ignore_polys,
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
        polys, num_polys, ignore_polys, num_ignore_polys = torch_python_function(
            polys,
            ignore_polys,
            function=pad_keypoints,
            num_outputs=4,
            batch_processing=True,
        )

        return (
            images,
            polys_pad,
            num_polys,
            ignore_polys_pad,
            num_ignore_polys,
            text_masks,
            kernel_masks,
            ignore_text_masks,
            ignore_kernel_masks,
        )


def get_loader(batch_size, dset):
    train_dir = f"data/processed/{dset}/train"
    val_dir = f"data/processed/{dset}/val"
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
            "images",
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
            "images",
            "polys",
            "num_polys",
            "ignore_polys",
            "num_ignore_polys",
            "text_masks",
            "kernel_masks",
            "ignore_text_masks",
            "ignore_kernel_masks",
        ],
    )
    return train_loader, val_loader
