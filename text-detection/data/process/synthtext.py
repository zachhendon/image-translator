from tqdm import tqdm
import cv2 as cv
from scipy.io import loadmat
from utils import *
import lmdb
import numpy as np


root_dir = "../raw/synthtext/SynthText"

gt = loadmat(f"{root_dir}/gt.mat")
num_images = gt["imnames"].shape[1]

env = lmdb.open("../processed/lmdb_dataset", map_size=150 * 2**30)

for i in tqdm(range(num_images)):
    path = gt["imnames"][0][i].item()
    image = cv.imread(f"{root_dir}/{path}")
    bboxes = gt["wordBB"][0][i].T.reshape(-1, 4, 2).astype(np.int32)
    bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0, image.shape[1] - 1)
    bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0, image.shape[0] - 1)

    encoded_image = bytes(cv.imencode(".jpg", image)[1])
    gt_kernel, gt_text, edge_weight = get_maps2(bboxes, image.shape[:-1])
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())

    labels = np.stack([edge_weight, gt_text, gt_kernel], axis=-1) * 255
    encoded_maps = bytes(cv.imencode('.jpg', labels)[1])

    with env.begin(write=True) as txn:
        txn.put(f"synthtext_image_{str(i).zfill(6)}".encode(), encoded_image)
        txn.put(f"synthtext_bboxes_{str(i).zfill(6)}".encode(), bboxes)
        txn.put(f"synthtext_maps_{str(i).zfill(6)}".encode(), encoded_maps)
