from tqdm import tqdm
import cv2 as cv
from scipy.io import loadmat
from utils import get_maps
import lmdb
import numpy as np


root_dir = "../raw/synthtext/SynthText"

gt = loadmat("../raw/synthtext/SynthText/gt.mat")
num_images = gt["imnames"].shape[1]

env = lmdb.open("../processed/lmdb_dataset", map_size=150 * 2**30)

with env.begin(write=True) as txn:
    for i in tqdm(range(num_images)):
        path = gt["imnames"][0][i].item()
        bboxes = gt["wordBB"][0][i].T.reshape(-1, 4, 2).astype(np.uint64)

        img = cv.imread(f"../raw/synthtext/SynthText/{path}")
        h, w, _ = img.shape

        gt_text, gt_kernel = get_maps(bboxes, (h, w))

        encoded_image = bytes(cv.imencode(".jpg", img)[1])
        txn.put(f"synthtext_image_{str(i).zfill(6)}".encode(), encoded_image)

        txn.put(
            f"synthtext_nbounds_{str(i).zfill(6)}".encode(), str(len(bboxes)).encode()
        )
        txn.put(f"synthtext_bounds_{str(i).zfill(6)}".encode(), bboxes)

        encoded_gt_text = bytes(cv.imencode(".jpg", gt_text * 255)[1])
        txn.put(f"synthtext_text_{str(i).zfill(6)}".encode(), encoded_gt_text)
        encoded_gt_kernel = bytes(cv.imencode(".jpg", gt_kernel * 255)[1])
        txn.put(f"synthtext_kernel_{str(i).zfill(6)}".encode(), encoded_gt_kernel)
