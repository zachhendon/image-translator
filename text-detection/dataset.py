import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
import math
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import time


def get_polygons(poly):
    G = Polygon(poly[:-1].reshape(-1, 2))
    D = G.area * (1 - 0.4**2) / G.length
    Gs = G.buffer(-D, join_style=2)
    Gd = G.buffer(D, join_style=2)
    return G, Gs, Gd, D


def get_bin_map(gt, size):
    map = np.zeros(size, dtype=np.float32)

    for poly in gt:
        G, _, _, _ = get_polygons(poly)
        x1, y1, x2, y2 = G.bounds
        x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)

        for i in range(x2 - x1 + 1):
            for j in range(y2 - y1 + 1):
                x = x1 + i
                y = y1 + j
                if x > size[0] or x < 0 or y > size[1] or y < 0:
                    continue

                point = Point(x, y)
                if G.contains(point):
                    map[x, y] = 1
    return map


def get_prob_map(gt, size):
    map = np.zeros(size, dtype=np.float32)

    for poly in gt:
        _, Gs, _, D = get_polygons(poly)
        x1, y1, x2, y2 = Gs.bounds
        x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)

        for i in range(x2 - x1 + 1):
            for j in range(y2 - y1 + 1):
                x = x1 + i
                y = y1 + j
                if x > size[0] or x < 0 or y > size[1] or y < 0:
                    continue

                point = Point(x, y)
                if Gs.contains(point):
                    map[x, y] = 1
    return map


def get_thresh_map(gt, size):
    map = np.zeros(size, dtype=np.float32)

    for poly in gt:
        G, Gs, Gd, D = get_polygons(poly)
        border = Gd.difference(Gs)
        x1, y1, x2, y2 = border.bounds
        x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)

        for i in range(x2 - x1 + 1):
            for j in range(y2 - y1 + 1):
                x = x1 + i
                y = y1 + j
                if x > size[0] or x < 0 or y > size[1] or y < 0:
                    continue

                point = Point(x, y)
                if border.contains(point):
                    p1, p2 = nearest_points(G.exterior, point)
                    distance = p1.distance(p2)
                    map[x, y] = max(map[x, y], 1 - np.clip(distance / D, 0, 1))
    return map


class ICDR2015Dataset(Dataset):
    def __init__(self, datadir, train=True):
        if train:
            imagedir = f"{datadir}/train_images"
            gtdir = f"{datadir}/train_gts"
        else:
            imagedir = f"{datadir}/test_images"
            gtdir = f"{datadir}/test_gts"

        self.image_paths = [
            f"{imagedir}/img_{i+1}.jpg" for i in range(len(os.listdir(imagedir)))
        ]
        self.gt_paths = [
            f"{gtdir}/gt_img_{i+1}.txt" for i in range(len(os.listdir(gtdir)))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        gt = pd.read_csv(self.gt_paths[idx], header=None)
        gt = gt.to_numpy()

        size = image.size
        bin_map = get_bin_map(gt, size)
        prob_map = get_prob_map(gt, size)
        thresh_map = get_thresh_map(gt, size)
        return image, (bin_map, prob_map, thresh_map)
