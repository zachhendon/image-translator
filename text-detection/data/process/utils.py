import numpy as np
import math
from shapely.geometry import Point, Polygon
import albumentations as A


def get_polygons(poly):
    G = Polygon(poly.reshape(-1, 2))
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
                if x >= size[0] or x < 0 or y >= size[1] or y < 0:
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
                if x >= size[0] or x < 0 or y >= size[1] or y < 0:
                    continue

                point = Point(x, y)
                if border.contains(point):
                    distance = point.distance(G.exterior)
                    map[x, y] = max(map[x, y], 1 - np.clip(distance / D, 0, 1))
    return map


transform = A.Compose([A.Resize(640, 640)])


def resize_image(image):
    return transform(image=image)["image"]
