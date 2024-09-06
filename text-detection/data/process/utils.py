import numpy as np
from shapely.geometry import Polygon
import albumentations as A
import pyclipper
import cv2 as cv


def get_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(
        xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
        xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
        point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
        (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
                     square_sin / square_distance + 1e-4)

    result[cosin < 0] = np.sqrt(
        np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def draw_polygon(polygon, thresh_map, bin_map, gt_map):
    polygon = polygon.reshape(-1, 2)
    G = Polygon(polygon)
    D = G.area * (1 - 0.4 ** 2) / G.length

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(np.array(polygon).reshape(-1, 2),
                pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    Gd = Polygon(pco.Execute(D)[0])

    Gs = pco.Execute(-D)
    # print(np.array(Gs).dtype, np.expand_dims(polygon, 0).astype(np.int64).dtype)
    if Gs != []:
        cv.fillPoly(bin_map, np.array(Gs), 1)
        cv.fillPoly(gt_map, np.expand_dims(polygon, 0).astype(np.int64), 1)

    xmin, ymin, xmax, ymax = [int(bound) for bound in Gd.bounds]

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    xs = np.broadcast_to(np.linspace(0, w - 1, w).reshape(1, w), (h, w))
    ys = np.broadcast_to(np.linspace(0, h - 1, h).reshape(h, 1), (h, w))
    xs, ys

    polygon[:, 0] -= xmin
    polygon[:, 1] -= ymin

    distance_map = np.zeros((polygon.shape[0], h, w), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = get_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / D, 0, 1)
    distance_map = distance_map.min(0)

    xmin_valid = min(max(0, xmin), thresh_map.shape[1] - 1)
    xmax_valid = min(max(0, xmax), thresh_map.shape[1] - 1)
    ymin_valid = min(max(0, ymin), thresh_map.shape[0] - 1)
    ymax_valid = min(max(0, ymax), thresh_map.shape[0] - 1)
    thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid-ymin:ymax_valid-ymax+h,
            xmin_valid-xmin:xmax_valid-xmax+w],
        thresh_map[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def get_maps(gt, size):
    thresh_map = np.zeros(size, dtype=np.float32)
    thresh_map = thresh_map * (0.7 - 0.3) + 0.3

    bin_map = np.zeros(size, dtype=np.float32)
    gt_map = np.zeros(size, dtype=np.float32)
    for polygon in gt:
        draw_polygon(polygon, thresh_map, bin_map, gt_map)
    return thresh_map, bin_map, gt_map


transform = A.Compose([A.Resize(640, 640)])


def resize_image(image):
    return transform(image=image)["image"]
