import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
import pyclipper


def dilate_polygons(polygons):
    dilated_polygons = []
    for poly in polygons:
        if len(poly) < 4:
            continue
        Gs = Polygon(poly)
        D = Gs.area * 1.5 / Gs.length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

        dilated_polygon = pco.Execute(D)
        if dilated_polygon == []: continue
        dilated_polygon = np.array(dilated_polygon[0]).reshape(-1, 2)
        dilated_polygons.append(dilated_polygon)
    return dilated_polygons


def seg_to_polygons(map):
    bin_map = (map >= 0.2).astype(np.uint8).squeeze()
    contours, _ = cv.findContours(
        bin_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 25 or area > 200000:
            continue
        approx = cv.approxPolyDP(cnt, 0.005 * cv.arcLength(cnt, True), True)
        polygons.append(approx.reshape(-1, 2))
    return polygons


def polygons_to_seg(polygons, size):
    map = np.zeros(size, np.float32)
    for poly in polygons:
        cv.fillPoly(map, np.expand_dims(poly, 0), 1)
    return map


def get_seg_and_poly(output_maps):
    output_maps = output_maps.detach().cpu().numpy()
    bin_maps = (output_maps >= 0.2).astype(np.uint8)

    gt_polygons = []
    gt_maps = np.empty_like(bin_maps, dtype=np.float32)
    for i, map in enumerate(bin_maps):
        polygons = seg_to_polygons(map)
        dilated_polygons = dilate_polygons(polygons)
        gt_polygons.append(dilated_polygons)
        gt_maps[i] = polygons_to_seg(dilated_polygons, (640, 640))
    return gt_polygons, gt_maps
