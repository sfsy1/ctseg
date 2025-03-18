import math

import numpy as np

AIR_VALUE = -1000  # HU of air


def expand_bbox(bbox, up, down, left, right):
    expanded_box = [bbox[0] - left, bbox[1] - up, bbox[2] + right, bbox[3] + down]
    return [max(0, x) for x in expanded_box]


def expand_bbox_to_multiple(bbox, n, margin, min_multiples):
    """
    expand bbox by margin, then expand to multiples of n
    bbox: xyxy
    n: multiples
    margin: margin
    """
    x1, y1, x2, y2 = bbox

    # expand bbox by margin
    x1 -= margin
    x2 += margin
    y1 -= margin
    y2 += margin
    width = x2 - x1
    height = y2 - y1
    center_x = math.ceil((x1 + x2) / 2)
    center_y = math.ceil((y1 + y2) / 2)

    # min dimension is min_multiples * n
    max_dim = max(height, width)
    new_width = max(min_multiples, max_dim // n + (max_dim % n > 0)) * n
    new_height = new_width


    left = int(new_width / 2)
    right = new_width - left
    top = int(new_height / 2)
    bottom = new_height - top

    new_x1 = center_x - left
    new_y1 = center_y - top
    new_x2 = center_x + right
    new_y2 = center_y + bottom

    return round(new_x1), round(new_y1), round(new_x2), round(new_y2)


def pad_to_multiples_of_32(image, pad_value=AIR_VALUE):
    """Pads image to dimensions that are multiples of 32 with a specified pad value."""
    h, w = image.shape[:2]
    hp = 32 - h % 32 if h % 32 else 0
    wp = 32 - w % 32 if w % 32 else 0
    tp, bp = hp // 2, hp - hp // 2
    lp, rp = wp // 2, wp - wp // 2
    return np.pad(image, ((tp, bp), (lp, rp)), mode='constant', constant_values=pad_value)


def pad_arr(arr, m, mode, pad_value):
    if mode == 'constant':
        return np.pad(arr, ((m, m), (m, m)), mode='constant', constant_values=pad_value)
    elif mode == 'edge':
        return np.pad(arr, ((m, m), (m, m)), mode='edge')


def crop_from_img(img: np.ndarray, crop_bbox, m, pad_value):
    """Crop bbox from img_arr. img_arr is padded so that margins work for crops near the edges"""
    img = pad_arr(img, m, pad_value)
    return img[crop_bbox[1]:crop_bbox[3] + 1 + 2 * m, crop_bbox[0]:crop_bbox[2] + 1 + 2 * m]
