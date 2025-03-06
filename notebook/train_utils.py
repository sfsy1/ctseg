import numpy as np

AIR_VALUE = -1000  # HU of air


def expand_bbox(bbox, up, down, left, right):
    expanded_box = [bbox[0] - left, bbox[1] - up, bbox[2] + right, bbox[3] + down]
    return [max(0, x) for x in expanded_box]


def expand_bbox_to_multiple(bbox, n, margin):
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
    center_x = round((x1 + x2) / 2)
    center_y = round((y1 + y2) / 2)

    min_multiples = 2
    new_width = max(min_multiples, width // n + (width % n > 0)) * n
    new_height = max(min_multiples, height // n + (height % n > 0)) * n

    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

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
