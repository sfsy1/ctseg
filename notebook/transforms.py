from functools import partial

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

VALUE_MIN = -500
VALUE_MAX = 1000


def clip_image(image, min_val, max_val, **kwargs):
    return np.clip(image, min_val, max_val)


def scale_image(image, min_val, max_val, **kwargs):
    return (image - min_val) / (max_val - min_val)


def pass_through(mask, **kwargs):
    return mask


clip_partial = partial(clip_image, min_val=VALUE_MIN, max_val=VALUE_MAX)
scale_partial = partial(scale_image, min_val=VALUE_MIN, max_val=VALUE_MAX)

D = 256

preprocess = [
    A.Lambda(image=clip_partial, mask=pass_through),
    A.Lambda(image=scale_partial, mask=pass_through)
]

resize = [
    A.LongestMaxSize(max_size=D),
    A.PadIfNeeded(min_height=D, min_width=D, border_mode=0, fill=0),
    ToTensorV2(),
]

val_transform = A.Compose([
    *preprocess,
    *resize,
])

train_transform = A.Compose([
    *preprocess,
    # pixel
    A.GaussianBlur(blur_limit=5, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), per_channel=False, p=0.5),
    # spatial
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Transpose(p=0.5),
    A.GridDistortion(p=0.5),  # produces black border at the btm/right
    *resize,
])
