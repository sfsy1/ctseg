import os
import random
from pathlib import Path

import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from evaluation.eval_utils import get_seg_bbox
from train_utils import crop_from_img, AIR_VALUE, expand_bbox_to_multiple, pad_arr, expand_bbox


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names=None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if image_names is None:
            self.image_names = [x for x in os.listdir(image_dir) if x.endswith('.npy')]
        else:
            self.image_names = image_names
        self.transform = transform
        # self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = Path(self.image_names[idx])
        image_path = self.image_dir / image_name
        mask_name = f"{image_name.stem}_seg"
        mask_path = self.mask_dir / mask_name

        image = np.load(image_path)
        mask, header = nrrd.read(str(mask_path))
        bbox = get_seg_bbox(mask)

        # add some margins to existing bbox and expand to multiples
        multiples = 32
        margin = 2
        bbox = expand_bbox_to_multiple(bbox, multiples, margin)

        # pad then crop image/mask from full slice
        max_margin = multiples + margin
        image = pad_arr(image, max_margin, 'edge', None)

        image_crop = image[
             bbox[1] + max_margin:bbox[3] + max_margin,
             bbox[0] + max_margin:bbox[2] + max_margin,
        ]
        mask = pad_arr(mask, max_margin, "constant", AIR_VALUE)
        mask_crop = mask[
             bbox[1] + max_margin:bbox[3] + max_margin,
             bbox[0] + max_margin:bbox[2] + max_margin,
        ]

        transformed = self.transform(image=image_crop, mask=mask_crop)
        image_crop = transformed['image']
        mask_crop = transformed['mask']

        # create box mask based on the transformed mask_crop
        bbox = get_seg_bbox(mask_crop)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # expand up to 20% of max_dim or 3px, whichever is smaller
        expand = (0, min(3, int(max(w, h) * 0.2)))
        x0, y0, x1, y1 = expand_bbox(bbox, *[random.randint(*expand) for _ in range(4)])
        box_mask_crop = torch.zeros(mask_crop.shape)
        box_mask_crop[y0:y1+1, x0:x1+1] = 1

        input_image = torch.concat([image_crop, box_mask_crop.unsqueeze(0)], dim=0)

        return {
            "input": input_image,
            "image": image_crop,
            "mask": mask_crop.unsqueeze(0),
            "box_mask": box_mask_crop,
            "name": str(image_name)
        }
