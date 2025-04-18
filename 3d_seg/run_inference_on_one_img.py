import sys
from pathlib import Path

import SimpleITK as sitk
import torch

from lighter_zoo import SegResNet
from ctfm_utils import inference


train_images = sys.argv[1]
ct_name = sys.argv[2]

train_images = Path(train_images)

model_name = "project-lighter/whole_body_segmentation"
device = "cuda"
model = SegResNet.from_pretrained(model_name).to(device)

ct_path = train_images / ct_name
ct_img = sitk.ReadImage(ct_path)

print("Running inference...")
with torch.no_grad():
    out = inference(model, ct_path)
out = out.squeeze().permute([2, 1, 0])  # same shape as ct
out = out / 117  # normalize to 0-1

mask = sitk.GetImageFromArray(out.numpy())
mask.CopyInformation(ct_img)

mask_path = train_images / ct_name.replace("_0000.nii.gz", "_0001.nii.gz")
sitk.WriteImage(mask, mask_path)
print("Saved:", mask_path)