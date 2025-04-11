import torch
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground, Invert,
    Activations, AsDiscrete, SaveImage, Spacing,
    KeepLargestConnectedComponent
)

from monai.inferers import SlidingWindowInferer


def get_inferer(inference_device='cuda'):
    return SlidingWindowInferer(
        roi_size=[96, 160, 160],
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
        sw_device=inference_device,
        device='cpu',
    )

def get_preprocess(device=torch.device("cpu")):
    return Compose([
        LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
        Spacing(pixdim=(2.0, 2.0, 2.0)),
        EnsureType(device=device),                         # Ensure correct data type
        Orientation(axcodes="SPL"),           # Standardize orientation
        ScaleIntensityRange(-1024, 2048, 0, 1, clip=True),
        CropForeground(allow_smaller=True),
    ])

def get_postprocess(preprocess):
    return Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True),
        # KeepLargestConnectedComponent(),
        Invert(transform=preprocess),
        # SaveImage(output_dir="./ct_fm_output")
    ])