import math
import torch
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground, Invert,
    Activations, AsDiscrete, KeepLargestConnectedComponent,
    SaveImage
)
from monai.inferers import SlidingWindowInferer
from torch.nn import Upsample


def create_inferer():
    return SlidingWindowInferer(
        roi_size=[96, 160, 160],  # Size of patches to process
        sw_batch_size=2,          # Number of windows to process in parallel
        overlap=0.5,            # Overlap between windows (reduces boundary artifacts)
        mode="gaussian",           # Gaussian weighting for overlap regions
        sw_device=torch.device("cuda"),
        device=torch.device('cpu'),
    )

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    Orientation(axcodes="SPL"),           # Standardize orientation
    # Scale intensity to [0,1] range, clipping outliers
    ScaleIntensityRange(
        a_min=-1024,    # Min HU value
        a_max=2048,     # Max HU value
        b_min=0,        # Target min
        b_max=1,        # Target max
        clip=True       # Clip values outside range
    ),
    CropForeground(allow_smaller=True)    # Remove background to reduce computation
])

# Postprocessing pipeline
postprocess = Compose([
    Activations(softmax=True),              # Apply softmax to get probabilities
    AsDiscrete(argmax=True, dtype=torch.int32),  # Convert to class labels
    # KeepLargestConnectedComponent(),        # Remove small disconnected regions
    Invert(transform=preprocess),           # Restore original space
    # Save the result
    # SaveImage(output_dir="./segmentations")
])

# downsampling scale
ds = (2, 2, 2)
upsample = Upsample(scale_factor=ds, mode='nearest')

def downsample_tensor_if_needed(x, max_size=8e7):
    if math.prod(x.shape) > max_size:
        # downsample 2x, check if odd_slices
        return x[:, ::ds[0], ::ds[1], ::ds[2]], True, bool(x.shape[1] % 2)
    return x, False, False


def inference(model, input_path, save_path=None):
    input_tensor = preprocess(input_path)
    input_tensor, downsampled, odd = downsample_tensor_if_needed(input_tensor)
    print(f"Preprocessed. {input_tensor.shape}")

    inferer = create_inferer()
    with torch.no_grad():
        print("Infering...")
        output = inferer(input_tensor.unsqueeze(dim=0), model)[0]
        print("Inferer done.")
    # Copy metadata from input
    output.applied_operations = input_tensor.applied_operations
    output.affine = input_tensor.affine

    result = postprocess(output[0])
    print("Postprocess done.")
    if downsampled:
        result = upsample(result.unsqueeze(0).float()).squeeze(0)
        if odd:  #  remove extra slice
            result = result[:,:,:,:-1]

    if save_path:
        print(f"Saving to {save_path}")
        SaveImage()(result, filename=save_path)
    return result