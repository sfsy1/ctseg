from monai.transforms import (
    Compose, LoadImage, Orientation,
    ScaleIntensityRange, Activations, AsDiscrete,
    Invert, Spacing
)


preprocess_seg = Compose([
    LoadImage(ensure_channel_first=True),
    Orientation(axcodes="SPL"),
    Spacing((1, 1, 1), mode="bilinear"),
])

postprocess_seg = Invert(preprocess_seg)

preprocess = Compose([
    LoadImage(ensure_channel_first=True),
    Orientation(axcodes="SPL"),
    Spacing((1, 1, 1), mode="bilinear"),
    ScaleIntensityRange(-1024, 2048, 0, 1, clip=True),
])

postprocess = Compose([
    Activations(softmax=True),
    AsDiscrete(argmax=True),
    Invert(preprocess)
    # KeepLargestConnectedComponent(),
])
