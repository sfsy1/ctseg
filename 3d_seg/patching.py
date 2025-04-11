import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from constants import PATCH_DIMS, PATCH_SIZE_MM
from utils.plot import transparent_cmap


def get_centroid(coords):
    assert type(coords) == np.ndarray
    return np.mean(coords, axis=0).round().astype(int)


def sample_points(coords):
    centroid = get_centroid(coords)
    coords = np.vstack([c for c in coords if np.any(c != centroid)])
    n_points = 2
    point_indices = np.random.choice(len(coords), n_points, replace=False)
    sampled_points = np.vstack([coords[point_indices], centroid]).astype(int)
    return np.unique(sampled_points, axis=0)


def get_xyz_range(point, spacing, patch_size_mm):
    if spacing:
        x_size = round(patch_size_mm[2] / spacing[0])
        y_size = round(patch_size_mm[1] / spacing[1])
        z_size = round(patch_size_mm[0] / spacing[2])
    else:
        # if no spacing preovided, patch_size_mm is pixel sizes
        x_size = patch_size_mm[2]
        y_size = patch_size_mm[1]
        z_size = patch_size_mm[0]

    x_start = point[2] - x_size // 2
    x_end = x_start + x_size
    y_start = point[1] - y_size // 2
    y_end = y_start + y_size
    z_start = point[0] - z_size // 2
    z_end = z_start + z_size
    return (x_start, x_end), (y_start, y_end), (z_start, z_end)


def calculate_padding(array, x_range, y_range, z_range):
    # array is zyx
    pad_x = (max(-x_range[0], 0), max(x_range[1] - array.shape[2], 0))
    pad_y = (max(-y_range[0], 0), max(y_range[1] - array.shape[1], 0))
    pad_z = (max(-z_range[0], 0), max(z_range[1] - array.shape[0], 0))
    return pad_x, pad_y, pad_z


def resize_volume(volume, new_shape):
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    resized_tensor = F.interpolate(tensor, size=new_shape, mode='trilinear')
    return resized_tensor.squeeze().numpy()


def get_lesion_patch(array, point, spacing=None, patch_dims=PATCH_DIMS, patch_size_mm=PATCH_SIZE_MM):
    """
    Create patch centered around point
    """
    # get ranges to crop
    x_range, y_range, z_range = get_xyz_range(point, spacing, patch_size_mm)

    # pad array so it fits within the range
    pad_x, pad_y, pad_z = calculate_padding(array, x_range, y_range, z_range)
    array_padded = np.pad(array, (pad_z, pad_y, pad_x), mode='reflect')  # mode='constant', constant_values=pad_value

    # adjust range after padding
    z_range = [z + pad_z[0] for z in z_range]
    y_range = [y + pad_y[0] for y in y_range]
    x_range = [x + pad_x[0] for x in x_range]

    # crop and resize
    patch = array_padded[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]
    return resize_volume(patch, patch_dims)


def get_lesion_patch_simple(array, point, dims):
    # get ranges to crop
    x_range, y_range, z_range = get_xyz_range(point, None, dims)

    # pad array so it fits within the range
    pad_x, pad_y, pad_z = calculate_padding(array, x_range, y_range, z_range)
    array_padded = np.pad(array, (pad_z, pad_y, pad_x), mode='reflect')  # mode='constant', constant_values=pad_value

    # adjust range after padding
    z_range = [z + pad_z[0] for z in z_range]
    y_range = [y + pad_y[0] for y in y_range]
    x_range = [x + pad_x[0] for x in x_range]

    # crop
    patch = array_padded[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]
    return torch.tensor(patch)


def get_n_windows(x, width, overlap):
    """Get number of windows that can fit, and remainder as a ratio of window"""
    windows = (x - overlap) / (width - overlap)
    remainder = windows % 1 * (width - overlap) / width
    return math.floor(windows), remainder


def get_step_size(x, width, n) -> int:
    """Calculate step size needed to fit n windows in x"""
    if n == 1:
        return 0
    overlap = (n * width - x) / (n - 1)
    return width - overlap


def get_range(dim, x, width, step) -> tuple:
    """get start and end range of a dimension given window width and step"""
    start = round(x * step)
    end = start + width
    if end > dim:  # if it exceeds the dimension
        end = dim
        start = end - width
    return start, end


def get_sliding_patches(volume, patch_size, overlap_ratio=0.5):
    """Generate sliding windows"""
    shape = list(volume.shape)  # zyx

    prelim_overlaps = [overlap_ratio * x for x in patch_size]
    n_windows = [get_n_windows(x, w, s)[0] for x, w, s in zip(shape, patch_size, prelim_overlaps)]
    remainders = [get_n_windows(x, w, s)[1] for x, w, s in zip(shape, patch_size, prelim_overlaps)]

    # add an extra window if there's remainder >10% of window
    n_windows = [(n + 1 if r > 0.1 else n) for n, r in zip(n_windows, remainders)]

    # calculate new step size to evenly distribute the windows
    steps = [get_step_size(x, w, n) for x, w, n in zip(shape, patch_size, n_windows)]

    # print("winds", n_windows)
    # print("steps", steps)

    patches = []
    zyx_ranges = []
    for z in range(n_windows[0]):
        z_start, z_end = get_range(shape[0], z, patch_size[0], steps[0])
        for y in range(n_windows[1]):
            y_start, y_end = get_range(shape[1], y, patch_size[1], steps[1])
            for x in range(n_windows[2]):
                x_start, x_end = get_range(shape[2], x, patch_size[2], steps[2])
                patches.append(volume[z_start:z_end, y_start:y_end, x_start:x_end])
                zyx_ranges.append([[z_start, z_end], [y_start, y_end], [x_start, x_end]])
    return patches, zyx_ranges


def visualize_patch_seg(ct_patch, seg_patch):
    """Visualize slices in a patch with the segment superimposed"""
    for i in range(len(ct_patch)):
        if seg_patch[i].sum() > 0:
            fig, axes = plt.subplots(1, 2, figsize=(3, 1.5), gridspec_kw = {'wspace':0, 'hspace':0})
            axes[0].imshow(ct_patch[i], cmap="gray")
            axes[1].imshow(ct_patch[i], cmap="gray")
            axes[1].imshow(seg_patch[i], cmap=transparent_cmap("r"), alpha=0.5)
            for a in axes:
                a.axis('off')
        else:
            plt.figure(figsize=(0.8, 0.8))
            plt.imshow(ct_patch[i], cmap="gray"); plt.axis('off')
        plt.tight_layout()
        plt.show()


def useful_patch_indices(seg_patches, min_voxels):
    """Get indices of patch with min number of voxel labels """
    indices = []
    for i, seg_patch in enumerate(seg_patches):
        if seg_patch.sum() >= min_voxels:
            indices.append(i)
    return indices