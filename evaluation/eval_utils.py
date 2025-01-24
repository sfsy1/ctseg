from pathlib import Path

import nrrd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
from skimage.measure import label, regionprops, marching_cubes
from scipy.ndimage import label as nd_label


def get_start_end_slice(seg_array, margin=1) -> tuple:
    nonzero_slice = np.nonzero(seg_array.sum(axis=(1,2)))
    start = max(0, np.min(nonzero_slice) - margin)
    end = min(seg_array.shape[0], np.max(nonzero_slice) + margin)
    return start, end


def plot_seg(image_slice, seg_slice):
    contours = measure.find_contours(seg_slice, level=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original CT Slice")
    axes[0].axis("off")

    axes[1].imshow(image_slice, cmap="gray")
    for contour in contours:
        axes[1].plot(contour[:, 1], contour[:, 0], color="red", linewidth=1)
    axes[1].set_title("CT with Segmentation Overlay")
    axes[1].axis("off")
    plt.tight_layout()


def load_seg(file_path: Path | str) -> np.ndarray:
    """
    Load 3D segmentation from a file.
    Replace this with actual loading logic (e.g., NIFTI, NumPy, etc.).
    """
    # Add appropriate file handling here based on the file format
    file_type = Path(file_path).suffix[1:].lower()
    if file_type == "nrrd":
        data, header = nrrd.read(file_path)
        return data
    elif file_type == "npy":
        return np.load(file_path)  # Example assuming a NumPy file
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Dice coefficient."""
    intersection = np.sum((pred > 0) & (gt > 0))
    dice = (2 * intersection) / (np.sum(pred > 0) + np.sum(gt > 0))
    return float(dice)


def volume_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate volume-based similarity."""
    vol_pred = np.sum(pred > 0)
    vol_gt = np.sum(gt > 0)
    vol_diff = abs(vol_pred - vol_gt)
    return float(1 - vol_diff / (vol_pred + vol_gt))


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute the Hausdorff distance between boundaries."""
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)
    distance = max(directed_hausdorff(pred_points, gt_points)[0],
                   directed_hausdorff(gt_points, pred_points)[0])
    return float(distance)


def surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate boundary distance similarity using mesh surface points."""
    verts_gt, _, _, _ = marching_cubes(gt, level=0.5)
    verts_pred, _, _, _ = marching_cubes(pred, level=0.5)

    dist = np.mean([np.min(np.linalg.norm(verts_pred - point, axis=1)) for point in verts_gt])
    return float(dist)


def shape_similarity(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Assess similarity in shape via region properties (centroid distance)."""
    labeled_pred, _ = nd_label(pred)
    labeled_gt, _ = nd_label(gt)

    props_pred = regionprops(labeled_pred)
    props_gt = regionprops(labeled_gt)

    if len(props_gt) > 0 and len(props_pred) > 0:
        centroid_gt = props_gt[0].centroid
        centroid_pred = props_pred[0].centroid
        centroid_dist = np.linalg.norm(np.array(centroid_gt) - np.array(centroid_pred))
        return float(centroid_dist)
    else:
        return None


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate precision (positive predictive value)."""
    tp = np.sum((pred > 0) & (gt > 0))
    fp = np.sum((pred > 0) & (gt == 0))
    return float(tp / (tp + fp) if (tp + fp) > 0 else 0)


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate recall (sensitivity)."""
    tp = np.sum((pred > 0) & (gt > 0))
    fn = np.sum((pred == 0) & (gt > 0))
    return float(tp / (tp + fn) if (tp + fn) > 0 else 0)


def evaluate_3d_metrics(pred_path: Path | str, gt_path: Path | str) -> dict:
    """
    Evaluate multiple 3D segmentation metrics given file paths to
    prediction and ground truth segmentations.
    """
    pred = load_seg(pred_path)
    gt = load_seg(gt_path)

    metrics = {
        "Dice Score": dice_score(pred, gt),
        "Precision": precision(pred, gt),
        "Recall": recall(pred, gt),
        "Volume Similarity": volume_similarity(pred, gt),
        "Hausdorff Distance": hausdorff_distance(pred, gt),
        "Surface Distance": surface_distance(pred, gt),
        "Shape Similarity": shape_similarity(pred, gt)
    }

    return metrics
