import io
import tempfile
import zipfile
from pathlib import Path

import SimpleITK as sitk
import nrrd
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
from skimage.measure import regionprops, marching_cubes


def get_start_end_slice(seg_array, margin=1) -> tuple:
    nonzero_slice = np.nonzero(seg_array.sum(axis=(1, 2)))
    start = max(0, np.min(nonzero_slice) - margin)
    end = min(seg_array.shape[0], np.max(nonzero_slice) + margin)
    return start, end


def get_seg_bbox(seg):
    mask = np.asarray(seg)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [xmin, ymin, xmax, ymax]


def expand_to_square_bbox(bbox, ratio, min_size=None):
    """
    Convert bbox to square w longest side, then expand it by a ratio
    """
    x_min, y_min, x_max, y_max = bbox
    side = max(x_max - x_min, y_max - y_min) * ratio
    if min_size:
        side = max(side, min_size)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_side = side / 2
    return [cx - half_side, cy - half_side, cx + half_side, cy + half_side]


def plot_seg(image_slice, seg_slice, gt_slice, zoom=True):
    contours = measure.find_contours(seg_slice, level=0.5)
    gt_contours = measure.find_contours(gt_slice, level=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_slice, cmap="gray_r")
    axes[0].set_title("Original CT Slice")

    axes[1].imshow(image_slice, cmap="gray_r")
    for contour in contours:
        axes[1].plot(contour[:, 1], contour[:, 0], color="purple", linewidth=1)
    for contour in gt_contours:
        axes[1].plot(contour[:, 1], contour[:, 0], color="g", linewidth=1)
    axes[1].set_title(f"GT(green): {int(gt_slice.sum())}  |  Pred(purple) {int(seg_slice.sum())}")


    if zoom:
        zoom_box = expand_to_square_bbox(get_seg_bbox(seg_slice + gt_slice), 3, min_size=100)
        for ax in axes:
            ax.set_xlim(zoom_box[0], zoom_box[2])
            ax.set_ylim(zoom_box[1], zoom_box[3])

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


def read_nii_gz_zip(zip_path: Path | str) -> sitk.Image:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        nii_file = next(f for f in zf.namelist() if f.endswith('.nii.gz'))
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True) as temp_file:
            temp_file.write(zf.read(nii_file))
            temp_file.flush()  # Ensure all data is written to disk
            return sitk.ReadImage(temp_file.name, imageIO="NiftiImageIO")


def load_seg(file_path: Path | str) -> np.ndarray:
    """
    Load 3D segmentation from a file.
    Replace this with actual loading logic (e.g., NIFTI, NumPy, etc.).
    """
    # Add appropriate file handling here based on the file format
    file_type = "".join(Path(file_path).suffixes).lower()
    if file_type == ".nrrd":
        data, header = nrrd.read(file_path)
        return data
    elif file_type == ".npy":
        return np.load(file_path)
    elif file_type == ".nii.gz":
        img = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(img)
    elif file_type == ".nii.gz.zip":
        img = read_nii_gz_zip(file_path)
        return sitk.GetArrayFromImage(img)
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
    return round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)


def iou_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate IoU for two binary 3D arrays."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.sum((pred_bool & gt_bool))
    union = np.sum((pred_bool | gt_bool))
    return 1.0 if union == 0 and intersection == 0 else intersection / union


def get_bbox(seg):
    mask = np.asarray(seg)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [xmin, ymin, xmax, ymax]


def get_bbox_aspect_ratio(seg):
    """Get aspect ratio of bounding box of seg projected onto the horizontal plane"""
    bbox = get_bbox(seg.sum(axis=0))
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin + 1, ymax - ymin + 1
    return max(w, h) / min(w, h)


def get_vertical_aspect_ratio(seg):
    """dim in vertical axis / geometric mean of dims of projection onto the horizontal plane"""
    bbox = get_bbox(seg.sum(axis=0))
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin + 1, ymax - ymin + 1

    indices = np.nonzero(seg)
    z = np.max(indices[0]) - np.min(indices[0]) + 1
    return z / np.sqrt(w**2 + h**2)


def evaluate_3d_metrics(pred_path: Path | str, gt_path: Path | str) -> dict:
    """
    Evaluate multiple 3D segmentation metrics given file paths to
    prediction and ground truth segmentations.
    """
    pred = load_seg(pred_path)
    gt = load_seg(gt_path)

    float_metrics = {
        "dice": dice_score(pred, gt),
        "iou": iou_3d(pred, gt),
        "p": precision(pred, gt),
        "r": recall(pred, gt),
        "volume_similarity": volume_similarity(pred, gt),
        "gt_volume": gt.sum(),
        "gt_ar_horizontal": get_bbox_aspect_ratio(gt),
        "gt_ar_vertical": get_vertical_aspect_ratio(gt),
        # "Hausdorff Distance": round(float(hausdorff_distance(pred, gt)), dp),
        # "Surface Distance": round(float(surface_distance(pred, gt)), dp),
        # "Shape Similarity": round(float(shape_similarity(pred, gt)), dp),
    }
    float_metrics = {k: round(float(x), 5) for k, x in float_metrics.items()}
    other_metrics = {}

    return float_metrics | other_metrics
