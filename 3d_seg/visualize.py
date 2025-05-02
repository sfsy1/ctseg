import os
from pathlib import Path

import matplotlib.pyplot as plt
import SimpleITK as sitk
from utils.plot import transparent_cmap


DATASETS = {
    1: "ct",
    2: "ct+seg",
    3: "ct+seg+box",
    5: "ct+seg+2box",
    6: "ct+seg+mask",
    7: "ct+seg+2mask",
}

def load_data(ct_filename, images_folder: Path, labels_folder: Path, res_folder: Path):
    name = ct_filename.replace("_0000.nii.gz", ".nii.gz")

    in_channels = []
    for channel in range(2):  # only read CT and mask, not the box and mask channels
        file_path = images_folder / name.replace(".nii.gz", f"_000{channel}.nii.gz")
        if os.path.exists(file_path):
            in_channels.append(sitk.GetArrayFromImage(sitk.ReadImage(file_path)))

    label_path = labels_folder / name
    label_img = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label_img)

    preds = []
    for dataset_n in DATASETS:
        dataset_name = [f for f in os.listdir(res_folder) if f.startswith(f"Dataset00{dataset_n}")][0]
        pred_folder = res_folder / dataset_name
        pred_path = pred_folder / "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/" / name
        preds.append(sitk.GetArrayFromImage(sitk.ReadImage(pred_path)))

    return in_channels, label, preds


def plot_all_outputs(in_channels, label, preds):
    """
    Plot all 3D input, label, and all preds, where every slice is a row.

    in_channels: list of input array of all channels
    label: ground truth array
    preds: list of prediction arrays from all datasets (ct, ct+seg, ...)
    """
    gt_alpha = 1
    pred_alpha = 0.5
    font = 10

    nc = len(in_channels)
    nd = len(preds)

    size = 2

    k = label.sum(axis=(1,2)).argmax()  # key slice index
    start = max(0, k - 3)
    end = max(label.shape[0] - 1, k + 3)
    slices = range(start, end + 1)

    # get slices with some info
    relevant_slices = [s for s in slices if in_channels[0][s].std() > 0]

    # layout
    rows = len(relevant_slices)
    cols = nc + nd
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))

    # plot every slice in a row
    for i, s in enumerate(relevant_slices):

        # ct + gt mask
        ax = axes[i][0]
        ax.imshow(in_channels[0][s], cmap="gray")
        ax.imshow(label[s], cmap=transparent_cmap('blue'), alpha=gt_alpha)
        ax.set_title(f"gt", fontsize=font)
        ax.axis("off")

        # seg mask
        ax = axes[i][1]
        ax.imshow(in_channels[1][s], cmap="tab20b")
        ax.set_title("ctfm seg", fontsize=font)
        ax.axis("off")

        for j, d in enumerate(DATASETS.keys()):
            ax = axes[i][nc + j]
            ax.imshow(in_channels[0][s], cmap="gray")
            ax.imshow(label[s], cmap=transparent_cmap('blue'), alpha=gt_alpha)
            ax.imshow(preds[j][s], cmap=transparent_cmap("red"), alpha=pred_alpha)
            ax.set_title(f"{DATASETS[d]}", fontsize=font)
            ax.axis("off")


def load_and_plot(ct_filename, images_folder: Path, labels_folder: Path, res_folder: Path):
    in_channels, label, preds = load_data(ct_filename, images_folder, labels_folder, res_folder)
    plot_all_outputs(in_channels, label, preds)