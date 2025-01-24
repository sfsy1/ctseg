import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from skimage import measure

# CT windows
LUNG_WIN = {'center': -600, 'width': 1500}
ABD_WIN = {'center': 40, 'width': 400}


def apply_window(image, center, width):
    window_min = center - width // 2
    window_max = center + width // 2
    return np.clip(image, window_min, window_max)


def plot_contours(ax, seg, color, thickness, original_axis=False):
    contours = measure.find_contours(seg, 0.5)  # 0.5 is the threshold level
    if original_axis:
        ax.imshow(seg, alpha=0)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=thickness, color=color)


def plot_text(ax, text, color, weight=500, align="left", size="medium", font="Ubuntu"):
    text_margin = 0.01
    if align == "left":
        x, y = text_margin, 1 - text_margin
        valign, halign = "top", "left"
    elif align == "right":
        x, y = 1 - text_margin, 1 - text_margin
        valign, halign = "top", "right"

    ax.text(
        x, y, text, transform=ax.transAxes, verticalalignment=valign, horizontalalignment=halign,
        fontdict=dict(c=color, weight=weight, font=font, fontsize=size),
    )


def get_slice_range(array, crop_ratio=None, pad_ratio=None, min_pad=10, safe_crop=True):
    n_slices = array.shape[0]
    relevant_slices = np.where(array)[0]

    if len(relevant_slices):
        top_slice = min(relevant_slices).item()
        btm_slice = max(relevant_slices).item()

        if pad_ratio is not None:
            # pad range by some slices above and below, at least `min_pad` slices
            slice_pad = max(min_pad, round(n_slices * pad_ratio))
            slice_range = max(0, top_slice - slice_pad), min(n_slices, btm_slice + slice_pad)

        elif crop_ratio is not None:
            # crop the top/bottom by a ratio
            if safe_crop:
                top_limit = top_slice
                btm_limit = btm_slice
            else:
                top_limit = 0
                btm_limit = n_slices

            slice_crop = max(0, round(n_slices * crop_ratio))
            slice_range = (min(top_limit, slice_crop), max(btm_limit, n_slices - slice_crop))
        else:
            slice_range = (0, n_slices)

    else:
        slice_range = (0, 1)

    return slice_range, n_slices


def add_border_to_axis(a, color="#333"):
    t = a.transAxes
    rect = Rectangle(
        (0, 0), 1, 1, fill=False, edgecolor=color, lw=1, transform=t, clip_on=False
    )
    a.add_patch(rect)


def transparent_cmap(color):
    """Create a colormap with transparency going from 0 to 1"""
    ncolors = 256
    color_array = plt.get_cmap('Greys')(range(ncolors))
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    color_array[:, 0:3] = to_rgb(color)
    return LinearSegmentedColormap.from_list(name=f'{color}_transparent', colors=color_array)


def add_current_slice_viz(ax, ct_array, slice_idx, text_size, color="limegreen"):
    """
    plot a horizontal line and add text to show "current_slice/total_slice"
    """
    ax.axhline(y=slice_idx, color=color, linestyle='-', linewidth=1)
    slice_text = f"{slice_idx + 1}/{ct_array.shape[0]}"
    plot_text(ax, slice_text, color, 700, align="right", size=text_size)


def window_ct(ct_slice):
    lung = apply_window(ct_slice, LUNG_WIN['center'], LUNG_WIN['width'])
    abdomen = apply_window(ct_slice, ABD_WIN['center'], ABD_WIN['width'])
    return lung, abdomen


def plot_slice_full(ct_array, seg_array, suv_array=None, slice_idx=0):
    # get slice
    ct_slice = ct_array[slice_idx]
    seg_slice = seg_array[slice_idx]

    lung, abdomen = window_ct(ct_slice)

    # col_widths = [4.5, 4.5, 3.5, 3.5]
    # if suv_array is not None:
    #     suv_clipped = np.clip(suv_array, 0, 10)

    fig, ax = plt.subplot_mosaic(
        "ABEF;CDEF",
        figsize=(16, 9),
        width_ratios=[4.5, 4.5, 4, 3],
    )
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("k")

    # styling
    title_size = "medium"
    legend_size = "medium"
    lc = "magenta"

    # top left
    a = ax["A"]
    a.imshow(lung, cmap='gray', origin='lower')
    plot_text(a, "Lung", "silver", size=title_size)

    # top right
    a = ax["B"]
    a.imshow(lung, cmap='gray', origin='lower')
    plot_contours(a, seg_slice, lc, 0.5)
    plot_text(a, "Labels", lc, size=legend_size)

    # btm left
    a = ax["C"]
    a.imshow(abdomen, cmap='gray', origin='lower')
    plot_text(a, "Abdomen", "silver", size=title_size)

    # btm right
    a = ax["D"]
    a.imshow(abdomen, cmap='gray', origin='lower')
    plot_contours(a, seg_slice, lc, 0.5)
    plot_text(a, "Labels", lc, size=legend_size)

    # right (frontal/coronal)
    a = ax["E"]
    a.set_anchor("W")  # align left
    if suv_array is not None:
        a.imshow(
            suv_array.max(axis=1),
            cmap='gray_r', origin="lower", aspect='auto',
            vmin=0, vmax=10
        );
    else:
        a.imshow(ct_array.mean(axis=1), cmap='gray', origin="lower", aspect='auto');
    plot_contours(a, seg_array.any(axis=1), lc, 0.5)
    # a.imshow(seg_array.any(axis=1), cmap=transparent_cmap(lc), origin="lower", aspect='auto', alpha=0.5)
    add_current_slice_viz(a, ct_array, slice_idx, legend_size)

    # right2 (sagittal)
    a = ax["F"]
    a.set_anchor("W")  # align left
    if suv_array is not None:
        a.imshow(
            suv_array.max(axis=2), cmap='gray_r', origin="lower", aspect='auto',
            vmin=0, vmax=10
        );
    else:
        a.imshow(ct_array.mean(axis=2), cmap='gray', origin="lower", aspect='auto');
    plot_contours(a, seg_array.any(axis=2), lc, 0.5)
    # a.imshow(seg_array.any(axis=2), cmap=transparent_cmap(lc), origin="lower", aspect='auto', alpha=0.5)
    add_current_slice_viz(a, ct_array, slice_idx, legend_size)

    # subplot border
    for k, a in ax.items():
        a.set_axis_off()
        a.set_facecolor("k")
        add_border_to_axis(a)

    return fig


def plot_and_save(args):
    ct_array, seg_array, suv_array, i, frames_folder = args
    fig = plot_slice_full(ct_array, seg_array, suv_array, i)
    frame_path = f"{frames_folder}/frame_{i:04d}"
    fig.savefig(f"{frame_path}.jpg", bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig)
