from csbdeep.utils import (
    Path,
    normalize,
    download_and_extract_zip_file,
)
import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import skimage.measure
from stardist import fill_label_holes, random_label_cmap
from tifffile import imread
from tqdm import tqdm

matplotlib.rcParams["image.interpolation"] = "None"
np.random.seed(42)


def plot_img_label(
    img,
    lbl,
    img_title="image",
    lbl_title="label",
    img_cmap="gray",
    lbl_cmap=random_label_cmap(),
    **kwargs,
):
    fig, (ai, al) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1))
    )
    im = ai.imshow(img, cmap=img_cmap, clim=(0, 1))
    ai.set_title(img_title)
    ai.set_axis_off()
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap, **kwargs)
    al.set_title(lbl_title)
    al.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def _get_min_max_label(lbl, img, kth):
    props = skimage.measure.regionprops_table(
        lbl, img, properties=["label", "minor_axis_length", "mean_intensity"]
    )
    keep = np.nonzero(props["minor_axis_length"])
    props["label"] = props["label"][keep]
    props["mean_intensity"] = props["mean_intensity"][keep]
    kth_v = min(kth, len(props["label"]) - 1)
    min_label = props["label"][np.argpartition(props["mean_intensity"], kth_v)[:kth]]
    max_label = props["label"][np.argpartition(props["mean_intensity"], -kth_v)[-kth:]]
    return min_label, max_label


def _get_bg(lbl_s, lbl):
    height, width = lbl_s.shape
    ys, xs = np.nonzero(0 < lbl_s)
    ys = np.clip(ys - 10, 0, height - 1)
    xs = np.clip(xs - 10, 0, width - 1)
    bg = np.zeros_like(lbl)
    bg[ys, xs] = 1
    return bg * (lbl == 0)


def to_sparse(X, Y, kth, include_bg=True, mode="minmax"):
    if mode == "minmax":
        Y_s = [
            np.where(np.isin(y, _get_min_max_label(y, x, kth)), y, -1)
            for x, y in tqdm(zip(X, Y))
        ]
    elif mode == "min":
        Y_s = [
            np.where(np.isin(y, _get_min_max_label(y, x, kth * 2)[0]), y, -1)
            for x, y in tqdm(zip(X, Y))
        ]
    elif mode == "max":
        Y_s = [
            np.where(np.isin(y, _get_min_max_label(y, x, kth * 2)[1]), y, -1)
            for x, y in tqdm(zip(X, Y))
        ]
    else:
        raise NotImplementedError()
    if include_bg:
        Y_s = [np.where(_get_bg(y_s, y), 0, y_s) for y_s, y in tqdm(zip(Y_s, Y))]
    for y_s in Y_s:
        if y_s.max() < 0:
            raise ValueError("seg label should have positive values")
    return Y_s


def plot_stats(
    stats_list,
    title,
    data_points=(1, 4, 16, 64, 256),
    xlabel="kth",
    ax2_ylim=[0, 3000],
    metrics=(
        "precision",
        "recall",
        "accuracy",
        "f1",
        "mean_true_score",
        "mean_matched_score",
        "panoptic_quality",
    )
):
    if len(stats_list) - 1 != len(data_points):
        raise ValueError(
            f"len(stats_list)={len(stats_list)} - 1 and len(data_points)={len(data_points)}"
            + " should be the same"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for m in metrics:
        p = ax1.plot(
            data_points,
            [s._asdict()[m] for s in stats_list[:-1]],
            "o-",
            lw=2,
            label=m,
        )
        ax1.plot(
            (data_points[0], data_points[-1]),
            (stats_list[-1]._asdict()[m],) * 2,
            "--",
            color=p[0].get_color(),
            lw=1,
        )
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Metric value")
    ax1.set_ylim([0, 1])
    ax1.grid()
    ax1.legend(loc=5)

    for m in ("fp", "tp", "fn"):
        p = ax2.plot(
            data_points,
            [s._asdict()[m] for s in stats_list[:-1]],
            "o-",
            lw=2,
            label=m,
        )
        ax2.plot(
            (data_points[0], data_points[-1]),
            (stats_list[-1]._asdict()[m],) * 2,
            "--",
            color=p[0].get_color(),
            lw=1,
        )
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Number #")
    ax2.set_ylim(ax2_ylim)
    ax2.grid()
    ax2.legend()

    plt.show()
    plt.close(fig)


def _read_images(base_dir):
    X_files = natsorted(list(Path(base_dir).glob("images/*.tif")))
    Y_files = natsorted(list(Path(base_dir).glob("masks/*.tif")))
    assert all(Path(x).name == Path(y).name for x, y in zip(X_files, Y_files))
    X = list(map(imread, X_files))
    Y = list(map(imread, Y_files))
    # X = [normalize(x, 1, 99.8, axis=(0, 1)) for x in tqdm(X)]
    X = [normalize(x, 0, 100, axis=(0, 1)) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]
    return X, Y


def get_data(target_dir="data"):
    download_and_extract_zip_file(
        url="https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip",
        targetdir=target_dir,
        verbose=1,
    )
    X_trn, Y_trn = _read_images(Path(target_dir) / "dsb2018/train")
    X_val, Y_val = _read_images(Path(target_dir) / "dsb2018/test")
    print("number of images for training:   %3d" % len(X_trn))
    print("number of images for validation: %3d" % len(X_val))
    return (X_trn, Y_trn), (X_val, Y_val)


def remove_small_labels(lbl, area_threshold):
    props = skimage.measure.regionprops(lbl)
    for prop in props:
        if prop.area < area_threshold:
            lbl[prop.slice] = 0


seg_cmap = matplotlib.colors.ListedColormap(
    np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
)
