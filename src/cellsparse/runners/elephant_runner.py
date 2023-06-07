import logging
from pathlib import Path

from elephant.common import run_train
from elephant.common import _get_seg_prediction
from elephant.common import _find_and_push_spots
from elephant.datasets import SegmentationDatasetNumpy
from elephant.losses import SegmentationLoss
from elephant.models import ResUNet
from elephant.models import UNet
from elephant.util import dilate
from elephant.util.ellipse import ellipse
from elephant.util.ellipsoid import ellipsoid
from elephant.util.scaled_moments import radii_and_rotation
import numpy as np
import skimage.measure
import torch
from tqdm import tqdm

from cellsparse.runners.base import BaseRunner
from cellsparse.utils import remove_small_labels


def get_instance_segmentation(
    img,
    models,
    keep_axials,
    device,
    patch_size=None,
    c_ratio=0.5,
    p_thresh=0.5,
    r_min=0,
    r_max=1e6,
):
    prediction = _get_seg_prediction(
        img[None, None], models, keep_axials, device, patch_size
    )
    spots = []
    _find_and_push_spots(
        spots,
        0,
        prediction[-1],  # last channel is the center label
        c_ratio=c_ratio,
        p_thresh=p_thresh,
        r_min=r_min,
        r_max=r_max,
    )
    label = np.zeros(img.shape, np.uint16)
    draw_func = ellipsoid if img.ndim == 3 else ellipse
    for i in range(16):
        for ind, spot in enumerate(spots):
            centroid = np.array(spot["pos"][::-1])
            centroid = centroid[-img.ndim :]
            covariance = np.array(spot["covariance"][::-1]).reshape(3, 3)
            covariance = covariance[-img.ndim :, -img.ndim :]
            radii, rotation = np.linalg.eigh(covariance)
            radii = np.sqrt(radii)
            indices = draw_func(
                centroid,
                radii * (1 - 0.05 * i),
                rotation,
                shape=img.shape,
            )
            label[indices] = ind + 1
    # ensure that each spot is labeled at least its center voxcel
    for ind, spot in enumerate(spots):
        centroid = np.array(spot["pos"][::-1])
        centroid = centroid[-img.ndim :]
        indices_center = tuple(int(centroid[i]) for i in range(img.ndim))
        label[indices_center] = ind + 1
    return label


def generate_seg(lbl, cr=0.5, min_area=9, is_3d=False):
    if is_3d:
        draw_func = ellipsoid
        dilate_func = dilate.dilate_3d_indices
    else:
        draw_func = ellipse
        dilate_func = dilate.dilate_2d_indices
    if lbl.min() == -1:
        seg = np.zeros(lbl.shape, dtype=np.uint8)
        seg[lbl == 0] = 1
    else:
        seg = np.ones(lbl.shape, dtype=np.uint8)
    regions = skimage.measure.regionprops(lbl.clip(0))
    for region in regions:
        if region.minor_axis_length == 0:
            continue
        try:
            radii, rotation = radii_and_rotation(region.moments_central, is_3d)
            if (radii == 0).any():
                raise RuntimeError("all radii should be positive")
        except RuntimeError as e:
            print(str(e))
            continue
        radii *= 2
        factor = 1.0
        while True:
            indices_outer = draw_func(
                region.centroid[: 2 + is_3d],
                radii * factor,
                rotation,
                shape=seg.shape,
            )
            if len(indices_outer[0]) < min_area:
                factor *= 1.1
            else:
                break
        factor = 1.0
        while True:
            indices_inner = draw_func(
                region.centroid[: 2 + is_3d],
                radii * cr * factor,
                rotation,
                shape=seg.shape,
            )
            if len(indices_inner[0]) < min_area:
                factor *= 1.1
            else:
                break
        indices_inner_p = dilate_func(*indices_inner, seg.shape)
        seg[indices_outer] = np.where(seg[indices_outer] < 2, 2, seg[indices_outer])
        seg[indices_inner_p] = 2
        seg[indices_inner] = 3
    return seg


class ElephantRunner(BaseRunner):
    def __init__(
        self,
        is_3d=True,
        device="cuda",
        scale_factor_base=0.2,
        rotation_angle=45,
        contrast=0.5,
        batch_size=8,
        num_workers=2,
        crop_size=(224, 224),
        keep_axials=(True,) * 4,
        class_weights=(1.0, 10.0, 10.0),
        lr=0.001,
        n_epochs=100,
        patch_size=None,
        backbone="unet",
        model_dir="models",
        log_dir="logs",
        log_path=None,
        log_interval=100,
        step_offset=0,
        epoch_start=0,
        increment_from=None,
        is_generate_seg_trn=True,
        is_generate_seg_val=True,
        p_thresh=0.8,
        r_min=0,
        r_max=100,
        min_area=0,
        n_crops=1,
    ) -> None:
        self.is_3d = is_3d
        self.device = device if torch.cuda.is_available() else "cpu"
        self.scale_factor_base = scale_factor_base
        self.rotation_angle = rotation_angle
        self.contrast = contrast
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.keep_axials = keep_axials
        self.weight_tensor = torch.tensor(class_weights)
        self.lr = lr
        self.n_epochs = n_epochs
        self.patch_size = patch_size
        self.backbone = backbone
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.log_path = log_path
        self.log_interval = log_interval
        self.step_offset = step_offset
        self.epoch_start = epoch_start
        self.increment_from = increment_from
        self.is_generate_seg_trn = is_generate_seg_trn
        self.is_generate_seg_val = is_generate_seg_val
        self.p_thresh = p_thresh
        self.r_min = r_min
        self.r_max = r_max
        self.min_area = min_area
        self.n_crops = n_crops

    def name(self):
        return "ELEPHANT"

    def _train(self, x_trn, y_trn, x_val, y_val, description):
        is_3d = self.is_3d and x_trn[0].ndim == 3
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        filename_stem = f"{self.backbone}_{description}"
        model_path = str(Path(self.model_dir) / f"{filename_stem}.pth")
        if self.log_path is None:
            log_path = str(Path(self.log_dir) / filename_stem)
        else:
            log_path = self.log_path
        if self.increment_from is not None:
            increment_from = (
                Path(self.model_dir) / f"{self.backbone}_{self.increment_from}.pth"
            )
            if increment_from.exists():
                checkpoint = torch.load(str(increment_from), map_location=self.device)
            else:
                print(f"increment_from: {increment_from} does not exist")
                checkpoint = None
        else:
            checkpoint = None

        if self.is_generate_seg_trn:
            y_trn = list(map(generate_seg, tqdm(y_trn)))
        if self.is_generate_seg_val:
            y_val = list(map(generate_seg, tqdm(y_val)))
        train_loader = torch.utils.data.DataLoader(
            SegmentationDatasetNumpy(
                x_trn,
                y_trn,
                crop_size=self.crop_size,
                keep_axials=self.keep_axials,
                scale_factor_base=self.scale_factor_base,
                rotation_angle=self.rotation_angle,
                contrast=self.contrast,
                n_crops=self.n_crops,
            ),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        eval_loader = torch.utils.data.DataLoader(
            SegmentationDatasetNumpy(
                x_val,
                y_val,
                keep_axials=self.keep_axials,
                is_eval=True,
            ),
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
        )
        if self.backbone.lower() == "unet":
            models = [
                UNet.three_class_segmentation(
                    device=self.device, is_3d=is_3d, state_dict=checkpoint
                )
            ]
        elif self.backbone.lower() == "resunet":
            models = [
                ResUNet.three_class_segmentation(
                    device=self.device, is_3d=is_3d, state_dict=checkpoint
                )
            ]
        else:
            raise NotImplementedError()
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.lr) for model in models
        ]
        loss_fn = SegmentationLoss(
            class_weights=self.weight_tensor,
            false_weight=1,
            is_3d=is_3d,
        )
        run_train(
            self.device,
            1,
            models,
            train_loader,
            optimizers,
            loss_fn,
            self.n_epochs,
            model_path,
            False,
            self.log_interval,
            log_path,
            self.step_offset,
            self.epoch_start,
            eval_loader,
            self.patch_size,
            self.device == "cpu",
            True,
        )

    def _eval(self, x_val, description):
        logging.getLogger().disabled = True
        is_3d = self.is_3d and x_val[0].ndim == 3
        try:
            filename_stem = f"{self.backbone}_{description}"
            model_path = Path(self.model_dir) / f"{filename_stem}.pth"
            if model_path.exists():
                checkpoint = torch.load(str(model_path), map_location=self.device)
            else:
                checkpoint = None
            if self.backbone.lower() == "unet":
                models = [
                    UNet.three_class_segmentation(
                        device=self.device,
                        is_3d=is_3d,
                        is_eval=True,
                        state_dict=checkpoint,
                    )
                ]
            elif self.backbone.lower() == "resunet":
                models = [
                    ResUNet.three_class_segmentation(
                        device=self.device,
                        is_3d=is_3d,
                        is_eval=True,
                        state_dict=checkpoint,
                    )
                ]
            else:
                raise NotImplementedError()

            labels = [
                get_instance_segmentation(
                    x,
                    models,
                    self.keep_axials,
                    self.device,
                    p_thresh=self.p_thresh,
                    r_min=self.r_min,
                    r_max=self.r_max,
                )
                for x in tqdm(x_val)
            ]
            if 0 < self.min_area:
                for lbl in labels:
                    remove_small_labels(lbl, self.min_area)
        finally:
            logging.getLogger().disabled = False
        return labels
