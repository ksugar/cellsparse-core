import os

import numpy as np
from stardist.models import (
    Config2D,
    StarDist2D,
)
import tensorflow as tf
from tqdm import tqdm

from cellsparse.runners.base import BaseRunner
from cellsparse.utils import remove_small_labels

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


class StarDistRunner(BaseRunner):
    def __init__(
        self,
        axes="YX",
        n_rays=32,
        n_channel_in=1,
        grid=(1, 1),
        n_classes=None,
        backbone="unet",
        basedir="stardist_models",
        prob_thresh=None,
        nms_thresh=None,
        min_area=0,
        **kwargs,
    ) -> None:
        self.config = Config2D(
            axes=axes,
            n_rays=n_rays,
            n_channel_in=n_channel_in,
            grid=grid,
            n_classes=n_classes,
            backbone=backbone,
            **kwargs,
        )
        if self.config.use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory

            # adjust as necessary: limit GPU memory to be used by TensorFlow to leave
            # some to OpenCL-based computations
            limit_gpu_memory(0.8)
            # alternatively, try this:
            # limit_gpu_memory(None, allow_growth=True)
        self.basedir = basedir
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.min_area = min_area

    def name(self):
        return "StarDist"

    def augmenter(self, x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        x, y = random_fliprot(x, y)
        x = random_intensity_change(x)
        # add some gaussian noise
        sig = 0.02 * np.random.uniform(0, 1)
        x = x + sig * np.random.normal(0, 1, x.shape)
        return x, y

    def _init_model(self, description):
        model = StarDist2D(self.config, name=description, basedir=self.basedir)
        model.keras_model.save_weights(
            str(model.logdir / model.config.train_checkpoint_last)
        )

    def _train(self, x_trn, y_trn, x_val, y_val, description):
        model = StarDist2D(self.config, name=description, basedir=self.basedir)
        if (model.logdir / self.config.train_checkpoint_last).exists():
            model.load_weights(self.config.train_checkpoint_last)
        model.train(
            x_trn,
            y_trn,
            validation_data=(x_val, y_val),
            augmenter=self.augmenter,
        )

    def _eval(self, x_val, description):
        tf.keras.utils.disable_interactive_logging()
        try:
            model = StarDist2D(self.config, name=description, basedir=self.basedir)
            if (model.logdir / self.config.train_checkpoint_last).exists():
                model.load_weights(self.config.train_checkpoint_last)
            labels = [
                model.predict_instances(
                    x,
                    n_tiles=model._guess_n_tiles(x),
                    show_tile_progress=False,
                    prob_thresh=self.prob_thresh,
                    nms_thresh=self.nms_thresh,
                    predict_kwargs={"verbose": 0},
                )[0]
                for x in tqdm(x_val)
            ]
            if 0 < self.min_area:
                for lbl in labels:
                    remove_small_labels(lbl, self.min_area)
        finally:
            tf.keras.utils.enable_interactive_logging()
        return labels
