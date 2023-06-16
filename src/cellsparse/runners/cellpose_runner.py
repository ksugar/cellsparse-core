import os
from pathlib import Path

from cellpose import (
    core,
    io,
    models,
)
from tqdm import tqdm

from cellsparse.runners.base import BaseRunner
from cellsparse.utils import remove_small_labels


class CellposeRunner(BaseRunner):
    def __init__(
        self,
        use_GPU=core.use_gpu(),
        initial_model=None,
        channels=[0, 0],
        save_path="cellpose_models",
        n_epochs=100,
        learning_rate=0.001,
        weight_decay=0.0001,
        nimg_per_epoch=8,
        min_area=0,
    ) -> None:
        self.use_GPU = use_GPU
        self.initial_model = initial_model
        self.channels = channels
        self.save_path = save_path
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nimg_per_epoch = nimg_per_epoch
        self.min_area = min_area
        io.logger_setup()

    def name(self):
        return "Cellpose"

    def _init_model(self, description):
        model_path = os.path.join(self.save_path, "models", description)
        model = models.CellposeModel(
            gpu=self.use_GPU,
            model_type=self.initial_model,
        )
        model.net.save_model(model_path)

    def _train(self, x_trn, y_trn, x_val, y_val, description):
        model_path = os.path.join(self.save_path, "models", description)
        if os.path.exists(model_path):
            model = models.CellposeModel(
                gpu=self.use_GPU,
                pretrained_model=model_path,
            )
        else:
            model = models.CellposeModel(
                gpu=self.use_GPU,
                model_type=self.initial_model,
            )
        model.train(
            x_trn.copy(),
            y_trn,
            test_data=x_val.copy(),
            test_labels=y_val.copy(),
            channels=self.channels,
            normalize=False,  # already normalized
            save_path=self.save_path,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            nimg_per_epoch=self.nimg_per_epoch,
            model_name=description,
            min_train_masks=1,
        )

    def _eval(self, x_val, description):
        model_path = os.path.join(self.save_path, "models", description)
        if not os.path.exists(model_path):
            model_path = None
        model = models.CellposeModel(gpu=self.use_GPU, pretrained_model=model_path)
        diam_labels = model.diam_labels.copy()
        labels = [
            model.eval(
                x,
                channels=self.channels,
                normalize=False,  # already normalized
                diameter=diam_labels,
            )[0]
            for x in tqdm(x_val)
        ]
        if 0 < self.min_area:
            for lbl in labels:
                remove_small_labels(lbl, self.min_area)
        return labels
