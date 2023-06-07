#!/usr/bin/env python
import argparse

from runners import (
    CellposeRunner,
    ElephantRunner,
    StarDistRunner,
)
from utils import get_data


def main(name: str) -> None:
    (x_trn, y_trn), (x_val, y_val) = get_data()
    cellpose_runner = CellposeRunner(
        save_path=f"/src/models/cellpose/{name}",
        n_epochs=100,
    )
    elephant_runner = ElephantRunner(
        is_3d=False,
        model_dir=f"/src/models/elephant/{name}",
        log_dir=f"/src/models/elephant/{name}/logs",
        n_epochs=100,
    )
    train_batch_size = 8
    stardist_runner = StarDistRunner(
        grid=(2, 2),
        basedir=f"/src/models/stardist/{name}",
        use_gpu=False,
        train_epochs=100,
        train_patch_size=(224, 224),
        train_batch_size=train_batch_size,
        train_steps_per_epoch=len(x_trn) // train_batch_size + 1,
    )
    for runner in (cellpose_runner, elephant_runner, stardist_runner):
        for include_bg in (False, True):
            for mode in ("min", "max", "minmax"):
                runner.run(
                    x_trn,
                    y_trn,
                    x_val,
                    y_val,
                    mode=mode,
                    is_train=True,
                    is_eval=False,
                    include_bg=include_bg,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Cellsparse training")
    parser.add_argument(
        "name",
        type=str,
        help="model directory name, e.g. paper01",
    )
    args = parser.parse_args()
    main(args.name)
