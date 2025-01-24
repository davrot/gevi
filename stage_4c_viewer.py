import os
import numpy as np

import matplotlib.pyplot as plt  # type:ignore

from functions.create_logger import create_logger
from functions.load_config import load_config

import argh


def main(
    *, config_filename: str = "config.json", experiment_id: int = 1, trial_id: int = 1
) -> None:

    experiment_name: str = f"Exp{experiment_id:03d}_Trial{trial_id:03d}"
    mylogger = create_logger(
        save_logging_messages=False,
        display_logging_messages=False,
        log_stage_name="stage_4c",
    )

    config = load_config(mylogger=mylogger, filename=config_filename)

    temp_path = os.path.join(
        config["export_path"], experiment_name + "_inspect_images.npz"
    )
    data = np.load(temp_path)

    acceptor = data["acceptor"][0, ...]
    donor = data["donor"][0, ...]
    oxygenation = data["oxygenation"][0, ...]
    volume = data["volume"][0, ...]

    plt.figure(1)
    plt.imshow(acceptor, cmap="hot")
    plt.title(f"Acceptor Experiment: {experiment_id:03d} Trial:{trial_id:03d}")
    plt.show(block=False)
    plt.figure(2)
    plt.imshow(donor, cmap="hot")
    plt.title(f"Donor Experiment: {experiment_id:03d} Trial:{trial_id:03d}")
    plt.show(block=False)
    plt.figure(3)
    plt.imshow(oxygenation, cmap="hot")
    plt.title(f"Oxygenation Experiment: {experiment_id:03d} Trial:{trial_id:03d}")
    plt.show(block=False)
    plt.figure(4)
    plt.imshow(volume, cmap="hot")
    plt.title(f"Volume Experiment: {experiment_id:03d} Trial:{trial_id:03d}")
    plt.show(block=True)

    return


if __name__ == "__main__":
    argh.dispatch_command(main)
