import os
import numpy as np

import matplotlib.pyplot as plt  # type:ignore
import matplotlib
from matplotlib.widgets import Button  # type:ignore

# pip install roipoly
from roipoly import RoiPoly  # type:ignore

from functions.create_logger import create_logger
from functions.load_config import load_config

import argh


def compose_image(image_3color: np.ndarray, mask: np.ndarray) -> np.ndarray:
    display_image = image_3color.copy()
    display_image[..., 2] = display_image[..., 0]
    display_image[mask == 0, :] = 1.0
    return display_image


def main(*, config_filename: str = "config.json") -> None:
    mylogger = create_logger(
        save_logging_messages=True,
        display_logging_messages=True,
        log_stage_name="stage_3",
    )

    config = load_config(mylogger=mylogger, filename=config_filename)

    path: str = config["ref_image_path"]
    use_channel: str = "donor"
    padding: int = 20

    image_ref_file: str = os.path.join(path, use_channel + ".npy")
    heartbeat_mask_file: str = os.path.join(path, "heartbeat_mask.npy")
    refined_mask_file: str = os.path.join(path, "mask_not_rotated.npy")

    mylogger.info(f"loading image reference file: {image_ref_file}")
    image_ref: np.ndarray = np.load(image_ref_file)
    image_ref /= image_ref.max()
    image_ref = np.pad(image_ref, pad_width=padding)

    mylogger.info(f"loading heartbeat mask: {heartbeat_mask_file}")
    mask: np.ndarray = np.load(heartbeat_mask_file)
    mask = np.pad(mask, pad_width=padding)

    image_3color = np.concatenate(
        (
            np.zeros_like(image_ref[..., np.newaxis]),
            image_ref[..., np.newaxis],
            np.zeros_like(image_ref[..., np.newaxis]),
        ),
        axis=-1,
    )

    mylogger.info("-==- DONE -==-")

    fig, ax_main = plt.subplots()

    display_image = compose_image(image_3color=image_3color, mask=mask)
    image_handle = ax_main.imshow(display_image, vmin=0, vmax=1, cmap="hot")

    mylogger.info("Add controls")

    def on_clicked_accept(event: matplotlib.backend_bases.MouseEvent) -> None:
        nonlocal mylogger
        nonlocal refined_mask_file
        nonlocal mask

        mylogger.info(f"Save mask to: {refined_mask_file}")
        mask = mask[padding:-padding, padding:-padding]
        np.save(refined_mask_file, mask)

        exit()

    def on_clicked_cancel(event: matplotlib.backend_bases.MouseEvent) -> None:
        nonlocal mylogger
        mylogger.info("Ended without saving the mask")
        exit()

    def on_clicked_add(event: matplotlib.backend_bases.MouseEvent) -> None:
        nonlocal new_roi  # type: ignore
        nonlocal mask
        nonlocal image_3color
        nonlocal display_image
        nonlocal mylogger
        if len(new_roi.x) > 0:
            mylogger.info(
                "A ROI with the following coordiantes has been added to the mask"
            )
            for i in range(0, len(new_roi.x)):
                mylogger.info(f"{round(new_roi.x[i], 1)} x {round(new_roi.y[i], 1)}")
            mylogger.info("")
            new_mask = new_roi.get_mask(display_image[:, :, 0])
            mask[new_mask] = 0.0
            display_image = compose_image(image_3color=image_3color, mask=mask)
            image_handle.set_data(display_image)
            for line in ax_main.lines:
                line.remove()
            plt.draw()

            new_roi = RoiPoly(ax=ax_main, color="r", close_fig=False, show_fig=False)

    def on_clicked_remove(event: matplotlib.backend_bases.MouseEvent) -> None:
        nonlocal new_roi  # type: ignore
        nonlocal mask
        nonlocal image_3color
        nonlocal display_image
        if len(new_roi.x) > 0:
            mylogger.info(
                "A ROI with the following coordiantes has been removed from the mask"
            )
            for i in range(0, len(new_roi.x)):
                mylogger.info(f"{round(new_roi.x[i], 1)} x {round(new_roi.y[i], 1)}")
            mylogger.info("")
            new_mask = new_roi.get_mask(display_image[:, :, 0])
            mask[new_mask] = 1.0
            display_image = compose_image(image_3color=image_3color, mask=mask)
            image_handle.set_data(display_image)
            for line in ax_main.lines:
                line.remove()
            plt.draw()
            new_roi = RoiPoly(ax=ax_main, color="r", close_fig=False, show_fig=False)

    axbutton_accept = fig.add_axes(rect=(0.3, 0.85, 0.2, 0.04))
    button_accept = Button(
        ax=axbutton_accept, label="Accept", image=None, color="0.85", hovercolor="0.95"
    )
    button_accept.on_clicked(on_clicked_accept)  # type: ignore

    axbutton_cancel = fig.add_axes(rect=(0.5, 0.85, 0.2, 0.04))
    button_cancel = Button(
        ax=axbutton_cancel, label="Cancel", image=None, color="0.85", hovercolor="0.95"
    )
    button_cancel.on_clicked(on_clicked_cancel)  # type: ignore

    axbutton_addmask = fig.add_axes(rect=(0.3, 0.9, 0.2, 0.04))
    button_addmask = Button(
        ax=axbutton_addmask,
        label="Add mask",
        image=None,
        color="0.85",
        hovercolor="0.95",
    )
    button_addmask.on_clicked(on_clicked_add)  # type: ignore

    axbutton_removemask = fig.add_axes(rect=(0.5, 0.9, 0.2, 0.04))
    button_removemask = Button(
        ax=axbutton_removemask,
        label="Remove mask",
        image=None,
        color="0.85",
        hovercolor="0.95",
    )
    button_removemask.on_clicked(on_clicked_remove)  # type: ignore

    # ax_main.cla()

    mylogger.info("Display")
    new_roi: RoiPoly = RoiPoly(ax=ax_main, color="r", close_fig=False, show_fig=False)

    plt.show()


if __name__ == "__main__":
    argh.dispatch_command(main)
