import matplotlib.pyplot as plt  # type:ignore
import matplotlib
import numpy as np
import torch
import os
import json

from jsmin import jsmin  # type:ignore
from matplotlib.widgets import Slider, Button  # type:ignore
from functools import partial
from functions.gauss_smear_individual import gauss_smear_individual
from functions.create_logger import create_logger
from functions.get_torch_device import get_torch_device

mylogger = create_logger(
    save_logging_messages=True, display_logging_messages=True, log_stage_name="stage_2"
)

mylogger.info("loading config file")
with open("config.json", "r") as file:
    config = json.loads(jsmin(file.read()))

threshold: float = 0.05
path: str = config["ref_image_path"]
use_channel: str = "donor"
spatial_width: float = 4.0
temporal_width: float = 0.1


image_ref_file: str = os.path.join(path, use_channel + ".npy")
image_var_file: str = os.path.join(path, use_channel + "_var.npy")
heartbeat_mask_file: str = os.path.join(path, "heartbeat_mask.npy")
heartbeat_mask_threshold_file: str = os.path.join(path, "heartbeat_mask_threshold.npy")

device = get_torch_device(mylogger, config["force_to_cpu"])


def next_frame(
    i: float, images: np.ndarray, image_handle: matplotlib.image.AxesImage
) -> None:
    global threshold
    threshold = i

    display_image: np.ndarray = images.copy()
    display_image[..., 2] = display_image[..., 0]
    mask: np.ndarray = np.where(images[..., 2] >= i, 1.0, np.nan)[..., np.newaxis]
    display_image *= mask
    display_image = np.nan_to_num(display_image, nan=1.0)

    image_handle.set_data(display_image)
    return


def on_clicked_accept(event: matplotlib.backend_bases.MouseEvent) -> None:
    global threshold
    global image_3color
    global path
    global mylogger
    global heartbeat_mask_file
    global heartbeat_mask_threshold_file

    mylogger.info(f"Threshold: {threshold}")

    mask: np.ndarray = image_3color[..., 2] >= threshold
    mylogger.info(f"Save mask to: {heartbeat_mask_file}")
    np.save(heartbeat_mask_file, mask)
    mylogger.info(f"Save threshold to: {heartbeat_mask_threshold_file}")
    np.save(heartbeat_mask_threshold_file, np.array([threshold]))
    exit()


def on_clicked_cancel(event: matplotlib.backend_bases.MouseEvent) -> None:
    exit()


mylogger.info(f"loading image reference file: {image_ref_file}")
image_ref: np.ndarray = np.load(image_ref_file)
image_ref /= image_ref.max()

mylogger.info(f"loading image heartbeat power: {image_var_file}")
image_var: np.ndarray = np.load(image_var_file)
image_var /= image_var.max()

mylogger.info("Smear the image heartbeat power patially")
temp, _ = gauss_smear_individual(
    input=torch.tensor(image_var[..., np.newaxis], device=device),
    spatial_width=spatial_width,
    temporal_width=temporal_width,
    use_matlab_mask=False,
)
temp /= temp.max()

mylogger.info("-==- DONE -==-")

image_3color = np.concatenate(
    (
        np.zeros_like(image_ref[..., np.newaxis]),
        image_ref[..., np.newaxis],
        temp.cpu().numpy(),
    ),
    axis=-1,
)

mylogger.info("Prepare image")

display_image = image_3color.copy()
display_image[..., 2] = display_image[..., 0]
mask = np.where(image_3color[..., 2] >= threshold, 1.0, np.nan)[..., np.newaxis]
display_image *= mask
display_image = np.nan_to_num(display_image, nan=1.0)

value_sort = np.sort(image_var.flatten())
value_sort_max = value_sort[int(value_sort.shape[0] * 0.95)]
mylogger.info("-==- DONE -==-")

mylogger.info("Create figure")

fig: matplotlib.figure.Figure = plt.figure()

image_handle = plt.imshow(display_image, vmin=0, vmax=1, cmap="hot")

mylogger.info("Add controls")

axfreq = fig.add_axes(rect=(0.4, 0.9, 0.3, 0.03))
slice_slider = Slider(
    ax=axfreq,
    label="Slice",
    valmin=0,
    valmax=value_sort_max,
    valinit=threshold,
    valstep=value_sort_max / 100.0,
)
axbutton_accept = fig.add_axes(rect=(0.3, 0.85, 0.2, 0.04))
button_accept = Button(
    ax=axbutton_accept, label="Accept", image=None, color="0.85", hovercolor="0.95"
)
button_accept.on_clicked(on_clicked_accept)  # type: ignore

axbutton_cancel = fig.add_axes(rect=(0.55, 0.85, 0.2, 0.04))
button_cancel = Button(
    ax=axbutton_cancel, label="Cancel", image=None, color="0.85", hovercolor="0.95"
)
button_cancel.on_clicked(on_clicked_cancel)  # type: ignore

slice_slider.on_changed(
    partial(next_frame, images=image_3color, image_handle=image_handle)
)

mylogger.info("Display")
plt.show()
