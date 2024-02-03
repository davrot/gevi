import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py  # type: ignore

from functions.preprocessing import preprocessing


if __name__ == "__main__":

    if torch.cuda.is_available():
        device_name: str = "cuda:0"
    else:
        device_name = "cpu"
    print(f"Using device: {device_name}")
    device: torch.device = torch.device(device_name)

    filename_metadata: str = "raw/Exp001_Trial001_Part001_meta.txt"
    filename_data: str = "Exp001_Trial001_Part001.mat"
    filename_mask: str = "2020-12-08maskPixelraw2.mat"

    first_none_ramp_frame: int = 100
    spatial_width: float = 2
    temporal_width: float = 0.1

    target_camera: list[str] = ["acceptor", "donor"]
    regressor_cameras: list[str] = ["oxygenation", "volume"]

    data_acceptor, data_donor, mask = preprocessing(
        filename_metadata=filename_metadata,
        filename_data=filename_data,
        filename_mask=filename_mask,
        device=device,
        first_none_ramp_frame=first_none_ramp_frame,
        spatial_width=spatial_width,
        temporal_width=temporal_width,
        target_camera=target_camera,
        regressor_cameras=regressor_cameras,
    )

    ratio_sequence: torch.Tensor = data_acceptor / data_donor

    new: np.ndarray = ratio_sequence.cpu().numpy()

    file_handle = h5py.File("old.mat", "r")
    old: np.ndarray = np.array(file_handle["ratioSequence"])
    # HDF5 loads everything backwards...
    old = np.moveaxis(old, 0, -1)
    old = np.moveaxis(old, 0, -2)

    pos_x = 25
    pos_y = 75

    plt.subplot(2, 1, 1)
    new_select = new[pos_x, pos_y, :]
    old_select = old[pos_x, pos_y, :]
    plt.plot(new_select, label="New")
    plt.plot(old_select, "--", label="Old")
    plt.plot(old_select - new_select + 1.0, label="Old - New + 1")
    plt.title(f"Position: {pos_x}, {pos_y}")
    plt.legend()

    plt.subplot(2, 1, 2)
    differences = (np.abs(new - old)).max(axis=-1)
    plt.imshow(differences)
    plt.title("Max of abs(new-old) along time")
    plt.colorbar()
    plt.show()
