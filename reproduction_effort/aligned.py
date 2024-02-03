import scipy.io as sio  # type: ignore
import torch
import numpy as np
import matplotlib.pyplot as plt

from functions.align_cameras import align_cameras

if __name__ == "__main__":

    if torch.cuda.is_available():
        device_name: str = "cuda:0"
    else:
        device_name = "cpu"
    print(f"Using device: {device_name}")
    device: torch.device = torch.device(device_name)
    dtype: torch.dtype = torch.float32

    filename_raw_json: str = "raw/Exp001_Trial001_Part001_meta.txt"
    filename_bin_mat: str = "bin_old/Exp001_Trial001_Part001.mat"
    batch_size: int = 200

    filename_aligned_mat: str = "aligned_old/Exp001_Trial001_Part001.mat"

    (
        acceptor,
        donor,
        oxygenation,
        volume,
        angle_donor_volume,
        tvec_donor_volume,
        angle_refref,
        tvec_refref,
    ) = align_cameras(
        filename_raw_json=filename_raw_json,
        filename_bin_mat=filename_bin_mat,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        fill_value=-1,
    )

    mat_data = torch.tensor(
        sio.loadmat(filename_aligned_mat)["data"].astype(dtype=np.float32),
        dtype=dtype,
        device=device,
    )

    old: list = []
    old.append(mat_data[..., 0].movedim(-1, 0))
    old.append(mat_data[..., 1].movedim(-1, 0))
    old.append(mat_data[..., 2].movedim(-1, 0))
    old.append(mat_data[..., 3].movedim(-1, 0))

    new: list = []
    new.append(acceptor)
    new.append(donor)
    new.append(oxygenation)
    new.append(volume)

    names: list = []
    new.append("acceptor")
    new.append("donor")
    new.append("oxygenation")
    new.append("volume")

    mask = torch.zeros(
        (acceptor.shape[-2], acceptor.shape[-1]),
        dtype=torch.bool,
        device=device,
    )

    mask[torch.any(acceptor < 0, dim=0)] = True
    mask[torch.any(donor < 0, dim=0)] = True
    mask[torch.any(oxygenation < 0, dim=0)] = True
    mask[torch.any(volume < 0, dim=0)] = True

    frame_id: int = 0
    image: list = []
    for channel_id in range(0, len(old)):
        temp = np.zeros((new[channel_id].shape[-2], new[channel_id].shape[-1], 3))
        temp[:, :, 0] = (
            old[channel_id][frame_id, ...] / old[channel_id][frame_id, ...].max()
        ).cpu()
        temp[:, :, 1] = (
            new[channel_id][frame_id, ...] / new[channel_id][frame_id, ...].max()
        ).cpu()
        temp[:, :, 2] = 0.0
        image.append(temp)

    subplot_position: int = 1
    for channel_id in range(0, len(old)):
        difference = (image[channel_id][..., 0] - image[channel_id][..., 1]) / (
            image[channel_id][..., 0] + image[channel_id][..., 1]
        )
        plt.subplot(4, 2, subplot_position)
        plt.imshow(difference, cmap="hot")
        plt.colorbar()
        subplot_position += 1

        plt.subplot(4, 2, subplot_position)
        plt.plot(np.sort(difference.flatten()))
        subplot_position += 1

    plt.show()
