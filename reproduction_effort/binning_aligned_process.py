import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import h5py  # type: ignore
import scipy.io as sio  # type: ignore


from functions.binning import binning
from functions.align_cameras import align_cameras
from functions.preprocessing import preprocessing
from functions.bandpass import bandpass
from functions.make_mask import make_mask
from functions.interpolate_along_time import interpolate_along_time

if torch.cuda.is_available():
    device_name: str = "cuda:0"
else:
    device_name = "cpu"
print(f"Using device: {device_name}")
device: torch.device = torch.device(device_name)
dtype: torch.dtype = torch.float32


filename_raw: str = f"raw{os.sep}Exp001_Trial001_Part001.npy"
filename_raw_json: str = f"raw{os.sep}Exp001_Trial001_Part001_meta.txt"
filename_mask: str = "2020-12-08maskPixelraw2.mat"

first_none_ramp_frame: int = 100
spatial_width: float = 2
temporal_width: float = 0.1

lower_freqency_bandpass: float = 5.0
upper_freqency_bandpass: float = 14.0

lower_frequency_heartbeat: float = 5.0
upper_frequency_heartbeat: float = 14.0
sample_frequency: float = 100.0

target_camera: list[str] = ["acceptor", "donor"]
regressor_cameras: list[str] = ["oxygenation", "volume"]
batch_size: int = 200
required_order: list[str] = ["acceptor", "donor", "oxygenation", "volume"]


test_overwrite_with_old_bining: bool = False
test_overwrite_with_old_aligned: bool = True
filename_data_binning_replace: str = "bin_old/Exp001_Trial001_Part001.mat"
filename_data_aligned_replace: str = "aligned_old/Exp001_Trial001_Part001.mat"

data = torch.tensor(np.load(filename_raw).astype(np.float32), dtype=dtype)

with open(filename_raw_json, "r") as file_handle:
    metadata: dict = json.load(file_handle)
channels: list[str] = metadata["channelKey"]

data = binning(data).to(device)

if test_overwrite_with_old_bining:
    data = torch.tensor(
        sio.loadmat(filename_data_binning_replace)["nparray"].astype(np.float32),
        dtype=dtype,
        device=device,
    )

ref_image = data[:, :, data.shape[-2] // 2, :].clone()

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
    channels=channels,
    data=data,
    ref_image=ref_image,
    device=device,
    dtype=dtype,
    batch_size=batch_size,
    fill_value=-1,
)
del data


camera_sequence: list[torch.Tensor] = []

for cam in required_order:
    if cam.startswith("acceptor"):
        camera_sequence.append(acceptor.movedim(0, -1).clone())
        del acceptor
    if cam.startswith("donor"):
        camera_sequence.append(donor.movedim(0, -1).clone())
        del donor
    if cam.startswith("oxygenation"):
        camera_sequence.append(oxygenation.movedim(0, -1).clone())
        del oxygenation
    if cam.startswith("volume"):
        camera_sequence.append(volume.movedim(0, -1).clone())
        del volume

if test_overwrite_with_old_aligned:

    data_aligned_replace: torch.Tensor = torch.tensor(
        sio.loadmat(filename_data_aligned_replace)["data"].astype(np.float32),
        device=device,
        dtype=dtype,
    )

    camera_sequence[0] = data_aligned_replace[..., 0].clone()
    camera_sequence[1] = data_aligned_replace[..., 1].clone()
    camera_sequence[2] = data_aligned_replace[..., 2].clone()
    camera_sequence[3] = data_aligned_replace[..., 3].clone()
    del data_aligned_replace

# ->


mask: torch.Tensor = make_mask(
    filename_mask=filename_mask,
    camera_sequence=camera_sequence,
    device=device,
    dtype=dtype,
)

mask_flatten = mask.flatten(start_dim=0, end_dim=-1)

interpolate_along_time(camera_sequence)

heartbeat_ts: torch.Tensor = bandpass(
    data=camera_sequence[channels.index("volume")].clone(),
    device=camera_sequence[channels.index("volume")].device,
    low_frequency=lower_freqency_bandpass,
    high_frequency=upper_freqency_bandpass,
    fs=sample_frequency,
    filtfilt_chuck_size=10,
)

heartbeat_ts_copy = heartbeat_ts.clone()

heartbeat_ts = heartbeat_ts.flatten(start_dim=0, end_dim=-2)
heartbeat_ts = heartbeat_ts[mask_flatten, :]

heartbeat_ts = heartbeat_ts.movedim(0, -1)
heartbeat_ts -= heartbeat_ts.mean(dim=0, keepdim=True)

volume_heartbeat, _, _ = torch.linalg.svd(heartbeat_ts, full_matrices=False)
volume_heartbeat = volume_heartbeat[:, 0]
volume_heartbeat -= volume_heartbeat[first_none_ramp_frame:].mean()
volume_heartbeat = volume_heartbeat.unsqueeze(0).unsqueeze(0)

heartbeat_coefficients: list[torch.Tensor] = []
for i in range(0, len(camera_sequence)):
    y = bandpass(
        data=camera_sequence[i].clone(),
        device=camera_sequence[i].device,
        low_frequency=lower_freqency_bandpass,
        high_frequency=upper_freqency_bandpass,
        fs=sample_frequency,
        filtfilt_chuck_size=10,
    )[..., first_none_ramp_frame:]
    y -= y.mean(dim=-1, keepdim=True)

    heartbeat_coefficients.append(
        (
            (volume_heartbeat[..., first_none_ramp_frame:] * y).sum(
                dim=-1, keepdim=True
            )
            / (volume_heartbeat[..., first_none_ramp_frame:] ** 2).sum(
                dim=-1, keepdim=True
            )
        )
        * mask.unsqueeze(-1)
    )
del y

donor_correction_factor = heartbeat_coefficients[channels.index("donor")].clone()
acceptor_correction_factor = heartbeat_coefficients[channels.index("acceptor")].clone()


for i in range(0, len(camera_sequence)):
    camera_sequence[i] -= heartbeat_coefficients[i] * volume_heartbeat


donor_factor: torch.Tensor = (donor_correction_factor + acceptor_correction_factor) / (
    2 * donor_correction_factor
)
acceptor_factor: torch.Tensor = (
    donor_correction_factor + acceptor_correction_factor
) / (2 * acceptor_correction_factor)


# mean_values: list = []
# for i in range(0, len(channels)):
#     mean_values.append(
#         camera_sequence[i][..., first_none_ramp_frame:].nanmean(dim=-1, keepdim=True)
#     )
#     camera_sequence[i] -= mean_values[i]

camera_sequence[channels.index("acceptor")] *= acceptor_factor * mask.unsqueeze(-1)
camera_sequence[channels.index("donor")] *= donor_factor * mask.unsqueeze(-1)

# for i in range(0, len(channels)):
#     camera_sequence[i] -= mean_values[i]

# exit()
# <-

data_acceptor, data_donor, mask = preprocessing(
    cameras=channels,
    camera_sequence=camera_sequence,
    filename_mask=filename_mask,
    device=device,
    first_none_ramp_frame=first_none_ramp_frame,
    spatial_width=spatial_width,
    temporal_width=temporal_width,
    target_camera=target_camera,
    regressor_cameras=regressor_cameras,
)

ratio_sequence: torch.Tensor = data_acceptor / data_donor

ratio_sequence /= ratio_sequence.mean(dim=-1, keepdim=True)
ratio_sequence = torch.nan_to_num(ratio_sequence, nan=0.0)

new: np.ndarray = ratio_sequence.cpu().numpy()

file_handle = h5py.File("old.mat", "r")
old: np.ndarray = np.array(file_handle["ratioSequence"])  # type:ignore
# HDF5 loads everything backwards...
old = np.moveaxis(old, 0, -1)
old = np.moveaxis(old, 0, -2)

pos_x = 25
pos_y = 75

plt.figure(1)
plt.subplot(2, 1, 1)
new_select = new[pos_x, pos_y, :]
old_select = old[pos_x, pos_y, :]
plt.plot(old_select, "r", label="Old")
plt.plot(new_select, "k", label="New")

# plt.plot(old_select - new_select + 1.0, label="Old - New + 1")
plt.title(f"Position: {pos_x}, {pos_y}")
plt.legend()

plt.subplot(2, 1, 2)
differences = (np.abs(new - old)).max(axis=-1) * mask.cpu().numpy()
plt.imshow(differences, cmap="hot")
plt.title("Max of abs(new-old) along time")
plt.colorbar()
plt.show()
