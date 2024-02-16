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

remove_heartbeat: bool = True

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

if remove_heartbeat:
    mask: torch.Tensor = make_mask(
        filename_mask=filename_mask,
        camera_sequence=camera_sequence,
        device=device,
        dtype=dtype,
    )

    mask_flatten = mask.flatten(start_dim=0, end_dim=-1)

    interpolate_along_time(camera_sequence)

    original_shape = camera_sequence[0].shape
    heartbeat_ts: list[torch.Tensor] = []
    for i in range(0, len(camera_sequence)):

        heartbeat_ts.append(
            bandpass(
                data=camera_sequence[i].clone(),
                device=camera_sequence[i].device,
                low_frequency=lower_freqency_bandpass,
                high_frequency=upper_freqency_bandpass,
                fs=100.0,
                filtfilt_chuck_size=10,
            )
        )

    for i in range(0, len(heartbeat_ts)):
        heartbeat_ts[i] = heartbeat_ts[i].flatten(start_dim=0, end_dim=-2)
        heartbeat_ts[i] = heartbeat_ts[i][mask_flatten, :]

        heartbeat_ts[i] = heartbeat_ts[i].movedim(0, -1)
        heartbeat_ts[i] -= heartbeat_ts[i].mean(dim=0, keepdim=True)

    heartbeat_ts_cat = torch.cat(
        (heartbeat_ts[0], heartbeat_ts[1], heartbeat_ts[2], heartbeat_ts[3]),
        dim=-1,
    )

    u_a, s_a, Vh_a = torch.linalg.svd(heartbeat_ts_cat, full_matrices=False)
    u_a = u_a[:, 0]
    Vh_a = Vh_a[0, :] * s_a[0]

    heart_beat_activity_map: list[torch.Tensor] = []

    start_pos: int = 0
    end_pos: int = 0
    for i in range(0, len(camera_sequence)):
        end_pos = start_pos + int(mask_flatten.sum())
        heart_beat_activity_map.append(
            torch.full(
                (original_shape[0], original_shape[1]),
                torch.nan,
                device=Vh_a.device,
                dtype=Vh_a.dtype,
            ).flatten(start_dim=0, end_dim=-1)
        )
        heart_beat_activity_map[-1][mask_flatten] = Vh_a[start_pos:end_pos]
        heart_beat_activity_map[-1] = heart_beat_activity_map[-1].reshape(
            (original_shape[0], original_shape[1])
        )
        start_pos = end_pos

    donor_power_factor = float(torch.abs(torch.nanmean(heart_beat_activity_map[0])))
    acceptor_power_factor = float(torch.abs(torch.nanmean(heart_beat_activity_map[1])))
    power_factors: None | list[float] = [donor_power_factor, acceptor_power_factor]

    for i in range(0, len(camera_sequence)):
        camera_sequence[i] -= heart_beat_activity_map[i].unsqueeze(-1) * u_a.unsqueeze(
            0
        ).unsqueeze(0)
else:
    power_factors = None
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
    lower_frequency_heartbeat=lower_frequency_heartbeat,
    upper_frequency_heartbeat=upper_frequency_heartbeat,
    sample_frequency=sample_frequency,
    power_factors=power_factors,
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
plt.show(block=False)


ratio_sequence_a = bandpass(
    data=data_acceptor,
    device=data_acceptor.device,
    low_frequency=lower_freqency_bandpass,
    high_frequency=upper_freqency_bandpass,
    fs=100.0,
    filtfilt_chuck_size=10,
)

ratio_sequence_b = bandpass(
    data=data_donor,
    device=data_donor.device,
    low_frequency=lower_freqency_bandpass,
    high_frequency=upper_freqency_bandpass,
    fs=100.0,
    filtfilt_chuck_size=10,
)

original_shape = ratio_sequence_a.shape

ratio_sequence_a = ratio_sequence_a.flatten(start_dim=0, end_dim=-2)
ratio_sequence_b = ratio_sequence_b.flatten(start_dim=0, end_dim=-2)

mask = mask.flatten(start_dim=0, end_dim=-1)
ratio_sequence_a = ratio_sequence_a[mask, :]
ratio_sequence_b = ratio_sequence_b[mask, :]

ratio_sequence_a = ratio_sequence_a.movedim(0, -1)
ratio_sequence_b = ratio_sequence_b.movedim(0, -1)

ratio_sequence_a -= ratio_sequence_a.mean(dim=0, keepdim=True)
ratio_sequence_b -= ratio_sequence_b.mean(dim=0, keepdim=True)

u_a, s_a, Vh_a = torch.linalg.svd(ratio_sequence_a, full_matrices=False)
u_a = u_a[:, 0]
s_a = s_a[0]
Vh_a = Vh_a[0, :]

u_a_std = torch.std(u_a)
u_a /= u_a_std
Vh_a *= u_a_std

heartbeatactivitmap_a = torch.zeros(
    (original_shape[0], original_shape[1]), device=Vh_a.device, dtype=Vh_a.dtype
).flatten(start_dim=0, end_dim=-1)

heartbeatactivitmap_a *= torch.nan
heartbeatactivitmap_a[mask] = s_a * Vh_a
heartbeatactivitmap_a = heartbeatactivitmap_a.reshape(
    (original_shape[0], original_shape[1])
)

u_b, s_b, Vh_b = torch.linalg.svd(ratio_sequence_b, full_matrices=False)
u_b = u_b[:, 0]
s_b = s_b[0]
Vh_b = Vh_b[0, :]

u_b_std = torch.std(u_b)
u_b /= u_b_std
Vh_b *= u_b_std

heartbeatactivitmap_b = torch.zeros(
    (original_shape[0], original_shape[1]), device=Vh_b.device, dtype=Vh_b.dtype
).flatten(start_dim=0, end_dim=-1)

heartbeatactivitmap_b *= torch.nan
heartbeatactivitmap_b[mask] = s_b * Vh_b
heartbeatactivitmap_b = heartbeatactivitmap_b.reshape(
    (original_shape[0], original_shape[1])
)

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(u_a.cpu(), label="aceptor")
plt.plot(u_b.cpu(), label="donor")
plt.legend()
plt.subplot(2, 1, 2)
plt.imshow(
    torch.cat(
        (
            heartbeatactivitmap_a,
            heartbeatactivitmap_b,
        ),
        dim=1,
    ).cpu()
)
plt.colorbar()
plt.show()
