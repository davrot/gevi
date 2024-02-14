import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import scipy.io as sio  # type: ignore

from functions.binning import binning
from functions.align_cameras import align_cameras
from functions.bandpass import bandpass
from functions.make_mask import make_mask

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
test_overwrite_with_old_aligned: bool = False
filename_data_binning_replace: str = "bin_old/Exp001_Trial001_Part001.mat"
filename_data_aligned_replace: str = "aligned_old/Exp001_Trial001_Part001.mat"

data = torch.tensor(np.load(filename_raw).astype(np.float32), dtype=dtype)

with open(filename_raw_json, "r") as file_handle:
    metadata: dict = json.load(file_handle)
channels: list[str] = metadata["channelKey"]


if test_overwrite_with_old_bining:
    data = torch.tensor(
        sio.loadmat(filename_data_binning_replace)["nparray"].astype(np.float32),
        dtype=dtype,
        device=device,
    )
else:
    data = binning(data).to(device)

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


mask: torch.Tensor = make_mask(
    filename_mask=filename_mask,
    camera_sequence=camera_sequence,
    device=device,
    dtype=dtype,
)

mask_flatten = mask.flatten(start_dim=0, end_dim=-1)

original_shape = camera_sequence[0].shape
for i in range(0, len(camera_sequence)):
    camera_sequence[i] = bandpass(
        data=camera_sequence[i].clone(),
        device=camera_sequence[i].device,
        low_frequency=lower_freqency_bandpass,
        high_frequency=upper_freqency_bandpass,
        fs=100.0,
        filtfilt_chuck_size=10,
    )

    camera_sequence[i] = camera_sequence[i].flatten(start_dim=0, end_dim=-2)
    camera_sequence[i] = camera_sequence[i][mask_flatten, :]
    if (i == 0) or (i == 1):
        camera_sequence[i] = camera_sequence[i][:, 1:]
    else:
        camera_sequence[i] = (
            camera_sequence[i][:, 1:] + camera_sequence[i][:, :-1]
        ) / 2.0

    camera_sequence[i] = camera_sequence[i].movedim(0, -1)
    camera_sequence[i] -= camera_sequence[i].mean(dim=0, keepdim=True)


camera_sequence_cat = torch.cat(
    (camera_sequence[0], camera_sequence[1], camera_sequence[2], camera_sequence[3]),
    dim=-1,
)

print(camera_sequence_cat.min(), camera_sequence_cat.max())

u_a, s_a, Vh_a = torch.linalg.svd(camera_sequence_cat, full_matrices=False)
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

full_image = torch.cat(heart_beat_activity_map, dim=1)


# I want to scale the time series to std unity
# and therefore need to increase the amplitudes of the maps
u_a_std = torch.std(u_a)
u_a /= u_a_std
full_image *= u_a_std

plt.subplot(2, 1, 1)
plt.plot(u_a.cpu())
plt.xlabel("Frame ID")
plt.title(
    f"Common heartbeat in {lower_freqency_bandpass}Hz - {upper_freqency_bandpass}Hz"
)
plt.subplot(2, 1, 2)
plt.imshow(full_image.cpu(), cmap="hot")
plt.colorbar()
plt.title("acceptor, donor, oxygenation, volume")
plt.show()
