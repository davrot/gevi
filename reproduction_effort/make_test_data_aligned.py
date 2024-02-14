import torch
import torchvision as tv  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  # type: ignore
import json

from functions.align_cameras import align_cameras

if torch.cuda.is_available():
    device_name: str = "cuda:0"
else:
    device_name = "cpu"
print(f"Using device: {device_name}")
device: torch.device = torch.device(device_name)
dtype: torch.dtype = torch.float32

filename_bin_mat: str = "bin_old/Exp001_Trial001_Part001.mat"
filename_bin_mat_fake: str = "Exp001_Trial001_Part001_fake.mat"
fill_value: float = 0.0

mat_data = torch.tensor(
    sio.loadmat(filename_bin_mat)["nparray"].astype(dtype=np.float32),
    dtype=dtype,
    device=device,
)

angle_refref_target = torch.tensor(
    [2],
    dtype=dtype,
    device=device,
)

tvec_refref_target = torch.tensor(
    [10, 3],
    dtype=dtype,
    device=device,
)


t = (
    torch.arange(
        0,
        mat_data.shape[-2],
        dtype=dtype,
        device=device,
    )
    - mat_data.shape[-2] // 2
) / float(mat_data.shape[-2] // 2)

f_a: float = 8
A_a: float = 2
a_target = A_a * torch.sin(2 * torch.pi * t * f_a)

f_x: float = 5
A_x: float = 10
x_target = A_x * torch.sin(2.0 * torch.pi * t * f_x)

f_y: float = 7
A_y: float = 7
y_target = A_y * torch.sin(2 * torch.pi * t * f_y)

master_images: torch.Tensor = mat_data[:, :, mat_data.shape[-2] // 2, 1]

master_images_2: torch.Tensor = master_images.unsqueeze(-1).tile(
    (1, 1, mat_data.shape[-1])
)

# Rotate and move the whole timeseries of the acceptor and oxygenation
master_images_2[..., 0] = tv.transforms.functional.affine(
    img=master_images_2[..., 0].unsqueeze(0),
    angle=-float(angle_refref_target),
    translate=[0, 0],
    scale=1.0,
    shear=0,
    interpolation=tv.transforms.InterpolationMode.BILINEAR,
    fill=fill_value,
).squeeze(0)

master_images_2[..., 0] = tv.transforms.functional.affine(
    img=master_images_2[..., 0].unsqueeze(0),
    angle=0,
    translate=[tvec_refref_target[1], tvec_refref_target[0]],
    scale=1.0,
    shear=0,
    interpolation=tv.transforms.InterpolationMode.BILINEAR,
    fill=fill_value,
).squeeze(0)

master_images_2[..., 2] = tv.transforms.functional.affine(
    img=master_images_2[..., 2].unsqueeze(0),
    angle=-float(angle_refref_target),
    translate=[0, 0],
    scale=1.0,
    shear=0,
    interpolation=tv.transforms.InterpolationMode.BILINEAR,
    fill=fill_value,
).squeeze(0)

master_images_2[..., 2] = tv.transforms.functional.affine(
    img=master_images_2[..., 2].unsqueeze(0),
    angle=0,
    translate=[tvec_refref_target[1], tvec_refref_target[0]],
    scale=1.0,
    shear=0,
    interpolation=tv.transforms.InterpolationMode.BILINEAR,
    fill=fill_value,
).squeeze(0)

fake_data = master_images_2.unsqueeze(-2).tile((1, 1, mat_data.shape[-2], 1)).clone()

for t_id in range(0, fake_data.shape[-2]):
    for c_id in range(0, fake_data.shape[-1]):
        fake_data[..., t_id, c_id] = tv.transforms.functional.affine(
            img=fake_data[..., t_id, c_id].unsqueeze(0),
            angle=-float(a_target[t_id]),
            translate=[0, 0],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

fake_data = fake_data.clone()

for t_id in range(0, fake_data.shape[-2]):
    for c_id in range(0, fake_data.shape[-1]):
        fake_data[..., t_id, c_id] = tv.transforms.functional.affine(
            img=fake_data[..., t_id, c_id].unsqueeze(0),
            angle=0.0,
            translate=[y_target[t_id], x_target[t_id]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

fake_data_np = fake_data.cpu().numpy()
mdic = {"nparray": fake_data_np}
sio.savemat(filename_bin_mat_fake, mdic)

# ----------------------------------------------------

batch_size: int = 200
filename_raw_json: str = "raw/Exp001_Trial001_Part001_meta.txt"

with open(filename_raw_json, "r") as file_handle:
    metadata: dict = json.load(file_handle)
channels: list[str] = metadata["channelKey"]


data = torch.tensor(
    sio.loadmat(filename_bin_mat_fake)["nparray"].astype(np.float32),
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


print("References Acceptor <-> Donor:")
print("Rotation:")
print(f"target: {float(angle_refref_target):.3f}")
print(f"found: {-float(angle_refref):.3f}")
print("Translation")
print(f"target: {float(tvec_refref_target[0]):.3f}, {float(tvec_refref_target[1]):.3f}")
print(f"found: {-float(tvec_refref[0]):.3f}, {-float(tvec_refref[1]):.3f}")

plt.subplot(3, 1, 1)
plt.plot(-angle_donor_volume.cpu(), "g", label="angle found")
plt.plot(a_target.cpu(), "--k", label="angle target")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(-tvec_donor_volume[:, 0].cpu(), "g", label="x found")
plt.plot(x_target.cpu(), "k--", label="x target")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(-tvec_donor_volume[:, 1].cpu(), "g", label="y found")
plt.plot(y_target.cpu(), "k--", label="y target")
plt.legend()

plt.show()
