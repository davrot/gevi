import torch
import torchvision as tv  # type: ignore
import numpy as np
import json
import scipy.io as sio  # type: ignore

from functions.align_refref import align_refref
from functions.perform_donor_volume_rotation import perform_donor_volume_rotation
from functions.perform_donor_volume_translation import perform_donor_volume_translation
from functions.ImageAlignment import ImageAlignment


@torch.no_grad()
def align_cameras(
    filename_raw_json: str,
    filename_bin_mat: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    fill_value: float = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    image_alignment = ImageAlignment(default_dtype=dtype, device=device)

    # --- Load data ---
    with open(filename_raw_json, "r") as file_handle:
        metadata: dict = json.load(file_handle)
    channels: list[str] = metadata["channelKey"]

    data = torch.tensor(
        sio.loadmat(filename_bin_mat)["nparray"].astype(np.float32),
        device=device,
        dtype=dtype,
    )
    # --==-- DONE --==--

    # --- Get reference image ---
    acceptor_index: int = channels.index("acceptor")
    donor_index: int = channels.index("donor")
    oxygenation_index: int = channels.index("oxygenation")
    volume_index: int = channels.index("volume")

    # --==-- DONE --==--

    # --- Sort data ---
    acceptor = data[..., acceptor_index].moveaxis(-1, 0).clone()
    donor = data[..., donor_index].moveaxis(-1, 0).clone()
    oxygenation = data[..., oxygenation_index].moveaxis(-1, 0).clone()
    volume = data[..., volume_index].moveaxis(-1, 0).clone()
    del data
    # --==-- DONE --==--

    # --- Calculate translation and rotation between the reference images ---
    angle_refref, tvec_refref, ref_image_acceptor, ref_image_donor = align_refref(
        ref_image_acceptor=acceptor[
            acceptor.shape[0] // 2,
            :,
            :,
        ],
        ref_image_donor=donor[
            donor.shape[0] // 2,
            :,
            :,
        ],
        image_alignment=image_alignment,
        batch_size=batch_size,
        fill_value=fill_value,
    )

    ref_image_oxygenation = tv.transforms.functional.affine(
        img=oxygenation[
            oxygenation.shape[0] // 2,
            :,
            :,
        ].unsqueeze(0),
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    ref_image_oxygenation = tv.transforms.functional.affine(
        img=ref_image_oxygenation,
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    ref_image_oxygenation = ref_image_oxygenation.squeeze(0)

    ref_image_volume = volume[
        volume.shape[0] // 2,
        :,
        :,
    ].clone()

    # --==-- DONE --==--

    # --- Rotate and translate the acceptor and oxygenation data accordingly ---
    acceptor = tv.transforms.functional.affine(
        img=acceptor,
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    acceptor = tv.transforms.functional.affine(
        img=acceptor,
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    oxygenation = tv.transforms.functional.affine(
        img=oxygenation,
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    oxygenation = tv.transforms.functional.affine(
        img=oxygenation,
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )
    # --==-- DONE --==--

    acceptor, donor, oxygenation, volume, angle_donor_volume = (
        perform_donor_volume_rotation(
            acceptor=acceptor,
            donor=donor,
            oxygenation=oxygenation,
            volume=volume,
            ref_image_donor=ref_image_donor,
            ref_image_volume=ref_image_volume,
            image_alignment=image_alignment,
            batch_size=batch_size,
            fill_value=fill_value,
        )
    )

    acceptor, donor, oxygenation, volume, tvec_donor_volume = (
        perform_donor_volume_translation(
            acceptor=acceptor,
            donor=donor,
            oxygenation=oxygenation,
            volume=volume,
            ref_image_donor=ref_image_donor,
            ref_image_volume=ref_image_volume,
            image_alignment=image_alignment,
            batch_size=batch_size,
            fill_value=fill_value,
        )
    )

    return (
        acceptor,
        donor,
        oxygenation,
        volume,
        angle_donor_volume,
        tvec_donor_volume,
        angle_refref,
        tvec_refref,
    )
