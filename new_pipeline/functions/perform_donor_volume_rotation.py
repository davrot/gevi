import torch
import torchvision as tv  # type: ignore
import logging
from functions.calculate_rotation import calculate_rotation
from functions.ImageAlignment import ImageAlignment


@torch.no_grad()
def perform_donor_volume_rotation(
    mylogger: logging.Logger,
    acceptor: torch.Tensor,
    donor: torch.Tensor,
    oxygenation: torch.Tensor,
    volume: torch.Tensor,
    ref_image_donor: torch.Tensor,
    ref_image_volume: torch.Tensor,
    image_alignment: ImageAlignment,
    batch_size: int,
    fill_value: float = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    mylogger.info("Calculate rotation between donor data and donor ref image")
    angle_donor = calculate_rotation(
        input=donor,
        reference_image=ref_image_donor,
        image_alignment=image_alignment,
        batch_size=batch_size,
    )

    mylogger.info("Calculate rotation between volume data and volume ref image")
    angle_volume = calculate_rotation(
        input=volume,
        reference_image=ref_image_volume,
        image_alignment=image_alignment,
        batch_size=batch_size,
    )

    mylogger.info("Average over both rotations")
    angle_donor_volume = (angle_donor + angle_volume) / 2.0

    mylogger.info("Rotate acceptor data based on the average rotation")
    for frame_id in range(0, angle_donor_volume.shape[0]):
        acceptor[frame_id, ...] = tv.transforms.functional.affine(
            img=acceptor[frame_id, ...].unsqueeze(0),
            angle=-float(angle_donor_volume[frame_id]),
            translate=[0, 0],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    mylogger.info("Rotate donor data based on the average rotation")
    for frame_id in range(0, angle_donor_volume.shape[0]):
        donor[frame_id, ...] = tv.transforms.functional.affine(
            img=donor[frame_id, ...].unsqueeze(0),
            angle=-float(angle_donor_volume[frame_id]),
            translate=[0, 0],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    mylogger.info("Rotate oxygenation data based on the average rotation")
    for frame_id in range(0, angle_donor_volume.shape[0]):
        oxygenation[frame_id, ...] = tv.transforms.functional.affine(
            img=oxygenation[frame_id, ...].unsqueeze(0),
            angle=-float(angle_donor_volume[frame_id]),
            translate=[0, 0],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    mylogger.info("Rotate volume data based on the average rotation")
    for frame_id in range(0, angle_donor_volume.shape[0]):
        volume[frame_id, ...] = tv.transforms.functional.affine(
            img=volume[frame_id, ...].unsqueeze(0),
            angle=-float(angle_donor_volume[frame_id]),
            translate=[0, 0],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    return (acceptor, donor, oxygenation, volume, angle_donor_volume)
