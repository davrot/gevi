import torch
import torchvision as tv  # type: ignore
import logging

from functions.calculate_translation import calculate_translation
from functions.ImageAlignment import ImageAlignment


@torch.no_grad()
def perform_donor_volume_translation(
    mylogger: logging.Logger,
    acceptor: torch.Tensor,
    donor: torch.Tensor,
    oxygenation: torch.Tensor,
    volume: torch.Tensor,
    ref_image_donor: torch.Tensor,
    ref_image_volume: torch.Tensor,
    batch_size: int,
    config: dict,
    fill_value: float = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    try:

        return perform_donor_volume_translation_internal(
            mylogger=mylogger,
            acceptor=acceptor,
            donor=donor,
            oxygenation=oxygenation,
            volume=volume,
            ref_image_donor=ref_image_donor,
            ref_image_volume=ref_image_volume,
            batch_size=batch_size,
            config=config,
            fill_value=fill_value,
        )

    except torch.cuda.OutOfMemoryError:

        (
            acceptor_cpu,
            donor_cpu,
            oxygenation_cpu,
            volume_cpu,
            tvec_donor_volume_cpu,
        ) = perform_donor_volume_translation_internal(
            mylogger=mylogger,
            acceptor=acceptor.cpu(),
            donor=donor.cpu(),
            oxygenation=oxygenation.cpu(),
            volume=volume.cpu(),
            ref_image_donor=ref_image_donor.cpu(),
            ref_image_volume=ref_image_volume.cpu(),
            batch_size=batch_size,
            config=config,
            fill_value=fill_value,
        )

        return (
            acceptor_cpu.to(device=acceptor.device),
            donor_cpu.to(device=acceptor.device),
            oxygenation_cpu.to(device=acceptor.device),
            volume_cpu.to(device=acceptor.device),
            tvec_donor_volume_cpu.to(device=acceptor.device),
        )


@torch.no_grad()
def perform_donor_volume_translation_internal(
    mylogger: logging.Logger,
    acceptor: torch.Tensor,
    donor: torch.Tensor,
    oxygenation: torch.Tensor,
    volume: torch.Tensor,
    ref_image_donor: torch.Tensor,
    ref_image_volume: torch.Tensor,
    batch_size: int,
    config: dict,
    fill_value: float = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    image_alignment = ImageAlignment(
        default_dtype=acceptor.dtype, device=acceptor.device
    )

    mylogger.info("Calculate translation between donor data and donor ref image")
    tvec_donor = calculate_translation(
        input=donor,
        reference_image=ref_image_donor,
        image_alignment=image_alignment,
        batch_size=batch_size,
    )

    mylogger.info("Calculate translation between volume data and volume ref image")
    tvec_volume = calculate_translation(
        input=volume,
        reference_image=ref_image_volume,
        image_alignment=image_alignment,
        batch_size=batch_size,
    )

    mylogger.info("Average over both translations")

    for i in range(0, 2):
        mylogger.info(f"Processing dimension {i}")
        donor_threshold: torch.Tensor = torch.sort(torch.abs(tvec_donor[:, i]))[0]
        donor_threshold = donor_threshold[
            int(
                donor_threshold.shape[0]
                * float(config["rotation_stabilization_threshold_border"])
            )
        ] * float(config["rotation_stabilization_threshold_factor"])

        volume_threshold: torch.Tensor = torch.sort(torch.abs(tvec_volume[:, i]))[0]
        volume_threshold = volume_threshold[
            int(
                volume_threshold.shape[0]
                * float(config["rotation_stabilization_threshold_border"])
            )
        ] * float(config["rotation_stabilization_threshold_factor"])

        donor_idx = torch.where(torch.abs(tvec_donor[:, i]) > donor_threshold)[0]
        volume_idx = torch.where(torch.abs(tvec_volume[:, i]) > volume_threshold)[0]
        mylogger.info(
            f"Border: {config['rotation_stabilization_threshold_border']}, "
            f"factor {config['rotation_stabilization_threshold_factor']}  "
        )
        mylogger.info(
            f"Donor threshold: {donor_threshold:.3e}, volume threshold: {volume_threshold:.3e}"
        )
        mylogger.info(
            f"Found broken rotation values: "
            f"donor {int(donor_idx.shape[0])}, "
            f"volume {int(volume_idx.shape[0])}"
        )
        tvec_donor[donor_idx, i] = tvec_volume[donor_idx, i]
        tvec_volume[volume_idx, i] = tvec_donor[volume_idx, i]

        donor_idx = torch.where(torch.abs(tvec_donor[:, i]) > donor_threshold)[0]
        volume_idx = torch.where(torch.abs(tvec_volume[:, i]) > volume_threshold)[0]
        mylogger.info(
            f"After fill in these broken rotation values remain: "
            f"donor {int(donor_idx.shape[0])}, "
            f"volume {int(volume_idx.shape[0])}"
        )
        tvec_donor[donor_idx, i] = 0.0
        tvec_volume[volume_idx, i] = 0.0

    tvec_donor_volume = (tvec_donor + tvec_volume) / 2.0

    mylogger.info("Translate acceptor data based on the average translation vector")
    for frame_id in range(0, tvec_donor_volume.shape[0]):
        acceptor[frame_id, ...] = tv.transforms.functional.affine(
            img=acceptor[frame_id, ...].unsqueeze(0),
            angle=0,
            translate=[tvec_donor_volume[frame_id, 1], tvec_donor_volume[frame_id, 0]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    mylogger.info("Translate donor data based on the average translation vector")
    for frame_id in range(0, tvec_donor_volume.shape[0]):
        donor[frame_id, ...] = tv.transforms.functional.affine(
            img=donor[frame_id, ...].unsqueeze(0),
            angle=0,
            translate=[tvec_donor_volume[frame_id, 1], tvec_donor_volume[frame_id, 0]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    mylogger.info("Translate oxygenation data based on the average translation vector")
    for frame_id in range(0, tvec_donor_volume.shape[0]):
        oxygenation[frame_id, ...] = tv.transforms.functional.affine(
            img=oxygenation[frame_id, ...].unsqueeze(0),
            angle=0,
            translate=[tvec_donor_volume[frame_id, 1], tvec_donor_volume[frame_id, 0]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    mylogger.info("Translate volume data based on the average translation vector")
    for frame_id in range(0, tvec_donor_volume.shape[0]):
        volume[frame_id, ...] = tv.transforms.functional.affine(
            img=volume[frame_id, ...].unsqueeze(0),
            angle=0,
            translate=[tvec_donor_volume[frame_id, 1], tvec_donor_volume[frame_id, 0]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

    return (acceptor, donor, oxygenation, volume, tvec_donor_volume)
