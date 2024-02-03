import torch
import torchvision as tv  # type: ignore
from functions.calculate_translation import calculate_translation
from functions.ImageAlignment import ImageAlignment


@torch.no_grad()
def perform_donor_volume_translation(
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

    tvec_donor = calculate_translation(
        input=donor,
        reference_image=ref_image_donor,
        image_alignment=image_alignment,
        batch_size=batch_size,
    )

    tvec_volume = calculate_translation(
        input=volume,
        reference_image=ref_image_volume,
        image_alignment=image_alignment,
        batch_size=batch_size,
    )

    tvec_donor_volume = (tvec_donor + tvec_volume) / 2.0

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

        donor[frame_id, ...] = tv.transforms.functional.affine(
            img=donor[frame_id, ...].unsqueeze(0),
            angle=0,
            translate=[tvec_donor_volume[frame_id, 1], tvec_donor_volume[frame_id, 0]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

        oxygenation[frame_id, ...] = tv.transforms.functional.affine(
            img=oxygenation[frame_id, ...].unsqueeze(0),
            angle=0,
            translate=[tvec_donor_volume[frame_id, 1], tvec_donor_volume[frame_id, 0]],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=fill_value,
        ).squeeze(0)

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
