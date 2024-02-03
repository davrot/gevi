import torch
import torchvision as tv  # type: ignore

from functions.ImageAlignment import ImageAlignment
from functions.calculate_translation import calculate_translation
from functions.calculate_rotation import calculate_rotation


@torch.no_grad()
def align_refref(
    ref_image_acceptor: torch.Tensor,
    ref_image_donor: torch.Tensor,
    image_alignment: ImageAlignment,
    batch_size: int,
    fill_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    angle_refref = calculate_rotation(
        image_alignment,
        ref_image_acceptor.unsqueeze(0),
        ref_image_donor,
        batch_size=batch_size,
    )

    ref_image_acceptor = tv.transforms.functional.affine(
        img=ref_image_acceptor.unsqueeze(0),
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    tvec_refref = calculate_translation(
        image_alignment,
        ref_image_acceptor,
        ref_image_donor,
        batch_size=batch_size,
    )

    tvec_refref = tvec_refref[0, :]

    ref_image_acceptor = tv.transforms.functional.affine(
        img=ref_image_acceptor,
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    ref_image_acceptor = ref_image_acceptor.squeeze(0)

    return angle_refref, tvec_refref, ref_image_acceptor, ref_image_donor
