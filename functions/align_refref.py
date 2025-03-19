import torch
import torchvision as tv  # type: ignore
import logging
from functions.ImageAlignment import ImageAlignment
from functions.calculate_translation import calculate_translation
from functions.calculate_rotation import calculate_rotation


@torch.no_grad()
def align_refref(
    mylogger: logging.Logger,
    ref_image_acceptor: torch.Tensor,
    ref_image_donor: torch.Tensor,
    batch_size: int,
    fill_value: float = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    image_alignment = ImageAlignment(
        default_dtype=ref_image_acceptor.dtype, device=ref_image_acceptor.device
    )

    mylogger.info("Rotate ref image acceptor onto donor")
    angle_refref = calculate_rotation(
        image_alignment=image_alignment,
        input=ref_image_acceptor.unsqueeze(0),
        reference_image=ref_image_donor,
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

    mylogger.info("Translate ref image acceptor onto donor")
    tvec_refref = calculate_translation(
        image_alignment=image_alignment,
        input=ref_image_acceptor,
        reference_image=ref_image_donor,
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
    ).squeeze(0)

    return angle_refref, tvec_refref, ref_image_acceptor, ref_image_donor
