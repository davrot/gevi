import torch
from functions.gauss_smear_individual import gauss_smear_individual


@torch.no_grad()
def gauss_smear(
    input_cameras: list[torch.Tensor],
    input_mask: torch.Tensor,
    spatial_width: float,
    temporal_width: float,
    use_matlab_mask: bool = True,
    epsilon: float = float(torch.finfo(torch.float64).eps),
) -> list[torch.Tensor]:
    assert len(input_cameras) == 4

    filtered_mask: torch.Tensor
    filtered_mask, _ = gauss_smear_individual(
        input=input_mask,
        spatial_width=spatial_width,
        temporal_width=temporal_width,
        use_matlab_mask=use_matlab_mask,
        epsilon=epsilon,
    )

    overwrite_fft_gauss: None | torch.Tensor = None
    for id in range(0, len(input_cameras)):

        input_cameras[id] *= input_mask.unsqueeze(-1)
        input_cameras[id], overwrite_fft_gauss = gauss_smear_individual(
            input=input_cameras[id],
            spatial_width=spatial_width,
            temporal_width=temporal_width,
            overwrite_fft_gauss=overwrite_fft_gauss,
            use_matlab_mask=use_matlab_mask,
            epsilon=epsilon,
        )

        input_cameras[id] /= filtered_mask + 1e-20
        input_cameras[id] += 1.0 - input_mask.unsqueeze(-1)

    return input_cameras
