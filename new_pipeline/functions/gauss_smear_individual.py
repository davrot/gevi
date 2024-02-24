import torch
import math


@torch.no_grad()
def gauss_smear_individual(
    input: torch.Tensor,
    spatial_width: float,
    temporal_width: float,
    overwrite_fft_gauss: None | torch.Tensor = None,
    use_matlab_mask: bool = True,
    epsilon: float = float(torch.finfo(torch.float64).eps),
) -> tuple[torch.Tensor, torch.Tensor]:

    dim_x: int = int(2 * math.ceil(2 * spatial_width) + 1)
    dim_y: int = int(2 * math.ceil(2 * spatial_width) + 1)
    dim_t: int = int(2 * math.ceil(2 * temporal_width) + 1)
    dims_xyt: torch.Tensor = torch.tensor(
        [dim_x, dim_y, dim_t], dtype=torch.int64, device=input.device
    )

    if input.ndim == 2:
        input = input.unsqueeze(-1)

    input_padded = torch.nn.functional.pad(
        input.unsqueeze(0),
        pad=(
            dim_t,
            dim_t,
            dim_y,
            dim_y,
            dim_x,
            dim_x,
        ),
        mode="replicate",
    ).squeeze(0)

    if overwrite_fft_gauss is None:
        center_x: int = int(math.ceil(input_padded.shape[0] / 2))
        center_y: int = int(math.ceil(input_padded.shape[1] / 2))
        center_z: int = int(math.ceil(input_padded.shape[2] / 2))
        grid_x: torch.Tensor = (
            torch.arange(0, input_padded.shape[0], device=input.device) - center_x + 1
        )
        grid_y: torch.Tensor = (
            torch.arange(0, input_padded.shape[1], device=input.device) - center_y + 1
        )
        grid_z: torch.Tensor = (
            torch.arange(0, input_padded.shape[2], device=input.device) - center_z + 1
        )

        grid_x = grid_x.unsqueeze(-1).unsqueeze(-1) ** 2
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1) ** 2
        grid_z = grid_z.unsqueeze(0).unsqueeze(0) ** 2

        gauss_kernel: torch.Tensor = (
            (grid_x / (spatial_width**2))
            + (grid_y / (spatial_width**2))
            + (grid_z / (temporal_width**2))
        )

        if use_matlab_mask:
            filter_radius: torch.Tensor = (dims_xyt - 1) // 2

            border_lower: list[int] = [
                center_x - int(filter_radius[0]) - 1,
                center_y - int(filter_radius[1]) - 1,
                center_z - int(filter_radius[2]) - 1,
            ]

            border_upper: list[int] = [
                center_x + int(filter_radius[0]),
                center_y + int(filter_radius[1]),
                center_z + int(filter_radius[2]),
            ]

            matlab_mask: torch.Tensor = torch.zeros_like(gauss_kernel)
            matlab_mask[
                border_lower[0] : border_upper[0],
                border_lower[1] : border_upper[1],
                border_lower[2] : border_upper[2],
            ] = 1.0

        gauss_kernel = torch.exp(-gauss_kernel / 2.0)
        if use_matlab_mask:
            gauss_kernel = gauss_kernel * matlab_mask

        gauss_kernel[gauss_kernel < (epsilon * gauss_kernel.max())] = 0

        sum_gauss_kernel: float = float(gauss_kernel.sum())

        if sum_gauss_kernel != 0.0:
            gauss_kernel = gauss_kernel / sum_gauss_kernel

        # FFT Shift
        gauss_kernel = torch.cat(
            (gauss_kernel[center_x - 1 :, :, :], gauss_kernel[: center_x - 1, :, :]),
            dim=0,
        )
        gauss_kernel = torch.cat(
            (gauss_kernel[:, center_y - 1 :, :], gauss_kernel[:, : center_y - 1, :]),
            dim=1,
        )
        gauss_kernel = torch.cat(
            (gauss_kernel[:, :, center_z - 1 :], gauss_kernel[:, :, : center_z - 1]),
            dim=2,
        )
        overwrite_fft_gauss = torch.fft.fftn(gauss_kernel)
        input_padded_gauss_filtered: torch.Tensor = torch.real(
            torch.fft.ifftn(torch.fft.fftn(input_padded) * overwrite_fft_gauss)
        )
    else:
        input_padded_gauss_filtered = torch.real(
            torch.fft.ifftn(torch.fft.fftn(input_padded) * overwrite_fft_gauss)
        )

    start = dims_xyt
    stop = (
        torch.tensor(input_padded.shape, device=dims_xyt.device, dtype=dims_xyt.dtype)
        - dims_xyt
    )

    output = input_padded_gauss_filtered[
        start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]
    ]

    return (output, overwrite_fft_gauss)
