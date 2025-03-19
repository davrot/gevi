import torch


@torch.no_grad()
def binning(
    data: torch.Tensor,
    kernel_size: int = 4,
    stride: int = 4,
    divisor_override: int | None = 1,
) -> torch.Tensor:

    try:
        return binning_internal(
            data=data,
            kernel_size=kernel_size,
            stride=stride,
            divisor_override=divisor_override,
        )
    except torch.cuda.OutOfMemoryError:
        return binning_internal(
            data=data.cpu(),
            kernel_size=kernel_size,
            stride=stride,
            divisor_override=divisor_override,
        ).to(device=data.device)


@torch.no_grad()
def binning_internal(
    data: torch.Tensor,
    kernel_size: int = 4,
    stride: int = 4,
    divisor_override: int | None = 1,
) -> torch.Tensor:

    assert data.ndim == 4
    return (
        torch.nn.functional.avg_pool2d(
            input=data.movedim(0, -1).movedim(0, -1),
            kernel_size=kernel_size,
            stride=stride,
            divisor_override=divisor_override,
        )
        .movedim(-1, 0)
        .movedim(-1, 0)
    )
