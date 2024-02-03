import torch


def binning(
    data: torch.Tensor,
    kernel_size: int = 4,
    stride: int = 4,
    divisor_override: int | None = 1,
) -> torch.Tensor:

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
