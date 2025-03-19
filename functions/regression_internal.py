import torch


def regression_internal(
    input_regressor: torch.Tensor, input_target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    regressor_offset = input_regressor.mean(keepdim=True, dim=-2)
    target_offset = input_target.mean(keepdim=True, dim=-1)

    regressor = input_regressor - regressor_offset
    target = input_target - target_offset

    try:
        coefficients, _, _, _ = torch.linalg.lstsq(regressor, target, rcond=None)
    except torch.cuda.OutOfMemoryError:
        coefficients_cpu, _, _, _ = torch.linalg.lstsq(
            regressor.cpu(), target.cpu(), rcond=None
        )
        coefficients = coefficients_cpu.to(regressor.device, copy=True)
        del coefficients_cpu

    intercept = target_offset.squeeze(-1) - (
        coefficients * regressor_offset.squeeze(-2)
    ).sum(dim=-1)

    return coefficients, intercept
