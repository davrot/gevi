import torch


def regression_internal(
    input_regressor: torch.Tensor, input_target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    regressor_offset = input_regressor.mean(keepdim=True, dim=-2)
    target_offset = input_target.mean(keepdim=True, dim=-1)

    regressor = input_regressor - regressor_offset
    target = input_target - target_offset

    coefficients, _, _, _ = torch.linalg.lstsq(regressor, target, rcond=None)  # None ?

    intercept = target_offset.squeeze(-1) - (
        coefficients * regressor_offset.squeeze(-2)
    ).sum(dim=-1)

    return coefficients, intercept
