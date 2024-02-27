import torch
from functions.regression_internal import regression_internal


@torch.no_grad()
def regression(
    target_camera_id: int,
    regressor_camera_ids: list[int],
    mask: torch.Tensor,
    camera_sequence: list[torch.Tensor],
    camera_sequence_filtered: list[torch.Tensor],
    first_none_ramp_frame: int,
) -> torch.Tensor:

    assert len(regressor_camera_ids) > 0

    # ------- Prepare the target signals ----------
    target_signals_train: torch.Tensor = (
        camera_sequence_filtered[target_camera_id][..., first_none_ramp_frame:].clone()
        - 1.0
    )
    target_signals_train[target_signals_train < -1] = 0.0

    target_signals_perform: torch.Tensor = (
        camera_sequence[target_camera_id].clone() - 1.0
    )

    # Check if everything is happy
    assert target_signals_train.ndim == 3
    assert target_signals_train.ndim == target_signals_perform.ndim
    assert target_signals_train.shape[0] == target_signals_perform.shape[0]
    assert target_signals_train.shape[1] == target_signals_perform.shape[1]
    assert (
        target_signals_train.shape[2] + first_none_ramp_frame
    ) == target_signals_perform.shape[2]
    # --==DONE==-

    # ------- Prepare the regressor signals ----------

    # --- Train ---

    regressor_signals_train: torch.Tensor = torch.zeros(
        (
            camera_sequence_filtered[0].shape[0],
            camera_sequence_filtered[0].shape[1],
            camera_sequence_filtered[0].shape[2],
            len(regressor_camera_ids) + 1,
        ),
        device=camera_sequence_filtered[0].device,
        dtype=camera_sequence_filtered[0].dtype,
    )

    # Copy the regressor signals -1
    for matrix_id, id in enumerate(regressor_camera_ids):
        regressor_signals_train[..., matrix_id] = camera_sequence_filtered[id] - 1.0

    regressor_signals_train[regressor_signals_train < -1] = 0.0

    # Linear regressor
    trend = torch.arange(
        0, regressor_signals_train.shape[-2], device=camera_sequence_filtered[0].device
    ) / float(regressor_signals_train.shape[-2] - 1)
    trend -= trend.mean()
    trend = trend.unsqueeze(0).unsqueeze(0)
    trend = trend.tile(
        (regressor_signals_train.shape[0], regressor_signals_train.shape[1], 1)
    )
    regressor_signals_train[..., -1] = trend

    regressor_signals_train = regressor_signals_train[:, :, first_none_ramp_frame:, :]

    # --- Perform ---

    regressor_signals_perform: torch.Tensor = torch.zeros(
        (
            camera_sequence[0].shape[0],
            camera_sequence[0].shape[1],
            camera_sequence[0].shape[2],
            len(regressor_camera_ids) + 1,
        ),
        device=camera_sequence[0].device,
        dtype=camera_sequence[0].dtype,
    )

    # Copy the regressor signals -1
    for matrix_id, id in enumerate(regressor_camera_ids):
        regressor_signals_perform[..., matrix_id] = camera_sequence[id] - 1.0

    # Linear regressor
    trend = torch.arange(
        0, regressor_signals_perform.shape[-2], device=camera_sequence[0].device
    ) / float(regressor_signals_perform.shape[-2] - 1)
    trend -= trend.mean()
    trend = trend.unsqueeze(0).unsqueeze(0)
    trend = trend.tile(
        (regressor_signals_perform.shape[0], regressor_signals_perform.shape[1], 1)
    )
    regressor_signals_perform[..., -1] = trend

    # --==DONE==-

    coefficients, intercept = regression_internal(
        input_regressor=regressor_signals_train, input_target=target_signals_train
    )

    target_signals_perform -= (
        regressor_signals_perform * coefficients.unsqueeze(-2)
    ).sum(dim=-1)

    target_signals_perform -= intercept.unsqueeze(-1)

    target_signals_perform[
        ~mask.unsqueeze(-1).tile((1, 1, target_signals_perform.shape[-1]))
    ] = 0.0

    target_signals_perform += 1.0

    return target_signals_perform
