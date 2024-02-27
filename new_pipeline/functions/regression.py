import torch
import logging
from functions.regression_internal import regression_internal


@torch.no_grad()
def regression(
    mylogger: logging.Logger,
    target_camera_id: int,
    regressor_camera_ids: list[int],
    mask: torch.Tensor,
    data: torch.Tensor,
    data_filtered: torch.Tensor,
    first_none_ramp_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert len(regressor_camera_ids) > 0

    mylogger.info("Prepare the target signal - 1.0 (from data_filtered)")
    target_signals_train: torch.Tensor = (
        data_filtered[target_camera_id, ..., first_none_ramp_frame:].clone() - 1.0
    )
    target_signals_train[target_signals_train < -1] = 0.0

    # Check if everything is happy
    assert target_signals_train.ndim == 3
    assert target_signals_train.ndim == data[target_camera_id, ...].ndim
    assert target_signals_train.shape[0] == data[target_camera_id, ...].shape[0]
    assert target_signals_train.shape[1] == data[target_camera_id, ...].shape[1]
    assert (target_signals_train.shape[2] + first_none_ramp_frame) == data[
        target_camera_id, ...
    ].shape[2]

    mylogger.info("Prepare the regressor signals (linear plus from data_filtered)")

    regressor_signals_train: torch.Tensor = torch.zeros(
        (
            data_filtered.shape[1],
            data_filtered.shape[2],
            data_filtered.shape[3],
            len(regressor_camera_ids) + 1,
        ),
        device=data_filtered.device,
        dtype=data_filtered.dtype,
    )

    mylogger.info("Copy the regressor signals - 1.0")
    for matrix_id, id in enumerate(regressor_camera_ids):
        regressor_signals_train[..., matrix_id] = data_filtered[id, ...] - 1.0

    regressor_signals_train[regressor_signals_train < -1] = 0.0

    mylogger.info("Create the linear regressor")
    trend = torch.arange(
        0, regressor_signals_train.shape[-2], device=data_filtered.device
    ) / float(regressor_signals_train.shape[-2] - 1)
    trend -= trend.mean()
    trend = trend.unsqueeze(0).unsqueeze(0)
    trend = trend.tile(
        (regressor_signals_train.shape[0], regressor_signals_train.shape[1], 1)
    )
    regressor_signals_train[..., -1] = trend

    regressor_signals_train = regressor_signals_train[:, :, first_none_ramp_frame:, :]

    mylogger.info("Calculating the regression coefficients")
    coefficients, intercept = regression_internal(
        input_regressor=regressor_signals_train, input_target=target_signals_train
    )
    del regressor_signals_train
    del target_signals_train

    mylogger.info("Prepare the target signal - 1.0 (from data)")
    target_signals_perform: torch.Tensor = data[target_camera_id, ...].clone() - 1.0

    mylogger.info("Prepare the regressor signals (linear plus from data)")
    regressor_signals_perform: torch.Tensor = torch.zeros(
        (
            data.shape[1],
            data.shape[2],
            data.shape[3],
            len(regressor_camera_ids) + 1,
        ),
        device=data.device,
        dtype=data.dtype,
    )

    mylogger.info("Copy the regressor signals - 1.0 ")
    for matrix_id, id in enumerate(regressor_camera_ids):
        regressor_signals_perform[..., matrix_id] = data[id] - 1.0

    mylogger.info("Create the linear regressor")
    trend = torch.arange(
        0, regressor_signals_perform.shape[-2], device=data[0].device
    ) / float(regressor_signals_perform.shape[-2] - 1)
    trend -= trend.mean()
    trend = trend.unsqueeze(0).unsqueeze(0)
    trend = trend.tile(
        (regressor_signals_perform.shape[0], regressor_signals_perform.shape[1], 1)
    )
    regressor_signals_perform[..., -1] = trend

    mylogger.info("Remove regressors")
    target_signals_perform -= (
        regressor_signals_perform * coefficients.unsqueeze(-2)
    ).sum(dim=-1)

    mylogger.info("Remove offset")
    target_signals_perform -= intercept.unsqueeze(-1)

    mylogger.info("Remove masked pixels")
    target_signals_perform[mask, :] = 0.0

    mylogger.info("Add an offset of 1.0")
    target_signals_perform += 1.0

    return target_signals_perform, coefficients
