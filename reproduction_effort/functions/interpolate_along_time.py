import torch


def interpolate_along_time(camera_sequence: list[torch.Tensor]) -> None:
    camera_sequence[2][:, :, 1:] = (
        camera_sequence[2][:, :, 1:] + camera_sequence[2][:, :, :-1]
    ) / 2.0

    camera_sequence[3][:, :, 1:] = (
        camera_sequence[3][:, :, 1:] + camera_sequence[3][:, :, :-1]
    ) / 2.0
