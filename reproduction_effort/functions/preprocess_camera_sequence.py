import torch


@torch.no_grad()
def preprocess_camera_sequence(
    camera_sequence: torch.Tensor,
    mask: torch.Tensor,
    first_none_ramp_frame: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:

    limit: torch.Tensor = torch.tensor(
        2**16 - 1,
        device=device,
        dtype=dtype,
    )

    camera_sequence = camera_sequence / camera_sequence[
        :, :, first_none_ramp_frame:
    ].mean(
        dim=2,
        keepdim=True,
    )

    camera_sequence = camera_sequence.nan_to_num(nan=0.0)

    camera_sequence_zero_idx = torch.any(camera_sequence == 0, dim=-1, keepdim=True)
    mask &= (~camera_sequence_zero_idx.squeeze(-1)).type(dtype=torch.bool)
    camera_sequence_zero_idx = torch.tile(
        camera_sequence_zero_idx, (1, 1, camera_sequence.shape[-1])
    )
    camera_sequence_zero_idx[:, :, :first_none_ramp_frame] = False
    camera_sequence[camera_sequence_zero_idx] = limit

    return camera_sequence, mask
