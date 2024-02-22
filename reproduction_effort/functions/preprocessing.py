import torch


from functions.make_mask import make_mask
from functions.preprocess_camera_sequence import preprocess_camera_sequence
from functions.gauss_smear import gauss_smear
from functions.regression import regression


@torch.no_grad()
def preprocessing(
    cameras: list[str],
    camera_sequence: list[torch.Tensor],
    filename_mask: str,
    device: torch.device,
    first_none_ramp_frame: int,
    spatial_width: float,
    temporal_width: float,
    target_camera: list[str],
    regressor_cameras: list[str],
    donor_correction_factor: torch.Tensor,
    acceptor_correction_factor: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mask: torch.Tensor = make_mask(
        filename_mask=filename_mask,
        camera_sequence=camera_sequence,
        device=device,
        dtype=dtype,
    )

    for num_cams in range(len(camera_sequence)):
        camera_sequence[num_cams], mask = preprocess_camera_sequence(
            camera_sequence=camera_sequence[num_cams],
            mask=mask,
            first_none_ramp_frame=first_none_ramp_frame,
            device=device,
            dtype=dtype,
        )

    camera_sequence_filtered: list[torch.Tensor] = []
    for id in range(0, len(camera_sequence)):
        camera_sequence_filtered.append(camera_sequence[id].clone())

    camera_sequence_filtered = gauss_smear(
        camera_sequence_filtered,
        mask.type(dtype=dtype),
        spatial_width=spatial_width,
        temporal_width=temporal_width,
    )

    regressor_camera_ids: list[int] = []

    for cam in regressor_cameras:
        regressor_camera_ids.append(cameras.index(cam))

    results: list[torch.Tensor] = []

    for channel_position in range(0, len(target_camera)):
        print(f"channel position: {channel_position}")
        target_camera_selected = target_camera[channel_position]
        target_camera_id: int = cameras.index(target_camera_selected)

        output = regression(
            target_camera_id=target_camera_id,
            regressor_camera_ids=regressor_camera_ids,
            mask=mask,
            camera_sequence=camera_sequence,
            camera_sequence_filtered=camera_sequence_filtered,
            first_none_ramp_frame=first_none_ramp_frame,
        )
        results.append(output)

    donor_factor: torch.Tensor = (
        donor_correction_factor + acceptor_correction_factor
    ) / (2 * donor_correction_factor)
    acceptor_factor: torch.Tensor = (
        donor_correction_factor + acceptor_correction_factor
    ) / (2 * acceptor_correction_factor)

    results[0] *= acceptor_factor * mask.unsqueeze(-1)
    results[1] *= donor_factor * mask.unsqueeze(-1)

    return results[0], results[1], mask
