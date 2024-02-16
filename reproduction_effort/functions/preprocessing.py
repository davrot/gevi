import torch


from functions.make_mask import make_mask
from functions.heart_beat_frequency import heart_beat_frequency
from functions.adjust_factor import adjust_factor
from functions.preprocess_camera_sequence import preprocess_camera_sequence
from functions.interpolate_along_time import interpolate_along_time
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
    lower_frequency_heartbeat: float,
    upper_frequency_heartbeat: float,
    sample_frequency: float,
    dtype: torch.dtype = torch.float32,
    power_factors: None | list[float] = None,
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

    # Interpolate in-between images
    if power_factors is None:
        interpolate_along_time(camera_sequence)

    camera_sequence_filtered: list[torch.Tensor] = []
    for id in range(0, len(camera_sequence)):
        camera_sequence_filtered.append(camera_sequence[id].clone())

    if power_factors is None:
        idx_volume: int = cameras.index("volume")
        heart_rate: None | float = heart_beat_frequency(
            input=camera_sequence_filtered[idx_volume],
            lower_frequency_heartbeat=lower_frequency_heartbeat,
            upper_frequency_heartbeat=upper_frequency_heartbeat,
            sample_frequency=sample_frequency,
            mask=mask,
        )
    else:
        heart_rate = None

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

    if heart_rate is not None:
        lower_frequency_heartbeat_selection: float = heart_rate - 3
        upper_frequency_heartbeat_selection: float = heart_rate + 3
    else:
        lower_frequency_heartbeat_selection = 0
        upper_frequency_heartbeat_selection = 0

    donor_correction_factor, acceptor_correction_factor = adjust_factor(
        input_acceptor=results[0],
        input_donor=results[1],
        lower_frequency_heartbeat=lower_frequency_heartbeat_selection,
        upper_frequency_heartbeat=upper_frequency_heartbeat_selection,
        sample_frequency=sample_frequency,
        mask=mask,
        power_factors=power_factors,
    )

    results[0] = acceptor_correction_factor * (
        results[0] - results[0].mean(dim=-1, keepdim=True)
    ) + results[0].mean(dim=-1, keepdim=True)

    results[1] = donor_correction_factor * (
        results[1] - results[1].mean(dim=-1, keepdim=True)
    ) + results[1].mean(dim=-1, keepdim=True)

    return results[0], results[1], mask
