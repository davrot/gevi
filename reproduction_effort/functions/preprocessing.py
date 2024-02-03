import scipy.io as sio  # type: ignore
import torch
import numpy as np
import json

from functions.make_mask import make_mask
from functions.convert_camera_sequenc_to_list import convert_camera_sequenc_to_list
from functions.preprocess_camera_sequence import preprocess_camera_sequence
from functions.interpolate_along_time import interpolate_along_time
from functions.gauss_smear import gauss_smear
from functions.regression import regression


@torch.no_grad()
def preprocessing(
    filename_metadata: str,
    filename_data: str,
    filename_mask: str,
    device: torch.device,
    first_none_ramp_frame: int,
    spatial_width: float,
    temporal_width: float,
    target_camera: list[str],
    regressor_cameras: list[str],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    data: torch.Tensor = torch.tensor(
        sio.loadmat(filename_data)["data"].astype(np.float32),
        device=device,
        dtype=dtype,
    )

    with open(filename_metadata, "r") as file_handle:
        metadata: dict = json.load(file_handle)
    cameras: list[str] = metadata["channelKey"]

    required_order: list[str] = ["acceptor", "donor", "oxygenation", "volume"]

    mask: torch.Tensor = make_mask(
        filename_mask=filename_mask, data=data, device=device, dtype=dtype
    )

    camera_sequence: list[torch.Tensor] = convert_camera_sequenc_to_list(
        data=data, required_order=required_order, cameras=cameras
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
    interpolate_along_time(camera_sequence)

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

    return results[0], results[1], mask
