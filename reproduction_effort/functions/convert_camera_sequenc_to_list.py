import torch


@torch.no_grad()
def convert_camera_sequenc_to_list(
    data: torch.Tensor, required_order: list[str], cameras: list[str]
) -> list[torch.Tensor]:
    camera_sequence: list[torch.Tensor] = []

    for cam in required_order:
        camera_sequence.append(data[:, :, :, cameras.index(cam)].clone())

    return camera_sequence
