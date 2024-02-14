import scipy.io as sio  # type: ignore

import torch


@torch.no_grad()
def make_mask(
    filename_mask: str,
    camera_sequence: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask: torch.Tensor = torch.tensor(
        sio.loadmat(filename_mask)["maskInfo"]["maskIdx2D"][0][0],
        device=device,
        dtype=torch.bool,
    )
    mask = mask > 0.5

    limit: torch.Tensor = torch.tensor(
        2**16 - 1,
        device=device,
        dtype=dtype,
    )

    for id in range(0, len(camera_sequence)):
        if torch.any(camera_sequence[id].flatten() >= limit):
            mask = mask & ~(torch.any(camera_sequence[id] >= limit, dim=-1))
        if torch.any(camera_sequence[id].flatten() < 0):
            mask = mask & ~(torch.any(camera_sequence[id] < 0, dim=-1))

    return mask
