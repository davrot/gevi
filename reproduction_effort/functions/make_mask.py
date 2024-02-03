import scipy.io as sio  # type: ignore

import torch


@torch.no_grad()
def make_mask(
    filename_mask: str, data: torch.Tensor, device: torch.device, dtype: torch.dtype
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

    if torch.any(data.flatten() >= limit):
        mask = mask & ~(torch.any(torch.any(data >= limit, dim=-1), dim=-1))

    return mask
