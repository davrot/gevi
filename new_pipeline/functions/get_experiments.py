import torch
import os
import glob


@torch.no_grad()
def get_experiments(path: str) -> torch.Tensor:
    filename_np: str = os.path.join(
        path,
        "Exp*_Part001.npy",
    )

    list_str = glob.glob(filename_np)
    list_int: list[int] = []
    for i in range(0, len(list_str)):
        list_int.append(int(list_str[i].split("Exp")[-1].split("_Trial")[0]))
    list_int = sorted(list_int)

    return torch.tensor(list_int).unique()
