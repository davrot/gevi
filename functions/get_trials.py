import torch
import os
import glob


@torch.no_grad()
def get_trials(path: str, experiment_id: int) -> torch.Tensor:
    filename_np: str = os.path.join(
        path,
        f"Exp{experiment_id:03d}_Trial*_Part001.npy",
    )

    list_str = glob.glob(filename_np)
    list_int: list[int] = []
    for i in range(0, len(list_str)):
        list_int.append(int(list_str[i].split("_Trial")[-1].split("_Part")[0]))

    list_int = sorted(list_int)
    return torch.tensor(list_int).unique()
