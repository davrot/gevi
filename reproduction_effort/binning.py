import numpy as np
import torch
import os
import scipy.io as sio  # type: ignore

from functions.binning import binning

filename_raw: str = f"raw{os.sep}Exp001_Trial001_Part001.npy"
filename_old_mat: str = "Exp001_Trial001_Part001.mat"

data = torch.tensor(np.load(filename_raw).astype(np.float32))

data = binning(data)

mat_data = torch.tensor(sio.loadmat(filename_old_mat)["nparray"].astype(np.float32))

diff = torch.abs(mat_data - data)
print(diff.min(), diff.max())
