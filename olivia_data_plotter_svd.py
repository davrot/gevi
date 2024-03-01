import numpy as np
import matplotlib.pyplot as plt
import os
from functions.create_logger import create_logger
from functions.load_config import load_config

from functions.get_trials import get_trials
import h5py  # type: ignore
import torch

control_file_handle = h5py.File("ROI_control_49.mat", "r")
control_roi = (np.array(control_file_handle["roi"]).T) > 0
control_file_handle.close()
control_roi = control_roi.reshape((control_roi.shape[0] * control_roi.shape[1]))

s_darken_file_handle = h5py.File("ROI_sDarken_49.mat", "r")
s_darken_roi = (np.array(s_darken_file_handle["roi"]).T) > 0
s_darken_file_handle.close()
s_darken_roi = s_darken_roi.reshape((s_darken_roi.shape[0] * s_darken_roi.shape[1]))

mylogger = create_logger(
    save_logging_messages=True, display_logging_messages=True, log_stage_name="test_xxx"
)
config = load_config(mylogger=mylogger)

experiment_id: int = 2

raw_data_path: str = os.path.join(
    config["basic_path"],
    config["recoding_data"],
    config["mouse_identifier"],
    config["raw_path"],
)

data_path: str = "output"

trails = get_trials(path=raw_data_path, experiment_id=experiment_id)
for i in range(0, trails.shape[0]):
    trial_id = int(trails[i])
    experiment_name: str = f"Exp{experiment_id:03d}_Trial{trial_id:03d}"
    mylogger.info(f"Loading files for {experiment_name}")

    data = np.load(os.path.join(data_path, f"{experiment_name}_ratio_sequence.npz"))
    rs = data["ratio_sequence"]
    rs = rs.reshape((rs.shape[0] * rs.shape[1], rs.shape[2]))
    rs_c = rs[control_roi, :]
    rs_c_core, _, _ = torch.linalg.svd(torch.tensor(rs_c.T), full_matrices=False)
    rs_c_core = rs_c_core[:, 0].numpy()

    rs_s = rs[s_darken_roi, :]
    rs_s_core, _, _ = torch.linalg.svd(torch.tensor(rs_s.T), full_matrices=False)
    rs_s_core = rs_s_core[:, 0].numpy()

    rs_s_core -= rs_s_core.mean(keepdims=True)
    rs_c_core -= rs_c_core.mean(keepdims=True)

    rs_c_core *= (rs_s_core * rs_c_core).sum() / (rs_c_core**2).sum()

    if i == 0:
        ratio_sequence = rs_s_core - rs_c_core
    else:
        ratio_sequence += rs_s_core - rs_c_core

ratio_sequence /= float(trails.shape[0])

t = np.arange(0, ratio_sequence.shape[0]) / 100.0

plt.plot(t, ratio_sequence, label="sDarken-control")
plt.legend()
plt.xlabel("Time [sec]")
plt.show()
