import numpy as np
import matplotlib.pyplot as plt
import os
from functions.create_logger import create_logger
from functions.load_config import load_config

from functions.get_trials import get_trials
import h5py  # type: ignore

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

experiment_id: int = 1

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
    if i == 0:
        ratio_sequence = data["ratio_sequence"]
    else:
        ratio_sequence += data["ratio_sequence"]

ratio_sequence /= float(trails.shape[0])

ratio_sequence = ratio_sequence.reshape(
    (ratio_sequence.shape[0] * ratio_sequence.shape[1], ratio_sequence.shape[2])
)

control = ratio_sequence[control_roi, :].mean(axis=0)
s_darken = ratio_sequence[s_darken_roi, :].mean(axis=0)

t = np.arange(0, control.shape[0]) / 100.0

plt.plot(t, control, label="control")
plt.plot(t, s_darken, label="sDarken")
plt.legend()
plt.xlabel("Time [sec]")
plt.show()
