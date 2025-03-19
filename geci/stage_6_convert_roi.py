import json
import os
import argh
from jsmin import jsmin  # type:ignore
import numpy as np
import h5py


def converter(filename: str = "config_M_Sert_Cre_49.json") -> None:

    if os.path.isfile(filename) is False:
        print(f"{filename} is missing")
        exit()

    with open(filename, "r") as file:
        config = json.loads(jsmin(file.read()))

    raw_data_path: str = os.path.join(
        config["basic_path"],
        config["recoding_data"],
        config["mouse_identifier"],
        config["raw_path"],
    )

    if os.path.isdir(raw_data_path) is False:
        print(f"ERROR: could not find raw directory {raw_data_path}!!!!")
        exit()

    roi_path: str = os.path.join(
        config["basic_path"], config["recoding_data"], config["mouse_identifier"]
    )
    roi_control_mat: str = os.path.join(roi_path, "ROI_control.mat")
    roi_sdarken_mat: str = os.path.join(roi_path, "ROI_sDarken.mat")

    if os.path.isfile(roi_control_mat):
        hf = h5py.File(roi_control_mat, "r")
        roi_control = np.array(hf["roi"]).T
        filename_out: str = f"roi_control{config['mouse_identifier']}.npy"
        np.save(filename_out, roi_control)
    else:
        print("ROI Control not found")

    if os.path.isfile(roi_sdarken_mat):
        hf = h5py.File(roi_sdarken_mat, "r")
        roi_darken = np.array(hf["roi"]).T
        filename_out: str = f"roi_sdarken{config['mouse_identifier']}.npy"
        np.save(filename_out, roi_darken)
    else:
        print("ROI sDarken not found")


if __name__ == "__main__":
    argh.dispatch_command(converter)
