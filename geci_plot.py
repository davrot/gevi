import numpy as np
import matplotlib.pyplot as plt
import h5py  # type: ignore
import argh
import scipy  # type: ignore
import json
import os
from jsmin import jsmin  # type:ignore
from functions.get_trials import get_trials


def func_pow(x, a, b, c):
    return -a * x**b + c


def func_exp(x, a, b, c):
    return a * np.exp(-x / b) + c


# mouse: int = 0, 1, 2, 3, 4
def plot(
    filename: str = "config_M_Sert_Cre_49.json",
    experiment: int = 4,
    skip_timesteps: int = 100,
    # If there is no special ROI... Get one! This is just a backup
    roi_control_path: str = "/data_1/hendrik/2023-03-15/ROI_control.mat",
    roi_sdraken_path: str = "/data_1/hendrik/2023-03-15/ROI_sDarken.mat",
    remove_fit: bool = True,
    fit_power: bool = False,  # True => -ax^b ; False => exp(-b)
) -> None:

    # lbs = [
    #     [
    #         "control",
    #         "visual control",
    #         "op20 50 5",
    #         "op20 100 5",
    #         "op20 50 grat",
    #         "op20 100 grat",
    #     ],
    #     [
    #         "control",
    #         "visual control",
    #         "op20 50 5",
    #         "op20 100 5",
    #         "op20 50 grat",
    #         "op20 100 grat",
    #     ],
    #     [
    #         "control",
    #         "visual control",
    #         "op20 50 5",
    #         "op20 100 5",
    #         "op20 50 grat",
    #         "op20 100 grat",
    #     ],
    #     ["control", "visual control", "op20 50 5", "op20 100 5"],
    #     ["control", "visual control", "op20 50 5", "op20 100 5"],
    # ]

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

    trials = get_trials(raw_data_path, experiment).numpy()
    assert trials.shape[0] > 0

    with open(os.path.join(raw_data_path, f"Exp{experiment:03d}_Trial{trials[0]:03d}_Part001_meta.txt"), "r") as file:
        metadata = json.loads(jsmin(file.read()))

    experiment_names = metadata['sessionMetaData']['experimentNames'][str(experiment)]

    roi_path: str = os.path.join(config["basic_path"], config["recoding_data"])
    roi_control_mat: str = os.path.join(roi_path, "ROI_control.mat")
    roi_sdarken_mat: str = os.path.join(roi_path, "ROI_sDarken.mat")

    if os.path.isdir(roi_control_mat):
        roi_control_path = roi_control_mat

    if os.path.isdir(roi_sdarken_mat):
        roi_sdraken_path = roi_sdarken_mat

    print("Load data...")
    data = np.load("dsq_" + config["mouse_identifier"] + ".npy", mmap_mode="r")

    print("Load light signal...")
    light = np.load("lsq_" + config["mouse_identifier"] + ".npy", mmap_mode="r")

    print("Load mask...")
    mask = np.load("msq_" + config["mouse_identifier"] + ".npy")

    hf = h5py.File(roi_control_path, "r")
    roi_lighten = np.array(hf["roi"]).T
    roi_lighten *= mask

    hf = h5py.File(roi_sdraken_path, "r")
    roi_darken = np.array(hf["roi"]).T
    roi_darken *= mask

    plt.figure(1)
    a_show = data[experiment - 1, :, :, 1000].copy()
    a_show[(roi_darken + roi_lighten) < 0.5] = np.nan
    plt.imshow(a_show)
    plt.title(f"{config["mouse_identifier"]} -- Experiment: {experiment}")
    plt.show(block=False)

    plt.figure(2)
    a_dontshow = data[experiment - 1, :, :, 1000].copy()
    a_dontshow[(roi_darken + roi_lighten) > 0.5] = np.nan
    plt.imshow(a_dontshow)
    plt.title(f"{config["mouse_identifier"]} -- Experiment: {experiment}")
    plt.show(block=False)

    plt.figure(3)
    light_exp = light[experiment - 1, :, :, skip_timesteps:].copy()
    light_exp[(roi_darken + roi_lighten) < 0.5, :] = 0.0
    light_signal = light_exp.mean(axis=(0, 1))
    light_signal -= light_signal.min()
    light_signal /= light_signal.max()

    a_exp = data[experiment - 1, :, :, skip_timesteps:].copy()

    if remove_fit:
        combined_matrix = (roi_darken + roi_lighten) > 0
        idx = np.where(combined_matrix)
        for idx_pos in range(0, idx[0].shape[0]):
            temp = a_exp[idx[0][idx_pos], idx[1][idx_pos], :]
            temp -= temp.mean()

            data_time = np.arange(0, temp.shape[0], dtype=np.float32) + skip_timesteps
            data_time /= 100.0

            data_min = temp.min()
            data_max = temp.max()
            data_delta = data_max - data_min
            a_min = data_min - data_delta
            b_min = 0.01
            a_max = data_max + data_delta
            if fit_power:
                b_max = 10.0
            else:
                b_max = 100.0
            c_min = data_min - data_delta
            c_max = data_max + data_delta

            try:
                if fit_power:
                    popt, _ = scipy.optimize.curve_fit(
                        f=func_pow,
                        xdata=data_time,
                        ydata=np.nan_to_num(temp),
                        bounds=([a_min, b_min, c_min], [a_max, b_max, c_max]),
                    )
                    pattern: np.ndarray | None = func_pow(data_time, *popt)
                else:
                    popt, _ = scipy.optimize.curve_fit(
                        f=func_exp,
                        xdata=data_time,
                        ydata=np.nan_to_num(temp),
                        bounds=([a_min, b_min, c_min], [a_max, b_max, c_max]),
                    )
                    pattern = func_exp(data_time, *popt)

                assert pattern is not None
                pattern -= pattern.mean()

                scale = (temp * pattern).sum() / (pattern**2).sum()
                pattern *= scale

            except ValueError:
                print(f"Fit failed: Position ({idx[0][idx_pos]}, {idx[1][idx_pos]}")
                pattern = None

            if pattern is not None:
                temp -= pattern

    darken = a_exp[roi_darken > 0.5, :].sum(axis=0) / (roi_darken > 0.5).sum()
    lighten = a_exp[roi_lighten > 0.5, :].sum(axis=0) / (roi_lighten > 0.5).sum()

    light_signal *= darken.max() - darken.min()
    light_signal += darken.min()

    time_axis = np.arange(0, lighten.shape[-1], dtype=np.float32) + skip_timesteps
    time_axis /= 100.0

    plt.plot(time_axis, light_signal, c="k", label="light")
    plt.plot(time_axis, darken, label="sDarken")
    plt.plot(time_axis, lighten, label="control")
    plt.title(f"{config["mouse_identifier"]} -- Experiment: {experiment} ({experiment_names})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    argh.dispatch_command(plot)
