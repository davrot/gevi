import numpy as np
import os
import json
from jsmin import jsmin  # type:ignore
import argh
from functions.get_trials import get_trials
from functions.get_experiments import get_experiments
import scipy  # type: ignore


def func_pow(x, a, b, c):
    return -a * x**b + c


def func_exp(x, a, b, c):
    return a * np.exp(-x / b) + c


def loader(
    filename: str = "config_M_Sert_Cre_49.json",
    fpath: str = "/data_1/hendrik/gevi",
    skip_timesteps: int = 100,
    # If there is no special ROI... Get one! This is just a backup
    roi_control_path_default: str = "roi_controlM_Sert_Cre_49.npy",
    roi_sdarken_path_default: str = "roi_sdarkenM_Sert_Cre_49.npy",
    remove_fit: bool = True,
    fit_power: bool = False,  # True => -ax^b ; False => exp(-b)
) -> None:

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

    if remove_fit:
        roi_control_path: str = f"roi_control{config["mouse_identifier"]}.npy"
        roi_sdarken_path: str = f"roi_sdarken{config["mouse_identifier"]}.npy"

        if os.path.isfile(roi_control_path) is False:
            print(f"Using replacement RIO: {roi_control_path_default}")
            roi_control_path = roi_control_path_default

        if os.path.isfile(roi_sdarken_path) is False:
            print(f"Using replacement RIO: {roi_sdarken_path_default}")
            roi_sdarken_path = roi_sdarken_path_default

        roi_control: np.ndarray = np.load(roi_control_path)
        roi_darken: np.ndarray = np.load(roi_sdarken_path)

    experiments = get_experiments(raw_data_path).numpy()
    n_exp = experiments.shape[0]

    first_run: bool = True

    for i_exp in range(0, n_exp):
        trials = get_trials(raw_data_path, experiments[i_exp]).numpy()
        n_tri = trials.shape[0]

        for i_tri in range(0, n_tri):

            experiment_name: str = (
                f"Exp{experiments[i_exp]:03d}_Trial{trials[i_tri]:03d}"
            )
            tmp_fname = os.path.join(
                fpath,
                "output_" + config["mouse_identifier"],
                experiment_name + "_acceptor_donor.npz",
            )
            print(f'Processing file "{tmp_fname}"...')
            tmp = np.load(tmp_fname)

            tmp_data_sequence = tmp["data_donor"]
            tmp_data_sequence = tmp_data_sequence[:, :, skip_timesteps:]
            tmp_light_signal = tmp["data_acceptor"]
            tmp_light_signal = tmp_light_signal[:, :, skip_timesteps:]

            if first_run:
                mask = tmp["mask"]
                new_shape = [n_exp, *tmp_data_sequence.shape]
                data_sequence = np.zeros(new_shape)
                light_signal = np.zeros(new_shape)
                first_run = False

                if remove_fit:
                    roi_control *= mask
                    assert roi_control.sum() > 0, "ROI control empty"
                    roi_darken *= mask
                    assert roi_darken.sum() > 0, "ROI sDarken empty"

            if remove_fit:
                combined_matrix = (roi_darken + roi_control) > 0
                idx = np.where(combined_matrix)
                for idx_pos in range(0, idx[0].shape[0]):

                    temp = tmp_data_sequence[idx[0][idx_pos], idx[1][idx_pos], :]
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
                    tmp_data_sequence[idx[0][idx_pos], idx[1][idx_pos], :] = temp

            data_sequence[i_exp] += tmp_data_sequence
            light_signal[i_exp] += tmp_light_signal
        data_sequence[i_exp] /= n_tri
        light_signal[i_exp] /= n_tri
    np.save("dsq_" + config["mouse_identifier"], data_sequence)
    np.save("lsq_" + config["mouse_identifier"], light_signal)
    np.save("msq_" + config["mouse_identifier"], mask)


if __name__ == "__main__":
    argh.dispatch_command(loader)
