import numpy as np
import matplotlib.pyplot as plt
import os
from functions.create_logger import create_logger
from functions.load_config import load_config

from functions.get_trials import get_trials
import h5py  # type: ignore
import torch
import scipy  # type: ignore

import argh
from functions.data_raw_loader import data_raw_loader

# def func(x, a, b, c, dt):
#     return a * (x - dt) ** b + c


# def fit(
#     ratio_sequence: np.ndarray,
#     t: np.ndarray,
#     config: dict,
# ) -> tuple[np.ndarray, np.ndarray]:

#     data_min = ratio_sequence[config["skip_frames_in_the_beginning"] :].min()
#     data_max = ratio_sequence[config["skip_frames_in_the_beginning"] :].max()
#     b_min = 1.0
#     b_max = 3.0

#     temp_1 = max([abs(data_min), abs(data_max)])
#     a_min = -temp_1 - 2 * abs(data_max - data_min)
#     a_max = +temp_1 + 2 * abs(data_max - data_min)

#     try:
#         popt, _ = scipy.optimize.curve_fit(
#             f=func,
#             xdata=t[config["skip_frames_in_the_beginning"] :],
#             ydata=np.nan_to_num(
#                 ratio_sequence[config["skip_frames_in_the_beginning"] :]
#             ),
#             bounds=([a_min, b_min, a_min, -t[-1]], [a_max, b_max, a_max, t[-1]]),
#         )
#         a: float | None = float(popt[0])
#         b: float | None = float(popt[1])
#         c: float | None = float(popt[2])
#         dt: float | None = float(popt[3])

#     except ValueError:
#         a = None
#         b = None
#         c = None
#         dt = None

#     print(a, b, c, dt)

#     f1 = func(t, a, b, c, dt)
#     ratio_sequence_f1 = ratio_sequence - f1
#     return ratio_sequence_f1, f1


def main(
    *,
    experiment_id: int = 4,
    config_filename: str = "config.json",
    highpass_freqency: float = 0.5,
    lowpass_freqency: float = 10.0,
    butter_worth_order: int = 4,
    log_stage_name: str = "olivia_svd",
    scale_before_substraction: bool = True,
    plot_show: bool = True,
) -> None:

    mylogger = create_logger(
        save_logging_messages=True,
        display_logging_messages=True,
        log_stage_name=log_stage_name,
    )
    config = load_config(mylogger=mylogger, filename=config_filename)

    roi_path: str = config["ref_image_path"]

    control_file_handle = h5py.File(os.path.join(roi_path, "ROI_control.mat"), "r")
    control_roi = (np.array(control_file_handle["roi"]).T) > 0
    control_file_handle.close()
    control_roi = control_roi.reshape((control_roi.shape[0] * control_roi.shape[1]))

    s_darken_file_handle = h5py.File(os.path.join(roi_path, "ROI_sDarken.mat"), "r")
    s_darken_roi = (np.array(s_darken_file_handle["roi"]).T) > 0
    s_darken_file_handle.close()
    s_darken_roi = s_darken_roi.reshape((s_darken_roi.shape[0] * s_darken_roi.shape[1]))

    raw_data_path: str = os.path.join(
        config["basic_path"],
        config["recoding_data"],
        config["mouse_identifier"],
        config["raw_path"],
    )

    data_path: str = str(config["export_path"])

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

        rs_s_core -= rs_s_core[config["skip_frames_in_the_beginning"] :].mean(
            keepdims=True
        )
        rs_c_core -= rs_c_core[config["skip_frames_in_the_beginning"] :].mean(
            keepdims=True
        )

        if scale_before_substraction:
            rs_c_core *= (
                rs_s_core[config["skip_frames_in_the_beginning"] :]
                * rs_c_core[config["skip_frames_in_the_beginning"] :]
            ).sum() / (rs_c_core[config["skip_frames_in_the_beginning"] :] ** 2).sum()

        if i == 0:
            ratio_sequence = rs_s_core - rs_c_core
        else:
            ratio_sequence += rs_s_core - rs_c_core

    ratio_sequence /= float(trails.shape[0])

    t = np.arange(0, ratio_sequence.shape[0]) / 100.0

    # ratio_sequence_f1, f1 = fit(
    #     ratio_sequence=ratio_sequence,
    #     t=t,
    #     config=config,
    # )

    first_trial_id: int = int(get_trials(raw_data_path, experiment_id).min())
    (
        meta_channels,
        meta_mouse_markings,
        meta_recording_date,
        meta_stimulation_times,
        meta_experiment_names,
        meta_trial_recording_duration,
        meta_frame_time,
        meta_mouse,
        data,
    ) = data_raw_loader(
        raw_data_path=raw_data_path,
        mylogger=mylogger,
        experiment_id=experiment_id,
        trial_id=first_trial_id,
        device=torch.device("cpu"),
        force_to_cpu_memory=True,
        config=config,
    )

    b, a = scipy.signal.butter(
        butter_worth_order,
        lowpass_freqency,
        btype="low",
        output="ba",
        fs=1.0 / meta_frame_time,
    )
    ratio_sequence_f1 = scipy.signal.filtfilt(b, a, ratio_sequence)

    b, a = scipy.signal.butter(
        butter_worth_order,
        highpass_freqency,
        btype="high",
        output="ba",
        fs=1.0 / meta_frame_time,
    )
    ratio_sequence_f2 = scipy.signal.filtfilt(b, a, ratio_sequence_f1)

    idx = config["required_order"].index("acceptor")
    acceptor = data[..., idx].mean(axis=0).mean(axis=0)
    acceptor -= acceptor[config["skip_frames_in_the_beginning"] :].min()
    acceptor /= acceptor[config["skip_frames_in_the_beginning"] :].max()

    acceptor *= (
        ratio_sequence_f2[config["skip_frames_in_the_beginning"] :].max()
        - ratio_sequence_f2[config["skip_frames_in_the_beginning"] :].min()
    )
    acceptor += ratio_sequence_f2[config["skip_frames_in_the_beginning"] :].min()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        ratio_sequence[config["skip_frames_in_the_beginning"] :],
        label=f"sDarken - control (scaled:{scale_before_substraction})",
    )
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        ratio_sequence_f1[config["skip_frames_in_the_beginning"] :],
        label=f"low pass {lowpass_freqency} Hz",
    )
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        ratio_sequence_f2[config["skip_frames_in_the_beginning"] :],
        label=f"high pass {highpass_freqency} Hz",
    )
    plt.xlabel("Time [sec]")
    plt.title(
        f"Experiment {experiment_id} {config['recoding_data']} {config['mouse_identifier']}"
    )
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        acceptor[config["skip_frames_in_the_beginning"] :],
        color=(0.5, 0.5, 0.5),
        label="light (acceptor)",
    )
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        ratio_sequence_f2[config["skip_frames_in_the_beginning"] :],
        label=f"high pass {highpass_freqency} Hz",
    )
    plt.legend()
    plt.xlabel("Time [sec]")

    plt.savefig(
        f"olivia_Exp{experiment_id}_{config['recoding_data']}_{config['mouse_identifier']}.png",
        dpi=300,
    )
    if plot_show:
        plt.show()


if __name__ == "__main__":
    argh.dispatch_command(main)
