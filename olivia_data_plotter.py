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


def main(
    *,
    experiment_id: int = 4,
    config_filename: str = "config.json",
    highpass_freqency: float = 0.5,
    lowpass_freqency: float = 10.0,
    butter_worth_order: int = 4,
    log_stage_name: str = "olivia",
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

    max_value = max(
        [
            control[config["skip_frames_in_the_beginning"] :].max(),
            s_darken[config["skip_frames_in_the_beginning"] :].max(),
        ]
    )
    min_value = min(
        [
            control[config["skip_frames_in_the_beginning"] :].min(),
            s_darken[config["skip_frames_in_the_beginning"] :].min(),
        ]
    )

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

    idx = config["required_order"].index("acceptor")
    acceptor = data[..., idx].mean(axis=0).mean(axis=0)
    acceptor -= acceptor[config["skip_frames_in_the_beginning"] :].min()
    acceptor /= acceptor[config["skip_frames_in_the_beginning"] :].max()

    acceptor_f0 = acceptor.clone()
    acceptor_f0 *= max_value - min_value
    acceptor_f0 += min_value

    b, a = scipy.signal.butter(
        butter_worth_order,
        lowpass_freqency,
        btype="low",
        output="ba",
        fs=1.0 / meta_frame_time,
    )
    control_f1 = scipy.signal.filtfilt(b, a, control)
    s_darken_f1 = scipy.signal.filtfilt(b, a, s_darken)

    b, a = scipy.signal.butter(
        butter_worth_order,
        highpass_freqency,
        btype="high",
        output="ba",
        fs=1.0 / meta_frame_time,
    )
    control_f1 = scipy.signal.filtfilt(b, a, control_f1)
    s_darken_f1 = scipy.signal.filtfilt(b, a, s_darken_f1)

    max_value = max(
        [
            control_f1[config["skip_frames_in_the_beginning"] :].max(),
            s_darken_f1[config["skip_frames_in_the_beginning"] :].max(),
        ]
    )
    min_value = min(
        [
            control_f1[config["skip_frames_in_the_beginning"] :].min(),
            s_darken_f1[config["skip_frames_in_the_beginning"] :].min(),
        ]
    )

    acceptor_f1 = acceptor.clone()
    acceptor_f1 *= max_value - min_value
    acceptor_f1 += min_value

    t = np.arange(0, control.shape[0]) / 100.0

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        acceptor_f0[config["skip_frames_in_the_beginning"] :],
        color=(0.5, 0.5, 0.5),
        label="light (acceptor)",
    )

    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        control[config["skip_frames_in_the_beginning"] :],
        label="control",
    )
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        s_darken[config["skip_frames_in_the_beginning"] :],
        label="sDarken",
    )
    plt.title(
        f"Experiment {experiment_id} {config['recoding_data']} {config['mouse_identifier']}"
    )

    plt.legend()
    plt.xlabel("Time [sec]")

    plt.subplot(2, 1, 2)

    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        acceptor_f1[config["skip_frames_in_the_beginning"] :],
        color=(0.5, 0.5, 0.5),
        label="light (acceptor)",
    )

    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        control_f1[config["skip_frames_in_the_beginning"] :],
        label=f"control ({highpass_freqency}Hz - {lowpass_freqency}Hz)",
    )
    plt.plot(
        t[config["skip_frames_in_the_beginning"] :],
        s_darken_f1[config["skip_frames_in_the_beginning"] :],
        label=f"sDarken ({highpass_freqency}Hz - {lowpass_freqency}Hz)",
    )

    plt.legend()
    plt.xlabel("Time [sec]")
    plt.savefig(
        f"olivia_both_Exp{experiment_id}_{config['recoding_data']}_{config['mouse_identifier']}.png",
        dpi=300,
    )
    if plot_show:
        plt.show()


if __name__ == "__main__":
    argh.dispatch_command(main)
