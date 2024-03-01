import os
import torch
import numpy as np
import argh

from functions.get_experiments import get_experiments
from functions.get_trials import get_trials
from functions.bandpass import bandpass
from functions.create_logger import create_logger
from functions.get_torch_device import get_torch_device
from functions.load_config import load_config
from functions.data_raw_loader import data_raw_loader


def main(*, config_filename: str = "config.json") -> None:
    mylogger = create_logger(
        save_logging_messages=True,
        display_logging_messages=True,
        log_stage_name="stage_1",
    )

    config = load_config(mylogger=mylogger, filename=config_filename)

    if config["binning_enable"] and (config["binning_at_the_end"] is False):
        device: torch.device = torch.device("cpu")
    else:
        device = get_torch_device(mylogger, config["force_to_cpu"])

    raw_data_path: str = os.path.join(
        config["basic_path"],
        config["recoding_data"],
        config["mouse_identifier"],
        config["raw_path"],
    )

    mylogger.info(f"Using data path: {raw_data_path}")

    first_experiment_id: int = int(get_experiments(raw_data_path).min())
    first_trial_id: int = int(get_trials(raw_data_path, first_experiment_id).min())

    meta_channels: list[str]
    meta_mouse_markings: str
    meta_recording_date: str
    meta_stimulation_times: dict
    meta_experiment_names: dict
    meta_trial_recording_duration: float
    meta_frame_time: float
    meta_mouse: str
    data: torch.Tensor

    if config["binning_enable"] and (config["binning_at_the_end"] is False):
        force_to_cpu_memory: bool = True
    else:
        force_to_cpu_memory = False

    mylogger.info("Loading data")

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
        experiment_id=first_experiment_id,
        trial_id=first_trial_id,
        device=device,
        force_to_cpu_memory=force_to_cpu_memory,
        config=config,
    )
    mylogger.info("-==- Done -==-")

    output_path = config["ref_image_path"]
    mylogger.info(f"Create directory {output_path} in the case it does not exist")
    os.makedirs(output_path, exist_ok=True)

    mylogger.info("Reference images")
    for i in range(0, len(meta_channels)):
        temp_path: str = os.path.join(output_path, meta_channels[i] + ".npy")
        mylogger.info(f"Extract and save: {temp_path}")
        frame_id: int = data.shape[-2] // 2
        mylogger.info(f"Will use frame id: {frame_id}")
        ref_image: np.ndarray = (
            data[:, :, frame_id, meta_channels.index(meta_channels[i])]
            .clone()
            .cpu()
            .numpy()
        )
        np.save(temp_path, ref_image)
    mylogger.info("-==- Done -==-")

    sample_frequency: float = 1.0 / meta_frame_time
    mylogger.info(
        (
            f"Heartbeat power {config['lower_freqency_bandpass']} Hz"
            f" - {config['upper_freqency_bandpass']} Hz,"
            f" sample-rate: {sample_frequency},"
            f" skipping the first {config['skip_frames_in_the_beginning']} frames"
        )
    )

    for i in range(0, len(meta_channels)):
        temp_path = os.path.join(output_path, meta_channels[i] + "_var.npy")
        mylogger.info(f"Extract and save: {temp_path}")

        heartbeat_ts: torch.Tensor = bandpass(
            data=data[..., i],
            low_frequency=config["lower_freqency_bandpass"],
            high_frequency=config["upper_freqency_bandpass"],
            fs=sample_frequency,
            filtfilt_chuck_size=10,
        )

        heartbeat_power = heartbeat_ts[
            ..., config["skip_frames_in_the_beginning"] :
        ].var(dim=-1)
        np.save(temp_path, heartbeat_power)

    mylogger.info("-==- Done -==-")


if __name__ == "__main__":
    argh.dispatch_command(main)
