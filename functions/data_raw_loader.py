import numpy as np
import torch
import os
import logging
import copy

from functions.get_experiments import get_experiments
from functions.get_trials import get_trials
from functions.get_parts import get_parts
from functions.load_meta_data import load_meta_data


def data_raw_loader(
    raw_data_path: str,
    mylogger: logging.Logger,
    experiment_id: int,
    trial_id: int,
    device: torch.device,
    force_to_cpu_memory: bool,
    config: dict,
) -> tuple[list[str], str, str, dict, dict, float, float, str, torch.Tensor]:

    meta_channels: list[str] = []
    meta_mouse_markings: str = ""
    meta_recording_date: str = ""
    meta_stimulation_times: dict = {}
    meta_experiment_names: dict = {}
    meta_trial_recording_duration: float = 0.0
    meta_frame_time: float = 0.0
    meta_mouse: str = ""
    data: torch.Tensor = torch.zeros((1))

    dtype_str = config["dtype"]
    mylogger.info(f"Data precision will be {dtype_str}")
    dtype: torch.dtype = getattr(torch, dtype_str)
    dtype_np: np.dtype = getattr(np, dtype_str)

    if os.path.isdir(raw_data_path) is False:
        mylogger.info(f"ERROR: could not find raw directory {raw_data_path}!!!!")
        assert os.path.isdir(raw_data_path)
        return (
            meta_channels,
            meta_mouse_markings,
            meta_recording_date,
            meta_stimulation_times,
            meta_experiment_names,
            meta_trial_recording_duration,
            meta_frame_time,
            meta_mouse,
            data,
        )

    if (torch.where(get_experiments(raw_data_path) == experiment_id)[0].shape[0]) != 1:
        mylogger.info(f"ERROR: could not find experiment id {experiment_id}!!!!")
        assert (
            torch.where(get_experiments(raw_data_path) == experiment_id)[0].shape[0]
        ) == 1
        return (
            meta_channels,
            meta_mouse_markings,
            meta_recording_date,
            meta_stimulation_times,
            meta_experiment_names,
            meta_trial_recording_duration,
            meta_frame_time,
            meta_mouse,
            data,
        )

    if (
        torch.where(get_trials(raw_data_path, experiment_id) == trial_id)[0].shape[0]
    ) != 1:
        mylogger.info(f"ERROR: could not find trial id {trial_id}!!!!")
        assert (
            torch.where(get_trials(raw_data_path, experiment_id) == trial_id)[0].shape[
                0
            ]
        ) == 1
        return (
            meta_channels,
            meta_mouse_markings,
            meta_recording_date,
            meta_stimulation_times,
            meta_experiment_names,
            meta_trial_recording_duration,
            meta_frame_time,
            meta_mouse,
            data,
        )

    available_parts: torch.Tensor = get_parts(raw_data_path, experiment_id, trial_id)
    if available_parts.shape[0] < 1:
        mylogger.info("ERROR: could not find any part files")
        assert available_parts.shape[0] >= 1

    experiment_name = f"Exp{experiment_id:03d}_Trial{trial_id:03d}"
    mylogger.info(f"Will work on: {experiment_name}")

    mylogger.info(f"We found {int(available_parts.shape[0])} parts.")

    first_run: bool = True

    mylogger.info("Compare meta data of all parts")
    for id in range(0, available_parts.shape[0]):
        part_id = available_parts[id]

        filename_meta: str = os.path.join(
            raw_data_path,
            f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}_meta.txt",
        )

        if os.path.isfile(filename_meta) is False:
            mylogger.info(f"Could not load meta data... {filename_meta}")
            assert os.path.isfile(filename_meta)
            return (
                meta_channels,
                meta_mouse_markings,
                meta_recording_date,
                meta_stimulation_times,
                meta_experiment_names,
                meta_trial_recording_duration,
                meta_frame_time,
                meta_mouse,
                data,
            )

        (
            meta_channels,
            meta_mouse_markings,
            meta_recording_date,
            meta_stimulation_times,
            meta_experiment_names,
            meta_trial_recording_duration,
            meta_frame_time,
            meta_mouse,
        ) = load_meta_data(
            mylogger=mylogger, filename_meta=filename_meta, silent_mode=True
        )

        if first_run:
            first_run = False
            master_meta_channels: list[str] = copy.deepcopy(meta_channels)
            master_meta_mouse_markings: str = meta_mouse_markings
            master_meta_recording_date: str = meta_recording_date
            master_meta_stimulation_times: dict = copy.deepcopy(meta_stimulation_times)
            master_meta_experiment_names: dict = copy.deepcopy(meta_experiment_names)
            master_meta_trial_recording_duration: float = meta_trial_recording_duration
            master_meta_frame_time: float = meta_frame_time
            master_meta_mouse: str = meta_mouse

        meta_channels_check = master_meta_channels == meta_channels

        # Check channel order
        if meta_channels_check:
            for channel_a, channel_b in zip(master_meta_channels, meta_channels):
                if channel_a != channel_b:
                    meta_channels_check = False

        meta_mouse_markings_check = master_meta_mouse_markings == meta_mouse_markings
        meta_recording_date_check = master_meta_recording_date == meta_recording_date
        meta_stimulation_times_check = (
            master_meta_stimulation_times == meta_stimulation_times
        )
        meta_experiment_names_check = (
            master_meta_experiment_names == meta_experiment_names
        )
        meta_trial_recording_duration_check = (
            master_meta_trial_recording_duration == meta_trial_recording_duration
        )
        meta_frame_time_check = master_meta_frame_time == meta_frame_time
        meta_mouse_check = master_meta_mouse == meta_mouse

        if meta_channels_check is False:
            mylogger.info(f"{filename_meta} failed: channels")
            assert meta_channels_check

        if meta_mouse_markings_check is False:
            mylogger.info(f"{filename_meta} failed: mouse_markings")
            assert meta_mouse_markings_check

        if meta_recording_date_check is False:
            mylogger.info(f"{filename_meta} failed: recording_date")
            assert meta_recording_date_check

        if meta_stimulation_times_check is False:
            mylogger.info(f"{filename_meta} failed: stimulation_times")
            assert meta_stimulation_times_check

        if meta_experiment_names_check is False:
            mylogger.info(f"{filename_meta} failed: experiment_names")
            assert meta_experiment_names_check

        if meta_trial_recording_duration_check is False:
            mylogger.info(f"{filename_meta} failed: trial_recording_duration")
            assert meta_trial_recording_duration_check

        if meta_frame_time_check is False:
            mylogger.info(f"{filename_meta} failed: frame_time_check")
            assert meta_frame_time_check

        if meta_mouse_check is False:
            mylogger.info(f"{filename_meta} failed: mouse")
            assert meta_mouse_check
    mylogger.info("-==- Done -==-")

    mylogger.info(f"Will use: {filename_meta} for meta data")
    (
        meta_channels,
        meta_mouse_markings,
        meta_recording_date,
        meta_stimulation_times,
        meta_experiment_names,
        meta_trial_recording_duration,
        meta_frame_time,
        meta_mouse,
    ) = load_meta_data(mylogger=mylogger, filename_meta=filename_meta)

    #################
    # Meta data end #
    #################

    first_run = True
    mylogger.info("Count the number of frames in the data of all parts")
    frame_count: int = 0
    for id in range(0, available_parts.shape[0]):
        part_id = available_parts[id]

        filename_data: str = os.path.join(
            raw_data_path,
            f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}.npy",
        )

        if os.path.isfile(filename_data) is False:
            mylogger.info(f"Could not load data... {filename_data}")
            assert os.path.isfile(filename_data)
            return (
                meta_channels,
                meta_mouse_markings,
                meta_recording_date,
                meta_stimulation_times,
                meta_experiment_names,
                meta_trial_recording_duration,
                meta_frame_time,
                meta_mouse,
                data,
            )
        data_np: np.ndarray = np.load(filename_data, mmap_mode="r")

        if data_np.ndim != 4:
            mylogger.info(f"ERROR: Data needs to have 4 dimensions {filename_data}")
            assert data_np.ndim == 4

        if first_run:
            first_run = False
            dim_0: int = int(data_np.shape[0])
            dim_1: int = int(data_np.shape[1])
            dim_3: int = int(data_np.shape[3])

        frame_count += int(data_np.shape[2])

        if int(data_np.shape[0]) != dim_0:
            mylogger.info(
                f"ERROR: Data dim 0 is broken {int(data_np.shape[0])} vs {dim_0} {filename_data}"
            )
            assert int(data_np.shape[0]) == dim_0

        if int(data_np.shape[1]) != dim_1:
            mylogger.info(
                f"ERROR: Data dim 1 is broken {int(data_np.shape[1])} vs {dim_1} {filename_data}"
            )
            assert int(data_np.shape[1]) == dim_1

        if int(data_np.shape[3]) != dim_3:
            mylogger.info(
                f"ERROR: Data dim 3 is broken {int(data_np.shape[3])} vs {dim_3} {filename_data}"
            )
            assert int(data_np.shape[3]) == dim_3

        mylogger.info(
            f"{filename_data}: {int(data_np.shape[2])} frames -> {frame_count} frames total"
        )

    if force_to_cpu_memory:
        mylogger.info("Using CPU memory for data")
        data = torch.empty(
            (dim_0, dim_1, frame_count, dim_3), dtype=dtype, device=torch.device("cpu")
        )
    else:
        mylogger.info("Using GPU memory for data")
        data = torch.empty(
            (dim_0, dim_1, frame_count, dim_3), dtype=dtype, device=device
        )

    start_position: int = 0
    end_position: int = 0
    for id in range(0, available_parts.shape[0]):
        part_id = available_parts[id]

        filename_data = os.path.join(
            raw_data_path,
            f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}.npy",
        )

        mylogger.info(f"Will work on {filename_data}")
        mylogger.info("Loading data file")
        data_np = np.load(filename_data).astype(dtype_np)

        end_position = start_position + int(data_np.shape[2])

        for i in range(0, len(config["required_order"])):
            mylogger.info(f"Move raw data channel: {config['required_order'][i]}")

            idx = meta_channels.index(config["required_order"][i])
            data[..., start_position:end_position, i] = torch.tensor(
                data_np[..., idx], dtype=dtype, device=data.device
            )
        start_position = end_position

    if start_position != int(data.shape[2]):
        mylogger.info("ERROR: data was not fulled fully!!!")
        assert start_position == int(data.shape[2])

    mylogger.info("-==- Done -==-")

    #################
    # Raw data end  #
    #################

    return (
        meta_channels,
        meta_mouse_markings,
        meta_recording_date,
        meta_stimulation_times,
        meta_experiment_names,
        meta_trial_recording_duration,
        meta_frame_time,
        meta_mouse,
        data,
    )
