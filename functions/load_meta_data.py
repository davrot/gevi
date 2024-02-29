import logging
import json


def load_meta_data(
    mylogger: logging.Logger, filename_meta: str, silent_mode=False
) -> tuple[list[str], str, str, dict, dict, float, float, str]:

    if silent_mode is False:
        mylogger.info("Loading meta data")
    with open(filename_meta, "r") as file_handle:
        metadata: dict = json.load(file_handle)

    channels: list[str] = metadata["channelKey"]

    if silent_mode is False:
        mylogger.info(f"meta data: channel order: {channels}")

    if "mouseMarkings" in metadata["sessionMetaData"]:
        mouse_markings: str = metadata["sessionMetaData"]["mouseMarkings"]
        if silent_mode is False:
            mylogger.info(f"meta data: mouse markings: {mouse_markings}")
    else:
        mouse_markings = ""
        if silent_mode is False:
            mylogger.info("meta data: no mouse markings")

    recording_date: str = metadata["sessionMetaData"]["date"]
    if silent_mode is False:
        mylogger.info(f"meta data: recording data: {recording_date}")

    stimulation_times: dict = metadata["sessionMetaData"]["stimulationTimes"]
    if silent_mode is False:
        mylogger.info(f"meta data: stimulation times: {stimulation_times}")

    experiment_names: dict = metadata["sessionMetaData"]["experimentNames"]
    if silent_mode is False:
        mylogger.info(f"meta data: experiment names: {experiment_names}")

    trial_recording_duration: float = float(
        metadata["sessionMetaData"]["trialRecordingDuration"]
    )
    if silent_mode is False:
        mylogger.info(
            f"meta data: trial recording duration: {trial_recording_duration} sec"
        )

    frame_time: float = float(metadata["sessionMetaData"]["frameTime"])
    if silent_mode is False:
        mylogger.info(
            f"meta data: frame time: {frame_time} sec ; frame rate: {1.0/frame_time}Hz"
        )

    mouse: str = metadata["sessionMetaData"]["mouse"]
    if silent_mode is False:
        mylogger.info(f"meta data: mouse: {mouse}")
        mylogger.info("-==- Done -==-")

    return (
        channels,
        mouse_markings,
        recording_date,
        stimulation_times,
        experiment_names,
        trial_recording_duration,
        frame_time,
        mouse,
    )
