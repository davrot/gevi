import torch
import logging


def get_torch_device(mylogger: logging.Logger, force_to_cpu: bool) -> torch.device:

    if torch.cuda.is_available():
        device_name: str = "cuda:0"
    else:
        device_name = "cpu"

    if force_to_cpu:
        device_name = "cpu"

    mylogger.info(f"Using device: {device_name}")
    device: torch.device = torch.device(device_name)
    return device
