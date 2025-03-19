# %%

import numpy as np
import torch
import torchvision as tv  # type: ignore

import os
import logging

from functions.create_logger import create_logger
from functions.get_torch_device import get_torch_device
from functions.load_config import load_config
from functions.get_experiments import get_experiments
from functions.get_trials import get_trials
from functions.binning import binning
from functions.align_refref import align_refref
from functions.perform_donor_volume_rotation import perform_donor_volume_rotation
from functions.perform_donor_volume_translation import perform_donor_volume_translation
from functions.data_raw_loader import data_raw_loader

import argh


@torch.no_grad()
def process_trial(
    config: dict,
    mylogger: logging.Logger,
    experiment_id: int,
    trial_id: int,
    device: torch.device,
):

    mylogger.info("")
    mylogger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    mylogger.info("~ TRIAL START ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    mylogger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    mylogger.info("")

    if device != torch.device("cpu"):
        torch.cuda.empty_cache()
        mylogger.info("Empty CUDA cache")
        cuda_total_memory: int = torch.cuda.get_device_properties(
            device.index
        ).total_memory
    else:
        cuda_total_memory = 0

    raw_data_path: str = os.path.join(
        config["basic_path"],
        config["recoding_data"],
        config["mouse_identifier"],
        config["raw_path"],
    )

    if config["binning_enable"] and (config["binning_at_the_end"] is False):
        force_to_cpu_memory: bool = True
    else:
        force_to_cpu_memory = False

    meta_channels: list[str]
    meta_mouse_markings: str
    meta_recording_date: str
    meta_stimulation_times: dict
    meta_experiment_names: dict
    meta_trial_recording_duration: float
    meta_frame_time: float
    meta_mouse: str
    data: torch.Tensor

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
        trial_id=trial_id,
        device=device,
        force_to_cpu_memory=force_to_cpu_memory,
        config=config,
    )
    experiment_name: str = f"Exp{experiment_id:03d}_Trial{trial_id:03d}"

    dtype_str = config["dtype"]
    dtype_np: np.dtype = getattr(np, dtype_str)

    dtype: torch.dtype = data.dtype

    if device != torch.device("cpu"):
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(f"CUDA memory: {free_mem // 1024} MByte")

    mylogger.info(f"Data shape: {data.shape}")
    mylogger.info("-==- Done -==-")

    mylogger.info("Finding limit values in the RAW data and mark them for masking")
    limit: float = (2**16) - 1
    for i in range(0, data.shape[3]):
        zero_pixel_mask: torch.Tensor = torch.any(data[..., i] >= limit, dim=-1)
        data[zero_pixel_mask, :, i] = -100.0
        mylogger.info(
            f"{meta_channels[i]}: "
            f"found {int(zero_pixel_mask.type(dtype=dtype).sum())} pixel "
            f"with limit values "
        )
    mylogger.info("-==- Done -==-")

    mylogger.info("Reference images and mask")

    ref_image_path: str = config["ref_image_path"]

    ref_image_path_acceptor: str = os.path.join(ref_image_path, "acceptor.npy")
    if os.path.isfile(ref_image_path_acceptor) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_acceptor}")
        assert os.path.isfile(ref_image_path_acceptor)
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_acceptor}")
    ref_image_acceptor: torch.Tensor = torch.tensor(
        np.load(ref_image_path_acceptor).astype(dtype_np),
        dtype=dtype,
        device=data.device,
    )

    ref_image_path_donor: str = os.path.join(ref_image_path, "donor.npy")
    if os.path.isfile(ref_image_path_donor) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_donor}")
        assert os.path.isfile(ref_image_path_donor)
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_donor}")
    ref_image_donor: torch.Tensor = torch.tensor(
        np.load(ref_image_path_donor).astype(dtype_np), dtype=dtype, device=data.device
    )

    ref_image_path_oxygenation: str = os.path.join(ref_image_path, "oxygenation.npy")
    if os.path.isfile(ref_image_path_oxygenation) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_oxygenation}")
        assert os.path.isfile(ref_image_path_oxygenation)
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_oxygenation}")
    ref_image_oxygenation: torch.Tensor = torch.tensor(
        np.load(ref_image_path_oxygenation).astype(dtype_np),
        dtype=dtype,
        device=data.device,
    )

    ref_image_path_volume: str = os.path.join(ref_image_path, "volume.npy")
    if os.path.isfile(ref_image_path_volume) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_volume}")
        assert os.path.isfile(ref_image_path_volume)
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_volume}")
    ref_image_volume: torch.Tensor = torch.tensor(
        np.load(ref_image_path_volume).astype(dtype_np), dtype=dtype, device=data.device
    )

    refined_mask_file: str = os.path.join(ref_image_path, "mask_not_rotated.npy")
    if os.path.isfile(refined_mask_file) is False:
        mylogger.info(f"Could not load mask file: {refined_mask_file}")
        assert os.path.isfile(refined_mask_file)
        return

    mylogger.info(f"Loading mask file data: {refined_mask_file}")
    mask: torch.Tensor = torch.tensor(
        np.load(refined_mask_file).astype(dtype_np), dtype=dtype, device=data.device
    )
    mylogger.info("-==- Done -==-")

    if config["binning_enable"] and (config["binning_at_the_end"] is False):
        mylogger.info("Binning of data")
        mylogger.info(
            (
                f"kernel_size={int(config['binning_kernel_size'])}, "
                f"stride={int(config['binning_stride'])}, "
                f"divisor_override={int(config['binning_divisor_override'])}"
            )
        )

        data = binning(
            data,
            kernel_size=int(config["binning_kernel_size"]),
            stride=int(config["binning_stride"]),
            divisor_override=int(config["binning_divisor_override"]),
        ).to(device=data.device)
        ref_image_acceptor = (
            binning(
                ref_image_acceptor.unsqueeze(-1).unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=int(config["binning_divisor_override"]),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        ref_image_donor = (
            binning(
                ref_image_donor.unsqueeze(-1).unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=int(config["binning_divisor_override"]),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        ref_image_oxygenation = (
            binning(
                ref_image_oxygenation.unsqueeze(-1).unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=int(config["binning_divisor_override"]),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        ref_image_volume = (
            binning(
                ref_image_volume.unsqueeze(-1).unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=int(config["binning_divisor_override"]),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        mask = (
            binning(
                mask.unsqueeze(-1).unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=int(config["binning_divisor_override"]),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        mylogger.info(f"Data shape: {data.shape}")
        mylogger.info("-==- Done -==-")

    mylogger.info("Preparing alignment")
    mylogger.info("Re-order Raw data")
    data = data.moveaxis(-2, 0).moveaxis(-1, 0)
    mylogger.info(f"Data shape: {data.shape}")
    mylogger.info("-==- Done -==-")

    mylogger.info("Alignment of the ref images and the mask")
    mylogger.info("Ref image of donor stays fixed.")
    mylogger.info("Ref image of volume and the mask doesn't need to be touched")
    mylogger.info("Calculate translation and rotation between the reference images")
    angle_refref, tvec_refref, ref_image_acceptor, ref_image_donor = align_refref(
        mylogger=mylogger,
        ref_image_acceptor=ref_image_acceptor,
        ref_image_donor=ref_image_donor,
        batch_size=config["alignment_batch_size"],
        fill_value=-100.0,
    )
    mylogger.info(f"Rotation: {round(float(angle_refref[0]), 2)} degree")
    mylogger.info(
        f"Translation: {round(float(tvec_refref[0]), 1)} x {round(float(tvec_refref[1]), 1)} pixel"
    )

    if config["save_alignment"]:
        temp_path: str = os.path.join(
            config["export_path"], experiment_name + "_angle_refref.npy"
        )
        mylogger.info(f"Save angle to {temp_path}")
        np.save(temp_path, angle_refref.cpu())

        temp_path = os.path.join(
            config["export_path"], experiment_name + "_tvec_refref.npy"
        )
        mylogger.info(f"Save translation vector to {temp_path}")
        np.save(temp_path, tvec_refref.cpu())

    mylogger.info("Moving & rotating the oxygenation ref image")
    ref_image_oxygenation = tv.transforms.functional.affine(  # type: ignore
        img=ref_image_oxygenation.unsqueeze(0),
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-100.0,
    )

    ref_image_oxygenation = tv.transforms.functional.affine(  # type: ignore
        img=ref_image_oxygenation,
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-100.0,
    ).squeeze(0)
    mylogger.info("-==- Done -==-")

    mylogger.info("Rotate and translate the acceptor and oxygenation data accordingly")
    acceptor_index: int = config["required_order"].index("acceptor")
    donor_index: int = config["required_order"].index("donor")
    oxygenation_index: int = config["required_order"].index("oxygenation")
    volume_index: int = config["required_order"].index("volume")

    mylogger.info("Rotate acceptor")
    data[acceptor_index, ...] = tv.transforms.functional.affine(  # type: ignore
        img=data[acceptor_index, ...],  # type: ignore
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-100.0,
    )

    mylogger.info("Translate acceptor")
    data[acceptor_index, ...] = tv.transforms.functional.affine(  # type: ignore
        img=data[acceptor_index, ...],
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-100.0,
    )

    mylogger.info("Rotate oxygenation")
    data[oxygenation_index, ...] = tv.transforms.functional.affine(  # type: ignore
        img=data[oxygenation_index, ...],
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-100.0,
    )

    mylogger.info("Translate oxygenation")
    data[oxygenation_index, ...] = tv.transforms.functional.affine(  # type: ignore
        img=data[oxygenation_index, ...],
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-100.0,
    )
    mylogger.info("-==- Done -==-")

    mylogger.info("Perform rotation between donor and volume and its ref images")
    mylogger.info("for all frames and then rotate all the data accordingly")

    (
        data[acceptor_index, ...],
        data[donor_index, ...],
        data[oxygenation_index, ...],
        data[volume_index, ...],
        angle_donor_volume,
    ) = perform_donor_volume_rotation(
        mylogger=mylogger,
        acceptor=data[acceptor_index, ...],
        donor=data[donor_index, ...],
        oxygenation=data[oxygenation_index, ...],
        volume=data[volume_index, ...],
        ref_image_donor=ref_image_donor,
        ref_image_volume=ref_image_volume,
        batch_size=config["alignment_batch_size"],
        fill_value=-100.0,
        config=config,
    )

    mylogger.info(
        f"angles: "
        f"min {round(float(angle_donor_volume.min()), 2)} "
        f"max {round(float(angle_donor_volume.max()), 2)} "
        f"mean {round(float(angle_donor_volume.mean()), 2)} "
    )

    if config["save_alignment"]:
        temp_path = os.path.join(
            config["export_path"], experiment_name + "_angle_donor_volume.npy"
        )
        mylogger.info(f"Save angles to {temp_path}")
        np.save(temp_path, angle_donor_volume.cpu())
    mylogger.info("-==- Done -==-")

    mylogger.info("Perform translation between donor and volume and its ref images")
    mylogger.info("for all frames and then translate all the data accordingly")

    (
        data_acceptor,
        data_donor,
        data_oxygenation,
        data_volume,
        _,
    ) = perform_donor_volume_translation(
        mylogger=mylogger,
        acceptor=data[acceptor_index, 0:1, ...],
        donor=data[donor_index, 0:1, ...],
        oxygenation=data[oxygenation_index, 0:1, ...],
        volume=data[volume_index, 0:1, ...],
        ref_image_donor=ref_image_donor,
        ref_image_volume=ref_image_volume,
        batch_size=config["alignment_batch_size"],
        fill_value=-100.0,
        config=config,
    )

    #

    temp_path = os.path.join(
        config["export_path"], experiment_name + "_inspect_images.npz"
    )
    mylogger.info(f"Save images for inspection to {temp_path}")
    np.savez(
        temp_path,
        acceptor=data_acceptor.cpu(),
        donor=data_donor.cpu(),
        oxygenation=data_oxygenation.cpu(),
        volume=data_volume.cpu(),
    )

    mylogger.info("")
    mylogger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    mylogger.info("~ TRIAL START ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    mylogger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    mylogger.info("")

    return


def main(
    *,
    config_filename: str = "config.json",
    experiment_id_overwrite: int = -1,
    trial_id_overwrite: int = -1,
) -> None:
    mylogger = create_logger(
        save_logging_messages=True,
        display_logging_messages=True,
        log_stage_name="stage_4b",
    )

    config = load_config(mylogger=mylogger, filename=config_filename)

    if (config["save_as_python"] is False) and (config["save_as_matlab"] is False):
        mylogger.info("No output will be created. ")
        mylogger.info("Change save_as_python and/or save_as_matlab in the config file")
        mylogger.info("ERROR: STOP!!!")
        exit()

    if (len(config["target_camera_donor"]) == 0) and (
        len(config["target_camera_acceptor"]) == 0
    ):
        mylogger.info(
            "Configure at least target_camera_donor or target_camera_acceptor correctly."
        )
        mylogger.info("ERROR: STOP!!!")
        exit()

    device = get_torch_device(mylogger, config["force_to_cpu"])

    mylogger.info(
        f"Create directory {config['export_path']} in the case it does not exist"
    )
    os.makedirs(config["export_path"], exist_ok=True)

    raw_data_path: str = os.path.join(
        config["basic_path"],
        config["recoding_data"],
        config["mouse_identifier"],
        config["raw_path"],
    )

    if os.path.isdir(raw_data_path) is False:
        mylogger.info(f"ERROR: could not find raw directory {raw_data_path}!!!!")
        exit()

    if experiment_id_overwrite == -1:
        experiments = get_experiments(raw_data_path)
    else:
        assert experiment_id_overwrite >= 0
        experiments = torch.tensor([experiment_id_overwrite])

    for experiment_counter in range(0, experiments.shape[0]):
        experiment_id = int(experiments[experiment_counter])

        if trial_id_overwrite == -1:
            trials = get_trials(raw_data_path, experiment_id)
        else:
            assert trial_id_overwrite >= 0
            trials = torch.tensor([trial_id_overwrite])

        for trial_counter in range(0, trials.shape[0]):
            trial_id = int(trials[trial_counter])

            mylogger.info("")
            mylogger.info(
                f"======= EXPERIMENT ID: {experiment_id} ==== TRIAL ID: {trial_id} ======="
            )
            mylogger.info("")

            try:
                process_trial(
                    config=config,
                    mylogger=mylogger,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    device=device,
                )
            except torch.cuda.OutOfMemoryError:
                mylogger.info("WARNING: RUNNING IN FAILBACK MODE!!!!")
                mylogger.info("Not enough GPU memory. Retry on CPU")
                process_trial(
                    config=config,
                    mylogger=mylogger,
                    experiment_id=experiment_id,
                    trial_id=trial_id,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    argh.dispatch_command(main)
