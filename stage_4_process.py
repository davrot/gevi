import numpy as np
import torch
import torchvision as tv  # type: ignore

import os
import logging
import h5py  # type: ignore

from functions.create_logger import create_logger
from functions.get_torch_device import get_torch_device
from functions.load_config import load_config
from functions.get_experiments import get_experiments
from functions.get_trials import get_trials
from functions.binning import binning
from functions.align_refref import align_refref
from functions.perform_donor_volume_rotation import perform_donor_volume_rotation
from functions.perform_donor_volume_translation import perform_donor_volume_translation
from functions.bandpass import bandpass
from functions.gauss_smear_individual import gauss_smear_individual
from functions.regression import regression
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
        data[acceptor_index, ...],
        data[donor_index, ...],
        data[oxygenation_index, ...],
        data[volume_index, ...],
        tvec_donor_volume,
    ) = perform_donor_volume_translation(
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
        f"translation dim 0: "
        f"min {round(float(tvec_donor_volume[:, 0].min()), 1)} "
        f"max {round(float(tvec_donor_volume[:, 0].max()), 1)} "
        f"mean {round(float(tvec_donor_volume[:, 0].mean()), 1)} "
    )
    mylogger.info(
        f"translation dim 1: "
        f"min {round(float(tvec_donor_volume[:, 1].min()), 1)} "
        f"max {round(float(tvec_donor_volume[:, 1].max()), 1)} "
        f"mean {round(float(tvec_donor_volume[:, 1].mean()), 1)} "
    )

    if config["save_alignment"]:
        temp_path = os.path.join(
            config["export_path"], experiment_name + "_tvec_donor_volume.npy"
        )
        mylogger.info(f"Save translation vector to {temp_path}")
        np.save(temp_path, tvec_donor_volume.cpu())
    mylogger.info("-==- Done -==-")

    mylogger.info("Finding zeros values in the RAW data and mark them for masking")
    for i in range(0, data.shape[0]):
        zero_pixel_mask = torch.any(data[i, ...] == 0, dim=0)
        data[i, :, zero_pixel_mask] = -100.0
        mylogger.info(
            f"{config['required_order'][i]}: "
            f"found {int(zero_pixel_mask.type(dtype=dtype).sum())} pixel "
            f"with zeros "
        )
    mylogger.info("-==- Done -==-")

    mylogger.info("Update mask with the new regions due to alignment")

    new_mask_area: torch.Tensor = torch.any(torch.any(data < -0.1, dim=0), dim=0).bool()
    mask = (mask == 0).bool()
    mask = torch.logical_or(mask, new_mask_area)
    mask_negative: torch.Tensor = mask.clone()
    mask_positve: torch.Tensor = torch.logical_not(mask)
    del mask

    mylogger.info("Update the data with the new mask")
    data *= mask_positve.unsqueeze(0).unsqueeze(0).type(dtype=dtype)
    mylogger.info("-==- Done -==-")

    mylogger.info("Interpolate the 'in-between' frames for oxygenation and volume")
    data[oxygenation_index, 1:, ...] = (
        data[oxygenation_index, 1:, ...] + data[oxygenation_index, :-1, ...]
    ) / 2.0
    data[volume_index, 1:, ...] = (
        data[volume_index, 1:, ...] + data[volume_index, :-1, ...]
    ) / 2.0
    mylogger.info("-==- Done -==-")

    sample_frequency: float = 1.0 / meta_frame_time

    if config["gevi"]:
        assert config["heartbeat_remove"]

    if config["heartbeat_remove"]:
        mylogger.info("Extract heartbeat from volume signal")
        heartbeat_ts: torch.Tensor = bandpass(
            data=data[volume_index, ...].movedim(0, -1).clone(),
            low_frequency=config["lower_freqency_bandpass"],
            high_frequency=config["upper_freqency_bandpass"],
            fs=sample_frequency,
            filtfilt_chuck_size=config["heartbeat_filtfilt_chuck_size"],
        )
        heartbeat_ts = heartbeat_ts.flatten(start_dim=0, end_dim=-2)
        mask_flatten: torch.Tensor = mask_positve.flatten(start_dim=0, end_dim=-1)

        heartbeat_ts = heartbeat_ts[mask_flatten, :]

        heartbeat_ts = heartbeat_ts.movedim(0, -1)
        heartbeat_ts -= heartbeat_ts.mean(dim=0, keepdim=True)

        try:
            volume_heartbeat, _, _ = torch.linalg.svd(heartbeat_ts, full_matrices=False)
        except torch.cuda.OutOfMemoryError:
            mylogger.info("torch.cuda.OutOfMemoryError: Fallback to cpu")
            volume_heartbeat_cpu, _, _ = torch.linalg.svd(
                heartbeat_ts.cpu(), full_matrices=False
            )
            volume_heartbeat = volume_heartbeat_cpu.to(heartbeat_ts.data, copy=True)
            del volume_heartbeat_cpu

        volume_heartbeat = volume_heartbeat[:, 0]
        volume_heartbeat -= volume_heartbeat[
            config["skip_frames_in_the_beginning"] :
        ].mean()

        del heartbeat_ts

        if device != torch.device("cpu"):
            torch.cuda.empty_cache()
            mylogger.info("Empty CUDA cache")
            free_mem = cuda_total_memory - max(
                [
                    torch.cuda.memory_reserved(device),
                    torch.cuda.memory_allocated(device),
                ]
            )
            mylogger.info(f"CUDA memory: {free_mem // 1024} MByte")

        if config["save_heartbeat"]:
            temp_path = os.path.join(
                config["export_path"], experiment_name + "_volume_heartbeat.npy"
            )
            mylogger.info(f"Save volume heartbeat to {temp_path}")
            np.save(temp_path, volume_heartbeat.cpu())
        mylogger.info("-==- Done -==-")

        volume_heartbeat = volume_heartbeat.unsqueeze(0).unsqueeze(0)
        norm_volume_heartbeat = (
            volume_heartbeat[..., config["skip_frames_in_the_beginning"] :] ** 2
        ).sum(dim=-1)

        heartbeat_coefficients: torch.Tensor = torch.zeros(
            (data.shape[0], data.shape[-2], data.shape[-1]),
            dtype=data.dtype,
            device=data.device,
        )
        for i in range(0, data.shape[0]):
            y = bandpass(
                data=data[i, ...].movedim(0, -1).clone(),
                low_frequency=config["lower_freqency_bandpass"],
                high_frequency=config["upper_freqency_bandpass"],
                fs=sample_frequency,
                filtfilt_chuck_size=config["heartbeat_filtfilt_chuck_size"],
            )[..., config["skip_frames_in_the_beginning"] :]
            y -= y.mean(dim=-1, keepdim=True)

            heartbeat_coefficients[i, ...] = (
                volume_heartbeat[..., config["skip_frames_in_the_beginning"] :] * y
            ).sum(dim=-1) / norm_volume_heartbeat

            heartbeat_coefficients[i, ...] *= mask_positve.type(
                dtype=heartbeat_coefficients.dtype
            )
        del y

        if config["save_heartbeat"]:
            temp_path = os.path.join(
                config["export_path"], experiment_name + "_heartbeat_coefficients.npy"
            )
            mylogger.info(f"Save heartbeat coefficients to {temp_path}")
            np.save(temp_path, heartbeat_coefficients.cpu())
        mylogger.info("-==- Done -==-")

        mylogger.info("Remove heart beat from data")
        data -= heartbeat_coefficients.unsqueeze(1) * volume_heartbeat.unsqueeze(
            0
        ).movedim(-1, 1)
        mylogger.info("-==- Done -==-")

        donor_heartbeat_factor = heartbeat_coefficients[donor_index, ...].clone()
        acceptor_heartbeat_factor = heartbeat_coefficients[acceptor_index, ...].clone()
        del heartbeat_coefficients

        if device != torch.device("cpu"):
            torch.cuda.empty_cache()
            mylogger.info("Empty CUDA cache")
            free_mem = cuda_total_memory - max(
                [
                    torch.cuda.memory_reserved(device),
                    torch.cuda.memory_allocated(device),
                ]
            )
            mylogger.info(f"CUDA memory: {free_mem // 1024} MByte")

        mylogger.info("Calculate scaling factor for donor and acceptor")
        donor_factor: torch.Tensor = (
            donor_heartbeat_factor + acceptor_heartbeat_factor
        ) / (2 * donor_heartbeat_factor)
        acceptor_factor: torch.Tensor = (
            donor_heartbeat_factor + acceptor_heartbeat_factor
        ) / (2 * acceptor_heartbeat_factor)

        del donor_heartbeat_factor
        del acceptor_heartbeat_factor

        if config["save_factors"]:
            temp_path = os.path.join(
                config["export_path"], experiment_name + "_donor_factor.npy"
            )
            mylogger.info(f"Save donor factor to {temp_path}")
            np.save(temp_path, donor_factor.cpu())

            temp_path = os.path.join(
                config["export_path"], experiment_name + "_acceptor_factor.npy"
            )
            mylogger.info(f"Save acceptor factor to {temp_path}")
            np.save(temp_path, acceptor_factor.cpu())
        mylogger.info("-==- Done -==-")

        mylogger.info("Scale acceptor to heart beat amplitude")
        mylogger.info("Calculate mean")
        mean_values_acceptor = data[
            acceptor_index, config["skip_frames_in_the_beginning"] :, ...
        ].nanmean(dim=0, keepdim=True)

        mylogger.info("Remove mean")
        data[acceptor_index, ...] -= mean_values_acceptor
        mylogger.info("Apply acceptor_factor and mask")
        data[acceptor_index, ...] *= acceptor_factor.unsqueeze(
            0
        ) * mask_positve.unsqueeze(0)
        mylogger.info("Add mean")
        data[acceptor_index, ...] += mean_values_acceptor
        mylogger.info("-==- Done -==-")

        mylogger.info("Scale donor to heart beat amplitude")
        mylogger.info("Calculate mean")
        mean_values_donor = data[
            donor_index, config["skip_frames_in_the_beginning"] :, ...
        ].nanmean(dim=0, keepdim=True)
        mylogger.info("Remove mean")
        data[donor_index, ...] -= mean_values_donor
        mylogger.info("Apply donor_factor and mask")
        data[donor_index, ...] *= donor_factor.unsqueeze(0) * mask_positve.unsqueeze(0)
        mylogger.info("Add mean")
        data[donor_index, ...] += mean_values_donor
        mylogger.info("-==- Done -==-")

        mylogger.info("Divide by mean over time")
        data /= data[:, config["skip_frames_in_the_beginning"] :, ...].nanmean(
            dim=1,
            keepdim=True,
        )

        mylogger.info("-==- Done -==-")

    data = data.nan_to_num(nan=0.0)
    mylogger.info("Preparation for regression -- Gauss smear")
    spatial_width = float(config["gauss_smear_spatial_width"])

    if config["binning_enable"] and (config["binning_at_the_end"] is False):
        spatial_width /= float(config["binning_kernel_size"])

    mylogger.info(
        f"Mask -- "
        f"spatial width: {spatial_width}, "
        f"temporal width: {float(config['gauss_smear_temporal_width'])}, "
        f"use matlab mode: {bool(config['gauss_smear_use_matlab_mask'])} "
    )

    input_mask = mask_positve.type(dtype=dtype).clone()

    filtered_mask: torch.Tensor
    filtered_mask, _ = gauss_smear_individual(
        input=input_mask,
        spatial_width=spatial_width,
        temporal_width=float(config["gauss_smear_temporal_width"]),
        use_matlab_mask=bool(config["gauss_smear_use_matlab_mask"]),
        epsilon=float(torch.finfo(input_mask.dtype).eps),
    )

    mylogger.info("creating a copy of the data")
    data_filtered = data.clone().movedim(1, -1)
    if device != torch.device("cpu"):
        torch.cuda.empty_cache()
        mylogger.info("Empty CUDA cache")
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(f"CUDA memory: {free_mem // 1024} MByte")

    overwrite_fft_gauss: None | torch.Tensor = None
    for i in range(0, data_filtered.shape[0]):
        mylogger.info(
            f"{config['required_order'][i]} -- "
            f"spatial width: {spatial_width}, "
            f"temporal width: {float(config['gauss_smear_temporal_width'])}, "
            f"use matlab mode: {bool(config['gauss_smear_use_matlab_mask'])} "
        )
        data_filtered[i, ...] *= input_mask.unsqueeze(-1)
        data_filtered[i, ...], overwrite_fft_gauss = gauss_smear_individual(
            input=data_filtered[i, ...],
            spatial_width=spatial_width,
            temporal_width=float(config["gauss_smear_temporal_width"]),
            overwrite_fft_gauss=overwrite_fft_gauss,
            use_matlab_mask=bool(config["gauss_smear_use_matlab_mask"]),
            epsilon=float(torch.finfo(input_mask.dtype).eps),
        )

        data_filtered[i, ...] /= filtered_mask + 1e-20
        data_filtered[i, ...] += 1.0 - input_mask.unsqueeze(-1)

    del filtered_mask
    del overwrite_fft_gauss
    del input_mask
    mylogger.info("data_filtered is populated")

    if device != torch.device("cpu"):
        torch.cuda.empty_cache()
        mylogger.info("Empty CUDA cache")
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(f"CUDA memory: {free_mem // 1024} MByte")
    mylogger.info("-==- Done -==-")

    mylogger.info("Preperation for Regression")
    mylogger.info("Move time dimensions of data to the last dimension")
    data = data.movedim(1, -1)

    dual_signal_mode: bool = True
    if len(config["target_camera_acceptor"]) > 0:
        mylogger.info("Regression Acceptor")
        mylogger.info(f"Target: {config['target_camera_acceptor']}")
        mylogger.info(
            f"Regressors: constant, linear and {config['regressor_cameras_acceptor']}"
        )
        target_id: int = config["required_order"].index(
            config["target_camera_acceptor"]
        )
        regressor_id: list[int] = []
        for i in range(0, len(config["regressor_cameras_acceptor"])):
            regressor_id.append(
                config["required_order"].index(config["regressor_cameras_acceptor"][i])
            )

        data_acceptor, coefficients_acceptor = regression(
            mylogger=mylogger,
            target_camera_id=target_id,
            regressor_camera_ids=regressor_id,
            mask=mask_negative,
            data=data,
            data_filtered=data_filtered,
            first_none_ramp_frame=int(config["skip_frames_in_the_beginning"]),
        )

        if config["save_regression_coefficients"]:
            temp_path = os.path.join(
                config["export_path"], experiment_name + "_coefficients_acceptor.npy"
            )
            mylogger.info(f"Save acceptor coefficients to {temp_path}")
            np.save(temp_path, coefficients_acceptor.cpu())
        del coefficients_acceptor

        mylogger.info("-==- Done -==-")
    else:
        dual_signal_mode = False
        target_id = config["required_order"].index("acceptor")
        data_acceptor = data[target_id, ...].clone()
        data_acceptor[mask_negative, :] = 0.0

    if len(config["target_camera_donor"]) > 0:
        mylogger.info("Regression Donor")
        mylogger.info(f"Target: {config['target_camera_donor']}")
        mylogger.info(
            f"Regressors: constant, linear and {config['regressor_cameras_donor']}"
        )
        target_id = config["required_order"].index(config["target_camera_donor"])
        regressor_id = []
        for i in range(0, len(config["regressor_cameras_donor"])):
            regressor_id.append(
                config["required_order"].index(config["regressor_cameras_donor"][i])
            )

        data_donor, coefficients_donor = regression(
            mylogger=mylogger,
            target_camera_id=target_id,
            regressor_camera_ids=regressor_id,
            mask=mask_negative,
            data=data,
            data_filtered=data_filtered,
            first_none_ramp_frame=int(config["skip_frames_in_the_beginning"]),
        )

        if config["save_regression_coefficients"]:
            temp_path = os.path.join(
                config["export_path"], experiment_name + "_coefficients_donor.npy"
            )
            mylogger.info(f"Save acceptor donor to {temp_path}")
            np.save(temp_path, coefficients_donor.cpu())
        del coefficients_donor
        mylogger.info("-==- Done -==-")
    else:
        dual_signal_mode = False
        target_id = config["required_order"].index("donor")
        data_donor = data[target_id, ...].clone()
        data_donor[mask_negative, :] = 0.0

    del data
    del data_filtered

    if device != torch.device("cpu"):
        torch.cuda.empty_cache()
        mylogger.info("Empty CUDA cache")
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(f"CUDA memory: {free_mem // 1024} MByte")

    # #####################

    if config["gevi"]:
        assert dual_signal_mode
    else:
        assert dual_signal_mode is False

    if dual_signal_mode is False:

        mylogger.info("mono signal model")

        mylogger.info("Remove nan")
        data_acceptor = torch.nan_to_num(data_acceptor, nan=0.0)
        data_donor = torch.nan_to_num(data_donor, nan=0.0)
        mylogger.info("-==- Done -==-")

        if config["binning_enable"] and config["binning_at_the_end"]:
            mylogger.info("Binning of data")
            mylogger.info(
                (
                    f"kernel_size={int(config['binning_kernel_size'])}, "
                    f"stride={int(config['binning_stride'])}, "
                    "divisor_override=None"
                )
            )

            data_acceptor = binning(
                data_acceptor.unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=None,
            ).squeeze(-1)

            data_donor = binning(
                data_donor.unsqueeze(-1),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=None,
            ).squeeze(-1)

            mask_positve = (
                binning(
                    mask_positve.unsqueeze(-1).unsqueeze(-1).type(dtype=dtype),
                    kernel_size=int(config["binning_kernel_size"]),
                    stride=int(config["binning_stride"]),
                    divisor_override=None,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            mask_positve = (mask_positve > 0).type(torch.bool)

        if config["save_as_python"]:

            temp_path = os.path.join(
                config["export_path"], experiment_name + "_acceptor_donor.npz"
            )
            mylogger.info(f"Save data donor and acceptor and mask to {temp_path}")
            np.savez_compressed(
                temp_path,
                data_acceptor=data_acceptor.cpu(),
                data_donor=data_donor.cpu(),
                mask=mask_positve.cpu(),
            )

        if config["save_as_matlab"]:
            temp_path = os.path.join(
                config["export_path"], experiment_name + "_acceptor_donor.hd5"
            )
            mylogger.info(f"Save data donor and acceptor and mask to {temp_path}")
            file_handle = h5py.File(temp_path, "w")

            mask_positve = mask_positve.movedim(0, -1)
            data_acceptor = data_acceptor.movedim(1, -1).movedim(0, -1)
            data_donor = data_donor.movedim(1, -1).movedim(0, -1)
            _ = file_handle.create_dataset(
                "mask",
                data=mask_positve.type(torch.uint8).cpu(),
                compression="gzip",
                compression_opts=9,
            )
            _ = file_handle.create_dataset(
                "data_acceptor",
                data=data_acceptor.cpu(),
                compression="gzip",
                compression_opts=9,
            )
            _ = file_handle.create_dataset(
                "data_donor",
                data=data_donor.cpu(),
                compression="gzip",
                compression_opts=9,
            )
            mylogger.info("Reminder: How to read with matlab:")
            mylogger.info(f"mask = h5read('{temp_path}','/mask');")
            mylogger.info(f"data_acceptor = h5read('{temp_path}','/data_acceptor');")
            mylogger.info(f"data_donor = h5read('{temp_path}','/data_donor');")
            file_handle.close()
        return
    # #####################

    mylogger.info("Calculate ratio sequence")

    if config["classical_ratio_mode"]:
        mylogger.info("via acceptor / donor")
        ratio_sequence: torch.Tensor = data_acceptor / data_donor
        mylogger.info("via / mean over time")
        ratio_sequence /= ratio_sequence.mean(dim=-1, keepdim=True)
    else:
        mylogger.info("via 1.0 + acceptor - donor")
        ratio_sequence = 1.0 + data_acceptor - data_donor

    mylogger.info("Remove nan")
    ratio_sequence = torch.nan_to_num(ratio_sequence, nan=0.0)
    mylogger.info("-==- Done -==-")

    if config["binning_enable"] and config["binning_at_the_end"]:
        mylogger.info("Binning of data")
        mylogger.info(
            (
                f"kernel_size={int(config['binning_kernel_size'])}, "
                f"stride={int(config['binning_stride'])}, "
                "divisor_override=None"
            )
        )

        ratio_sequence = binning(
            ratio_sequence.unsqueeze(-1),
            kernel_size=int(config["binning_kernel_size"]),
            stride=int(config["binning_stride"]),
            divisor_override=None,
        ).squeeze(-1)

        mask_positve = (
            binning(
                mask_positve.unsqueeze(-1).unsqueeze(-1).type(dtype=dtype),
                kernel_size=int(config["binning_kernel_size"]),
                stride=int(config["binning_stride"]),
                divisor_override=None,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        mask_positve = (mask_positve > 0).type(torch.bool)

    if config["save_as_python"]:
        temp_path = os.path.join(
            config["export_path"], experiment_name + "_ratio_sequence.npz"
        )
        mylogger.info(f"Save ratio_sequence and mask to {temp_path}")
        np.savez_compressed(
            temp_path, ratio_sequence=ratio_sequence.cpu(), mask=mask_positve.cpu()
        )

    if config["save_as_matlab"]:
        temp_path = os.path.join(
            config["export_path"], experiment_name + "_ratio_sequence.hd5"
        )
        mylogger.info(f"Save ratio_sequence and mask to {temp_path}")
        file_handle = h5py.File(temp_path, "w")

        mask_positve = mask_positve.movedim(0, -1)
        ratio_sequence = ratio_sequence.movedim(1, -1).movedim(0, -1)
        _ = file_handle.create_dataset(
            "mask",
            data=mask_positve.type(torch.uint8).cpu(),
            compression="gzip",
            compression_opts=9,
        )
        _ = file_handle.create_dataset(
            "ratio_sequence",
            data=ratio_sequence.cpu(),
            compression="gzip",
            compression_opts=9,
        )
        mylogger.info("Reminder: How to read with matlab:")
        mylogger.info(f"mask = h5read('{temp_path}','/mask');")
        mylogger.info(f"ratio_sequence = h5read('{temp_path}','/ratio_sequence');")
        file_handle.close()

    del ratio_sequence
    del mask_positve
    del mask_negative

    mylogger.info("")
    mylogger.info("***********************************************")
    mylogger.info("* TRIAL END ***********************************")
    mylogger.info("***********************************************")
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
        log_stage_name="stage_4",
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
