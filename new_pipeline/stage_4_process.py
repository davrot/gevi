import numpy as np
import torch
import torchvision as tv  # type: ignore

import os
import logging

from functions.create_logger import create_logger
from functions.get_torch_device import get_torch_device
from functions.load_config import load_config
from functions.load_meta_data import load_meta_data

from functions.get_experiments import get_experiments
from functions.get_trials import get_trials
from functions.get_parts import get_parts
from functions.binning import binning
from functions.ImageAlignment import ImageAlignment
from functions.align_refref import align_refref
from functions.perform_donor_volume_rotation import perform_donor_volume_rotation
from functions.perform_donor_volume_translation import perform_donor_volume_translation
from functions.bandpass import bandpass
from functions.gauss_smear_individual import gauss_smear_individual

import matplotlib.pyplot as plt


@torch.no_grad()
def process_trial(
    config: dict,
    mylogger: logging.Logger,
    experiment_id: int,
    trial_id: int,
    device: torch.device,
):
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

    if os.path.isdir(raw_data_path) is False:
        mylogger.info(f"ERROR: could not find raw directory {raw_data_path}!!!!")
        return

    if (torch.where(get_experiments(raw_data_path) == experiment_id)[0].shape[0]) != 1:
        mylogger.info(f"ERROR: could not find experiment id {experiment_id}!!!!")
        return

    if (
        torch.where(get_trials(raw_data_path, experiment_id) == trial_id)[0].shape[0]
    ) != 1:
        mylogger.info(f"ERROR: could not find trial id {trial_id}!!!!")
        return

    if get_parts(raw_data_path, experiment_id, trial_id).shape[0] != 1:
        mylogger.info("ERROR: this has more than one part. NOT IMPLEMENTED YET!!!!")
        assert get_parts(raw_data_path, experiment_id, trial_id).shape[0] == 1
    part_id: int = 1

    experiment_name = f"Exp{experiment_id:03d}_Trial{trial_id:03d}"
    mylogger.info(f"Will work on: {experiment_name}")

    filename_data: str = os.path.join(
        raw_data_path,
        f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}.npy",
    )

    mylogger.info(f"Will use: {filename_data} for data")

    filename_meta: str = os.path.join(
        raw_data_path,
        f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}_meta.txt",
    )

    mylogger.info(f"Will use: {filename_meta} for meta data")

    if os.path.isfile(filename_meta) is False:
        mylogger.info(f"Could not load meta data... {filename_meta}")
        mylogger.info(f"ERROR: skipping {experiment_name}!!!!")
        return

    meta_channels: list[str]
    meta_mouse_markings: str
    meta_recording_date: str
    meta_stimulation_times: dict
    meta_experiment_names: dict
    meta_trial_recording_duration: float
    meta_frame_time: float
    meta_mouse: str

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

    dtype_str = config["dtype"]
    mylogger.info(f"Data precision will be {dtype_str}")
    dtype: torch.dtype = getattr(torch, dtype_str)
    dtype_np: np.dtype = getattr(np, dtype_str)

    mylogger.info("Loading raw data")

    if device != torch.device("cpu"):
        free_mem: int = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(f"CUDA memory before loading RAW data: {free_mem//1024} MByte")

    data_np: np.ndarray = np.load(filename_data, mmap_mode="r").astype(dtype_np)
    data: torch.Tensor = torch.zeros(data_np.shape, dtype=dtype, device=device)
    for i in range(0, len(config["required_order"])):
        mylogger.info(f"Move raw data to PyTorch device: {config['required_order'][i]}")
        idx = meta_channels.index(config["required_order"][i])
        data[..., i] = torch.tensor(data_np[..., idx], dtype=dtype, device=device)

    if device != torch.device("cpu"):
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(f"CUDA memory after loading RAW data: {free_mem//1024} MByte")

    del data_np
    mylogger.info(f"Data shape: {data.shape}")
    mylogger.info("-==- Done -==-")

    mylogger.info("Reference images and mask")

    ref_image_path: str = config["ref_image_path"]

    ref_image_path_acceptor: str = os.path.join(ref_image_path, "acceptor.npy")
    if os.path.isfile(ref_image_path_acceptor) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_acceptor}")
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_acceptor}")
    ref_image_acceptor: torch.Tensor = torch.tensor(
        np.load(ref_image_path_acceptor).astype(dtype_np), dtype=dtype, device=device
    )

    ref_image_path_donor: str = os.path.join(ref_image_path, "donor.npy")
    if os.path.isfile(ref_image_path_donor) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_donor}")
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_donor}")
    ref_image_donor: torch.Tensor = torch.tensor(
        np.load(ref_image_path_donor).astype(dtype_np), dtype=dtype, device=device
    )

    ref_image_path_oxygenation: str = os.path.join(ref_image_path, "oxygenation.npy")
    if os.path.isfile(ref_image_path_oxygenation) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_oxygenation}")
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_oxygenation}")
    ref_image_oxygenation: torch.Tensor = torch.tensor(
        np.load(ref_image_path_oxygenation).astype(dtype_np), dtype=dtype, device=device
    )

    ref_image_path_volume: str = os.path.join(ref_image_path, "volume.npy")
    if os.path.isfile(ref_image_path_volume) is False:
        mylogger.info(f"Could not load ref file: {ref_image_path_volume}")
        return

    mylogger.info(f"Loading ref file data: {ref_image_path_volume}")
    ref_image_volume: torch.Tensor = torch.tensor(
        np.load(ref_image_path_volume).astype(dtype_np), dtype=dtype, device=device
    )

    refined_mask_file: str = os.path.join(ref_image_path, "mask_not_rotated.npy")
    if os.path.isfile(refined_mask_file) is False:
        mylogger.info(f"Could not load mask file: {refined_mask_file}")
        return

    mylogger.info(f"Loading mask file data: {refined_mask_file}")
    mask: torch.Tensor = torch.tensor(
        np.load(refined_mask_file).astype(dtype_np), dtype=dtype, device=device
    )
    mylogger.info("-==- Done -==-")

    if config["binning_enable"] and config["binning_before_alignment"]:
        mylogger.info("Binning of data")
        mylogger.info(
            (
                f"kernel_size={int(config['binning_kernel_size'])},"
                f"stride={int(config['binning_stride'])},"
                f"divisor_override={int(config['binning_divisor_override'])}"
            )
        )

        data = binning(
            data,
            kernel_size=int(config["binning_kernel_size"]),
            stride=int(config["binning_stride"]),
            divisor_override=int(config["binning_divisor_override"]),
        )
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
    image_alignment = ImageAlignment(default_dtype=dtype, device=device)

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
        image_alignment=image_alignment,
        batch_size=config["alignment_batch_size"],
        fill_value=-1.0,
    )
    mylogger.info(f"Rotation: {round(float(angle_refref[0]),2)} degree")
    mylogger.info(
        f"Translation: {round(float(tvec_refref[0]),1)} x {round(float(tvec_refref[1]),1)} pixel"
    )

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
    ref_image_oxygenation = tv.transforms.functional.affine(
        img=ref_image_oxygenation.unsqueeze(0),
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )

    ref_image_oxygenation = tv.transforms.functional.affine(
        img=ref_image_oxygenation,
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    ).squeeze(0)
    mylogger.info("-==- Done -==-")

    mylogger.info("Rotate and translate the acceptor and oxygenation data accordingly")
    acceptor_index: int = config["required_order"].index("acceptor")
    donor_index: int = config["required_order"].index("donor")
    oxygenation_index: int = config["required_order"].index("oxygenation")
    volume_index: int = config["required_order"].index("volume")

    mylogger.info("Rotate acceptor")
    data[acceptor_index, ...] = tv.transforms.functional.affine(
        img=data[acceptor_index, ...],
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )

    mylogger.info("Translate acceptor")
    data[acceptor_index, ...] = tv.transforms.functional.affine(
        img=data[acceptor_index, ...],
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )

    mylogger.info("Rotate oxygenation")
    data[oxygenation_index, ...] = tv.transforms.functional.affine(
        img=data[oxygenation_index, ...],
        angle=-float(angle_refref),
        translate=[0, 0],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )

    mylogger.info("Translate oxygenation")
    data[oxygenation_index, ...] = tv.transforms.functional.affine(
        img=data[oxygenation_index, ...],
        angle=0,
        translate=[tvec_refref[1], tvec_refref[0]],
        scale=1.0,
        shear=0,
        interpolation=tv.transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )
    mylogger.info("-==- Done -==-")

    mylogger.info("Perform rotation between donor and volume and its ref images")
    mylogger.info("for all frames and then rotate all the data accordingly")
    perform_donor_volume_rotation
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
        image_alignment=image_alignment,
        batch_size=config["alignment_batch_size"],
        fill_value=-1.0,
    )

    mylogger.info(
        f"angles: "
        f"min {round(float(angle_donor_volume.min()),2)} "
        f"max {round(float(angle_donor_volume.max()),2)} "
        f"mean {round(float(angle_donor_volume.mean()),2)} "
    )

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
        image_alignment=image_alignment,
        batch_size=config["alignment_batch_size"],
        fill_value=-1.0,
    )

    mylogger.info(
        f"translation dim 0: "
        f"min {round(float(tvec_donor_volume[:,0].min()),1)} "
        f"max {round(float(tvec_donor_volume[:,0].max()),1)} "
        f"mean {round(float(tvec_donor_volume[:,0].mean()),1)} "
    )
    mylogger.info(
        f"translation dim 1: "
        f"min {round(float(tvec_donor_volume[:,1].min()),1)} "
        f"max {round(float(tvec_donor_volume[:,1].max()),1)} "
        f"mean {round(float(tvec_donor_volume[:,1].mean()),1)} "
    )

    temp_path = os.path.join(
        config["export_path"], experiment_name + "_tvec_donor_volume.npy"
    )
    mylogger.info(f"Save translation vector to {temp_path}")
    np.save(temp_path, tvec_donor_volume.cpu())
    mylogger.info("-==- Done -==-")

    mylogger.info("Update mask with the new regions due to alignment")

    new_mask_area: torch.Tensor = torch.any(torch.any(data < -0.1, dim=0), dim=0).bool()
    mask = (mask == 0).bool()
    mask = torch.logical_or(mask, new_mask_area)
    mask_positve: torch.Tensor = torch.logical_not(mask)

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

    mylogger.info("Extract heartbeat from volume signal")
    heartbeat_ts: torch.Tensor = bandpass(
        data=data[volume_index, ...].movedim(0, -1).clone(),
        device=data.device,
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

    volume_heartbeat, _, _ = torch.linalg.svd(heartbeat_ts, full_matrices=False)
    volume_heartbeat = volume_heartbeat[:, 0]
    volume_heartbeat -= volume_heartbeat[
        config["skip_frames_in_the_beginning"] :
    ].mean()

    del heartbeat_ts

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
            device=data.device,
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

    temp_path = os.path.join(
        config["export_path"], experiment_name + "_heartbeat_coefficients.npy"
    )
    mylogger.info(f"Save heartbeat coefficients to {temp_path}")
    np.save(temp_path, heartbeat_coefficients.cpu())
    mylogger.info("-==- Done -==-")

    mylogger.info("Remove heart beat from data")
    data -= heartbeat_coefficients.unsqueeze(1) * volume_heartbeat.unsqueeze(0).movedim(
        -1, 1
    )
    mylogger.info("-==- Done -==-")

    donor_heartbeat_factor = heartbeat_coefficients[donor_index, ...].clone()
    acceptor_heartbeat_factor = heartbeat_coefficients[acceptor_index, ...].clone()
    del heartbeat_coefficients

    mylogger.info("Calculate scaling factor for donor and acceptor")
    donor_factor: torch.Tensor = (
        donor_heartbeat_factor + acceptor_heartbeat_factor
    ) / (2 * donor_heartbeat_factor)
    acceptor_factor: torch.Tensor = (
        donor_heartbeat_factor + acceptor_heartbeat_factor
    ) / (2 * acceptor_heartbeat_factor)

    del donor_heartbeat_factor
    del acceptor_heartbeat_factor

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
    data[acceptor_index, ...] *= acceptor_factor.unsqueeze(0) * mask.unsqueeze(0)
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
    data[donor_index, ...] *= donor_factor.unsqueeze(0) * mask.unsqueeze(0)
    mylogger.info("Add mean")
    data[donor_index, ...] += mean_values_donor
    mylogger.info("-==- Done -==-")

    mylogger.info("Divide by mean over time")
    data /= data[:, config["skip_frames_in_the_beginning"] :, ...].nanmean(
        dim=1,
        keepdim=True,
    )
    data = data.nan_to_num(nan=0.0)
    mylogger.info("-==- Done -==-")

    mylogger.info("Preparation for regression -- Gauss smear")
    spatial_width = float(config["gauss_smear_spatial_width"])

    if config["binning_enable"] and config["binning_before_alignment"]:
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
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(
            f"CUDA memory after reserving RAM for data_filtered: {free_mem//1024} MByte"
        )

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
        free_mem = cuda_total_memory - max(
            [torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)]
        )
        mylogger.info(
            f"CUDA memory after data_filtered is populated: {free_mem//1024} MByte"
        )

    mylogger.info("-==- Done -==-")

    exit()

    return


mylogger = create_logger(
    save_logging_messages=True, display_logging_messages=True, log_stage_name="stage_4"
)
config = load_config(mylogger=mylogger)
device = get_torch_device(mylogger, config["force_to_cpu"])

mylogger.info(f"Create directory {config['export_path']} in the case it does not exist")
os.makedirs(config["export_path"], exist_ok=True)

process_trial(
    config=config, mylogger=mylogger, experiment_id=1, trial_id=1, device=device
)
