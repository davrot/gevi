# pip install roipoly natsort numpy matplotlib
# Also install: torch torchaudio torchvision
# (for details see https://pytorch.org/get-started/locally/ )
# Tested on Python 3.11

import glob
import json
import logging
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import natsort
import numpy as np
import torch
import torchaudio as ta
import torchvision as tv
from roipoly import RoiPoly

from ImageAlignment import ImageAlignment


class DataContainer(torch.nn.Module):
    ref_image_acceptor: torch.Tensor | None = None
    ref_image_donor: torch.Tensor | None = None

    acceptor: torch.Tensor | None = None
    donor: torch.Tensor | None = None
    oxygenation: torch.Tensor | None = None
    volume: torch.Tensor | None = None

    acceptor_whiten_mean: torch.Tensor | None = None
    acceptor_whiten_k: torch.Tensor | None = None
    acceptor_eigenvalues: torch.Tensor | None = None
    acceptor_residuum: torch.Tensor | None = None

    donor_whiten_mean: torch.Tensor | None = None
    donor_whiten_k: torch.Tensor | None = None
    donor_eigenvalues: torch.Tensor | None = None
    donor_residuum: torch.Tensor | None = None

    oxygenation_whiten_mean: torch.Tensor | None = None
    oxygenation_whiten_k: torch.Tensor | None = None
    oxygenation_eigenvalues: torch.Tensor | None = None
    oxygenation_residuum: torch.Tensor | None = None

    volume_whiten_mean: torch.Tensor | None = None
    volume_whiten_k: torch.Tensor | None = None
    volume_eigenvalues: torch.Tensor | None = None
    volume_residuum: torch.Tensor | None = None

    # -------
    image_alignment: ImageAlignment

    acceptor_index: int
    donor_index: int
    oxygenation_index: int
    volume_index: int

    path: str
    channels: list[str]
    device: torch.device

    batch_size: int = 200

    fill_value: float = -0.1

    filtfilt_chuck_size: int = 10

    level0 = str("=")
    level1 = str("==")
    level2 = str("===")
    level3 = str("====")

    @torch.no_grad()
    def __init__(
        self,
        path: str,
        device: torch.device,
        display_logging_messages: bool = False,
        save_logging_messages: bool = False,
    ) -> None:
        super().__init__()
        self.device = device

        assert path is not None
        self.path = path
        now = datetime.now()
        dt_string_filename = now.strftime("%Y_%m_%d_%H_%M_%S")

        self.logger = logging.getLogger("DataContainer")
        self.logger.setLevel(logging.DEBUG)

        if save_logging_messages is True:
            time_format = "%b %-d %Y %H:%M:%S"
            logformat = "%(asctime)s %(message)s"
            file_formatter = logging.Formatter(fmt=logformat, datefmt=time_format)

            file_handler = logging.FileHandler(f"log_{dt_string_filename}.txt")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        if display_logging_messages is True:
            time_format = "%b %-d %Y %H:%M:%S"
            logformat = "%(asctime)s %(message)s"
            stream_formatter = logging.Formatter(fmt=logformat, datefmt=time_format)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)

        file_input_ref_image = self._find_ref_image_file()

        data = np.load(file_input_ref_image, mmap_mode="r")
        ref_image = torch.tensor(
            data[:, :, data.shape[2] // 2, :].astype(np.float32),
            device=self.device,
            dtype=torch.float32,
        )

        json_postfix: str = "_meta.txt"
        found_name_json: str = file_input_ref_image.replace(".npy", json_postfix)

        assert os.path.isfile(found_name_json) is True

        with open(found_name_json, "r") as file_handle:
            metadata = json.load(file_handle)
        self.channels = metadata["channelKey"]

        self.acceptor_index = self.channels.index("acceptor")
        self.donor_index = self.channels.index("donor")
        self.oxygenation_index = self.channels.index("oxygenation")
        self.volume_index = self.channels.index("volume")

        self.ref_image_acceptor: torch.Tensor = ref_image[:, :, self.acceptor_index]
        self.ref_image_donor: torch.Tensor = ref_image[:, :, self.donor_index]

        self.image_alignment = ImageAlignment(
            default_dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def get_trials(self, experiment_id: int) -> torch.Tensor:
        filename_np: str = os.path.join(
            self.path,
            f"Exp{experiment_id:03d}_Trial*_Part001.npy",
        )

        list_str = glob.glob(filename_np)
        list_int: list[int] = []
        for i in range(0, len(list_str)):
            list_int.append(int(list_str[i].split("_Trial")[-1].split("_Part")[0]))
        list_int = sorted(list_int)
        return torch.tensor(list_int).unique()

    @torch.no_grad()
    def get_experiments(
        self,
    ) -> torch.Tensor:
        filename_np: str = os.path.join(
            self.path,
            "Exp*_Part001.npy",
        )

        list_str = glob.glob(filename_np)
        list_int: list[int] = []
        for i in range(0, len(list_str)):
            list_int.append(int(list_str[i].split("Exp")[-1].split("_Trial")[0]))
        list_int = sorted(list_int)

        return torch.tensor(list_int).unique()

    @torch.no_grad()
    def load_data(  # start_position_coefficients: OK
        self,
        experiment_id: int,
        trial_id: int,
        align: bool = True,
        enable_secondary_data: bool = True,
        mmap_mode: bool = True,
        start_position_coefficients: int = 0,
    ):
        self.acceptor = None
        self.donor = None
        self.oxygenation = None
        self.volume = None

        part_id: int = 1
        filename_np: str = os.path.join(
            self.path,
            f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}.npy",
        )

        filename_meta: str = os.path.join(
            self.path,
            f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}_meta.txt",
        )

        while (os.path.isfile(filename_np) is True) and (
            os.path.isfile(filename_meta) is True
        ):
            self.logger.info(f"{self.level3} work in {filename_np}")
            # Check if channel asignment is still okay
            with open(filename_meta, "r") as file_handle:
                metadata = json.load(file_handle)
            channels = metadata["channelKey"]

            assert len(channels) == len(self.channels)
            for i in range(0, len(channels)):
                assert channels[i] == self.channels[i]

            # Load the data...
            self.logger.info(f"{self.level3} np.load")
            if mmap_mode is True:
                temp: np.ndarray = np.load(filename_np, mmap_mode="r")
            else:
                temp = np.load(filename_np)

            self.logger.info(f"{self.level3} organize acceptor")
            if self.acceptor is None:
                self.acceptor = torch.tensor(
                    temp[:, :, :, self.acceptor_index].astype(np.float32),
                    device=self.device,
                    dtype=torch.float32,
                )

            else:
                assert self.acceptor is not None
                assert self.acceptor.ndim == temp.ndim
                assert self.acceptor.shape[0] == temp.shape[0]
                assert self.acceptor.shape[1] == temp.shape[1]
                assert self.acceptor.shape[3] == temp.shape[3]
                self.acceptor = torch.cat(
                    (
                        self.acceptor,
                        torch.tensor(
                            temp[:, :, :, self.acceptor_index].astype(np.float32),
                            device=self.device,
                            dtype=torch.float32,
                        ),
                    ),
                    dim=2,
                )

            self.logger.info(f"{self.level3} organize donor")
            if self.donor is None:
                self.donor = torch.tensor(
                    temp[:, :, :, self.donor_index].astype(np.float32),
                    device=self.device,
                    dtype=torch.float32,
                )

            else:
                assert self.donor is not None
                assert self.donor.ndim == temp.ndim
                assert self.donor.shape[0] == temp.shape[0]
                assert self.donor.shape[1] == temp.shape[1]
                assert self.donor.shape[3] == temp.shape[3]
                self.donor = torch.cat(
                    (
                        self.donor,
                        torch.tensor(
                            temp[:, :, :, self.donor_index].astype(np.float32),
                            device=self.device,
                            dtype=torch.float32,
                        ),
                    ),
                    dim=2,
                )

            if enable_secondary_data is True:
                self.logger.info(f"{self.level3} organize oxygenation")
                if self.oxygenation is None:
                    self.oxygenation = torch.tensor(
                        temp[:, :, :, self.oxygenation_index].astype(np.float32),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    assert self.oxygenation is not None
                    assert self.oxygenation.ndim == temp.ndim
                    assert self.oxygenation.shape[0] == temp.shape[0]
                    assert self.oxygenation.shape[1] == temp.shape[1]
                    assert self.oxygenation.shape[3] == temp.shape[3]
                    self.oxygenation = torch.cat(
                        (
                            self.oxygenation,
                            torch.tensor(
                                temp[:, :, :, self.oxygenation_index].astype(
                                    np.float32
                                ),
                                device=self.device,
                                dtype=torch.float32,
                            ),
                        ),
                        dim=2,
                    )

                if self.volume is None:
                    self.logger.info(f"{self.level3} organize volume")
                    self.volume = torch.tensor(
                        temp[:, :, :, self.volume_index].astype(np.float32),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    assert self.volume is not None
                    assert self.volume.ndim == temp.ndim
                    assert self.volume.shape[0] == temp.shape[0]
                    assert self.volume.shape[1] == temp.shape[1]
                    assert self.volume.shape[3] == temp.shape[3]
                    self.volume = torch.cat(
                        (
                            self.volume,
                            torch.tensor(
                                temp[:, :, :, self.volume_index].astype(np.float32),
                                device=self.device,
                                dtype=torch.float32,
                            ),
                        ),
                        dim=2,
                    )

            part_id += 1
            filename_np = os.path.join(
                self.path,
                f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}.npy",
            )
            filename_meta = os.path.join(
                self.path,
                f"Exp{experiment_id:03d}_Trial{trial_id:03d}_Part{part_id:03d}_meta.txt",
            )

        self.logger.info(f"{self.level3} move axis")
        assert self.acceptor is not None
        assert self.donor is not None
        self.acceptor = self.acceptor.moveaxis(-1, 0)
        self.donor = self.donor.moveaxis(-1, 0)

        if enable_secondary_data is True:
            assert self.oxygenation is not None
            assert self.volume is not None
            self.oxygenation = self.oxygenation.moveaxis(-1, 0)
            self.volume = self.volume.moveaxis(-1, 0)

        if align is True:
            self.logger.info(f"{self.level3} move intra timeseries")
            self._move_intra_timeseries(
                enable_secondary_data=enable_secondary_data,
                start_position_coefficients=start_position_coefficients,
            )
            self.logger.info(f"{self.level3} rotate inter timeseries")
            self._rotate_inter_timeseries(
                enable_secondary_data=enable_secondary_data,
                start_position_coefficients=start_position_coefficients,
            )
            self.logger.info(f"{self.level3} move inter timeseries")
            self._move_inter_timeseries(
                enable_secondary_data=enable_secondary_data,
                start_position_coefficients=start_position_coefficients,
            )

        # reset svd
        self.acceptor_whiten_mean = None
        self.acceptor_whiten_k = None
        self.acceptor_eigenvalues = None

        self.donor_whiten_mean = None
        self.donor_whiten_k = None
        self.donor_eigenvalues = None

        self.oxygenation_whiten_mean = None
        self.oxygenation_whiten_k = None
        self.oxygenation_eigenvalues = None

        self.volume_whiten_mean = None
        self.volume_whiten_k = None
        self.volume_eigenvalues = None

    @torch.no_grad()
    def _find_ref_image_file(self) -> str:
        filename_postfix: str = "Exp*.npy"
        new_list = glob.glob(os.path.join(self.path, filename_postfix))
        new_list = natsort.natsorted(new_list)

        found_name: str | None = None
        for filename in new_list:
            if (filename.find("Trial") != -1) and (filename.find("Part") != -1):
                found_name = filename
                break
        assert found_name is not None

        return found_name

    @torch.no_grad()
    def _calculate_translation(  # start_position_coefficients: OK
        self,
        input: torch.Tensor,
        reference_image: torch.Tensor,
        start_position_coefficients: int = 0,
    ) -> torch.Tensor:
        tvec = torch.zeros((input.shape[0], 2))

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input[start_position_coefficients:, ...]),
            batch_size=self.batch_size,
            shuffle=False,
        )
        start_position: int = 0
        for input_batch in data_loader:
            assert len(input_batch) == 1

            end_position = start_position + input_batch[0].shape[0]

            tvec_temp = self.image_alignment.dry_run_translation(
                input=input_batch[0],
                new_reference_image=reference_image,
            )

            assert tvec_temp is not None

            tvec[start_position:end_position, :] = tvec_temp

            start_position += input_batch[0].shape[0]

        tvec = torch.round(torch.median(tvec, dim=0)[0])
        return tvec

    @torch.no_grad()
    def _calculate_rotation(  # start_position_coefficients: OK
        self,
        input: torch.Tensor,
        reference_image: torch.Tensor,
        start_position_coefficients: int = 0,
    ) -> torch.Tensor:
        angle = torch.zeros((input.shape[0]))

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input[start_position_coefficients:, ...]),
            batch_size=self.batch_size,
            shuffle=False,
        )
        start_position: int = 0
        for input_batch in data_loader:
            assert len(input_batch) == 1

            end_position = start_position + input_batch[0].shape[0]

            angle_temp = self.image_alignment.dry_run_angle(
                input=input_batch[0],
                new_reference_image=reference_image,
            )

            assert angle_temp is not None

            angle[start_position:end_position] = angle_temp

            start_position += input_batch[0].shape[0]

        angle = torch.where(angle >= 180, 360.0 - angle, angle)
        angle = torch.where(angle <= -180, 360.0 + angle, angle)
        angle = torch.median(angle, dim=0)[0]

        return angle

    @torch.no_grad()
    def _move_intra_timeseries(  # start_position_coefficients: OK
        self,
        enable_secondary_data: bool = True,
        start_position_coefficients: int = 0,
    ) -> None:
        # donor_volume
        assert self.donor is not None
        assert self.ref_image_donor is not None
        tvec_donor = self._calculate_translation(
            self.donor,
            self.ref_image_donor,
            start_position_coefficients=start_position_coefficients,
        )

        self.donor = tv.transforms.functional.affine(
            img=self.donor,
            angle=0,
            translate=[tvec_donor[1], tvec_donor[0]],
            scale=1.0,
            shear=0,
            fill=self.fill_value,
        )

        if enable_secondary_data is True:
            assert self.volume is not None
            self.volume = tv.transforms.functional.affine(
                img=self.volume,
                angle=0,
                translate=[tvec_donor[1], tvec_donor[0]],
                scale=1.0,
                shear=0,
                fill=self.fill_value,
            )

        # acceptor_oxy
        assert self.acceptor is not None
        assert self.ref_image_acceptor is not None
        tvec_acceptor = self._calculate_translation(
            self.acceptor,
            self.ref_image_acceptor,
            start_position_coefficients=start_position_coefficients,
        )

        self.acceptor = tv.transforms.functional.affine(
            img=self.acceptor,
            angle=0,
            translate=[tvec_acceptor[1], tvec_acceptor[0]],
            scale=1.0,
            shear=0,
            fill=self.fill_value,
        )
        if enable_secondary_data is True:
            assert self.oxygenation is not None
            self.oxygenation = tv.transforms.functional.affine(
                img=self.oxygenation,
                angle=0,
                translate=[tvec_acceptor[1], tvec_acceptor[0]],
                scale=1.0,
                shear=0,
                fill=self.fill_value,
            )

    @torch.no_grad()
    def _move_inter_timeseries(  # start_position_coefficients: OK
        self,
        enable_secondary_data: bool = True,
        start_position_coefficients: int = 0,
    ) -> None:
        # acceptor_oxy
        assert self.acceptor is not None
        assert self.ref_image_donor is not None
        tvec = self._calculate_translation(
            self.acceptor,
            self.ref_image_donor,
            start_position_coefficients=start_position_coefficients,
        )

        self.acceptor = tv.transforms.functional.affine(
            img=self.acceptor,
            angle=0,
            translate=[tvec[1], tvec[0]],
            scale=1.0,
            shear=0,
            fill=self.fill_value,
        )

        if enable_secondary_data is True:
            assert self.oxygenation is not None
            self.oxygenation = tv.transforms.functional.affine(
                img=self.oxygenation,
                angle=0,
                translate=[tvec[1], tvec[0]],
                scale=1.0,
                shear=0,
                fill=self.fill_value,
            )

    @torch.no_grad()
    def _rotate_inter_timeseries(  # start_position_coefficients: OK
        self,
        enable_secondary_data: bool = True,
        start_position_coefficients: int = 0,
    ) -> None:
        # acceptor_oxy
        assert self.acceptor is not None
        assert self.ref_image_donor is not None
        angle = self._calculate_rotation(
            self.acceptor,
            self.ref_image_donor,
            start_position_coefficients=start_position_coefficients,
        )

        self.acceptor = tv.transforms.functional.affine(
            img=self.acceptor,
            angle=-float(angle),
            translate=[0, 0],
            scale=1.0,
            shear=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            fill=self.fill_value,
        )

        if enable_secondary_data is True:
            assert self.oxygenation is not None
            self.oxygenation = tv.transforms.functional.affine(
                img=self.oxygenation,
                angle=-float(angle),
                translate=[0, 0],
                scale=1.0,
                shear=0,
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                fill=self.fill_value,
            )

    @torch.no_grad()
    def _svd(  # start_position_coefficients: OK
        self,
        input: torch.Tensor,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selection = torch.flatten(
            input[start_position_coefficients:, ...].clone().movedim(0, -1),
            start_dim=0,
            end_dim=1,
        )
        whiten_mean = torch.mean(selection, dim=-1)
        selection -= whiten_mean.unsqueeze(-1)
        whiten_mean = whiten_mean.reshape((input.shape[1], input.shape[2]))

        if lowrank_method is False:
            svd_u, svd_s, _ = torch.linalg.svd(selection, full_matrices=False)
        else:
            svd_u, svd_s, _ = torch.svd_lowrank(selection, q=lowrank_q)

        whiten_k = (
            torch.sign(svd_u[0, :]).unsqueeze(0) * svd_u / (svd_s.unsqueeze(0) + 1e-20)
        )
        whiten_k = whiten_k.reshape((input.shape[1], input.shape[2], svd_s.shape[0]))
        eigenvalues = svd_s

        return whiten_mean, whiten_k, eigenvalues

    @torch.no_grad()
    def _to_remove(  # start_position_coefficients: OK
        self,
        input: torch.Tensor | None,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
        assert input is not None

        id: int = 0
        (
            input_whiten_mean,
            input_whiten_k,
            input_eigenvalues,
        ) = self._svd(
            input,
            lowrank_method=lowrank_method,
            lowrank_q=lowrank_q,
            start_position_coefficients=start_position_coefficients,
        )

        assert input_whiten_mean is not None
        assert input_whiten_k is not None
        assert input_eigenvalues is not None

        eigenvalue = float(input_eigenvalues[id])
        whiten_mean = input_whiten_mean
        whiten_k = input_whiten_k[:, :, 0]

        data = (input - input_whiten_mean.unsqueeze(0)) * input_whiten_k[
            :, :, id
        ].unsqueeze(0)

        input_svd = data.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
        factor = (data * input_svd).sum(dim=0, keepdim=True) / (input_svd**2).sum(
            dim=0, keepdim=True
        )
        to_remove = input_svd * factor
        to_remove /= input_whiten_k[:, :, id].unsqueeze(0) + 1e-20
        to_remove += input_whiten_mean.unsqueeze(0)

        output = input - to_remove

        return output, to_remove, eigenvalue, whiten_mean, whiten_k

    @torch.no_grad()
    def acceptor_svd_remove(  # start_position_coefficients: OK
        self,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        self.acceptor, to_remove, eigenvalue, whiten_mean, whiten_k = self._to_remove(
            input=self.acceptor,
            lowrank_method=lowrank_method,
            lowrank_q=lowrank_q,
            start_position_coefficients=start_position_coefficients,
        )

        return to_remove, eigenvalue, whiten_mean, whiten_k

    @torch.no_grad()
    def donor_svd_remove(  # start_position_coefficients: OK
        self,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        self.donor, to_remove, eigenvalue, whiten_mean, whiten_k = self._to_remove(
            input=self.donor,
            lowrank_method=lowrank_method,
            lowrank_q=lowrank_q,
            start_position_coefficients=start_position_coefficients,
        )

        return to_remove, eigenvalue, whiten_mean, whiten_k

    @torch.no_grad()
    def volume_svd_remove(  # start_position_coefficients: OK
        self,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        self.volume, to_remove, eigenvalue, whiten_mean, whiten_k = self._to_remove(
            input=self.volume,
            lowrank_method=lowrank_method,
            lowrank_q=lowrank_q,
            start_position_coefficients=start_position_coefficients,
        )

        return to_remove, eigenvalue, whiten_mean, whiten_k

    @torch.no_grad()
    def oxygenation_svd_remove(  # start_position_coefficients: OK
        self,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        (
            self.oxygenation,
            to_remove,
            eigenvalue,
            whiten_mean,
            whiten_k,
        ) = self._to_remove(
            input=self.oxygenation,
            lowrank_method=lowrank_method,
            lowrank_q=lowrank_q,
            start_position_coefficients=start_position_coefficients,
        )

        return to_remove, eigenvalue, whiten_mean, whiten_k

    @torch.no_grad()
    def remove_heartbeat(  # start_position_coefficients: OK
        self,
        iterations: int = 2,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        enable_secondary_data: bool = True,
        start_position_coefficients: int = 0,
    ):
        self.acceptor_residuum = None
        self.donor_residuum = None
        self.oxygenation_residuum = None
        self.volume_residuum = None

        for _ in range(0, iterations):
            to_remove, _, _, _ = self.acceptor_svd_remove(
                lowrank_method=lowrank_method,
                lowrank_q=lowrank_q,
                start_position_coefficients=start_position_coefficients,
            )
            if self.acceptor_residuum is None:
                self.acceptor_residuum = to_remove
            else:
                self.acceptor_residuum += to_remove

            to_remove, _, _, _ = self.donor_svd_remove(
                lowrank_method=lowrank_method,
                lowrank_q=lowrank_q,
                start_position_coefficients=start_position_coefficients,
            )
            if self.donor_residuum is None:
                self.donor_residuum = to_remove
            else:
                self.donor_residuum += to_remove

            if enable_secondary_data is True:
                to_remove, _, _, _ = self.volume_svd_remove(
                    lowrank_method=lowrank_method,
                    lowrank_q=lowrank_q,
                    start_position_coefficients=start_position_coefficients,
                )
                if self.volume_residuum is None:
                    self.volume_residuum = to_remove
                else:
                    self.volume_residuum += to_remove

                to_remove, _, _, _ = self.oxygenation_svd_remove(
                    lowrank_method=lowrank_method,
                    lowrank_q=lowrank_q,
                    start_position_coefficients=start_position_coefficients,
                )
                if self.oxygenation_residuum is None:
                    self.oxygenation_residuum = to_remove
                else:
                    self.oxygenation_residuum += to_remove

    @torch.no_grad()
    def remove_mean_data(self, enable_secondary_data: bool = True) -> None:
        assert self.donor is not None
        assert self.acceptor is not None
        self.donor -= self.donor.mean(dim=0, keepdim=True)
        self.acceptor -= self.acceptor.mean(dim=0, keepdim=True)

        if enable_secondary_data is True:
            assert self.volume is not None
            assert self.oxygenation is not None
            self.volume -= self.volume.mean(dim=0, keepdim=True)
            self.oxygenation -= self.oxygenation.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def remove_mean_residuum(self, enable_secondary_data: bool = True) -> None:
        assert self.donor_residuum is not None
        assert self.acceptor_residuum is not None
        self.donor_residuum -= self.donor_residuum.mean(dim=0, keepdim=True)
        self.acceptor_residuum -= self.acceptor_residuum.mean(dim=0, keepdim=True)

        if enable_secondary_data is True:
            assert self.volume_residuum is not None
            assert self.oxygenation_residuum is not None
            self.volume_residuum -= self.volume_residuum.mean(dim=0, keepdim=True)
            self.oxygenation_residuum -= self.oxygenation_residuum.mean(
                dim=0, keepdim=True
            )

    @torch.no_grad()
    def _calculate_linear_trend_data(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 3
        time_beam: torch.Tensor = torch.arange(
            0, input.shape[0], dtype=torch.float32, device=self.device
        )
        time_beam -= time_beam.mean()
        input_mean = input.mean(dim=0, keepdim=True)
        factor = (time_beam.unsqueeze(-1).unsqueeze(-1) * (input - input_mean)).sum(
            dim=0, keepdim=True
        ) / (time_beam**2).sum(dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)

        output = factor * time_beam.unsqueeze(-1).unsqueeze(-1) + input_mean

        return output

    @torch.no_grad()
    def remove_linear_trend_data(self, enable_secondary_data: bool = True) -> None:
        assert self.donor is not None
        assert self.acceptor is not None
        self.donor -= self._calculate_linear_trend_data(self.donor)
        self.acceptor -= self._calculate_linear_trend_data(self.acceptor)

        if enable_secondary_data is True:
            assert self.volume is not None
            assert self.oxygenation is not None
            self.volume -= self._calculate_linear_trend_data(self.volume)
            self.oxygenation -= self._calculate_linear_trend_data(self.oxygenation)

    @torch.no_grad()
    def remove_linear_trend_residuum(
        self,
        enable_secondary_data: bool = True,
    ) -> None:
        assert self.donor_residuum is not None
        assert self.acceptor_residuum is not None

        self.donor_residuum -= self._calculate_linear_trend_data(self.donor_residuum)
        self.acceptor_residuum -= self._calculate_linear_trend_data(
            self.acceptor_residuum
        )

        if enable_secondary_data is True:
            assert self.volume_residuum is not None
            assert self.oxygenation_residuum is not None
            self.volume_residuum -= self._calculate_linear_trend_data(
                self.volume_residuum
            )
            self.oxygenation_residuum -= self._calculate_linear_trend_data(
                self.oxygenation_residuum
            )

    @torch.no_grad()
    def frame_shift(
        self,
        enable_secondary_data: bool = True,
    ):
        assert self.donor is not None
        assert self.acceptor is not None
        self.donor = self.donor[1:, :, :]
        self.acceptor = self.acceptor[1:, :, :]

        if enable_secondary_data is True:
            assert self.volume is not None
            assert self.oxygenation is not None
            self.volume = (self.volume[1:, :, :] + self.volume[:-1, :, :]) / 2.0
            self.oxygenation = (
                self.oxygenation[1:, :, :] + self.oxygenation[:-1, :, :]
            ) / 2.0

        if self.donor_residuum is not None:
            self.donor_residuum = self.donor_residuum[1:, :, :]

        if self.acceptor_residuum is not None:
            self.acceptor_residuum = self.acceptor_residuum[1:, :, :]

        if enable_secondary_data is True:
            if self.volume_residuum is not None:
                self.volume_residuum = (
                    self.volume_residuum[1:, :, :] + self.volume_residuum[:-1, :, :]
                ) / 2.0

            if self.oxygenation_residuum is not None:
                self.oxygenation_residuum = (
                    self.oxygenation_residuum[1:, :, :]
                    + self.oxygenation_residuum[:-1, :, :]
                ) / 2.0

    @torch.no_grad()
    def cleaned_load_data(
        self,
        experiment_id: int,
        trial_id: int,
        align: bool = True,
        iterations: int = 1,
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        remove_heartbeat: bool = True,
        remove_mean: bool = True,
        remove_linear: bool = True,
        remove_heartbeat_mean: bool = False,
        remove_heartbeat_linear: bool = False,
        bin_size: int = 4,
        do_frame_shift: bool = True,
        enable_secondary_data: bool = True,
        mmap_mode: bool = True,
        initital_mask: torch.Tensor | None = None,
        start_position_coefficients: int = 0,
    ) -> None:
        self.logger.info(f"{self.level2} start load_data")
        self.load_data(
            experiment_id=experiment_id,
            trial_id=trial_id,
            align=align,
            enable_secondary_data=enable_secondary_data,
            mmap_mode=mmap_mode,
            start_position_coefficients=start_position_coefficients,
        )
        assert self.donor is not None
        assert self.acceptor is not None

        if bin_size > 1:
            self.logger.info(f"{self.level2} spatial pooling")
            pool = torch.nn.AvgPool2d((bin_size, bin_size), stride=(bin_size, bin_size))
            self.donor = pool(self.donor)
            self.acceptor = pool(self.acceptor)
            if enable_secondary_data is True:
                assert self.volume is not None
                assert self.oxygenation is not None
                self.volume = pool(self.volume)
                self.oxygenation = pool(self.oxygenation)

        if initital_mask is not None:
            self.logger.info(f"{self.level2} initial mask is applied on the data")
            assert self.acceptor is not None
            assert self.donor is not None
            assert initital_mask.ndim == 2
            assert initital_mask.shape[0] == self.donor.shape[1]
            assert initital_mask.shape[1] == self.donor.shape[2]

            self.acceptor *= initital_mask.unsqueeze(0)
            self.donor *= initital_mask.unsqueeze(0)

            if enable_secondary_data is True:
                assert self.oxygenation is not None
                assert self.volume is not None
                self.oxygenation *= initital_mask.unsqueeze(0)
                self.volume *= initital_mask.unsqueeze(0)

        if remove_heartbeat is True:
            self.logger.info(f"{self.level2} remove the heart beat via SVD")
            self.remove_heartbeat(
                iterations=iterations,
                lowrank_method=lowrank_method,
                lowrank_q=lowrank_q,
                enable_secondary_data=enable_secondary_data,
                start_position_coefficients=start_position_coefficients,
            )

        if remove_mean is True:
            self.logger.info(f"{self.level2} remove mean")
            self.remove_mean_data(enable_secondary_data=enable_secondary_data)

        if remove_linear is True:
            self.logger.info(f"{self.level2} remove linear trends")
            self.remove_linear_trend_data(enable_secondary_data=enable_secondary_data)

        if remove_heartbeat is True:
            if remove_heartbeat_mean is True:
                self.logger.info(f"{self.level2} remove mean (heart beat signal)")
                self.remove_mean_residuum(enable_secondary_data=enable_secondary_data)
            if remove_heartbeat_linear is True:
                self.logger.info(
                    f"{self.level2} remove linear trends (heart beat signal)"
                )
                self.remove_linear_trend_residuum(
                    enable_secondary_data=enable_secondary_data
                )

        if do_frame_shift is True:
            self.logger.info(f"{self.level2} frame shift")
            self.frame_shift(enable_secondary_data=enable_secondary_data)

    @torch.no_grad()
    def remove_other_signals(  # start_position_coefficients: OK
        self,
        start_position_coefficients: int = 0,
        match_iterations: int = 25,
        export_parameters: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        float,
        float,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        assert self.acceptor is not None
        assert self.donor is not None
        assert self.oxygenation is not None
        assert self.volume is not None

        index_full_dataset = torch.arange(
            0, self.acceptor.shape[1], device=self.device, dtype=torch.int64
        )

        result_a: torch.Tensor = torch.zeros_like(self.acceptor)
        result_d: torch.Tensor = torch.zeros_like(self.donor)

        max_scale_value_a = 0.0
        initial_scale_value_a = 0.0
        max_scale_value_d = 0.0
        initial_scale_value_d = 0.0

        parameter_a: torch.Tensor | None = None
        parameter_d: torch.Tensor | None = None

        for chunk in self._chunk_iterator(index_full_dataset, self.filtfilt_chuck_size):
            a: torch.Tensor = self.acceptor[:, chunk, :].detach().clone()
            d: torch.Tensor = self.donor[:, chunk, :].detach().clone()

            o: torch.Tensor = self.oxygenation[:, chunk, :].detach().clone()
            v: torch.Tensor = self.volume[:, chunk, :].detach().clone()

            a_mean = a[start_position_coefficients:, ...].mean(dim=0, keepdim=True)
            a_mean_full = a.mean(dim=0, keepdim=True)
            a -= a_mean_full
            a_correction = a_mean - a_mean_full

            d_mean = d[start_position_coefficients:, ...].mean(dim=0, keepdim=True)
            d_mean_full = d.mean(dim=0, keepdim=True)
            d -= d_mean_full
            d_correction = d_mean - d_mean_full

            o_mean = o[start_position_coefficients:, ...].mean(dim=0, keepdim=True)
            o_mean_full = o.mean(dim=0, keepdim=True)
            o -= o_mean
            o_correction = o_mean - o_mean_full
            o_norm = 1.0 / (
                (o[start_position_coefficients:, ...] ** 2).sum(dim=0) + 1e-20
            )

            v_mean = v[start_position_coefficients:, ...].mean(dim=0, keepdim=True)
            v_mean_full = v.mean(dim=0, keepdim=True)
            v -= v_mean
            v_correction = v_mean - v_mean_full
            v_norm = 1.0 / (
                (v[start_position_coefficients:, ...] ** 2).sum(dim=0) + 1e-20
            )

            linear: torch.Tensor = (
                torch.arange(0, o.shape[0], device=self.device, dtype=torch.float32)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            l_mean = linear[start_position_coefficients:, ...].mean(dim=0, keepdim=True)
            l_mean_full = linear.mean(dim=0, keepdim=True)
            linear -= l_mean
            l_correction = l_mean - l_mean_full
            linear_norm = 1.0 / (
                (linear[start_position_coefficients:, ...] ** 2).sum(dim=0) + 1e-20
            )
            linear = torch.tile(linear, (1, o.shape[1], o.shape[2]))
            linear_norm = torch.tile(linear_norm, (o.shape[1], o.shape[2]))
            l_correction = torch.tile(l_correction, (1, o.shape[1], o.shape[2]))

            data = torch.cat(
                (linear.unsqueeze(-1), o.unsqueeze(-1), v.unsqueeze(-1)), dim=-1
            )
            del linear
            del o
            del v

            data_mean_correction = torch.cat(
                (
                    l_correction.unsqueeze(-1),
                    o_correction.unsqueeze(-1),
                    v_correction.unsqueeze(-1),
                ),
                dim=-1,
            )

            data_norm = torch.cat(
                (linear_norm.unsqueeze(-1), o_norm.unsqueeze(-1), v_norm.unsqueeze(-1)),
                dim=-1,
            )
            del linear_norm
            del o_norm
            del v_norm

            if export_parameters is True:
                parameter_a_temp: torch.Tensor | None = torch.zeros_like(data_norm)
                parameter_d_temp: torch.Tensor | None = torch.zeros_like(data_norm)
            else:
                parameter_a_temp = None
                parameter_d_temp = None

            for mode_a in [True, False]:
                if mode_a is True:
                    result = a.detach().clone()
                    result_mean_correct = a_correction

                else:
                    result = d.detach().clone()
                    result_mean_correct = d_correction

                for i in range(0, match_iterations):
                    scale = (
                        (
                            data[start_position_coefficients:, ...]
                            * (
                                result[start_position_coefficients:, ...]
                                + result_mean_correct
                            ).unsqueeze(-1)
                        ).sum(dim=0)
                    ) * data_norm

                    idx = torch.abs(scale).argmax(dim=-1)
                    scale = torch.gather(scale, -1, idx.unsqueeze(-1)).squeeze(-1)

                    idx_3d = torch.tile(idx.unsqueeze(0), (data.shape[0], 1, 1))
                    data_selected = torch.gather(
                        (data - data_mean_correction), -1, idx_3d.unsqueeze(-1)
                    ).squeeze(-1)

                    result -= data_selected * scale.unsqueeze(0)

                    if mode_a is True:
                        if i == 0:
                            initial_scale_value_a = max(
                                [max_scale_value_a, float(scale.max())]
                            )
                        if parameter_a_temp is not None:
                            parameter_a_temp.scatter_add_(
                                -1, idx.unsqueeze(-1), scale.unsqueeze(-1)
                            )

                    else:
                        if i == 0:
                            initial_scale_value_d = max(
                                [max_scale_value_d, float(scale.max())]
                            )
                        if parameter_d_temp is not None:
                            parameter_d_temp.scatter_add_(
                                -1, idx.unsqueeze(-1), scale.unsqueeze(-1)
                            )

                if mode_a is True:
                    result_a[:, chunk, :] = result.detach().clone()
                    max_scale_value_a = max([max_scale_value_a, float(scale.max())])
                    if parameter_a_temp is not None:
                        parameter_a_temp = torch.cat(
                            (parameter_a_temp, a_mean_full.squeeze(0).unsqueeze(-1)),
                            dim=-1,
                        )
                else:
                    result_d[:, chunk, :] = result.detach().clone()
                    max_scale_value_d = max([max_scale_value_d, float(scale.max())])
                    if parameter_d_temp is not None:
                        parameter_d_temp = torch.cat(
                            (parameter_d_temp, d_mean_full.squeeze(0).unsqueeze(-1)),
                            dim=-1,
                        )
            if export_parameters is True:
                if (parameter_a is None) and (parameter_a_temp is not None):
                    parameter_a = torch.zeros(
                        (
                            self.acceptor.shape[1],
                            parameter_a_temp.shape[1],
                            parameter_a_temp.shape[2],
                        ),
                        device=self.device,
                        dtype=torch.float32,
                    )
                if (parameter_a is not None) and (parameter_a_temp is not None):
                    parameter_a[chunk, ...] = parameter_a_temp

                if (parameter_d is None) and (parameter_d_temp is not None):
                    parameter_d = torch.zeros(
                        (
                            self.acceptor.shape[1],
                            parameter_d_temp.shape[1],
                            parameter_d_temp.shape[2],
                        ),
                        device=self.device,
                        dtype=torch.float32,
                    )
                if (parameter_d is not None) and (parameter_d_temp is not None):
                    parameter_d[chunk, ...] = parameter_d_temp

        self.logger.info(
            f"{self.level3} acceptor -- Progression scale: {initial_scale_value_a} -> {max_scale_value_a}"
        )
        self.logger.info(
            f"{self.level3} donor -- Progression scale: {initial_scale_value_d} -> {max_scale_value_d}"
        )
        return (
            result_a,
            result_d,
            max_scale_value_a,
            initial_scale_value_a,
            max_scale_value_d,
            initial_scale_value_d,
            parameter_a,
            parameter_d,
        )

    @torch.no_grad()
    def _filtfilt(
        self,
        input: torch.Tensor,
        butter_a: torch.Tensor,
        butter_b: torch.Tensor,
    ) -> torch.Tensor:
        assert butter_a.ndim == 1
        assert butter_b.ndim == 1
        assert butter_a.shape[0] == butter_b.shape[0]

        process_data: torch.Tensor = input.movedim(0, -1).detach().clone()

        padding_length = 12 * int(butter_a.shape[0])
        left_padding = 2 * process_data[..., 0].unsqueeze(-1) - process_data[
            ..., 1 : padding_length + 1
        ].flip(-1)
        right_padding = 2 * process_data[..., -1].unsqueeze(-1) - process_data[
            ..., -(padding_length + 1) : -1
        ].flip(-1)
        process_data_padded = torch.cat(
            (left_padding, process_data, right_padding), dim=-1
        )

        output = ta.functional.filtfilt(
            process_data_padded.unsqueeze(0), butter_a, butter_b, clamp=False
        ).squeeze(0)
        output = output[..., padding_length:-padding_length].movedim(-1, 0)
        return output

    @torch.no_grad()
    def _butter_bandpass(
        self, low_frequency: float = 5, high_frequency: float = 15, fs: float = 100.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import scipy

        butter_b_np, butter_a_np = scipy.signal.butter(
            4, [low_frequency, high_frequency], btype="bandpass", output="ba", fs=fs
        )
        butter_a = torch.tensor(butter_a_np, device=self.device, dtype=torch.float32)
        butter_b = torch.tensor(butter_b_np, device=self.device, dtype=torch.float32)
        return butter_a, butter_b

    @torch.no_grad()
    def _chunk_iterator(self, array: torch.Tensor, chunk_size: int):
        for i in range(0, array.shape[0], chunk_size):
            yield array[i : i + chunk_size]

    @torch.no_grad()
    def heartbeat_scale(  # start_position_coefficients: OK
        self,
        low_frequency: float = 5,
        high_frequency: float = 15,
        fs: float = 100.0,
        apply_to_data: bool = False,
        threshold: float | None = 0.5,
        start_position_coefficients: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.donor_residuum is not None
        assert self.acceptor_residuum is not None

        butter_a, butter_b = self._butter_bandpass(
            low_frequency=low_frequency, high_frequency=high_frequency, fs=fs
        )

        butter_a, butter_b = self._butter_bandpass(
            low_frequency=low_frequency, high_frequency=high_frequency, fs=100.0
        )
        self.logger.info(f"{self.level3} apply bandpass donor_residuum (filtfilt)")

        index_full_dataset: torch.Tensor = torch.arange(
            0, self.donor_residuum.shape[1], device=self.device, dtype=torch.int64
        )

        hb_d = torch.zeros_like(self.donor_residuum[start_position_coefficients:, ...])
        for chunk in self._chunk_iterator(index_full_dataset, self.filtfilt_chuck_size):
            temp_filtfilt = self._filtfilt(
                self.donor_residuum[start_position_coefficients:, chunk, :],
                butter_a=butter_a,
                butter_b=butter_b,
            )
            hb_d[:, chunk, :] = temp_filtfilt

        hb_d = hb_d[start_position:, ...]
        hb_d -= hb_d.mean(dim=0, keepdim=True)

        self.logger.info(f"{self.level3} apply bandpass acceptor_residuum (filtfilt)")

        index_full_dataset = torch.arange(
            0, self.acceptor_residuum.shape[1], device=self.device, dtype=torch.int64
        )
        hb_a = torch.zeros_like(self.donor_residuum[start_position_coefficients:, ...])
        for chunk in self._chunk_iterator(index_full_dataset, self.filtfilt_chuck_size):
            temp_filtfilt = self._filtfilt(
                self.acceptor_residuum[start_position_coefficients:, chunk, :],
                butter_a=butter_a,
                butter_b=butter_b,
            )
            hb_a[:, chunk, :] = temp_filtfilt

        hb_a = hb_a[start_position:, ...]
        hb_a -= hb_a.mean(dim=0, keepdim=True)

        scale = (hb_a * hb_d).sum(dim=0) / (hb_a**2).sum(dim=0)

        heartbeat_a = torch.sqrt(scale)
        heartbeat_d = 1.0 / (heartbeat_a + 1e-20)

        if apply_to_data is True:
            if self.donor is not None:
                self.donor *= heartbeat_d.unsqueeze(0)
            if self.volume is not None:
                self.volume *= heartbeat_d.unsqueeze(0)
            if self.acceptor is not None:
                self.acceptor *= heartbeat_a.unsqueeze(0)
            if self.oxygenation is not None:
                self.oxygenation *= heartbeat_a.unsqueeze(0)

        if threshold is not None:
            self.logger.info(f"{self.level3} calculate mask")
            mask = torch.where(hb_d.std(dim=0) > threshold, 1.0, 0.0) * torch.where(
                hb_a.std(dim=0) > threshold, 1.0, 0.0
            )
        else:
            mask = None

        return heartbeat_a, heartbeat_d, mask

    @torch.no_grad()
    def measure_heartbeat_frequency(  # start_position_coefficients: OK
        self,
        low_frequency: float = 5,
        high_frequency: float = 15,
        fs: float = 100.0,
        use_input_source: str = "volume",
        start_position_coefficients: int = 0,
        half_width_frequency_window: float = 3.0,  # Hz (on side )
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_input_source == "donor":
            assert self.donor is not None
            hb: torch.Tensor = self.donor[start_position_coefficients:, ...]

        elif use_input_source == "acceptor":
            assert self.acceptor is not None
            hb = self.acceptor[start_position_coefficients:, ...]

        elif use_input_source == "volume":
            assert self.volume is not None
            hb = self.volume[start_position_coefficients:, ...]

        else:
            assert self.oxygenation is not None
            hb = self.oxygenation[start_position_coefficients:, ...]

        frequency_axis: torch.Tensor = (
            torch.fft.rfftfreq(hb.shape[0]).to(device=self.device) * fs
        )

        delta_idx = int(
            math.ceil(
                half_width_frequency_window
                / (float(frequency_axis[1]) - float(frequency_axis[0]))
            )
        )

        idx: torch.Tensor = torch.where(
            (frequency_axis >= low_frequency) * (frequency_axis <= high_frequency)
        )[0]

        power_hb: torch.Tensor = torch.abs(torch.fft.rfft(hb, dim=0)) ** 2
        power_hb = power_hb[idx, :, :].argmax(dim=0) + idx[0]
        power_hb_low = power_hb - delta_idx
        power_hb_low = power_hb_low.clamp(min=0)
        power_hb_high = power_hb + delta_idx
        power_hb_high = power_hb_high.clamp(max=frequency_axis.shape[0])

        return power_hb_low, power_hb_high, frequency_axis

    @torch.no_grad()
    def measure_heartbeat_power(  # start_position_coefficients: OK
        self,
        use_input_source: str = "donor",
        start_position_coefficients: int = 0,
        power_hb_low: torch.Tensor | None = None,
        power_hb_high: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if use_input_source == "donor":
            assert self.donor is not None
            hb: torch.Tensor = self.donor[start_position_coefficients:, ...]

        elif use_input_source == "acceptor":
            assert self.acceptor is not None
            hb = self.acceptor[start_position_coefficients:, ...]

        elif use_input_source == "volume":
            assert self.volume is not None
            hb = self.volume[start_position_coefficients:, ...]

        else:
            assert self.oxygenation is not None
            hb = self.oxygenation[start_position_coefficients:, ...]

        counter: torch.Tensor = torch.zeros(
            (hb.shape[1], hb.shape[2]),
            dtype=hb.dtype,
            device=self.device,
        )

        index_full_dataset = torch.arange(
            0, hb.shape[1], device=self.device, dtype=torch.int64
        )

        power_hb: torch.Tensor | None = None
        for chunk in self._chunk_iterator(index_full_dataset, self.filtfilt_chuck_size):
            temp_power = torch.abs(torch.fft.rfft(hb[:, chunk, :], dim=0)) ** 2
            if power_hb is None:
                power_hb = torch.zeros(
                    (temp_power.shape[0], hb.shape[1], temp_power.shape[2]),
                    dtype=temp_power.dtype,
                    device=temp_power.device,
                )
            assert power_hb is not None
            power_hb[:, chunk, :] = temp_power

        assert power_hb is not None
        for pos in range(0, power_hb.shape[0]):
            pos_torch = torch.tensor(pos, dtype=torch.int64, device=self.device)
            slice_temp = (
                (pos_torch >= power_hb_low) * (pos_torch < power_hb_high)
            ).type(dtype=power_hb.dtype)
            power_hb[pos, ...] *= slice_temp
            counter += slice_temp
        power_hb = power_hb.sum(dim=0) / counter

        return power_hb

    @torch.no_grad()
    def automatic_load(  # start_position_coefficients: OK
        self,
        experiment_id: int = 1,
        trial_id: int = 1,
        start_position: int = 0,
        start_position_coefficients: int = 100,
        fs: float = 100.0,
        use_regression: bool | None = None,
        # Heartbeat
        remove_heartbeat: bool = False,  # i.e. use SVD
        low_frequency: float = 5,  # Hz Butter Bandpass Heartbeat
        high_frequency: float = 15,  # Hz Butter Bandpass Heartbeat
        threshold: float | None = 0.5,  # For the mask
        # Extra exposed parameters:
        align: bool = True,
        iterations: int = 1,  # SVD iterations: Do not touch! Keep at 1
        lowrank_method: bool = True,
        lowrank_q: int = 6,
        remove_heartbeat_mean: bool = False,
        remove_heartbeat_linear: bool = False,
        bin_size: int = 4,
        do_frame_shift: bool = True,
        half_width_frequency_window: float = 3.0,  # Hz (on side ) measure_heartbeat_frequency
        mmap_mode: bool = True,
        initital_mask_name: str | None = None,
        initital_mask_update: bool = True,
        initital_mask_roi: bool = True,
        gaussian_blur_kernel_size: int | None = None,
        gaussian_blur_sigma: float = 1.0,
        bin_size_post: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.logger.info(f"{self.level0} start automatic_load")

        if use_regression is None:
            use_regression = not remove_heartbeat

        initital_mask: torch.Tensor | None = None

        if (initital_mask_name is not None) and os.path.isfile(
            initital_mask_name
        ) is True:
            initital_mask = torch.tensor(
                np.load(initital_mask_name), device=self.device, dtype=torch.float32
            )
            self.logger.info(f"{self.level1} try to load previous mask: found")
        else:
            self.logger.info(f"{self.level1} try to load previous mask: NOT found")

        self.logger.info(f"{self.level1} start cleaned_load_data")
        self.cleaned_load_data(
            experiment_id=experiment_id,
            trial_id=trial_id,
            remove_heartbeat=remove_heartbeat,
            remove_mean=not use_regression,
            remove_linear=not use_regression,
            enable_secondary_data=use_regression,
            align=align,
            iterations=iterations,
            lowrank_method=lowrank_method,
            lowrank_q=lowrank_q,
            remove_heartbeat_mean=remove_heartbeat_mean,
            remove_heartbeat_linear=remove_heartbeat_linear,
            bin_size=bin_size,
            do_frame_shift=do_frame_shift,
            mmap_mode=mmap_mode,
            initital_mask=initital_mask,
            start_position_coefficients=start_position_coefficients,
        )

        heartbeat_a: torch.Tensor | None = None
        heartbeat_d: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        power_hb_low: torch.Tensor | None = None
        power_hb_high: torch.Tensor | None = None

        if remove_heartbeat is True:
            self.logger.info(f"{self.level1} remove heart beat (heartbeat_scale)")
            heartbeat_a, heartbeat_d, mask = self.heartbeat_scale(
                low_frequency=low_frequency,
                high_frequency=high_frequency,
                fs=fs,
                apply_to_data=False,
                threshold=threshold,
                start_position_coefficients=start_position_coefficients,
            )
        else:
            self.logger.info(
                f"{self.level1} measure heart rate (measure_heartbeat_frequency)"
            )
            (
                power_hb_low,
                power_hb_high,
                _,
            ) = self.measure_heartbeat_frequency(
                low_frequency=low_frequency,
                high_frequency=high_frequency,
                fs=fs,
                use_input_source="volume",
                start_position_coefficients=start_position_coefficients,
                half_width_frequency_window=half_width_frequency_window,
            )

        if use_regression is True:
            self.logger.info(f"{self.level1} use regression")
            (
                result_a,
                result_d,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.remove_other_signals(
                start_position_coefficients=start_position_coefficients,
                match_iterations=25,
                export_parameters=False,
            )
            result_a = result_a[start_position:, ...]
            result_d = result_d[start_position:, ...]
        else:
            self.logger.info(f"{self.level1} don't use regression")
            assert self.acceptor is not None
            assert self.donor is not None
            result_a = self.acceptor[start_position:, ...].clone()
            result_d = self.donor[start_position:, ...].clone()

        if mask is not None:
            result_a *= mask.unsqueeze(0)
            result_d *= mask.unsqueeze(0)

        if remove_heartbeat is False:
            self.logger.info(
                f"{self.level1} donor: measure heart beat spectral power (measure_heartbeat_power)"
            )
            temp_d = self.measure_heartbeat_power(
                use_input_source="donor",
                start_position_coefficients=start_position_coefficients,
                power_hb_low=power_hb_low,
                power_hb_high=power_hb_high,
            )
            self.logger.info(
                f"{self.level1} acceptor: measure heart beat spectral power (measure_heartbeat_power)"
            )
            temp_a = self.measure_heartbeat_power(
                use_input_source="acceptor",
                start_position_coefficients=start_position_coefficients,
                power_hb_low=power_hb_low,
                power_hb_high=power_hb_high,
            )
            scale = temp_d / (temp_a + 1e-20)

            heartbeat_a = torch.sqrt(scale)
            heartbeat_d = 1.0 / (heartbeat_a + 1e-20)

        self.logger.info(f"{self.level1} scale acceptor and donor signals")
        if heartbeat_a is not None:
            result_a *= heartbeat_a.unsqueeze(0)
        if heartbeat_d is not None:
            result_d *= heartbeat_d.unsqueeze(0)

        if mask is not None:
            if initital_mask_update is True:
                self.logger.info(f"{self.level1} update inital mask")
                if initital_mask is None:
                    initital_mask = mask.clone()
                else:
                    initital_mask *= mask

                if (initital_mask_roi is True) and (initital_mask is not None):
                    self.logger.info(f"{self.level1} enter roi mask drawing modus")
                    yes_choices = ["yes", "y"]
                    contiue_roi: bool = True

                    image: np.ndarray = (result_a - result_d)[0, ...].cpu().numpy()
                    image[initital_mask.cpu().numpy() == 0] = float("NaN")

                    while contiue_roi is True:
                        user_input = input(
                            "Mask: Do you want to remove more pixel (yes/no)? "
                        )

                        if user_input.lower() in yes_choices:
                            plt.imshow(image, cmap="hot")
                            plt.title("Select a region for removal")

                            temp_roi = RoiPoly(color="g")
                            plt.show()

                            if len(temp_roi.x) > 0:
                                new_mask = temp_roi.get_mask(image)
                                new_mask_np = new_mask.astype(np.float32)
                                new_mask_torch = torch.tensor(
                                    new_mask_np,
                                    device=self.device,
                                    dtype=torch.float32,
                                )

                                plt.imshow(image, cmap="hot")
                                temp_roi.display_roi()
                                plt.title("Selected region for removal")
                                print("Please close figure when ready...")
                                plt.show()
                                user_input = input(
                                    "Mask: Remove these pixel (yes/no)? "
                                )

                                if user_input.lower() in yes_choices:
                                    initital_mask *= 1.0 - new_mask_torch
                                    image[new_mask] = float("NaN")

                        else:
                            contiue_roi = False

                if initital_mask_name is not None:
                    self.logger.info(f"{self.level1} save mask")
                    np.save(initital_mask_name, initital_mask.cpu().numpy())

        self.logger.info(f"{self.level0} end automatic_load")

        result = result_a - result_d

        if (gaussian_blur_kernel_size is not None) and (gaussian_blur_kernel_size > 0):
            gaussian_blur = tv.transforms.GaussianBlur(
                kernel_size=[gaussian_blur_kernel_size, gaussian_blur_kernel_size],
                sigma=gaussian_blur_sigma,
            )
            result = gaussian_blur(result)

        if (bin_size_post is not None) and (bin_size_post > 1):
            pool = torch.nn.AvgPool2d(
                (bin_size_post, bin_size_post), stride=(bin_size_post, bin_size_post)
            )
            result = pool(result)

            if mask is not None:
                mask = (
                    (pool(mask.unsqueeze(0)) > 0).type(dtype=torch.float32).squeeze(0)
                )

        return result, mask


if __name__ == "__main__":
    from Anime import Anime

    # path: str = "/data_1/robert/2021-05-05/M3852M/raw"
    path: str = "/data_1/robert/2021-05-21/M3852M/raw"
    initital_mask_name: str | None = None
    initital_mask_update: bool = True
    initital_mask_roi: bool = False  # default: True

    experiment_id: int = 2
    trial_id: int = 180
    start_position: int = 0
    start_position_coefficients: int = 100
    remove_heartbeat: bool = True  # i.e. use SVD
    bin_size: int = 4

    example_position_x: int = 280
    example_position_y: int = 440

    display_logging_messages: bool = False
    save_logging_messages: bool = False

    show_example_timeseries: bool = True
    play_movie: bool = True

    # Post data processing modifiations
    gaussian_blur_kernel_size: int | None = 3
    gaussian_blur_sigma: float = 1.0
    bin_size_post: int | None = None

    # ------------------------
    example_position_x = example_position_x // bin_size
    example_position_y = example_position_y // bin_size
    if bin_size_post is not None:
        example_position_x = example_position_x // bin_size_post
        example_position_y = example_position_y // bin_size_post

    torch_device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    af = DataContainer(
        path=path,
        device=torch_device,
        display_logging_messages=display_logging_messages,
        save_logging_messages=save_logging_messages,
    )
    result, mask = af.automatic_load(
        experiment_id=experiment_id,
        trial_id=trial_id,
        start_position=start_position,
        remove_heartbeat=remove_heartbeat,  # i.e. use SVD
        bin_size=bin_size,
        initital_mask_name=initital_mask_name,
        initital_mask_update=initital_mask_update,
        initital_mask_roi=initital_mask_roi,
        start_position_coefficients=start_position_coefficients,
        gaussian_blur_kernel_size=gaussian_blur_kernel_size,
        gaussian_blur_sigma=gaussian_blur_sigma,
        bin_size_post=bin_size_post,
    )

    if show_example_timeseries is True:
        plt.plot(result[:, example_position_x, example_position_y].cpu())
        plt.show()

    if play_movie is True:
        ani = Anime()
        ani.show(result, mask=mask, vmin_scale=0.5, vmax_scale=0.5)
