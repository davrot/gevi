import torch
import torchvision as tv

# The source code is based on:
# https://github.com/matejak/imreg_dft

# The original LICENSE:
# Copyright (c) 2014, Matěj Týč
# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the {organization} nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class ImageAlignment(torch.nn.Module):
    device: torch.device
    default_dtype: torch.dtype
    excess_const: float = 1.1
    exponent: str = "inf"
    success: torch.Tensor | None = None

    # The factor that detmines how many
    # sub-pixel we will shift
    scale_factor: int = 4

    reference_image: torch.Tensor | None = None

    last_scale: torch.Tensor | None = None
    last_angle: torch.Tensor | None = None
    last_tvec: torch.Tensor | None = None

    # Cache
    image_reference_dft: torch.Tensor | None = None
    filt: torch.Tensor
    pcorr_shape: torch.Tensor
    log_base: torch.Tensor
    image_reference_logp: torch.Tensor

    def __init__(
        self,
        device: torch.device | None = None,
        default_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        assert device is not None
        assert default_dtype is not None
        self.device = device
        self.default_dtype = default_dtype

    def set_new_reference_image(self, new_reference_image: torch.Tensor | None = None):
        assert new_reference_image is not None
        assert new_reference_image.ndim == 2
        self.reference_image = (
            new_reference_image.detach()
            .clone()
            .to(device=self.device)
            .type(dtype=self.default_dtype)
        )
        self.image_reference_dft = None

    def forward(
        self, input: torch.Tensor, new_reference_image: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert input.ndim == 3

        if new_reference_image is not None:
            self.set_new_reference_image(new_reference_image)

        assert self.reference_image is not None
        assert self.reference_image.ndim == 2
        assert input.shape[-2] == self.reference_image.shape[-2]
        assert input.shape[-1] == self.reference_image.shape[-1]

        self.last_scale, self.last_angle, self.last_tvec, output = self.similarity(
            self.reference_image,
            input.to(device=self.device).type(dtype=self.default_dtype),
        )

        return output

    def dry_run(
        self, input: torch.Tensor, new_reference_image: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        assert input.ndim == 3

        if new_reference_image is not None:
            self.set_new_reference_image(new_reference_image)

        assert self.reference_image is not None
        assert self.reference_image.ndim == 2
        assert input.shape[-2] == self.reference_image.shape[-2]
        assert input.shape[-1] == self.reference_image.shape[-1]

        images_todo = input.to(device=self.device).type(dtype=self.default_dtype)
        image_reference = self.reference_image

        assert image_reference.ndim == 2
        assert images_todo.ndim == 3

        bgval: torch.Tensor = self.get_borderval(img=images_todo, radius=5)

        self.last_scale, self.last_angle, self.last_tvec = self._similarity(
            image_reference,
            images_todo,
            bgval,
        )

        return self.last_scale, self.last_angle, self.last_tvec

    def dry_run_translation(
        self, input: torch.Tensor, new_reference_image: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert input.ndim == 3

        if new_reference_image is not None:
            self.set_new_reference_image(new_reference_image)

        assert self.reference_image is not None
        assert self.reference_image.ndim == 2
        assert input.shape[-2] == self.reference_image.shape[-2]
        assert input.shape[-1] == self.reference_image.shape[-1]

        images_todo = input.to(device=self.device).type(dtype=self.default_dtype)
        image_reference = self.reference_image

        assert image_reference.ndim == 2
        assert images_todo.ndim == 3

        tvec, _ = self._translation(image_reference, images_todo)

        return tvec

    # ---------------

    def dry_run_angle(
        self,
        input: torch.Tensor,
        new_reference_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input.ndim == 3

        if new_reference_image is not None:
            self.set_new_reference_image(new_reference_image)

        constraints_dynamic_angle_0: torch.Tensor = torch.zeros(
            (input.shape[0]), dtype=self.default_dtype, device=self.device
        )
        constraints_dynamic_angle_1: torch.Tensor | None = None
        constraints_dynamic_scale_0: torch.Tensor = torch.ones(
            (input.shape[0]), dtype=self.default_dtype, device=self.device
        )
        constraints_dynamic_scale_1: torch.Tensor | None = None

        assert self.reference_image is not None
        assert self.reference_image.ndim == 2
        assert input.shape[-2] == self.reference_image.shape[-2]
        assert input.shape[-1] == self.reference_image.shape[-1]

        images_todo = input.to(device=self.device).type(dtype=self.default_dtype)
        image_reference = self.reference_image

        assert image_reference.ndim == 2
        assert images_todo.ndim == 3

        _, newangle = self._get_ang_scale(
            image_reference,
            images_todo,
            constraints_dynamic_scale_0,
            constraints_dynamic_scale_1,
            constraints_dynamic_angle_0,
            constraints_dynamic_angle_1,
        )

        return newangle

    # ---------------

    def _get_pcorr_shape(self, shape: torch.Size) -> tuple[int, int]:
        ret = (int(max(shape[-2:]) * 1.0),) * 2
        return ret

    def _get_log_base(self, shape: torch.Size, new_r: torch.Tensor) -> torch.Tensor:
        old_r = torch.tensor(
            (float(shape[-2]) * self.excess_const) / 2.0,
            dtype=self.default_dtype,
            device=self.device,
        )
        log_base = torch.exp(torch.log(old_r) / new_r)
        return log_base

    def wrap_angle(
        self, angles: torch.Tensor, ceil: float = 2 * torch.pi
    ) -> torch.Tensor:
        angles += ceil / 2.0
        angles %= ceil
        angles -= ceil / 2.0
        return angles

    def get_borderval(
        self, img: torch.Tensor, radius: int | None = None
    ) -> torch.Tensor:
        assert img.ndim == 3
        if radius is None:
            mindim = min([int(img.shape[-2]), int(img.shape[-1])])
            radius = max(1, mindim // 20)
        mask = torch.zeros(
            (int(img.shape[-2]), int(img.shape[-1])),
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, :radius] = True
        mask[:, -radius:] = True
        mask[:radius, :] = True
        mask[-radius:, :] = True

        mean = torch.median(img[:, mask], dim=-1)[0]
        return mean

    def get_apofield(self, shape: torch.Size, aporad: int) -> torch.Tensor:
        if aporad == 0:
            return torch.ones(
                shape[-2:],
                dtype=self.default_dtype,
                device=self.device,
            )

        assert int(shape[-2]) > aporad * 2
        assert int(shape[-1]) > aporad * 2

        apos = torch.hann_window(
            aporad * 2, dtype=self.default_dtype, periodic=False, device=self.device
        )

        toapp_0 = torch.ones(
            shape[-2],
            dtype=self.default_dtype,
            device=self.device,
        )
        toapp_0[:aporad] = apos[:aporad]
        toapp_0[-aporad:] = apos[-aporad:]

        toapp_1 = torch.ones(
            shape[-1],
            dtype=self.default_dtype,
            device=self.device,
        )
        toapp_1[:aporad] = apos[:aporad]
        toapp_1[-aporad:] = apos[-aporad:]

        apofield = torch.outer(toapp_0, toapp_1)

        return apofield

    def _get_subarr(
        self, array: torch.Tensor, center: torch.Tensor, rad: int
    ) -> torch.Tensor:
        assert array.ndim == 3
        assert center.ndim == 2
        assert array.shape[0] == center.shape[0]
        assert center.shape[1] == 2

        dim = 1 + 2 * rad
        subarr = torch.zeros(
            (array.shape[0], dim, dim), dtype=self.default_dtype, device=self.device
        )

        corner = center - rad
        idx_p = range(0, corner.shape[0])
        for ii in range(0, dim):
            yidx = corner[:, 0] + ii
            yidx %= array.shape[-2]
            for jj in range(0, dim):
                xidx = corner[:, 1] + jj
                xidx %= array.shape[-1]
                subarr[:, ii, jj] = array[idx_p, yidx, xidx]

        return subarr

    def _argmax_2d(self, array: torch.Tensor) -> torch.Tensor:
        assert array.ndim == 3

        max_pos = array.reshape(
            (array.shape[0], array.shape[1] * array.shape[2])
        ).argmax(dim=1)
        pos_0 = max_pos // array.shape[2]
        max_pos -= pos_0 * array.shape[2]
        ret = torch.zeros(
            (array.shape[0], 2), dtype=self.default_dtype, device=self.device
        )
        ret[:, 0] = pos_0
        ret[:, 1] = max_pos
        return ret.type(dtype=torch.int64)

    def _apodize(self, what: torch.Tensor) -> torch.Tensor:
        mindim = min([int(what.shape[-2]), int(what.shape[-1])])
        aporad = int(mindim * 0.12)

        apofield = self.get_apofield(what.shape, aporad).unsqueeze(0)

        res = what * apofield
        bg = self.get_borderval(what, aporad // 2).unsqueeze(-1).unsqueeze(-1)
        res += bg * (1 - apofield)
        return res

    def _logpolar_filter(self, shape: torch.Size) -> torch.Tensor:
        yy = torch.linspace(
            -torch.pi / 2.0,
            torch.pi / 2.0,
            shape[-2],
            dtype=self.default_dtype,
            device=self.device,
        ).unsqueeze(1)

        xx = torch.linspace(
            -torch.pi / 2.0,
            torch.pi / 2.0,
            shape[-1],
            dtype=self.default_dtype,
            device=self.device,
        ).unsqueeze(0)

        rads = torch.sqrt(yy**2 + xx**2)
        filt = 1.0 - torch.cos(rads) ** 2

        filt[torch.abs(rads) > torch.pi / 2] = 1
        return filt

    def _get_angles(self, shape: torch.Tensor) -> torch.Tensor:
        ret = torch.zeros(
            (int(shape[-2]), int(shape[-1])),
            dtype=self.default_dtype,
            device=self.device,
        )
        ret -= torch.linspace(
            0,
            torch.pi,
            int(shape[-2] + 1),
            dtype=self.default_dtype,
            device=self.device,
        )[:-1].unsqueeze(-1)

        return ret

    def _get_lograd(self, shape: torch.Tensor, log_base: torch.Tensor) -> torch.Tensor:
        ret = torch.zeros(
            (int(shape[-2]), int(shape[-1])),
            dtype=self.default_dtype,
            device=self.device,
        )
        ret += torch.pow(
            log_base,
            torch.arange(
                0,
                int(shape[-1]),
                dtype=self.default_dtype,
                device=self.device,
            ),
        ).unsqueeze(0)
        return ret

    def _logpolar(
        self, image: torch.Tensor, shape: torch.Tensor, log_base: torch.Tensor
    ) -> torch.Tensor:
        assert image.ndim == 3

        imshape: torch.Tensor = torch.tensor(
            image.shape[-2:],
            dtype=self.default_dtype,
            device=self.device,
        )

        center: torch.Tensor = imshape.clone() / 2

        theta: torch.Tensor = self._get_angles(shape)
        radius_x: torch.Tensor = self._get_lograd(shape, log_base)
        radius_y: torch.Tensor = radius_x.clone()

        ellipse_coef: torch.Tensor = imshape[0] / imshape[1]
        radius_x /= ellipse_coef

        y = radius_y * torch.sin(theta) + center[0]
        y /= float(image.shape[-2])
        y *= 2
        y -= 1

        x = radius_x * torch.cos(theta) + center[1]
        x /= float(image.shape[-1])
        x *= 2
        x -= 1

        idx_x = torch.where(torch.abs(x) <= 1.0, 1.0, 0.0)
        idx_y = torch.where(torch.abs(y) <= 1.0, 1.0, 0.0)

        normalized_coords = torch.cat(
            (
                x.unsqueeze(-1),
                y.unsqueeze(-1),
            ),
            dim=-1,
        ).unsqueeze(0)

        output = torch.empty(
            (int(image.shape[0]), int(y.shape[0]), int(y.shape[1])),
            dtype=self.default_dtype,
            device=self.device,
        )

        for id in range(0, int(image.shape[0])):
            bgval: torch.Tensor = torch.quantile(image[id, :, :], q=1.0 / 100.0)

            temp = torch.nn.functional.grid_sample(
                image[id, :, :].unsqueeze(0).unsqueeze(0),
                normalized_coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            output[id, :, :] = torch.where((idx_x * idx_y) == 0.0, bgval, temp)

        return output

    def _argmax_ext(self, array: torch.Tensor, exponent: float | str) -> torch.Tensor:
        assert array.ndim == 3

        if exponent == "inf":
            ret = self._argmax_2d(array)
        else:
            assert isinstance(exponent, float) or isinstance(exponent, int)

            col = (
                torch.arange(
                    0, array.shape[-2], dtype=self.default_dtype, device=self.device
                )
                .unsqueeze(-1)
                .unsqueeze(0)
            )
            row = (
                torch.arange(
                    0, array.shape[-1], dtype=self.default_dtype, device=self.device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            arr2 = torch.pow(array, float(exponent))
            arrsum = arr2.sum(dim=-2).sum(dim=-1)

            ret = torch.zeros(
                (array.shape[0], 2), dtype=self.default_dtype, device=self.device
            )

            arrprody = (arr2 * col).sum(dim=-1).sum(dim=-1) / arrsum
            arrprodx = (arr2 * row).sum(dim=-1).sum(dim=-1) / arrsum

            ret[:, 0] = arrprody.squeeze(-1).squeeze(-1)
            ret[:, 1] = arrprodx.squeeze(-1).squeeze(-1)

            idx = torch.where(arrsum == 0.0)[0]
            ret[idx, :] = 0.0
        return ret

    def _interpolate(
        self, array: torch.Tensor, rough: torch.Tensor, rad: int = 2
    ) -> torch.Tensor:
        assert array.ndim == 3
        assert rough.ndim == 2

        rough = torch.round(rough).type(torch.int64)

        surroundings = self._get_subarr(array, rough, rad)

        com = self._argmax_ext(surroundings, 1.0)

        offset = com - rad
        ret = rough + offset

        ret += 0.5
        ret %= (
            torch.tensor(array.shape[-2:], dtype=self.default_dtype, device=self.device)
            .type(dtype=torch.int64)
            .unsqueeze(0)
        )
        ret -= 0.5
        return ret

    def _get_success(
        self, array: torch.Tensor, coord: torch.Tensor, radius: int = 2
    ) -> torch.Tensor:
        assert array.ndim == 3
        assert coord.ndim == 2
        assert array.shape[0] == coord.shape[0]
        assert coord.shape[1] == 2

        coord = torch.round(coord).type(dtype=torch.int64)
        subarr = self._get_subarr(
            array, coord, 2
        )  # Not my fault. They want a 2 there. Not radius

        theval = subarr.sum(dim=-1).sum(dim=-1)

        theval2 = array[range(0, coord.shape[0]), coord[:, 0], coord[:, 1]]

        success = torch.sqrt(theval * theval2)
        return success

    def _get_constraint_mask(
        self,
        shape: torch.Size,
        log_base: torch.Tensor,
        constraints_scale_0: torch.Tensor,
        constraints_scale_1: torch.Tensor | None,
        constraints_angle_0: torch.Tensor,
        constraints_angle_1: torch.Tensor | None,
    ) -> torch.Tensor:
        assert constraints_scale_0 is not None
        assert constraints_angle_0 is not None
        assert constraints_scale_0.ndim == 1
        assert constraints_angle_0.ndim == 1

        assert constraints_scale_0.shape[0] == constraints_angle_0.shape[0]

        mask: torch.Tensor = torch.ones(
            (constraints_scale_0.shape[0], int(shape[-2]), int(shape[-1])),
            device=self.device,
            dtype=self.default_dtype,
        )

        scale: torch.Tensor = constraints_scale_0.clone()
        if constraints_scale_1 is not None:
            sigma: torch.Tensor | None = constraints_scale_1.clone()
        else:
            sigma = None

        scales = torch.fft.ifftshift(
            self._get_lograd(
                torch.tensor(shape[-2:], device=self.device, dtype=self.default_dtype),
                log_base,
            )
        )

        scales *= log_base ** (-shape[-1] / 2.0)
        scales = scales.unsqueeze(0) - (1.0 / scale).unsqueeze(-1).unsqueeze(-1)

        if sigma is not None:
            assert sigma.shape[0] == constraints_scale_0.shape[0]

            for p_id in range(0, sigma.shape[0]):
                if sigma[p_id] == 0:
                    ascales = torch.abs(scales[p_id, ...])
                    scale_min = ascales.min()
                    binary_mask = torch.where(ascales > scale_min, 0.0, 1.0)
                    mask[p_id, ...] *= binary_mask
                else:
                    mask[p_id, ...] *= torch.exp(
                        -(torch.pow(scales[p_id, ...], 2)) / torch.pow(sigma[p_id], 2)
                    )

        angle: torch.Tensor = constraints_angle_0.clone()
        if constraints_angle_1 is not None:
            sigma = constraints_angle_1.clone()
        else:
            sigma = None

        angles = self._get_angles(
            torch.tensor(shape[-2:], device=self.device, dtype=self.default_dtype)
        )

        angles = angles.unsqueeze(0) + torch.deg2rad(angle).unsqueeze(-1).unsqueeze(-1)

        angles = torch.rad2deg(angles)

        if sigma is not None:
            assert sigma.shape[0] == constraints_scale_0.shape[0]

            for p_id in range(0, sigma.shape[0]):
                if sigma[p_id] == 0:
                    aangles = torch.abs(angles[p_id, ...])
                    angle_min = aangles.min()
                    binary_mask = torch.where(aangles > angle_min, 0.0, 1.0)
                    mask[p_id, ...] *= binary_mask
                else:
                    mask *= torch.exp(
                        -(torch.pow(angles[p_id, ...], 2)) / torch.pow(sigma[p_id], 2)
                    )

        mask = torch.fft.fftshift(mask, dim=(-2, -1))

        return mask

    def argmax_angscale(
        self,
        array: torch.Tensor,
        log_base: torch.Tensor,
        constraints_scale_0: torch.Tensor,
        constraints_scale_1: torch.Tensor | None,
        constraints_angle_0: torch.Tensor,
        constraints_angle_1: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert array.ndim == 3
        assert constraints_scale_0 is not None
        assert constraints_angle_0 is not None
        assert constraints_scale_0.ndim == 1
        assert constraints_angle_0.ndim == 1

        mask = self._get_constraint_mask(
            array.shape[-2:],
            log_base,
            constraints_scale_0,
            constraints_scale_1,
            constraints_angle_0,
            constraints_angle_1,
        )

        array_orig = array.clone()

        array *= mask
        ret = self._argmax_ext(array, self.exponent)

        ret_final = self._interpolate(array, ret)

        success = self._get_success(array_orig, ret_final, 0)

        return ret_final, success

    def argmax_translation(
        self, array: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert array.ndim == 3

        array_orig = array.clone()

        ashape = torch.tensor(array.shape[-2:], device=self.device).type(
            dtype=torch.int64
        )

        aporad = (ashape // 6).min()
        mask2 = self.get_apofield(torch.Size(ashape), aporad).unsqueeze(0)
        array *= mask2

        tvec = self._argmax_ext(array, "inf")
        tvec = self._interpolate(array_orig, tvec)

        success = self._get_success(array_orig, tvec, 2)

        return tvec, success

    def transform_img(
        self,
        img: torch.Tensor,
        scale: torch.Tensor | None = None,
        angle: torch.Tensor | None = None,
        tvec: torch.Tensor | None = None,
        bgval: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert img.ndim == 3

        if scale is None:
            scale = torch.ones(
                (img.shape[0],), dtype=self.default_dtype, device=self.device
            )
        assert scale.ndim == 1
        assert scale.shape[0] == img.shape[0]

        if angle is None:
            angle = torch.zeros(
                (img.shape[0],), dtype=self.default_dtype, device=self.device
            )
        assert angle.ndim == 1
        assert angle.shape[0] == img.shape[0]

        if tvec is None:
            tvec = torch.zeros(
                (img.shape[0], 2), dtype=self.default_dtype, device=self.device
            )
        assert tvec.ndim == 2
        assert tvec.shape[0] == img.shape[0]
        assert tvec.shape[1] == 2

        if bgval is None:
            bgval = self.get_borderval(img)
        assert bgval.ndim == 1
        assert bgval.shape[0] == img.shape[0]

        # Otherwise we need to decompose it and put it back together
        assert torch.is_complex(img) is False

        output = torch.zeros_like(img)

        for pos in range(0, img.shape[0]):
            image_processed = img[pos, :, :].unsqueeze(0).clone()

            temp_shift = [
                int(round(tvec[pos, 1].item() * self.scale_factor)),
                int(round(tvec[pos, 0].item() * self.scale_factor)),
            ]

            image_processed = torch.nn.functional.interpolate(
                image_processed.unsqueeze(0),
                scale_factor=self.scale_factor,
                mode="bilinear",
            ).squeeze(0)

            image_processed = tv.transforms.functional.affine(
                img=image_processed,
                angle=-float(angle[pos]),
                translate=temp_shift,
                scale=float(scale[pos]),
                shear=[0, 0],
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                fill=float(bgval[pos]),
                center=None,
            )

            image_processed = torch.nn.functional.interpolate(
                image_processed.unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="bilinear",
            ).squeeze(0)

            image_processed = tv.transforms.functional.center_crop(
                image_processed, img.shape[-2:]
            )

            output[pos, ...] = image_processed.squeeze(0)

        return output

    def transform_img_dict(
        self,
        img: torch.Tensor,
        scale: torch.Tensor | None = None,
        angle: torch.Tensor | None = None,
        tvec: torch.Tensor | None = None,
        bgval: torch.Tensor | None = None,
        invert=False,
    ) -> torch.Tensor:
        if invert:
            if scale is not None:
                scale = 1.0 / scale
            if angle is not None:
                angle *= -1
            if tvec is not None:
                tvec *= -1

        res = self.transform_img(img, scale, angle, tvec, bgval=bgval)
        return res

    def _phase_correlation(
        self, image_reference: torch.Tensor, images_todo: torch.Tensor, callback, *args
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert image_reference.ndim == 3
        assert image_reference.shape[0] == 1
        assert images_todo.ndim == 3

        assert callback is not None

        image_reference_fft = torch.fft.fft2(image_reference, dim=(-2, -1))
        images_todo_fft = torch.fft.fft2(images_todo, dim=(-2, -1))

        eps = torch.abs(images_todo_fft).max(dim=-1)[0].max(dim=-1)[0] * 1e-15

        cps = abs(
            torch.fft.ifft2(
                (image_reference_fft * images_todo_fft.conj())
                / (
                    torch.abs(image_reference_fft) * torch.abs(images_todo_fft)
                    + eps.unsqueeze(-1).unsqueeze(-1)
                )
            )
        )

        scps = torch.fft.fftshift(cps, dim=(-2, -1))

        ret, success = callback(scps, *args)

        ret[:, 0] -= image_reference_fft.shape[-2] // 2
        ret[:, 1] -= image_reference_fft.shape[-1] // 2

        return ret, success

    def _translation(
        self, im0: torch.Tensor, im1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert im0.ndim == 2
        ret, succ = self._phase_correlation(
            im0.unsqueeze(0), im1, self.argmax_translation
        )
        return ret, succ

    def _get_ang_scale(
        self,
        image_reference: torch.Tensor,
        images_todo: torch.Tensor,
        constraints_scale_0: torch.Tensor,
        constraints_scale_1: torch.Tensor | None,
        constraints_angle_0: torch.Tensor,
        constraints_angle_1: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert image_reference.ndim == 2
        assert images_todo.ndim == 3
        assert image_reference.shape[-1] == images_todo.shape[-1]
        assert image_reference.shape[-2] == images_todo.shape[-2]
        assert constraints_scale_0.shape[0] == images_todo.shape[0]
        assert constraints_angle_0.shape[0] == images_todo.shape[0]

        if constraints_scale_1 is not None:
            assert constraints_scale_1.shape[0] == images_todo.shape[0]

        if constraints_angle_1 is not None:
            assert constraints_angle_1.shape[0] == images_todo.shape[0]

        if self.image_reference_dft is None:
            image_reference_apod = self._apodize(image_reference.unsqueeze(0))
            self.image_reference_dft = torch.fft.fftshift(
                torch.fft.fft2(image_reference_apod, dim=(-2, -1)), dim=(-2, -1)
            )
            self.filt = self._logpolar_filter(image_reference.shape).unsqueeze(0)
            self.image_reference_dft *= self.filt
            self.pcorr_shape = torch.tensor(
                self._get_pcorr_shape(image_reference.shape[-2:]),
                dtype=self.default_dtype,
                device=self.device,
            )
            self.log_base = self._get_log_base(
                image_reference.shape,
                self.pcorr_shape[1],
            )
            self.image_reference_logp = self._logpolar(
                torch.abs(self.image_reference_dft), self.pcorr_shape, self.log_base
            )

        images_todo_apod = self._apodize(images_todo)
        images_todo_dft = torch.fft.fftshift(
            torch.fft.fft2(images_todo_apod, dim=(-2, -1)), dim=(-2, -1)
        )

        images_todo_dft *= self.filt

        images_todo_lopg = self._logpolar(
            torch.abs(images_todo_dft), self.pcorr_shape, self.log_base
        )

        temp, _ = self._phase_correlation(
            self.image_reference_logp,
            images_todo_lopg,
            self.argmax_angscale,
            self.log_base,
            constraints_scale_0,
            constraints_scale_1,
            constraints_angle_0,
            constraints_angle_1,
        )

        arg_ang = temp[:, 0].clone()
        arg_rad = temp[:, 1].clone()

        angle = -torch.pi * arg_ang / float(self.pcorr_shape[0])
        angle = torch.rad2deg(angle)

        angle = self.wrap_angle(angle, 360)

        scale = torch.pow(self.log_base, arg_rad)

        angle = -angle
        scale = 1.0 / scale

        assert torch.where(scale < 2)[0].shape[0] == scale.shape[0]
        assert torch.where(scale > 0.5)[0].shape[0] == scale.shape[0]

        return scale, angle

    def translation(
        self, im0: torch.Tensor, im1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        angle = torch.zeros(
            (im1.shape[0]), dtype=self.default_dtype, device=self.device
        )
        assert im1.ndim == 3
        assert im0.shape[-2] == im1.shape[-2]
        assert im0.shape[-1] == im1.shape[-1]

        tvec, succ = self._translation(im0, im1)
        tvec2, succ2 = self._translation(im0, torch.rot90(im1, k=2, dims=[-2, -1]))

        assert tvec.shape[0] == tvec2.shape[0]
        assert tvec.ndim == 2
        assert tvec2.ndim == 2
        assert tvec.shape[1] == 2
        assert tvec2.shape[1] == 2
        assert succ.shape[0] == succ2.shape[0]
        assert succ.ndim == 1
        assert succ2.ndim == 1
        assert tvec.shape[0] == succ.shape[0]
        assert angle.shape[0] == tvec.shape[0]
        assert angle.ndim == 1

        for pos in range(0, angle.shape[0]):
            pick_rotated = False
            if succ2[pos] > succ[pos]:
                pick_rotated = True

            if pick_rotated:
                tvec[pos, :] = tvec2[pos, :]
                succ[pos] = succ2[pos]
                angle[pos] += 180

        return tvec, succ, angle

    def _similarity(
        self,
        image_reference: torch.Tensor,
        images_todo: torch.Tensor,
        bgval: torch.Tensor,
    ):
        assert image_reference.ndim == 2
        assert images_todo.ndim == 3
        assert image_reference.shape[-1] == images_todo.shape[-1]
        assert image_reference.shape[-2] == images_todo.shape[-2]

        # We are going to iterate and precise scale and angle estimates
        scale: torch.Tensor = torch.ones(
            (images_todo.shape[0]), dtype=self.default_dtype, device=self.device
        )
        angle: torch.Tensor = torch.zeros(
            (images_todo.shape[0]), dtype=self.default_dtype, device=self.device
        )

        constraints_dynamic_angle_0: torch.Tensor = torch.zeros(
            (images_todo.shape[0]), dtype=self.default_dtype, device=self.device
        )
        constraints_dynamic_angle_1: torch.Tensor | None = None
        constraints_dynamic_scale_0: torch.Tensor = torch.ones(
            (images_todo.shape[0]), dtype=self.default_dtype, device=self.device
        )
        constraints_dynamic_scale_1: torch.Tensor | None = None

        newscale, newangle = self._get_ang_scale(
            image_reference,
            images_todo,
            constraints_dynamic_scale_0,
            constraints_dynamic_scale_1,
            constraints_dynamic_angle_0,
            constraints_dynamic_angle_1,
        )
        scale *= newscale
        angle += newangle

        im2 = self.transform_img(images_todo, scale, angle, bgval=bgval)

        tvec, self.success, res_angle = self.translation(image_reference, im2)

        angle += res_angle

        angle = self.wrap_angle(angle, 360)

        return scale, angle, tvec

    def similarity(
        self,
        image_reference: torch.Tensor,
        images_todo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert image_reference.ndim == 2
        assert images_todo.ndim == 3

        bgval: torch.Tensor = self.get_borderval(img=images_todo, radius=5)

        scale, angle, tvec = self._similarity(
            image_reference,
            images_todo,
            bgval,
        )

        im2 = self.transform_img_dict(
            img=images_todo,
            scale=scale,
            angle=angle,
            tvec=tvec,
            bgval=bgval,
        )

        return scale, angle, tvec, im2
