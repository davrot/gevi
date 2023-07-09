import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation


class Anime:
    def __init__(self) -> None:
        super().__init__()

    def show(
        self,
        input: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray | None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "hot",
        axis_off: bool = True,
        show_frame_count: bool = True,
        interval: int = 100,
        repeat: bool = False,
        colorbar: bool = True,
        vmin_scale: float | None = None,
        vmax_scale: float | None = None,
    ) -> None:
        assert input.ndim == 3

        if isinstance(input, torch.Tensor):
            input_np: np.ndarray = input.cpu().numpy()
            if mask is not None:
                mask_np: np.ndarray | None = (mask == 0).cpu().numpy()
            else:
                mask_np = None
        else:
            input_np = input
            if mask is not None:
                mask_np = mask == 0  # type: ignore
            else:
                mask_np = None

        if vmin is None:
            vmin = float(np.where(np.isfinite(input_np), input_np, 0.0).min())
        if vmax is None:
            vmax = float(np.where(np.isfinite(input_np), input_np, 0.0).max())

        if vmin_scale is not None:
            vmin *= vmin_scale

        if vmax_scale is not None:
            vmax *= vmax_scale

        fig = plt.figure()
        image = np.nan_to_num(input_np[0, ...], copy=True, nan=0.0)
        if mask_np is not None:
            image[mask_np] = float("NaN")
        image_handle = plt.imshow(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if colorbar is True:
            plt.colorbar()

        if axis_off is True:
            plt.axis("off")

        def next_frame(i: int) -> None:
            image = np.nan_to_num(input_np[i, ...], copy=True, nan=0.0)
            if mask_np is not None:
                image[mask_np] = float("NaN")

            image_handle.set_data(image)
            if show_frame_count is True:
                bar_length: int = 10
                filled_length = int(round(bar_length * i / input_np.shape[0]))
                bar = "\u25A0" * filled_length + "\u25A1" * (bar_length - filled_length)
                plt.title(f"{bar} {i} of {int(input_np.shape[0]-1)}", loc="left")
            return

        _ = matplotlib.animation.FuncAnimation(
            fig,
            next_frame,
            frames=int(input.shape[0]),
            interval=interval,
            repeat=repeat,
        )

        plt.show()
