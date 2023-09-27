import torch
from functions.Anime import Anime
from functions.DataContainer import DataContainer
import matplotlib.pyplot as plt
import numpy as np
import argh
import os


def main(
    path: str = "/data_1/hendrik/2021-06-17/M3859M/raw",
    experiment_id: int = 1,
    trial_id: int = 1,
    use_svd: bool = True,  # i.e. use SVD
    mask_threshold: float | None = 0.0025,  # Between 0 and 1.0
    show_example_timeseries: bool = True,
    example_position_x: int = 280,
    example_position_y: int = 440,
    movie_play: bool = True,
    movie_vmin_scale: float | None = 0.05,
    movie_vmax_scale: float | None = 0.1,
    movie_enable_mask: bool = True,
    movie_export: bool = False,
    export_results: bool = True,
    export_path: str = "Export",
):
    if use_svd:
        print("SVD mode")
    else:
        print("Classic mode")

    if movie_export is False:
        movie_file: str | None = None
    else:
        if use_svd:
            movie_file = f"SVD_Exp{experiment_id}_Trial{trial_id}.mp4"
        else:
            movie_file = f"Classic_Exp{experiment_id}_Trial{trial_id}.mp4"

    if export_results:
        os.makedirs(export_path, exist_ok=True)

    initital_mask_name: str | None = None
    initital_mask_update: bool = True
    initital_mask_roi: bool = False  # default: True

    start_position: int = 0
    start_position_coefficients: int = 100

    svd_iterations: int = 1  # SVD iterations: Do not touch! Keep at 1
    bin_size: int = 4

    display_logging_messages: bool = False
    save_logging_messages: bool = False

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
        remove_heartbeat=use_svd,  # i.e. use SVD
        iterations=svd_iterations,
        bin_size=bin_size,
        initital_mask_name=initital_mask_name,
        initital_mask_update=initital_mask_update,
        initital_mask_roi=initital_mask_roi,
        start_position_coefficients=start_position_coefficients,
        gaussian_blur_kernel_size=gaussian_blur_kernel_size,
        gaussian_blur_sigma=gaussian_blur_sigma,
        bin_size_post=bin_size_post,
        threshold=mask_threshold,
    )

    if show_example_timeseries:
        plt.plot(result[:, example_position_x, example_position_y].cpu())
        plt.show()

    if export_results:
        if use_svd:
            np.save(
                os.path.join(export_path, f"SVD_{experiment_id}_{trial_id}_data.npy"),
                result.cpu().numpy(),
            )

            if mask is not None:
                np.save(
                    os.path.join(
                        export_path, f"SVD_{experiment_id}_{trial_id}_mask.npy"
                    ),
                    result.cpu().numpy(),
                )

        else:
            np.save(
                os.path.join(
                    export_path, f"Classic_{experiment_id}_{trial_id}_data.npy"
                ),
                result.cpu().numpy(),
            )

            if mask is not None:
                np.save(
                    os.path.join(
                        export_path, f"Classic_{experiment_id}_{trial_id}_mask.npy"
                    ),
                    result.cpu().numpy(),
                )

    if movie_play:
        ani = Anime()
        if movie_enable_mask:
            ani.show(
                result - 1.0,
                mask=mask,
                vmin_scale=movie_vmin_scale,
                vmax_scale=movie_vmax_scale,
                movie_file=movie_file,
            )
        else:
            ani.show(
                result - 1.0,
                vmin_scale=movie_vmin_scale,
                vmax_scale=movie_vmax_scale,
                movie_file=movie_file,
            )


if __name__ == "__main__":
    argh.dispatch_command(main)
