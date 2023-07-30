from functions.DataContainer import DataContainer
import torch
import matplotlib.pyplot as plt
import argh
import os
import numpy as np


@torch.no_grad()
def _calculate_cross_corelation(
    a: torch.Tensor | None, b: torch.Tensor | None, data_shape: torch.Size
) -> torch.Tensor:
    assert a is not None
    assert b is not None
    assert a.ndim == 3
    assert b.ndim == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == b.shape[1]
    assert a.shape[2] == b.shape[2]

    output = (
        (
            torch.fft.fftshift(
                torch.fft.irfft(
                    a * b.conj(),
                    dim=0,
                ),
                dim=0,
            )
            / int(data_shape[0])
        )
        .mean(-1)
        .mean(-1)
    )
    output = output[data_shape[0] // 2 : -data_shape[0] // 2]

    return output


@torch.no_grad()
def _prepare_data(input: torch.Tensor) -> torch.Tensor:
    input -= input.mean(dim=0, keepdim=True)
    input /= input.std(dim=0, keepdim=True) + 1e-20
    input = torch.cat(
        (torch.zeros_like(input), input, torch.zeros_like(input)),
        dim=0,
    )
    input = torch.fft.rfft(
        input,
        dim=0,
    )

    return input


@torch.no_grad()
def process_combinations(
    path: str,
    torch_device: torch.device,
    remove_heartbeat: bool = True,
    experiment_id: int = 1,
    trial_id: int = 1,
    remove_linear: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    af = DataContainer(
        path=path,
        device=torch_device,
        display_logging_messages=False,
        save_logging_messages=False,
    )

    af.cleaned_load_data(
        experiment_id=experiment_id,
        trial_id=trial_id,
        align=True,
        iterations=1,
        lowrank_method=True,
        lowrank_q=6,
        remove_heartbeat=remove_heartbeat,
        remove_mean=False,
        remove_linear=remove_linear,
        remove_heartbeat_mean=False,
        remove_heartbeat_linear=False,
        bin_size=4,
        do_frame_shift=True,
        enable_secondary_data=True,
        mmap_mode=True,
        initital_mask=None,
        start_position_coefficients=0,
    )
    assert af.acceptor is not None
    assert af.donor is not None
    assert af.oxygenation is not None
    assert af.volume is not None

    data_shape = af.acceptor.shape

    af.acceptor = _prepare_data(af.acceptor)
    af.donor = _prepare_data(af.donor)
    af.oxygenation = _prepare_data(af.oxygenation)
    af.volume = _prepare_data(af.volume)

    x_aa = _calculate_cross_corelation(
        a=af.acceptor, b=af.acceptor, data_shape=data_shape
    )
    time_axis = (
        torch.arange(0, x_aa.shape[0], device=x_aa.device, dtype=x_aa.dtype)
        - float(torch.argmax(x_aa))
    ) / 100.0

    x_dd = _calculate_cross_corelation(a=af.donor, b=af.donor, data_shape=data_shape)
    x_oo = _calculate_cross_corelation(
        a=af.oxygenation, b=af.oxygenation, data_shape=data_shape
    )
    x_vv = _calculate_cross_corelation(a=af.volume, b=af.volume, data_shape=data_shape)
    x_ad = _calculate_cross_corelation(a=af.acceptor, b=af.donor, data_shape=data_shape)
    x_ao = _calculate_cross_corelation(
        a=af.acceptor, b=af.oxygenation, data_shape=data_shape
    )
    x_av = _calculate_cross_corelation(
        a=af.acceptor, b=af.volume, data_shape=data_shape
    )
    x_da = _calculate_cross_corelation(a=af.donor, b=af.acceptor, data_shape=data_shape)
    x_do = _calculate_cross_corelation(
        a=af.donor, b=af.oxygenation, data_shape=data_shape
    )
    x_dv = _calculate_cross_corelation(a=af.donor, b=af.volume, data_shape=data_shape)

    x_vo = _calculate_cross_corelation(
        a=af.volume, b=af.oxygenation, data_shape=data_shape
    )

    return (x_aa, time_axis, x_dd, x_oo, x_vv, x_ad, x_ao, x_av, x_da, x_do, x_dv, x_vo)


def make_a_x_correlation_plot(
    x_aa: torch.Tensor,
    time_axis: torch.Tensor,
    x_dd: torch.Tensor,
    x_oo: torch.Tensor,
    x_vv: torch.Tensor,
    x_ad: torch.Tensor,
    x_ao: torch.Tensor,
    x_av: torch.Tensor,
    x_da: torch.Tensor,
    x_do: torch.Tensor,
    x_dv: torch.Tensor,
    x_vo: torch.Tensor,
) -> None:
    plt.subplot(2, 2, 1)
    plt.plot(time_axis.cpu(), x_aa.cpu(), label="acceptor")
    plt.plot(time_axis.cpu(), x_dd.cpu(), label="donor")
    plt.plot(time_axis.cpu(), x_oo.cpu(), label="oxygenation")
    plt.plot(time_axis.cpu(), x_vv.cpu(), label="volume")
    plt.legend()
    plt.ylabel("Auto-Correlation")
    plt.xlabel("Tau [sec]")

    plt.subplot(2, 2, 2)
    plt.plot(time_axis.cpu(), x_ad.cpu(), label="donor")
    plt.plot(time_axis.cpu(), x_ao.cpu(), label="oxygenation")
    plt.plot(time_axis.cpu(), x_av.cpu(), label="volume")
    plt.legend()
    plt.ylabel("X-Correlation with acceptor")
    plt.xlabel("Tau [sec]")

    plt.subplot(2, 2, 3)
    plt.plot(time_axis.cpu(), x_da.cpu(), label="acceptor")
    plt.plot(time_axis.cpu(), x_do.cpu(), label="oxygenation")
    plt.plot(time_axis.cpu(), x_dv.cpu(), label="volume")
    plt.legend()
    plt.ylabel("X-Correlation with donor")
    plt.xlabel("Tau [sec]")

    plt.subplot(2, 2, 4)
    plt.plot(time_axis.cpu(), x_vo.cpu(), label="volume -> oxygenation")
    plt.legend()
    plt.ylabel("X-Correlation")
    plt.xlabel("Tau [sec]")

    plt.show()


@torch.no_grad()
def main(
    path: str = "/data_1/hendrik/2023-07-17/M_Sert_Cre_41/raw",
    use_svd: bool = True,
    remove_linear_trend: bool = False,
    experiment_id: int = 1,
    trial_id: int = 1,
    plot_results: bool = True,
    export_results: bool = True,
    export_path: str = "Export_Correlation",
) -> None:
    if use_svd:
        print("SVD mode")
    else:
        print("Classic mode")

    if export_results:
        os.makedirs(export_path, exist_ok=True)

    torch_device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    (
        x_aa,
        time_axis,
        x_dd,
        x_oo,
        x_vv,
        x_ad,
        x_ao,
        x_av,
        x_da,
        x_do,
        x_dv,
        x_vo,
    ) = process_combinations(
        path=path,
        torch_device=torch_device,
        experiment_id=experiment_id,
        trial_id=trial_id,
        remove_heartbeat=use_svd,
        remove_linear=remove_linear_trend,
    )

    if export_results:
        if use_svd:
            np.savez(
                os.path.join(export_path, f"SVD_{experiment_id}_{trial_id}_data.npz"),
                time_axis=time_axis.cpu().numpy(),
                x_aa=x_aa.cpu().numpy(),
                x_dd=x_dd.cpu().numpy(),
                x_oo=x_oo.cpu().numpy(),
                x_vv=x_vv.cpu().numpy(),
                x_ad=x_ad.cpu().numpy(),
                x_ao=x_ao.cpu().numpy(),
                x_av=x_av.cpu().numpy(),
                x_da=x_da.cpu().numpy(),
                x_do=x_do.cpu().numpy(),
                x_dv=x_dv.cpu().numpy(),
                x_vo=x_vo.cpu().numpy(),
            )
        else:
            np.savez(
                os.path.join(
                    export_path, f"Classic_{experiment_id}_{trial_id}_data.npz"
                ),
                time_axis=time_axis.cpu().numpy(),
                x_aa=x_aa.cpu().numpy(),
                x_dd=x_dd.cpu().numpy(),
                x_oo=x_oo.cpu().numpy(),
                x_vv=x_vv.cpu().numpy(),
                x_ad=x_ad.cpu().numpy(),
                x_ao=x_ao.cpu().numpy(),
                x_av=x_av.cpu().numpy(),
                x_da=x_da.cpu().numpy(),
                x_do=x_do.cpu().numpy(),
                x_dv=x_dv.cpu().numpy(),
                x_vo=x_vo.cpu().numpy(),
            )

    if plot_results:
        make_a_x_correlation_plot(
            x_aa, time_axis, x_dd, x_oo, x_vv, x_ad, x_ao, x_av, x_da, x_do, x_dv, x_vo
        )


if __name__ == "__main__":
    argh.dispatch_command(main)
