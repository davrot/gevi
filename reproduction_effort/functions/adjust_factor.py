import torch
import math


def adjust_factor(
    input_acceptor: torch.Tensor,
    input_donor: torch.Tensor,
    lower_frequency_heartbeat: float,
    upper_frequency_heartbeat: float,
    sample_frequency: float,
    mask: torch.Tensor,
) -> tuple[float, float]:

    number_of_active_pixel: torch.Tensor = mask.type(dtype=torch.float32).sum()
    signal_acceptor: torch.Tensor = (input_acceptor * mask.unsqueeze(-1)).sum(
        dim=0
    ).sum(dim=0) / number_of_active_pixel

    signal_donor: torch.Tensor = (input_donor * mask.unsqueeze(-1)).sum(dim=0).sum(
        dim=0
    ) / number_of_active_pixel

    signal_acceptor_offset = signal_acceptor.mean()
    signal_donor_offset = signal_donor.mean()

    signal_acceptor = signal_acceptor - signal_acceptor_offset
    signal_donor = signal_donor - signal_donor_offset

    blackman_window = torch.blackman_window(
        window_length=signal_acceptor.shape[0],
        periodic=True,
        dtype=signal_acceptor.dtype,
        device=signal_acceptor.device,
    )

    signal_acceptor *= blackman_window
    signal_donor *= blackman_window
    nfft: int = int(2 ** math.ceil(math.log2(signal_donor.shape[0])))
    nfft = max([256, nfft])

    signal_acceptor_fft: torch.Tensor = torch.fft.rfft(signal_acceptor, n=nfft)
    signal_donor_fft: torch.Tensor = torch.fft.rfft(signal_donor, n=nfft)

    frequency_axis: torch.Tensor = (
        torch.fft.rfftfreq(nfft, device=signal_acceptor_fft.device) * sample_frequency
    )

    signal_acceptor_power: torch.Tensor = torch.abs(signal_acceptor_fft) ** 2
    signal_acceptor_power[1:-1] *= 2

    signal_donor_power: torch.Tensor = torch.abs(signal_donor_fft) ** 2
    signal_donor_power[1:-1] *= 2

    if frequency_axis[-1] != (sample_frequency / 2.0):
        signal_acceptor_power[-1] *= 2
        signal_donor_power[-1] *= 2

    signal_acceptor_power /= blackman_window.sum() ** 2
    signal_donor_power /= blackman_window.sum() ** 2

    idx = torch.where(
        (frequency_axis >= lower_frequency_heartbeat)
        * (frequency_axis <= upper_frequency_heartbeat)
    )[0]

    frequency_axis = frequency_axis[idx]
    signal_acceptor_power = signal_acceptor_power[idx]
    signal_donor_power = signal_donor_power[idx]

    acceptor_range = signal_acceptor_power.max() - signal_acceptor_power.min()

    donor_range = signal_donor_power.max() - signal_donor_power.min()

    acceptor_correction_factor: float = float(
        0.5
        * (
            1
            + (signal_acceptor_offset * torch.sqrt(donor_range))
            / (signal_donor_offset * torch.sqrt(acceptor_range))
        )
    )

    donor_correction_factor: float = float(
        acceptor_correction_factor / (2 * acceptor_correction_factor - 1)
    )

    return donor_correction_factor, acceptor_correction_factor
