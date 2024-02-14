import torch


def heart_beat_frequency(
    input: torch.Tensor,
    lower_frequency_heartbeat: float,
    upper_frequency_heartbeat: float,
    sample_frequency: float,
    mask: torch.Tensor,
) -> float:

    number_of_active_pixel: torch.Tensor = mask.type(dtype=torch.float32).sum()
    signal: torch.Tensor = (input * mask.unsqueeze(-1)).sum(dim=0).sum(
        dim=0
    ) / number_of_active_pixel
    signal = signal - signal.mean()

    hamming_window = torch.hamming_window(
        window_length=signal.shape[0],
        periodic=True,
        alpha=0.54,
        beta=0.46,
        dtype=signal.dtype,
        device=signal.device,
    )

    signal *= hamming_window

    signal_fft: torch.Tensor = torch.fft.rfft(signal)
    frequency_axis: torch.Tensor = (
        torch.fft.rfftfreq(signal.shape[0], device=input.device) * sample_frequency
    )
    signal_power: torch.Tensor = torch.abs(signal_fft) ** 2
    signal_power[1:-1] *= 2

    if frequency_axis[-1] != (sample_frequency / 2.0):
        signal_power[-1] *= 2
    signal_power /= hamming_window.sum() ** 2

    idx = torch.where(
        (frequency_axis > lower_frequency_heartbeat)
        * (frequency_axis < upper_frequency_heartbeat)
    )[0]
    frequency_axis = frequency_axis[idx]
    signal_power = signal_power[idx]

    heart_rate = float(frequency_axis[torch.argmax(signal_power)])

    return heart_rate
