import torchaudio as ta  # type: ignore
import torch


@torch.no_grad()
def filtfilt(
    input: torch.Tensor,
    butter_a: torch.Tensor,
    butter_b: torch.Tensor,
) -> torch.Tensor:
    assert butter_a.ndim == 1
    assert butter_b.ndim == 1
    assert butter_a.shape[0] == butter_b.shape[0]

    process_data: torch.Tensor = input.detach().clone()

    padding_length = 12 * int(butter_a.shape[0])
    left_padding = 2 * process_data[..., 0].unsqueeze(-1) - process_data[
        ..., 1 : padding_length + 1
    ].flip(-1)
    right_padding = 2 * process_data[..., -1].unsqueeze(-1) - process_data[
        ..., -(padding_length + 1) : -1
    ].flip(-1)
    process_data_padded = torch.cat((left_padding, process_data, right_padding), dim=-1)

    output = ta.functional.filtfilt(
        process_data_padded.unsqueeze(0), butter_a, butter_b, clamp=False
    ).squeeze(0)

    output = output[..., padding_length:-padding_length]
    return output


@torch.no_grad()
def butter_bandpass(
    device: torch.device,
    low_frequency: float = 0.1,
    high_frequency: float = 1.0,
    fs: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    import scipy  # type: ignore

    butter_b_np, butter_a_np = scipy.signal.butter(
        4, [low_frequency, high_frequency], btype="bandpass", output="ba", fs=fs
    )
    butter_a = torch.tensor(butter_a_np, device=device, dtype=torch.float32)
    butter_b = torch.tensor(butter_b_np, device=device, dtype=torch.float32)
    return butter_a, butter_b


@torch.no_grad()
def chunk_iterator(array: torch.Tensor, chunk_size: int):
    for i in range(0, array.shape[0], chunk_size):
        yield array[i : i + chunk_size]


@torch.no_grad()
def bandpass(
    data: torch.Tensor,
    low_frequency: float = 0.1,
    high_frequency: float = 1.0,
    fs=30.0,
    filtfilt_chuck_size: int = 10,
) -> torch.Tensor:

    try:
        return bandpass_internal(
            data=data,
            low_frequency=low_frequency,
            high_frequency=high_frequency,
            fs=fs,
            filtfilt_chuck_size=filtfilt_chuck_size,
        )

    except torch.cuda.OutOfMemoryError:

        return bandpass_internal(
            data=data.cpu(),
            low_frequency=low_frequency,
            high_frequency=high_frequency,
            fs=fs,
            filtfilt_chuck_size=filtfilt_chuck_size,
        ).to(device=data.device)


@torch.no_grad()
def bandpass_internal(
    data: torch.Tensor,
    low_frequency: float = 0.1,
    high_frequency: float = 1.0,
    fs=30.0,
    filtfilt_chuck_size: int = 10,
) -> torch.Tensor:
    butter_a, butter_b = butter_bandpass(
        device=data.device,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        fs=fs,
    )

    index_full_dataset: torch.Tensor = torch.arange(
        0, data.shape[1], device=data.device, dtype=torch.int64
    )

    for chunk in chunk_iterator(index_full_dataset, filtfilt_chuck_size):
        temp_filtfilt = filtfilt(
            data[:, chunk, :],
            butter_a=butter_a,
            butter_b=butter_b,
        )
        data[:, chunk, :] = temp_filtfilt

    return data
