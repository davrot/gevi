import torch
import matplotlib.pyplot as plt


from functions.preprocessing import preprocessing
from functions.bandpass import bandpass


if __name__ == "__main__":

    if torch.cuda.is_available():
        device_name: str = "cuda:0"
    else:
        device_name = "cpu"
    print(f"Using device: {device_name}")
    device: torch.device = torch.device(device_name)

    filename_metadata: str = "raw/Exp001_Trial001_Part001_meta.txt"
    filename_data: str = "Exp001_Trial001_Part001.mat"
    filename_mask: str = "2020-12-08maskPixelraw2.mat"

    first_none_ramp_frame: int = 100
    spatial_width: float = 2
    temporal_width: float = 0.1

    lower_freqency_bandpass: float = 5.0
    upper_freqency_bandpass: float = 14.0

    target_camera: list[str] = ["acceptor", "donor"]
    regressor_cameras: list[str] = ["oxygenation", "volume"]

    ratio_sequence_a, ratio_sequence_b, mask = preprocessing(
        filename_metadata=filename_metadata,
        filename_data=filename_data,
        filename_mask=filename_mask,
        device=device,
        first_none_ramp_frame=first_none_ramp_frame,
        spatial_width=spatial_width,
        temporal_width=temporal_width,
        target_camera=target_camera,
        regressor_cameras=regressor_cameras,
    )

    ratio_sequence_a = bandpass(
        data=ratio_sequence_a,
        device=ratio_sequence_a.device,
        low_frequency=lower_freqency_bandpass,
        high_frequency=upper_freqency_bandpass,
        fs=100.0,
        filtfilt_chuck_size=10,
    )

    ratio_sequence_b = bandpass(
        data=ratio_sequence_b,
        device=ratio_sequence_b.device,
        low_frequency=lower_freqency_bandpass,
        high_frequency=upper_freqency_bandpass,
        fs=100.0,
        filtfilt_chuck_size=10,
    )

    original_shape = ratio_sequence_a.shape

    ratio_sequence_a = ratio_sequence_a.flatten(start_dim=0, end_dim=-2)
    ratio_sequence_b = ratio_sequence_b.flatten(start_dim=0, end_dim=-2)

    mask = mask.flatten(start_dim=0, end_dim=-1)
    ratio_sequence_a = ratio_sequence_a[mask, :]
    ratio_sequence_b = ratio_sequence_b[mask, :]

    ratio_sequence_a = ratio_sequence_a.movedim(0, -1)
    ratio_sequence_b = ratio_sequence_b.movedim(0, -1)

    ratio_sequence_a -= ratio_sequence_a.mean(dim=0, keepdim=True)
    ratio_sequence_b -= ratio_sequence_b.mean(dim=0, keepdim=True)

    u_a, s_a, Vh_a = torch.linalg.svd(ratio_sequence_a, full_matrices=False)
    u_a = u_a[:, 0]
    s_a = s_a[0]
    Vh_a = Vh_a[0, :]

    heartbeatactivitmap_a = torch.zeros(
        (original_shape[0], original_shape[1]), device=Vh_a.device, dtype=Vh_a.dtype
    ).flatten(start_dim=0, end_dim=-1)

    heartbeatactivitmap_a *= torch.nan
    heartbeatactivitmap_a[mask] = s_a * Vh_a
    heartbeatactivitmap_a = heartbeatactivitmap_a.reshape(
        (original_shape[0], original_shape[1])
    )

    u_b, s_b, Vh_b = torch.linalg.svd(ratio_sequence_b, full_matrices=False)
    u_b = u_b[:, 0]
    s_b = s_b[0]
    Vh_b = Vh_b[0, :]

    heartbeatactivitmap_b = torch.zeros(
        (original_shape[0], original_shape[1]), device=Vh_b.device, dtype=Vh_b.dtype
    ).flatten(start_dim=0, end_dim=-1)

    heartbeatactivitmap_b *= torch.nan
    heartbeatactivitmap_b[mask] = s_b * Vh_b
    heartbeatactivitmap_b = heartbeatactivitmap_b.reshape(
        (original_shape[0], original_shape[1])
    )

    plt.subplot(2, 1, 1)
    plt.plot(u_a.cpu(), label="aceptor")
    plt.plot(u_b.cpu(), label="donor")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.imshow(
        torch.cat(
            (
                heartbeatactivitmap_a,
                heartbeatactivitmap_b,
            ),
            dim=1,
        ).cpu()
    )
    plt.colorbar()
    plt.show()
