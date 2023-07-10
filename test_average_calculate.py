import torch
from DataContainer import DataContainer
import numpy as np
from tqdm import trange

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
threshold: float | None = 0.1  # Between 0 and 1.0


display_logging_messages: bool = False
save_logging_messages: bool = False

# Post data processing modifiations
gaussian_blur_kernel_size: int | None = 3
gaussian_blur_sigma: float = 1.0
bin_size_post: int | None = None

# ------------------------


torch_device: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

af = DataContainer(
    path=path,
    device=torch_device,
    display_logging_messages=display_logging_messages,
    save_logging_messages=save_logging_messages,
)


list_of_experiments = af.get_experiments()
print(
    f"The following experiments have been found:\n {list_of_experiments.cpu().numpy()}"
)
assert experiment_id in list_of_experiments
print(f"Continue with experiment: {experiment_id}")
list_of_trials = af.get_trials(experiment_id).cpu().numpy()
print(f"The following trials have been found:\n {list_of_trials}")


# mask
_, mask = af.automatic_load(
    experiment_id=experiment_id,
    trial_id=int(list_of_trials[0]),
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
    threshold=threshold,
)

if mask is not None:
    np.save("mask.npy", mask.cpu())


# data
result: torch.Tensor | None = None
count_not_nan: torch.Tensor | None = None

n: float = 0
for trial_id in trange(0, len(list_of_trials)):
    result_temp, _ = af.automatic_load(  # type: ignore
        experiment_id=experiment_id,
        trial_id=int(list_of_trials[trial_id]),
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
        threshold=None,
    )
    n += 1.0
    if result is None:
        result = (result_temp - 1.0).nan_to_num(nan=0.0)
        count_not_nan = torch.isfinite(result_temp).type(torch.float32)
    else:
        result += (result_temp - 1.0).nan_to_num(nan=0.0)
        count_not_nan += torch.isfinite(result_temp).type(torch.float32)

    assert result is not None
    if trial_id % 10 == 0:
        np.save("result.npy", (result / count_not_nan).cpu())
        np.save("count_not_nan.npy", (count_not_nan / n).cpu())

assert result is not None
assert count_not_nan is not None

np.save("result.npy", (result / count_not_nan).cpu())
np.save("count_not_nan.npy", (count_not_nan / n).cpu())
