import numpy as np
import matplotlib.pyplot as plt
import os

export_path: str = "Export"
experiment_id: int = 1
trial_id: int = 1
example_position_x: int = 280
example_position_y: int = 440
bin_size: int = 4

svd_1 = np.load(os.path.join(export_path, f"SVD_{experiment_id}_{trial_id}_data.npy"))
classic = np.load(
    os.path.join(export_path, f"Classic_{experiment_id}_{trial_id}_data.npy")
)


example_position_x = example_position_x // bin_size
example_position_y = example_position_y // bin_size


dim = 2
plt.subplot(dim, 1, 1)
plt.plot(svd_1[:, example_position_x, example_position_y] - 1.0)
plt.title("SVD - 1.0")

plt.subplot(dim, 1, 2)
plt.plot(classic[:, example_position_x, example_position_y] - 1.0)
plt.title("Classic - 1.0")

plt.show()
