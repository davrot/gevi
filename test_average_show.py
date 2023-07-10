from Anime import Anime
import matplotlib.pyplot as plt
import numpy as np

show_example_timeseries: bool = True
play_movie: bool = True

example_position_x: int = 280
example_position_y: int = 440
bin_size: int = 4
bin_size_post: int | None = None

data = np.load("result.npz")
result = data["result"]
mask = data["mask"]

example_position_x = example_position_x // bin_size
example_position_y = example_position_y // bin_size
if bin_size_post is not None:
    example_position_x = example_position_x // bin_size_post
    example_position_y = example_position_y // bin_size_post

if show_example_timeseries:
    plt.plot(result[:, example_position_x, example_position_y])
    plt.show()

if play_movie:
    ani = Anime()
    ani.show(
        result - 1.0, mask=mask, vmin_scale=0.5, vmax_scale=0.5
    )  # , vmin=0.98)  # , vmin=1.0, vmax_scale=1.0)
