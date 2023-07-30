from functions.Anime import Anime
import matplotlib.pyplot as plt
import numpy as np

show_example_timeseries: bool = True
play_movie: bool = True

example_position_x: int = 280
example_position_y: int = 440
bin_size: int = 4
bin_size_post: int | None = None

result = np.load("result.npy")
mask = np.load("mask.npy")
count_not_nan = np.load("count_not_nan.npy")

# plt.imshow(count_not_nan.mean(axis=0))
# plt.colorbar()
# plt.title("Not nan")
# plt.show()
# exit()

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
    ani.show(result, mask=mask, vmin_scale=0.01, vmax_scale=0.01)
