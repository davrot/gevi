# %%
import numpy as np
import matplotlib.pyplot as plt
import os

output_path = 'output'

recording_name = 'M0134M_2024-12-04_SessionA'
n_trials_per_experiment = [30, 0, 30, 30, 30, 30, 30, 30, 30,]
name_experiment = ['none', 'visual', '2 uA', '5 uA', '7 uA', '10 uA', '15 uA', '30 uA', '60 uA']

#   recording_name = 'M0134M_2024-11-06_SessionB'
#   n_trials_per_experiment = [15, 15,]
#   name_experiment = ['none', 'visual',]

i_experiment = 8

r_avg = None
ad_avg = None
for i_trial in range(n_trials_per_experiment[i_experiment]):

    folder = output_path + os.sep + recording_name 
    file = f"Exp{i_experiment + 1:03}_Trial{i_trial + 1:03}_ratio_sequence.npz"
    fullpath = folder + os.sep + file

    print(f'Loading file "{fullpath}"...')
    data = np.load(fullpath)

    print(f"FIle contents: {data.files}")
    ratio_sequence = data["ratio_sequence"]
    if 'data_acceptor' in data.files:
        data_acceptor = data["data_acceptor"]
        data_donor = data["data_donor"]

    mask = data["mask"][:, :, np.newaxis]

    if i_trial == 0:
        r_avg = ratio_sequence
        if 'data_acceptor' in data.files:
            ad_avg = (data_acceptor / data_donor) * mask + 1 - mask
    else:
        r_avg += ratio_sequence
        if 'data_acceptor' in data.files:
            ad_avg += (data_acceptor / data_donor) * mask + 1 - mask

if r_avg is not None:
    r_avg /= n_trials_per_experiment[i_experiment]
if ad_avg is not None:
    ad_avg /= n_trials_per_experiment[i_experiment]

# %%
for t in range(200, 300, 5):
    plt.imshow(r_avg[:, :, t], vmin=0.99, vmax=1.01, cmap='seismic')
    plt.colorbar()
    plt.show()

