# gevi

## Main files:

Anime.py : Is a class used for displaying 2d Movies from a 3D Matrix.

ImageAlignment.py: Is a class used for aligning two images (move and rotate as well as an unused scale option). This source code is based on https://github.com/matejak/imreg_dft . Is was ported to PyTorch such it can run in GPUs.

DataContailer.py: Main class for data pre-processing of the gevi raw data.  

## TODO / Known problems

Support for several part files was included but, due to missing data with several part files, never tested. 

## Installation 

The code was tested on a Python 3.11.2 (Linux) with the following pip packages installed:

numpy scipy pandas flake8 pep8-naming black matplotlib seaborn ipython jupyterlab mypy dataclasses-json dataconf mat73 ipympl torch torchtext pywavelets scikit-image opencv-python scikit-learn tensorflow_datasets tensorboard tqdm argh sympy jsmin pybind11 pybind11-stubgen pigar asciichartpy torchvision torchaudio tensorflow natsort roipoly 

Not all packages are necessary (probably these are enougth: torch torchaudio torchvision roipoly natsort numpy matplotlib) but this is our default in-house installation plus roipoly. 

We used a RTX 3090 as test GPU. 


## Data processing chain

### SVD (requires donor and acceptor time series)

- start automatic_load
  - try to load previous mask
  - start cleaned_load_data
    - start load_data
    - work in XXXX.npy
      - np.load
      - organize acceptor (to GPU memory)
      - organize donor (to GPU memory)
      - move axis (move the time axis of the tensor)
      - move intra timeseries
        - donor time series and donor reference image
        - acceptor time series and acceptor reference image
      - rotate inter timeseries
      - move inter timeseries
    - spatial pooling
    - data(x,y,t) = data(x,y,t) / data(x,y,t).mean(t) + 1
    - remove the heart beat via SVD
    - remove mean
    - remove linear trends
  - remove heart beat (heartbeat_scale)
    - apply bandpass donor_residuum (filtfilt)
    - apply bandpass acceptor_residuum (filtfilt)
    - calculate mask (optinal)
  - don't use regression
  - scale acceptor signal (result_a(x,y,t)) and donor signal (result_d(x,y,t))
  - result(x,y,t) = 1.0 + result_a(x,y,t) - result_d(x,y,t)
  - update inital mask
- end automatic_load

### Classic (requires donor, acceptor, volume, and oxygenation time series)

- start automatic_load
    - try to load previous mask
    - start cleaned_load_data
        - start load_data
            - work in XXXX.npy
            - np.load
            - organize acceptor (to GPU memory)
            - organize donor (to GPU memory)
            - organize oxygenation (to GPU memory)
            - organize volume (to GPU memory)
            - move axis (move the time axis of the tensor)
            - move intra timeseries
              - donor time series and donor reference image; transformation also used on volume
              - acceptor time series and acceptor reference image; transformation also used on oxygenation
            - rotate inter timeseries
            - move inter timeseries
        - spatial pooling
        - data(x,y,t) = data(x,y,t) / data(x,y,t).mean(t) + 1
        - frame shift
    - measure heart rate (measure_heartbeat_frequency)
    - use "regression" (i.e. iterative non-orthogonal basis decomposition)
    - donor: measure heart beat spectral power (measure_heartbeat_power)
    - acceptor: measure heart beat spectral power (measure_heartbeat_power)
    - scale acceptor and donor signals
    - result(x,y,t) = 1.0 + result_a(x,y,t) - result_d(x,y,t)
- end automatic_load
