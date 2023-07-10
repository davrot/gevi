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
  - start: cleaned_load_data
    - start: load_data
    - work on XXXX.npy
      - np.load
      - organize acceptor (move to GPU memory)
      - organize donor (move to GPU memory)
      - move axis (move the time axis of the tensor)
      - move intra timeseries
        - donor time series and donor reference image
        - acceptor time series and acceptor reference image
      - rotate inter timeseries
        - acceptor time series and donor reference image
      - move inter timeseries
        - acceptor time series and donor reference image
    - spatial pooling (i.e. 2d average pooling layer)
        - acceptor(x,y,t) = acceptor(x,y,t) / acceptor(x,y,t).mean(t) + 1
        - donor(x,y,t) = donor(x,y,t) / donor(x,y,t).mean(t) + 1
    - remove the heart beat via SVD from donor and acceptor
      - copy donor and acceptor and work on the copy with the SVD 
      - remove the mean (over time)
      - use Cholesky whitening on data with SVD
      - scale the time series accoring the spatial whitening
      - average time series over the spatial dimension (which is the global heart beat)
      - use a normalized scalar product for getting spatial scaling factors
      - scale the heartbeat with the spatial scaling factors into donor_residuum and acceptor_residuum
      - store the heartbeat as well as substract it from the original donor and acceptor timeseries
    - remove mean from donor and acceptor timeseries (- mean over time)
    - remove linear trends from donor and acceptor timeseries (create a linear function and use a normalized scalar product for getting spatial scaling factors)
  - use the SVD heart beat for determining the scaling factors for donor and acceptor (heartbeat_scale)
    - apply bandpass donor_residuum (filtfilt)
    - apply bandpass acceptor_residuum (filtfilt)
    - a normalized scalar product is used to determine the scale factor scale(x,y) between donor_residuum(x,y,t) and acceptor_residuum(x,y,t)
    - calculate mask (optional) ; based on the heart beat power at the spatial positions
  - scale acceptor signal (heartbeat_scale_a(x,y) * result_a(x,y,t)) and donor signal (heartbeat_scale_d(x,y) * result_d(x,y,t))
    - heartbeat_scale_a = torch.sqrt(scale)
    - heartbeat_scale_d = 1.0 / (heartbeat_scale_a + 1e-20)
  - result(x,y,t) = 1.0 + result_a(x,y,t) - result_d(x,y,t)
  - update inital mask (optional)
- end automatic_load

### Classic (requires donor, acceptor, volume, and oxygenation time series)

- start automatic_load
    - try to load previous mask
    - start cleaned_load_data
        - start load_data
          - work on XXXX.npy
            - np.load (load one trial)
            - organize acceptor (move to GPU memory)
            - organize donor (move to GPU memory)
            - organize oxygenation (move to GPU memory)
            - organize volume (move to GPU memory)
            - move axis (move the time axis of the tensor)
            - move intra timeseries
              - donor time series and donor reference image; transformation also used on volume
              - acceptor time series and acceptor reference image; transformation also used on oxygenation
            - rotate inter timeseries
              - acceptor time series and donor reference image; transformation also used on volume
            - move inter timeseries
              - acceptor time series and donor reference image; transformation also used on volume
        - spatial pooling (i.e. 2d average pooling layer)
        - acceptor(x,y,t) = acceptor(x,y,t) / acceptor(x,y,t).mean(t) + 1
        - donor(x,y,t) = donor(x,y,t) / donor(x,y,t).mean(t) + 1
        - oxygenation(x,y,t) = oxygenation(x,y,t) / oxygenation(x,y,t).mean(t) + 1
        - volume(x,y,t) = volume(x,y,t) / volume(x,y,t).mean(t) + 1
        - frame shift
          - the first frame of donor and acceptor time series is dropped
          - the oxygenation and volume time series are interpolated between two frames (to compensate for the 5ms delay)
    - measure heart rate (measure_heartbeat_frequency) i.e. find the frequency f_HB(x,y) with the highest power in the frequency band in the volume signal
    - use "regression" (i.e. iterative non-orthogonal basis decomposition); remove offset, linear trend, oxygenation and volume timeseries
    - donor: measure heart beat spectral power (measure_heartbeat_power) f_HB(x,y) +/- 3Hz; results in power_d(x,y)
    - acceptor: measure heart beat spectral power (measure_heartbeat_power) f_HB(x,y) +/- 3Hz ; results in power_a(x,y)
    - scale acceptor and donor signals via the powers
      - scale(x,y) = power_d(x,y) / (power_a(x,y) + 1e-20)
      - heartbeat_scale_a = torch.sqrt(scale)
      - heartbeat_scale_d = 1.0 / (heartbeat_scale_a + 1e-20)
    - result(x,y,t) = 1.0 + result_a(x,y,t) - result_d(x,y,t)
- end automatic_load
