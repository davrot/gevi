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

For installing torch under Windows see here: https://pytorch.org/get-started/locally/ 


## Data processing chain

### SVD (requires donor and acceptor time series) remove_heartbeat: bool = True

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
    - spatial pooling (i.e. 2d average pooling layer; optional)
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
  - gauss blur (optional)
  - spatial binning (optional)
  - update inital mask (optional)
- end automatic_load

### Classic (requires donor, acceptor, volume, and oxygenation time series) remove_heartbeat: bool = False

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
        - spatial pooling (i.e. 2d average pooling layer; optional)
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
    - gauss blur (optional)
    - spatial binning (optional)
- end automatic_load

## DataContailer.py

### Constructor

    def __init__(
        self,
        path: str, # Path to the raw data
        device: torch.device, # e.g. torch.device("cpu") or torch.device("cuda:0")
        display_logging_messages: bool = False, # displays log messages on screen
        save_logging_messages: bool = False, # writes log messages into a log file
    ) -> None:

### automatic_load
    def automatic_load(  
        self,
        experiment_id: int = 1, # number of experiment
        trial_id: int = 1, # number of the trial
        start_position: int = 0, # number of frames cut away from the beginning of the time series
        start_position_coefficients: int = 100, # number of frames ignored for calculating scaling factors and such
        fs: float = 100.0, # sampling rate in Hz
        use_regression: bool | None = None, # controls the "regression"; None is automatic mode. Needs all four time series!
        remove_heartbeat: bool = True,  # i.e. use SVD=True or Classic=False
        low_frequency: float = 5,  # Hz Butter Bandpass Heartbeat
        high_frequency: float = 15,  # Hz Butter Bandpass Heartbeat
        threshold: float | None = 0.5,  # threshold for the mask; None is no mask. valid values are between 0.0 and 1.0.
        align: bool = True, # align the time series after loading them 
        iterations: int = 1,  # SVD iterations: Do not touch! Keep at 1
        lowrank_method: bool = True, # saves computational time if we know that we don't need all SVD values
        lowrank_q: int = 6, # parameter for the low rank method. need to be at least 2-3x larger than the number of SVD values we want
        remove_heartbeat_mean: bool = False, # allows us to remove a offset from the SVD heart signals (don't need that because of a bandpass filter)
        remove_heartbeat_linear: bool = False, # allows us to remove a linear treand from the SVD heart signals (don't need that because of a bandpass filter)
        bin_size: int = 4, # size of the kernel of the first 2d average pooling layer 
        do_frame_shift: bool = True, # Do the frame shift or not. 
        half_width_frequency_window: float = 3.0,  # Hz (on side ) measure_heartbeat_frequency
        mmap_mode: bool = True, # controls the np.load 
        initital_mask_name: str | None = None, # allows to store the map into a file (give filename here or None if you don't want to save it)
        initital_mask_update: bool = True, # the mask is updated with new information from this trial
        initital_mask_roi: bool = False, # enables a tool to refine the automatic map
        gaussian_blur_kernel_size: int | None = 3, # parameter of a gauss blur layer: kernel_size
        gaussian_blur_sigma: float = 1.0, # parameter of a gauss blur layer: sigma
        bin_size_post: int | None = None, # size of the kernel of the second 2d average pooling layer 
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
