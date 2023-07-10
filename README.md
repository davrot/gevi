# gevi


## TODO / Known problems

Support for several part files was included but, due to missing data with several part files, never tested. 

The code was tested on a Python 3.11.2 (Linux) with the following pip packages installed:

numpy scipy pandas flake8 pep8-naming black matplotlib seaborn ipython jupyterlab mypy dataclasses-json dataconf mat73 ipympl torch torchtext pywavelets scikit-image opencv-python scikit-learn tensorflow_datasets tensorboard tqdm argh sympy jsmin pybind11 pybind11-stubgen pigar asciichartpy torchvision torchaudio tensorflow natsort roipoly 

Not all packages are necessary (probably these are enougth: torch torchaudio torchvision roipoly natsort numpy matplotlib) but this is our default in-house installation plus roipoly. 

We used a RTX 3090 as test GPU. 
