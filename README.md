# MuS2-SR dataset
MuS2-SR hyperspectral images super-resolution reconstruction evaluation dataset code.

## Prerequisites
Code was tested on Python.3.9 and Windows dataset. Install requirements within the previously created Python virtual 
environment with:

```sh
pip install -r requirements.txt
```
Please mind, that for GPU support you may need to change the PyTorch version and other associated with it packages, 
which will match you CuDNN version.

The Geospatial Data Abstraction Library (GDAL) Python package has to be installed separately. To do so for Windows 
first download the matching wheel file from:

```sh
https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
```

and later install it with:

```sh
pip install path-to-wheel-file.whl
```

## Usage

To run the MuS2-SR dataset builder run `python build_dataset --raw_data_path raw_data_path --out_data_path dataset` 
where `raw_data_path` is the path to the raw Sentinel-2 and WorldView data structured as it was shown in the original 
paper. For further help run `python evaluate -h`.


Tu run the evaluation process set up all evaluation parameters in `config.yaml` file and run 
`python evaluate --dataset_path dataset_path`. For further help run `python evaluate -h`.