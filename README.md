# DADVI: Deterministic ADVI

### Getting started

To install this repository, please first install `pymc`. The suggested way of doing this is to run (see [here](https://www.pymc.io/projects/docs/en/latest/installation.html)):

```
conda create -c conda-forge -n pymc_env "pymc>=4" python=3.8 bambi
conda activate pymc_env
```

Note that `python=3.8` is required only if you'd like to use the `viabel` functionality.

You can then install the package using

```
pip install -e .
```

which will install the remaining dependencies. If you'd like to also install the dependencies for `viabel`, you can run:

```
pip install -e .[viabel]
```

The latest version of `viabel`, which includes `RAABBVI`, is not yet on `PyPI`. Please install it by cloning https://github.com/jhuggins/viabel and running `pip install -e .` if you would like to use it.

As of today (12th September 2022), running the POTUS model requires update `pymc` code that has not yet been merged back into `main`. To run the `POTUS` model, please clone the `pymc` environment and run `pip install -e .` in the main directory to install the latest version.