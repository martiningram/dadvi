# DADVI: Deterministic ADVI

### Getting started

To install this repository, please first install `pymc`. The suggested way of doing this is to run (see [here](https://www.pymc.io/projects/docs/en/latest/installation.html)):

```
conda create -c conda-forge -n dadvi "pymc>=5" bambi
conda activate dadvi
```

Note that `bambi` is optional and only needed for some models.

You can then install the package using

```
pip install -e .
```

To see how to use the package with PyMC, please take a look at the notebook
"Radon example.ipynb" in the jupyter folder.
