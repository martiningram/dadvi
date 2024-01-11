# DADVI: Deterministic ADVI


### Getting started using Docker

After installing [Docker](https://docs.docker.com/get-docker/), run the following commands in the
root of the present repository.

```
docker build --tag 'dadvi_paper' .
```

Then you can run

```
docker run -it --platform linux/amd64 dadvi_paper /bin/bash
```

and, at the prompt, type

```
conda activate dadvi
```

This should put you in a conda environment with the `dadvi` package and its dependencies
successfully installed.


### Getting started using conda

To install this repository, please first install `pymc`. The suggested way of doing this 
(see [here](https://www.pymc.io/projects/docs/en/latest/installation.html)) is to run:

```
conda create -c conda-forge -n dadvi "pymc>=5" bambi
conda activate dadvi
```

Note that `bambi` is optional and only needed for some models.

You can then install the package using

```
pip install -e .
```


### Running unit tests

To run the tests, run the following command in the root of the repository:

```
python3 tests/run_tests.sh
```


### Using the package

To see how to use the package with PyMC, please take a look at the notebook
"Radon example.ipynb" in the jupyter folder.

To run the experiments from our paper,  "Black Box Variational Inference with a
Deterministic Objective: Faster, More Accurate, and Even More Black Box", by
Giordano, Ingram and Broderick, please look at 
[github.com/martiningram/dadvi-experiments](https://github.com/martiningram/dadvi-experiments).