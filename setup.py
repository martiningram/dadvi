from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="dadvi",
    version=getenv("VERSION", "LOCAL"),
    description="Deterministic ADVI",
    packages=find_packages(),
    install_requires=["numpyro", "pymc>=4", "scikit-learn", "toolz"],
    extras_require={
        "viabel": [
            "pystan==2.19.1.1",
        ],
    },
)
