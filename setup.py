from setuptools import setup
from setuptools import find_packages


setup(
    name="dadvi",
    version="0.0.1",
    description="Deterministic ADVI",
    packages=find_packages(),
    install_requires=["numpyro", "pymc>=4", "scikit-learn", "toolz", "dill"],
    extras_require={
        "viabel": [
            "pystan==2.19.1.1",
        ],
    },
)
