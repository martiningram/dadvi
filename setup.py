from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="dadvi",
    version=getenv("VERSION", "LOCAL"),
    description="Deterministic ADVI",
    packages=find_packages(),
)
