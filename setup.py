# setup.py
from setuptools import setup, find_packages

setup(
    name="satellite_aukf_assignment",
    version="0.1.0",
    packages=find_packages(),  # will find the `aukf` package
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib", "pytest", "pyarrow", "pint"
    ],
)
