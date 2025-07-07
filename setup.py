# setup.py
from setuptools import setup, find_packages

setup(
    name="aukf",
    version="0.1.0",
    description="Adaptive Unscented Kalman Filter for GNSS",
    author="Naziha Aslam",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "pandas>=2.0",
        "matplotlib",
        "pyarrow",
        "pint",
    ],
    python_requires=">=3.8,<3.12",
)
