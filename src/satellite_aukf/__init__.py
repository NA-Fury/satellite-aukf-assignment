"""Public top-level API for the *satellite_aukf* package."""

# src/satellite_aukf/__init__.py
from importlib import metadata as _metadata

# configuration
from . import config

# core filter + enums
from .aukf import AdaptiveMethod, AdaptiveUKF, AUKFParameters

# utilities
from .utils import (
    CoordinateTransforms,
    DataPreprocessor,
    FilterTuning,
    OrbitPropagator,
    ecef_to_eci,
    ecef_to_eci_simple,
    eci_to_ecef,
    measurement_model,
    motion_model_ecef,
    save_figure,
    simple_ecef_to_eci,
)

# Back-compat alias: FilterParameters → AUKFParameters
FilterParameters = AUKFParameters

__all__ = [
    # filter
    "AdaptiveUKF",
    "AUKFParameters",
    "AdaptiveMethod",
    "FilterParameters",
    # utils
    "CoordinateTransforms",
    "OrbitPropagator",
    "DataPreprocessor",
    "FilterTuning",
    "measurement_model",
    "eci_to_ecef",
    "ecef_to_eci",
    "ecef_to_eci_simple",
    "simple_ecef_to_eci",
    "motion_model_ecef",
    "save_figure",
    # config
    "config",
]

# package version helper -------------------------------------------------------
try:
    __version__ = _metadata.version("satellite_aukf")
except _metadata.PackageNotFoundError:  # dev / editable install
    __version__ = "0.0.0.dev0"
