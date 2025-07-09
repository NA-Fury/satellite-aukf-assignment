"""
satellite_aukf package
~~~~~~~~~~~~~~~~~~~~~~

Adaptive Unscented Kalman Filter utilities for GNSS-based satellite
tracking.

The top-level package re-exports the two objects that external code and
unit tests need most often.
"""

from __future__ import annotations

from .aukf import AdaptiveUKF, FilterParameters  # noqa: F401  (re-export)
from .utils import (  # handy helpers that people might want
    load_and_preprocess_gnss_data,
    measurement_model,
    simple_ecef_to_eci,
)

__all__ = [
    "AdaptiveUKF",
    "FilterParameters",
    "load_and_preprocess_gnss_data",
    "measurement_model",
    "simple_ecef_to_eci",
]
