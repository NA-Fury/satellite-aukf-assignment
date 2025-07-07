# tests/test_sigma_points.py

import numpy as np
from aukf import _sigma_points

def test_sigma_point_mean_recovery():
    """Unscented transform must reproduce the mean exactly."""
    n = 3
    x = np.arange(n, dtype=float)
    P = np.eye(n)
    χ, Wm, _ = _sigma_points(x, P)
    x_rec = χ.T @ Wm
    assert np.allclose(x, x_rec), "UT mean mismatch"
