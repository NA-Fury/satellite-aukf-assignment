import numpy as np, pytest
from aukf import UnscentedKalman, _sigma_points

def test_sigma_pts_dimension():
    x = np.zeros(6); P = np.eye(6)
    chi, Wm, Wc = _sigma_points(x, P, 1e-3, 2.0, 0.0)
    assert chi.shape == (13, 6)
    assert np.isclose(Wm.sum(), 1.0)