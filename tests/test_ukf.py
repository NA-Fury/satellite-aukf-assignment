import numpy as np
from aukf import _sigma_points

def test_sigma_points_shape():
    x = np.zeros(6)
    P = np.eye(6)
    chi, Wm, Wc = _sigma_points(x, P, 1e-3, 2.0, 0.0)
    assert chi.shape == (2*6+1, 6)