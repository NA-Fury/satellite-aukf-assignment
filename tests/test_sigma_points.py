import numpy as np
from aukf import _sigma_points

def test_weights_sum():
    x  = np.zeros(6)
    P  = np.eye(6)
    χ, Wm, Wc = _sigma_points(x, P, 1e-3, 2.0, 0.0)
    assert np.isclose(Wm.sum(), 1.0)
    assert np.isclose(Wc.sum(), 4.0)      # see note in docs
