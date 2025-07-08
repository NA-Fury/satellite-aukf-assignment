# tests/test_predict_update.py

import numpy as np
from aukf import AUKF

def cv_model(x: np.ndarray, dt: float) -> np.ndarray:
    # simple constant‐velocity model
    pos = x[:3] + x[3:] * dt
    return np.hstack((pos, x[3:]))

def test_perfect_measurement_overwrites_state():
    cols = ["px","py","pz","vx","vy","vz"]
    # q0=0 makes it “unitless” and r0 small just to satisfy signature
    ukf = AUKF(cols, cv_model, q0=0.0, r0=1e-6)

    # start at zero
    ukf.init_from_measurement(0.0, np.zeros(6))

    # predict does something (we don’t care here)
    ukf.predict(10.0)

    # feed a “perfect” measurement
    z = np.array([10.0, 20.0, -5.0, 1.5, -2.2, 0.0], dtype=float)
    ukf.update(z)

    # after a perfect measurement, state must exactly equal z
    assert np.allclose(ukf.x, z, atol=1e-8)