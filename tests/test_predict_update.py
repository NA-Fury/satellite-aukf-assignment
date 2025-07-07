# tests/test_predict_update.py
"""
Predict-update sanity check with a *linear* CV model so we have ground truth.
"""
import numpy as np
from aukf import AUKF

def cv_model(x: np.ndarray, dt: float) -> np.ndarray:
    pos = x[:3] + x[3:] * dt
    return np.hstack((pos, x[3:]))

def test_single_step_linear():
    cols = ["px","py","pz","vx","vy","vz"]  # dummy
    ukf = AUKF(cols, cv_model, q0=0.0, r0=1e-2)
    ukf.init_from_measurement(0.0, np.zeros(6))

    # truth state
    x_true = np.array([10., 0., 0., 1., 0., 0.])  # after 10 s
    z_meas = x_true.copy()                        # perfect meas

    ukf.predict(dt=10.0)
    ukf.update(z_meas)

    err = np.abs(ukf.x - x_true)
    assert err.max() < 1e-3, "Predict/update failed on linear model"
