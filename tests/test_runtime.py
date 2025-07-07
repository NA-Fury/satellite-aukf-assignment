# tests/test_runtime.py

import time
import numpy as np
from aukf import AUKF

def dummy_f(x: np.ndarray, dt: float) -> np.ndarray:
    return np.hstack((x[:3] + x[3:] * dt, x[3:]))

def test_step_under_5ms():
    ukf = AUKF(["pos_x","pos_y","pos_z","vel_x","vel_y","vel_z"],
               dummy_f, q0=0.0, r0=1.0)
    ukf.init_from_measurement(0.0, np.zeros(6))
    t0 = time.perf_counter()
    ukf.predict(1.0)
    ukf.update(np.zeros(6))
    assert time.perf_counter() - t0 < 0.005
