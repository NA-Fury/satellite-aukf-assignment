# tests/test_predict_update.py

import numpy as np
from aukf import AUKF

def cv_model(x: np.ndarray, dt: float) -> np.ndarray:
    # simple constant‐velocity model
    pos = x[:3] + x[3:] * dt
    return np.hstack((pos, x[3:]))