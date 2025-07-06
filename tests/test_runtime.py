# tests/test_runtime.py

import time
import json, pandas as pd
from aukf import UnscentedKalman

def test_one_step_under_5ms():
    cols = json.load(open("meas_cols.json"))
    z = pd.read_parquet("GPS_clean.parquet")[cols].iloc[0].values
    ukf = UnscentedKalman(cols, q0=1e-2, r0=25.0)
    ukf.init_from_measurement(0.0, z)
    t0 = time.perf_counter()
    ukf.step(1.0, z)
    assert time.perf_counter() - t0 < 0.005
