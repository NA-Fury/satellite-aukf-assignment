# tests/test_clean_parquet.py

import pandas as pd

def test_parquet_load():
    df = pd.read_parquet("GPS_clean.parquet")
    assert 1_400_000 < len(df) < 1_500_000
    assert 6.0e6 < df["position_x"].abs().max() < 7.0e6
