# data_utils.py
import pandas as pd
def load_clean():
    """Return the pre-processed GPS dataframe (positions/velocities in m, m/s)."""
    return pd.read_parquet("GPS_clean.parquet")