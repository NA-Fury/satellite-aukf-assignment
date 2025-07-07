# data_utils.py

import pandas as pd

def load_clean() -> pd.DataFrame:
    """Return the pre-processed GPS dataframe (metres & m/s)."""
    return pd.read_parquet("GPS_clean.parquet")
