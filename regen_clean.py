# regen_clean.py  ── run once ──────────────────────────────────────────────
import os, json, pandas as pd, numpy as np, pathlib

RAW = "GPS_measurements.parquet"
CLEAN = "GPS_clean.parquet"

# 1) long → wide
gps_long = (pd.read_parquet(RAW)
              .rename(columns={"datetime": "time"}))

gps_wide = (gps_long
            .pivot_table(index="time", columns="ECEF",
                         values=["position", "velocity"])
            .reset_index())
gps_wide.columns = ["time"] + [
    f"{c[0]}_{c[1]}" for c in gps_wide.columns[1:]
]
pos_cols = [c for c in gps_wide if c.startswith("position_")]
vel_cols = [c for c in gps_wide if c.startswith("velocity_")]

# 2) conversions
df = gps_wide.copy()
df["time"] = pd.to_datetime(df["time"]).dt.floor("s")
df[pos_cols] *= 1_000          # km → m
df[vel_cols] /= 10             # dm/s → m/s

# 3) 3-σ filter
r_km = np.linalg.norm(df[pos_cols].values, axis=1) / 1_000
mask = np.abs((r_km - r_km.mean()) / r_km.std()) <= 3
df_clean = df[mask].reset_index(drop=True)

# 4) float32 + Brotli-11
for c in df_clean.select_dtypes("float64"):
    df_clean[c] = df_clean[c].astype("float32")

df_clean.to_parquet(
    CLEAN, compression="brotli", compression_level=11, index=False
)
print("✅  wrote", CLEAN,
      round(os.path.getsize(CLEAN)/1e6,1), "MB",
      "rows:", len(df_clean))

# 5) measurement columns
with open("meas_cols.json", "w") as f:
    json.dump(pos_cols + vel_cols, f)
print("✅  wrote meas_cols.json")
