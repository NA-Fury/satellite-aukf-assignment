# ─────────────────────────── regen_clean.py ────────────────────────────
"""
GNSS pre-processing
───────────────────
* long → wide pivot
* km→m, dm s⁻¹→m s⁻¹
* 3 σ radius filter
* float32 Snappy parquet
* outputs  GPS_clean.parquet  +  meas_cols.json
"""
import json, hashlib, os, numpy as np, pandas as pd

RAW, CLEAN = "data/GPS_measurements.parquet", "GPS_clean.parquet"

# ── long → wide ---------------------------------------------------------
gps_long = pd.read_parquet(RAW).rename(columns={"datetime": "time"})
gps_wide = (gps_long
            .pivot_table(index="time", columns="ECEF",
                         values=["position", "velocity"])
            .reset_index())
gps_wide.columns = ["time"] + [f"{a}_{b}" for a, b in gps_wide.columns[1:]]

pos_cols = [c for c in gps_wide if c.startswith("position_")]
vel_cols = [c for c in gps_wide if c.startswith("velocity_")]

# ── unit conversions ----------------------------------------------------
df = gps_wide.copy()
df["time"]     = pd.to_datetime(df["time"]).dt.floor("s")
df[pos_cols]  *= 1_000.0        # km → m
df[vel_cols]  /=   10.0         # dm s⁻¹ → m s⁻¹

# ── 3-σ sanity filter ---------------------------------------------------
r_km = np.linalg.norm(df[pos_cols], axis=1) / 1_000.0
mask = np.abs((r_km - r_km.mean())/r_km.std()) <= 3
df   = df.loc[mask].reset_index(drop=True)
print(f"Kept {len(df):,}/{len(gps_wide):,} rows after 3σ filter")

# ── save ----------------------------------------------------------------
df[pos_cols + vel_cols] = df[pos_cols + vel_cols].astype("float32")
df.to_parquet(CLEAN, compression="snappy", index=False)
json.dump(pos_cols + vel_cols, open("meas_cols.json", "w"))

sha = hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:8]
sz  = round(os.path.getsize(CLEAN)/1e6,1)
print(f"✅  wrote {CLEAN}  ({sz} MB)  hash={sha}")
